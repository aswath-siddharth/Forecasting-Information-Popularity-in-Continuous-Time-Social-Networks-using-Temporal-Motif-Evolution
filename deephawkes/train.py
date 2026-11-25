"""Train and evaluate DeepHawkes under the exact controlled Weibo protocol.

Controlled invariants (held identical to CasMotif):
  - Data splits: chronological 70/15/15 aligned to weibo_{train,val,test}.npz
  - Target: future increment Delta P = R_24h - R_{T_obs}, transformed as log2(Delta P + 1)
  - Loss & primary metric: MSE in log2 space (= MSLE_log2)
  - Batch size: 64
  - Optimizer: AdamW(lr=1e-3, weight_decay=1e-4)
  - LR schedule: ReduceLROnPlateau(factor=0.5, patience=3)
  - Gradient clipping: max_norm=1.0
  - Early stopping: patience=8, max_epochs=60
  - Model selection: restore checkpoint with lowest validation MSLE
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pipeline_w.train_weibo import regression_metrics

from .model import DeepHawkes


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DeepHawkesDataset(Dataset):
    """Loads compact flat path arrays produced by ``deephawkes_data.py``."""

    def __init__(self, npz_path: Path) -> None:
        d = np.load(npz_path, allow_pickle=True)
        self.cascade_id: list[str] = [str(x) for x in d["cascade_id"].tolist()]
        self.y_raw: np.ndarray = d["y_increment"].astype(np.float32)
        self.cascade_path_offsets: np.ndarray = d["cascade_path_offsets"]
        self.path_node_offsets: np.ndarray = d["path_node_offsets"]
        self.node_ids: np.ndarray = d["node_ids"]
        self.path_time_bins: np.ndarray = d["path_time_bins"]
        self.y_log: np.ndarray = (np.log(self.y_raw + 1.0) / np.log(2.0)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.cascade_id)

    def __getitem__(self, idx: int) -> tuple[int, np.float32, np.float32]:
        return idx, self.y_log[idx], self.y_raw[idx]


def make_collate(ds: DeepHawkesDataset):
    """Assemble a batch of cascades directly from the flat numpy buffers."""
    c_off = ds.cascade_path_offsets
    p_off = ds.path_node_offsets
    nodes = ds.node_ids
    tbins = ds.path_time_bins

    def collate(items):
        idxs, y_logs, y_raws = zip(*items)
        n_cascades = len(idxs)

        # Count total paths and max path length in this batch
        total_paths = 0
        max_len = 1
        for ci in idxs:
            ps, pe = int(c_off[ci]), int(c_off[ci + 1])
            total_paths += pe - ps
            lens = p_off[ps + 1 : pe + 1] - p_off[ps:pe]
            if len(lens) > 0:
                ml = int(lens.max())
                if ml > max_len:
                    max_len = ml

        path_users = np.zeros((total_paths, max_len), dtype=np.int64)
        path_lens = np.empty(total_paths, dtype=np.int64)
        time_bins = np.empty(total_paths, dtype=np.int64)
        cascade_idx = np.empty(total_paths, dtype=np.int64)

        cursor = 0
        for bi, ci in enumerate(idxs):
            ps, pe = int(c_off[ci]), int(c_off[ci + 1])
            cnt = pe - ps
            time_bins[cursor : cursor + cnt] = tbins[ps:pe]
            cascade_idx[cursor : cursor + cnt] = bi
            for p in range(ps, pe):
                ns, ne = int(p_off[p]), int(p_off[p + 1])
                ln = ne - ns
                path_users[cursor, :ln] = nodes[ns:ne]
                path_lens[cursor] = ln
                cursor += 1

        batch = {
            "path_users": torch.from_numpy(path_users),
            "path_lens": torch.from_numpy(path_lens),
            "time_bins": torch.from_numpy(time_bins),
            "cascade_idx": torch.from_numpy(cascade_idx),
            "n_cascades": n_cascades,
        }
        return (
            batch,
            torch.tensor(y_logs, dtype=torch.float32).unsqueeze(1),
            torch.tensor(y_raws, dtype=torch.float32).unsqueeze(1),
        )

    return collate


def to_device(batch: dict, dev: torch.device) -> dict:
    return {
        "path_users": batch["path_users"].to(dev, non_blocking=True),
        "path_lens": batch["path_lens"],  # stays on CPU for pack_padded_sequence
        "time_bins": batch["time_bins"].to(dev, non_blocking=True),
        "cascade_idx": batch["cascade_idx"].to(dev, non_blocking=True),
        "n_cascades": batch["n_cascades"],
    }


def compute_full_metrics(pred_log2: np.ndarray, true_log2: np.ndarray, true_raw: np.ndarray) -> dict[str, float]:
    m = regression_metrics(pred_log2, true_log2, true_raw)
    nz = true_log2 > 0
    m["MAPE_log2"] = float(np.mean(np.abs((true_log2[nz] - pred_log2[nz]) / true_log2[nz])))
    pred_raw = np.clip(np.exp2(np.clip(pred_log2, -30, 30)) - 1.0, 0.0, None)
    pos = true_raw > 0
    m["MAPE_count"] = float(np.mean(np.abs((true_raw[pos] - pred_raw[pos]) / true_raw[pos])))
    return m


def run_epoch(model, loader, criterion, device, optimizer=None, emb_l2: float = 0.0) -> float:
    training = optimizer is not None
    model.train(training)
    total_se = 0.0
    total_n = 0
    for batch, y_log, _ in loader:
        b = to_device(batch, device)
        y_log = y_log.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            pred = model(b)
            mse = criterion(pred, y_log)
            if training:
                loss = mse
                if emb_l2 > 0.0:
                    uniq = torch.unique(b["path_users"])
                    uniq = uniq[uniq > 0]
                    if uniq.numel() > 0:
                        loss = loss + 0.5 * emb_l2 * (model.user_emb.weight[uniq] ** 2).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        bs = y_log.shape[0]
        total_se += float(mse.item()) * bs
        total_n += bs
    return total_se / max(total_n, 1)


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, trues_log, trues_raw = [], [], []
    for batch, y_log, y_raw in loader:
        out = model(to_device(batch, device))
        preds.append(out.cpu().numpy().ravel())
        trues_log.append(y_log.numpy().ravel())
        trues_raw.append(y_raw.numpy().ravel())
    return np.concatenate(preds), np.concatenate(trues_log), np.concatenate(trues_raw)


def main() -> None:
    p = argparse.ArgumentParser(description="Train DeepHawkes under the controlled Weibo protocol.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--observation-time", type=int, required=True)
    p.add_argument("--embedding-dim", type=int, default=50)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--mlp-hidden-1", type=int, default=32)
    p.add_argument("--mlp-hidden-2", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--emb-lr", type=float, default=1e-4,
                   help="Separate LR for the sparse user embedding table (CaoQi92 uses 0.1x network LR).")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--emb-l2", type=float, default=0.01,
                   help="Explicit L2 penalty on active batch user embeddings (matching CaoQi92 model_sparse.py).")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--early-stop", type=int, default=8)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    vocab_size = int(meta["vocab"]["vocab_size_with_special"])
    n_time_intervals = int(meta["n_time_intervals"])

    ds = {s: DeepHawkesDataset(data_dir / f"deephawkes_{s}.npz") for s in ("train", "val", "test")}
    loaders = {
        s: DataLoader(
            ds[s],
            batch_size=args.batch_size,
            shuffle=(s == "train"),
            collate_fn=make_collate(ds[s]),
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        for s in ("train", "val", "test")
    }

    mean_log = float(ds["train"].y_log.mean())
    model = DeepHawkes(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        n_time_intervals=n_time_intervals,
        mlp_hidden_1=args.mlp_hidden_1,
        mlp_hidden_2=args.mlp_hidden_2,
        dropout=args.dropout,
    ).to(device)
    model.init_target_bias(mean_log)
    pcounts = model.parameter_counts()

    print("=" * 72)
    print(f" DeepHawkes (Cao et al., CIKM 2017) | window={args.observation_time}s | seed={args.seed}")
    print("=" * 72)
    print(f"• Device: {device} | cascades: train={len(ds['train']):,} val={len(ds['val']):,} test={len(ds['test']):,}")
    print(f"• Params: total={pcounts['total']:,} (non_emb={pcounts['non_embedding']:,}, emb={pcounts['embedding']:,})")
    print(f"• Mean train log2(y+1)={mean_log:.4f} | intervals={n_time_intervals}", flush=True)

    criterion = nn.MSELoss()
    emb_params = list(model.user_emb.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("user_emb.")]
    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": emb_params, "lr": args.emb_lr, "weight_decay": args.weight_decay},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience
    )

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    ckpt = out_dir / "best_deephawkes.pt"
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        ep_t0 = time.time()
        tr = run_epoch(model, loaders["train"], criterion, device, optimizer, emb_l2=args.emb_l2)
        va = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step(va)
        lr_now = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": ep,
            "train_msle": tr,
            "val_msle": va,
            "lr": lr_now,
            "elapsed_sec": time.time() - ep_t0,
        })
        print(
            f"[deephawkes s={args.seed}] ep {ep:02d}/{args.epochs:02d}  "
            f"train_MSLE={tr:.4f}  val_MSLE={va:.4f}  lr={lr_now:.2e}  ({time.time()-ep_t0:.1f}s)",
            flush=True,
        )
        if va < best_val - 1e-5:
            best_val = va
            best_epoch = ep
            stale = 0
            torch.save(model.state_dict(), ckpt)
        else:
            stale += 1
            if stale >= args.early_stop:
                print(f"[deephawkes s={args.seed}] early stop at epoch {ep} (best ep={best_epoch}, val={best_val:.4f})", flush=True)
                break

    train_secs = time.time() - t0
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    pred_test, true_log, true_raw = collect_predictions(model, loaders["test"], device)
    metrics = compute_full_metrics(pred_test, true_log, true_raw)
    metrics["best_val_MSLE_log2"] = float(best_val)
    metrics["best_epoch"] = int(best_epoch)
    metrics["train_wall_seconds"] = float(train_secs)

    learned_time_weights = [float(x) for x in model.time_weight.detach().cpu().tolist()]

    np.savez(
        out_dir / "test_predictions.npz",
        cascade_id=np.asarray(ds["test"].cascade_id),
        pred_log2=pred_test.astype(np.float32),
        true_log2=true_log.astype(np.float32),
        true_raw=true_raw.astype(np.float32),
    )

    report = {
        "model": "DeepHawkes (PyTorch reimplementation of Cao et al., ACM CIKM 2017)",
        "architecture": {
            "vocab_size": vocab_size,
            "embedding_dim": args.embedding_dim,
            "hidden_size": args.hidden_size,
            "n_time_intervals": n_time_intervals,
            "mlp_hidden": [args.mlp_hidden_1, args.mlp_hidden_2],
            "dropout": args.dropout,
            "params": pcounts,
            "learned_time_weights": learned_time_weights,
        },
        "protocol": {
            "observation_window_seconds": args.observation_time,
            "prediction_horizon_seconds": 86400,
            "split": "strictly chronological 70/15/15 aligned to weibo_{train,val,test}.npz",
            "target": "future increment Delta P = R_24h - R_{T_obs}, log2(Delta P + 1)",
            "loss": "MSLE (MSE in log2 space)",
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "lr": args.lr,
            "emb_lr": args.emb_lr,
            "weight_decay": args.weight_decay,
            "emb_l2": args.emb_l2,
            "scheduler": f"ReduceLROnPlateau(factor=0.5, patience={args.patience})",
            "grad_clip_max_norm": 1.0,
            "early_stop_patience": args.early_stop,
            "max_epochs": args.epochs,
            "checkpoint": "best validation MSLE epoch",
            "seed": args.seed,
            "device": str(device),
        },
        "data": {s: len(ds[s]) for s in ds},
        "metrics": metrics,
    }
    (out_dir / "deephawkes_evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f" DeepHawkes TEST RESULTS ({args.observation_time}s, seed={args.seed})")
    print("=" * 72)
    for k in ("MSLE_log2", "MSLE_ln", "R2_log2", "PCC_log2", "MAPE_log2", "MAE_count", "RMSE_count", "MedAE_count"):
        print(f"  {k:<14} {metrics[k]:.4f}")
    print(f"  learned_time_weights: {[round(x, 3) for x in learned_time_weights]}")
    print(f"• Report -> {out_dir / 'deephawkes_evaluation_report.json'}", flush=True)


if __name__ == "__main__":
    main()

