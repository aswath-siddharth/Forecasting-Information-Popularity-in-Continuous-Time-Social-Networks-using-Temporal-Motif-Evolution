"""Train and evaluate DeepHawkes on APS cascades under the exact controlled benchmark.

Optimization harness:
  - Target: future increment Delta P = R_{20y} - R_{T_obs}, log2(Delta P + 1)
  - Loss: MSE in log2 space (= MSLE_log2)
  - Batch size: 64
  - Optimizer: AdamW (non-emb lr=1e-3, emb lr=1e-4, wd=1e-4)
  - Scheduler: ReduceLROnPlateau(factor=0.5, patience=3)
  - Gradient clipping: max_norm=1.0
  - Early stopping: patience=8, max_epochs=60
  - Model selection: restore lowest validation MSLE checkpoint
  - Metrics: MSLE_log2, RMSLE, R2, PCC, MAPE, MAE, MedAE, RMSE, runtime
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

from .model_deephawkes import DeepHawkes
from .train_casmotif import compute_metrics, pick_device, set_seed


class DeepHawkesDataset(Dataset):
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
    c_off = ds.cascade_path_offsets
    p_off = ds.path_node_offsets
    nodes = ds.node_ids
    tbins = ds.path_time_bins

    def collate(items):
        idxs, y_logs, y_raws = zip(*items)
        n_cascades = len(idxs)

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
            for pi in range(ps, pe):
                ns, ne = int(p_off[pi]), int(p_off[pi + 1])
                l = ne - ns
                if l > 0:
                    path_users[cursor, :l] = nodes[ns:ne]
                    path_lens[cursor] = l
                else:
                    path_users[cursor, 0] = 1  # UNK
                    path_lens[cursor] = 1

                time_bins[cursor] = tbins[pi]
                cascade_idx[cursor] = bi
                cursor += 1

        batch = {
            "path_users": torch.from_numpy(path_users),
            "path_lens": torch.from_numpy(path_lens),
            "time_bins": torch.from_numpy(time_bins),
            "cascade_idx": torch.from_numpy(cascade_idx),
            "n_cascades": n_cascades,
        }
        y_log_t = torch.tensor(y_logs, dtype=torch.float32).unsqueeze(1)
        y_raw_t = torch.tensor(y_raws, dtype=torch.float32).unsqueeze(1)
        return batch, y_log_t, y_raw_t

    return collate


def to_device(b: dict[str, torch.Tensor | int], dev: torch.device) -> dict[str, torch.Tensor | int]:
    return {
        "path_users": b["path_users"].to(dev, non_blocking=True),
        "path_lens": b["path_lens"].to(dev, non_blocking=True),
        "time_bins": b["time_bins"].to(dev, non_blocking=True),
        "cascade_idx": b["cascade_idx"].to(dev, non_blocking=True),
        "n_cascades": b["n_cascades"],
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    t0 = time.time()
    preds: list[np.ndarray] = []
    trues_log: list[np.ndarray] = []
    trues_raw: list[np.ndarray] = []

    with torch.no_grad():
        for b, y_log, y_raw in loader:
            b_dev = to_device(b, device)
            out = model(b_dev)
            preds.append(out.detach().cpu().numpy())
            trues_log.append(y_log.numpy())
            trues_raw.append(y_raw.numpy())

    infer_sec = time.time() - t0
    p = np.concatenate(preds, axis=0).ravel()
    tl = np.concatenate(trues_log, axis=0).ravel()
    tr = np.concatenate(trues_raw, axis=0).ravel()
    msle = float(np.mean((p - tl) ** 2))
    return msle, p, tl, tr, infer_sec


def train_deephawkes(
    data_dir: Path,
    output_dir: Path,
    observation_days: int = 1095,
    batch_size: int = 64,
    lr: float = 1e-3,
    emb_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    emb_l2: float = 0.01,
    patience: int = 3,
    early_stop: int = 8,
    epochs: int = 60,
    seed: int = 42,
    device_name: str = "cuda",
) -> dict:
    set_seed(seed)
    device = pick_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[DeepHawkes] Window={observation_days}d | Seed={seed} | Device={device}", flush=True)

    vocab = json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))
    vocab_size = max(vocab.values()) + 1

    ds_train = DeepHawkesDataset(data_dir / "deephawkes_train.npz")
    ds_val = DeepHawkesDataset(data_dir / "deephawkes_val.npz")
    ds_test = DeepHawkesDataset(data_dir / "deephawkes_test.npz")

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=make_collate(ds_train))
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=make_collate(ds_val))
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, collate_fn=make_collate(ds_test))

    mean_target = float(ds_train.y_log.mean())
    model = DeepHawkes(vocab_size=vocab_size, embedding_dim=50, hidden_size=64, n_time_intervals=6).to(device)
    model.init_target_bias(mean_target)

    pcounts = model.parameter_counts()
    print(f"• DeepHawkes instantiated: {pcounts['total']:,} total params ({pcounts['embedding']:,} emb, {pcounts['non_embedding']:,} non-emb) | Target mean={mean_target:.4f}", flush=True)

    emb_params = list(model.user_emb.parameters())
    non_emb_params = [p for n, p in model.named_parameters() if not n.startswith("user_emb.")]
    optimizer = torch.optim.AdamW([
        {"params": non_emb_params, "lr": lr, "weight_decay": weight_decay},
        {"params": emb_params, "lr": emb_lr, "weight_decay": weight_decay},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience)
    criterion = nn.MSELoss()

    best_val_msle = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    epoch_times: list[float] = []

    t_train_start = time.time()
    ckpt_path = output_dir / "best_deephawkes.pt"

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        train_loss_acc = 0.0
        n_batches = 0

        for b, y_log, _ in train_loader:
            b_dev = to_device(b, device)
            y_log_dev = y_log.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(b_dev)
            mse_loss = criterion(pred, y_log_dev)

            # Active-batch L2 penalty on accessed user embeddings
            active_users = b_dev["path_users"]
            mask = active_users > 0
            if mask.any() and emb_l2 > 0:
                active_emb = model.user_emb(active_users[mask])
                reg = emb_l2 * torch.mean(active_emb ** 2)
            else:
                reg = 0.0

            loss = mse_loss + reg
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_acc += mse_loss.item()
            n_batches += 1

        ep_duration = time.time() - ep_t0
        epoch_times.append(ep_duration)
        tr_msle = train_loss_acc / max(1, n_batches)

        val_msle, _, _, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_msle)
        curr_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_msle": tr_msle,
            "val_msle": val_msle,
            "lr": curr_lr,
            "elapsed_seconds": ep_duration,
        })

        is_best = " *" if val_msle < best_val_msle - 1e-5 else ""
        print(f"[DeepHawkes s={seed}] ep {epoch:02d}/{epochs:02d} | train_MSLE={tr_msle:.4f} | val_MSLE={val_msle:.4f}{is_best} | lr={curr_lr:.2e} ({ep_duration:.1f}s)", flush=True)

        if val_msle < best_val_msle - 1e-5:
            best_val_msle = val_msle
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            stale_epochs += 1
            if stale_epochs >= early_stop:
                print(f"[DeepHawkes s={seed}] Early stopping at epoch {epoch} (best epoch {best_epoch}, val MSLE={best_val_msle:.4f})", flush=True)
                break

    train_wall_seconds = time.time() - t_train_start

    # Final evaluation on restored best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_msle, test_pred, test_true_log, test_true_raw, test_infer_sec = evaluate(model, test_loader, device)
    metrics = compute_metrics(test_pred, test_true_log, test_true_raw)

    np.savez_compressed(
        output_dir / "test_predictions.npz",
        cascade_id=np.asarray(ds_test.cascade_id, dtype=object),
        pred_log2=test_pred.astype(np.float32),
        true_log2=test_true_log.astype(np.float32),
        true_raw=test_true_raw.astype(np.float32),
    )

    report = {
        "model": "DeepHawkes (PyTorch reimplementation of Cao et al., ACM CIKM 2017)",
        "parameters": pcounts,
        "seed": seed,
        "observation_time_days": observation_days,
        "best_epoch": best_epoch,
        "best_val_msle": best_val_msle,
        "metrics": metrics,
        "timing": {
            "train_wall_seconds": train_wall_seconds,
            "test_inference_seconds": test_infer_sec,
            "avg_seconds_per_epoch": float(np.mean(epoch_times)) if epoch_times else 0.0,
            "total_epochs_trained": len(history),
        },
    }

    (output_dir / "deephawkes_evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(
        f"• [DeepHawkes s={seed}] Complete: Test MSLE={metrics['MSLE_log2']:.4f} | "
        f"R2={metrics['R2_log2']:.4f} | Train Time={train_wall_seconds:.1f}s | Infer Time={test_infer_sec:.2f}s",
        flush=True,
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Train DeepHawkes on APS benchmark.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--observation-time", type=int, default=1095)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--emb-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--emb-l2", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--early-stop", type=int, default=8)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    train_deephawkes(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        observation_days=args.observation_time,
        batch_size=args.batch_size,
        lr=args.lr,
        emb_lr=args.emb_lr,
        weight_decay=args.weight_decay,
        emb_l2=args.emb_l2,
        patience=args.patience,
        early_stop=args.early_stop,
        epochs=args.epochs,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
