"""Train DeepCas on Reddit cascades under the controlled benchmark protocol."""

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
from torch.utils.data import DataLoader

from .model import DeepCasModel
from .reddit_data import RedditDeepCasDataset, collate_deepcas


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(pred_log2: np.ndarray, true_log2: np.ndarray, true_raw: np.ndarray) -> dict[str, float]:
    pred_log2 = np.asarray(pred_log2, dtype=np.float64).ravel()
    true_log2 = np.asarray(true_log2, dtype=np.float64).ravel()
    true_raw = np.asarray(true_raw, dtype=np.float64).ravel()

    residuals = pred_log2 - true_log2
    msle_log2 = float(np.mean(residuals ** 2))
    rmsle_log2 = float(math.sqrt(msle_log2))
    msle_ln = float(msle_log2 * (math.log(2.0) ** 2))

    pred_raw = np.clip(np.exp2(np.clip(pred_log2, -30, 30)) - 1.0, 0.0, None)
    err = np.abs(pred_raw - true_raw)

    mae_count = float(np.mean(err))
    medae_count = float(np.median(err))
    mape = float(np.mean(err / (true_raw + 1.0)) * 100.0)

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((true_log2 - true_log2.mean()) ** 2))
    r2_log2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    p_std, t_std = np.std(pred_log2), np.std(true_log2)
    if p_std > 1e-8 and t_std > 1e-8:
        pcc_log2 = float(np.corrcoef(pred_log2, true_log2)[0, 1])
    else:
        pcc_log2 = 0.0

    return {
        "MSLE_log2": msle_log2,
        "RMSLE_log2": rmsle_log2,
        "MSLE_ln": msle_ln,
        "R2_log2": r2_log2,
        "PCC_log2": pcc_log2,
        "MAPE": mape,
        "MAE_count": mae_count,
        "MedAE_count": medae_count,
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_trues, all_raws = [], [], []

    with torch.no_grad():
        for walks, obs_sz, y_log, y_raw in loader:
            walks = walks.to(device)
            obs_sz = obs_sz.to(device)
            preds = model(walks, obs_sz)
            all_preds.append(preds.squeeze(-1).cpu().numpy())
            all_trues.append(y_log.numpy())
            all_raws.append(y_raw)

    p_log = np.concatenate(all_preds, axis=0)
    t_log = np.concatenate(all_trues, axis=0)
    t_raw = np.concatenate(all_raws, axis=0)
    metrics = compute_metrics(p_log, t_log, t_raw)
    return metrics, p_log, t_log, t_raw


def main() -> None:
    p = argparse.ArgumentParser(description="Train DeepCas on Reddit cascades.")
    p.add_argument("--cache-dir", default="reddit_benchmark/cache")
    p.add_argument("--output-dir", default="reddit_deepcas/runs_3600/s42")
    p.add_argument("--observation-time", type=int, default=3600)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--early-stop", type=int, default=8)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-cascades", type=int, default=0, help="Optional cap for quick testing.")
    p.add_argument("--embedding-dim", type=int, default=50)
    p.add_argument("--hidden-gru", type=int, default=32)
    p.add_argument("--n-walks", type=int, default=32)
    p.add_argument("--walk-length", type=int, default=10)
    args = p.parse_args()

    set_seed(args.seed)
    dev = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f" DEEPCAS REDDIT BENCHMARK | window={args.observation_time}s | seed={args.seed} | device={dev}")
    print("=" * 76, flush=True)

    # Read vocabulary size from DeepHawkes metadata
    meta_path = Path(args.cache_dir) / f"deephawkes_{args.observation_time}" / "metadata.json"
    vocab_size = 200000
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        vocab_size = meta["vocab"]["vocab_size_with_special"] + 5
        print(f"• Loaded vocab size from metadata: {vocab_size:,}")

    train_ds = RedditDeepCasDataset(
        args.cache_dir, "train", args.observation_time,
        n_walks=args.n_walks, walk_length=args.walk_length,
        seed=args.seed, max_cascades=args.max_cascades,
    )
    val_ds = RedditDeepCasDataset(
        args.cache_dir, "val", args.observation_time,
        n_walks=args.n_walks, walk_length=args.walk_length,
        seed=args.seed, max_cascades=args.max_cascades,
    )
    test_ds = RedditDeepCasDataset(
        args.cache_dir, "test", args.observation_time,
        n_walks=args.n_walks, walk_length=args.walk_length,
        seed=args.seed, max_cascades=args.max_cascades,
    )

    print(f"• Splits: train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_deepcas, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_deepcas, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_deepcas, num_workers=0)

    model = DeepCasModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_gru=args.hidden_gru,
        walk_length=args.walk_length,
    ).to(dev)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"• DeepCas model parameters: {total_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.patience)
    criterion = nn.MSELoss()

    best_val_msle = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_weights_path = out_dir / "best_deepcas.pt"
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for walks, obs_sz, y_log, _ in train_loader:
            walks = walks.to(dev)
            obs_sz = obs_sz.to(dev)
            y_log = y_log.to(dev)

            optimizer.zero_grad()
            preds = model(walks, obs_sz).squeeze(-1)
            loss = criterion(preds, y_log)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(1, n_batches)
        val_metrics, _, _, _ = evaluate(model, val_loader, dev)
        val_msle = val_metrics["MSLE_log2"]
        scheduler.step(val_msle)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | Train MSE: {train_loss:.4f} | "
            f"Val MSLE_log2: {val_msle:.4f} (R2: {val_metrics['R2_log2']:+.3f}) | lr: {optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )

        if val_msle < best_val_msle:
            best_val_msle = val_msle
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_weights_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop:
                print(f"• Early stopping triggered at epoch {epoch} (best epoch: {best_epoch})", flush=True)
                break

    # Restore best checkpoint
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    test_metrics, p_log, t_log, t_raw = evaluate(model, test_loader, dev)
    train_time = time.time() - t_start

    print("-" * 76)
    print(f"DeepCas Test Results (Seed {args.seed}) | Window: {args.observation_time}s | Time: {train_time:.1f}s")
    for k, v in test_metrics.items():
        print(f"  {k:<15}: {v:.4f}")
    print("-" * 76, flush=True)

    np.savez_compressed(
        out_dir / "test_predictions.npz",
        pred_log2=p_log,
        true_log2=t_log,
        true_raw=t_raw,
        cascade_id=test_ds.cascade_id,
    )

    report = {
        "model": "DeepCas",
        "dataset": "Reddit",
        "observation_time": args.observation_time,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "train_time_sec": train_time,
        "params": total_params,
        "metrics": test_metrics,
    }
    (out_dir / "deepcas_evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"• Saved test report -> {out_dir / 'deepcas_evaluation_report.json'}", flush=True)


if __name__ == "__main__":
    main()
