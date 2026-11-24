"""Training and evaluation loop for DeepCas on Twitter cascades.

Implements all specifications from the user's hyperparameter protocol:
- Optimization: AdamW, weight_decay=1e-4, initial_lr=1e-3
- Batch size: 64
- LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Gradient Clipping: max_norm=1.0
- Early Stopping: patience=8 epochs (max 60 epochs)
- Model Checkpoint: restore weights from lowest validation MSLE epoch
- Metrics: MSLE_log2, RMSLE_log2, MSLE_ln, R2_log2, PCC_log2, MAPE, MAE_count, MedAE_count
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import TwitterDeepCasDataset, collate_deepcas
from .model import DeepCasModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_metrics(pred_log2: np.ndarray, true_log2: np.ndarray, true_raw: np.ndarray) -> dict[str, float]:
    """Calculate comprehensive evaluation metrics in log2 space and raw counts."""
    pred_log2 = np.asarray(pred_log2, dtype=np.float64).ravel()
    true_log2 = np.asarray(true_log2, dtype=np.float64).ravel()
    true_raw = np.asarray(true_raw, dtype=np.float64).ravel()

    # Residuals in log2(y + 1) space
    residuals = pred_log2 - true_log2
    msle_log2 = float(np.mean(residuals ** 2))
    rmsle_log2 = float(math.sqrt(msle_log2))
    msle_ln = float(msle_log2 * (math.log(2.0) ** 2))

    # Raw counts
    pred_raw = np.clip(np.exp2(np.clip(pred_log2, -30, 30)) - 1.0, 0.0, None)
    err = np.abs(pred_raw - true_raw)

    mae_count = float(np.mean(err))
    medae_count = float(np.median(err))

    # MAPE: denominator smoothed by +1 to prevent division by zero
    mape = float(np.mean(err / (true_raw + 1.0)) * 100.0)

    # R2 log2
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((true_log2 - true_log2.mean()) ** 2))
    r2_log2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # PCC log2
    if len(pred_log2) > 1 and pred_log2.std() > 0 and true_log2.std() > 0:
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
        "n_samples": int(len(true_raw)),
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, count = 0.0, 0

    for walks, obs_sz, y_log, _ in loader:
        walks = walks.to(device)
        obs_sz = obs_sz.to(device)
        y_log = y_log.to(device)

        with torch.set_grad_enabled(is_train):
            pred = model(walks, obs_sz)
            loss = criterion(pred, y_log)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += float(loss.item()) * len(y_log)
        count += len(y_log)

    return total_loss / max(count, 1)


@torch.no_grad()
def evaluate_model(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, trues_log, trues_raw = [], [], []

    for walks, obs_sz, y_log, y_raw in loader:
        walks = walks.to(device)
        obs_sz = obs_sz.to(device)
        pred = model(walks, obs_sz)

        preds.append(pred.cpu().numpy().ravel())
        trues_log.append(y_log.cpu().numpy().ravel())
        trues_raw.append(y_raw)

    return (
        np.concatenate(preds),
        np.concatenate(trues_log),
        np.concatenate(trues_raw),
    )


def train_deepcas(
    data_dir: str,
    output_dir: str,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lr_patience: int = 3,
    early_stop_patience: int = 8,
    embedding_dim: int = 50,
    hidden_gru: int = 32,
    dropout: float = 0.2,
    seed: int = 42,
    device_str: str | None = None,
) -> dict:
    set_seed(seed)
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str) if device_str else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"\n=======================================================")
    print(f" TRAINING DEEPCAS (WWW 2017) ON TWITTER BENCHMARK")
    print(f"=======================================================")
    print(f"• Device: {device}")
    print(f"• Data directory: {data_path}")
    print(f"• Output directory: {out_path}")
    print(f"• Hyperparameters: batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}")
    print(f"  early_stop_patience={early_stop_patience}, lr_patience={lr_patience}, max_epochs={epochs}")

    # Load dataset metadata
    with open(data_path / "vocab_meta.json", "r") as f:
        meta = json.load(f)
    vocab_size = meta["vocab_size"]
    walk_length = meta["walk_length"]
    n_walks = meta["n_walks"]

    # Datasets
    train_ds = TwitterDeepCasDataset(data_path / "twitter_train.npz")
    val_ds = TwitterDeepCasDataset(data_path / "twitter_val.npz")
    test_ds = TwitterDeepCasDataset(data_path / "twitter_test.npz")

    mean_train_log = float(train_ds.y_log2.mean())
    print(f"• Split sizes: Train={len(train_ds):,}, Val={len(val_ds):,}, Test={len(test_ds):,}")
    print(f"• Vocabulary: {vocab_size:,} nodes, walk_length={walk_length}, n_walks={n_walks}")
    print(f"• Mean training log2(increment + 1): {mean_train_log:.4f}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_deepcas)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_deepcas)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_deepcas)

    # Reference Mean Baseline (sanity check)
    mean_preds = np.full(len(test_ds), mean_train_log, dtype=np.float64)
    baseline_metrics = compute_metrics(mean_preds, test_ds.y_log2, test_ds.y_raw)
    print(f"• Naive Mean Baseline Test MSLE: {baseline_metrics['MSLE_log2']:.4f}")

    # Build Model
    model = DeepCasModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_gru=hidden_gru,
        walk_length=walk_length,
        dropout=dropout,
    ).to(device)
    model.init_target_bias(mean_train_log)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_patience
    )

    best_val_msle = float("inf")
    best_epoch = 0
    stale_epochs = 0
    ckpt_path = out_path / "best_deepcas.pt"
    history = []

    print("\n--- Starting Training ---")
    for epoch in range(1, epochs + 1):
        tr_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        lr_current = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_msle": tr_loss,
            "val_msle": va_loss,
            "lr": lr_current,
        })

        is_best = va_loss < best_val_msle - 1e-5
        if is_best:
            best_val_msle = va_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
            tag = "★ (best saved)"
        else:
            stale_epochs += 1
            tag = ""

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train MSLE: {tr_loss:.4f} | Val MSLE: {va_loss:.4f} | LR: {lr_current:.2e} {tag}",
            flush=True,
        )

        if stale_epochs >= early_stop_patience:
            print(f"\n[Early Stopping] No improvement in validation MSLE for {early_stop_patience} consecutive epochs.")
            break

    # Restore best validation checkpoint for final testing
    print(f"\n• Restoring best checkpoint from epoch {best_epoch} (Val MSLE: {best_val_msle:.4f})...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Evaluate on Test Set
    test_preds, test_true_log, test_true_raw = evaluate_model(model, test_loader, device)
    test_metrics = compute_metrics(test_preds, test_true_log, test_true_raw)

    print("\n" + "=" * 60)
    print(" FINAL EVALUATION RESULTS ON TEST SET")
    print("=" * 60)
    print(f"  MSLE (log2)        : {test_metrics['MSLE_log2']:.4f}  (Naive baseline: {baseline_metrics['MSLE_log2']:.4f})")
    print(f"  RMSLE (log2)       : {test_metrics['RMSLE_log2']:.4f}")
    print(f"  MSLE (ln)          : {test_metrics['MSLE_ln']:.4f}")
    print(f"  R² (log2)          : {test_metrics['R2_log2']:.4f}")
    print(f"  PCC (log2)         : {test_metrics['PCC_log2']:.4f}")
    print(f"  MAPE (%)           : {test_metrics['MAPE']:.2f}%")
    print(f"  MAE (count)        : {test_metrics['MAE_count']:.2f}")
    print(f"  MedAE (count)      : {test_metrics['MedAE_count']:.2f}")
    print(f"  Test samples       : {test_metrics['n_samples']:,}")
    print("=" * 60)

    # Save artifacts
    report = {
        "model": "DeepCas (Li et al., WWW 2017)",
        "dataset": "Twitter Cascades (cascades/twitter.parquet)",
        "observation_window_seconds": meta["t_obs"],
        "observation_window_days": meta["t_obs"] / 86400,
        "prediction_horizon_days": 15,
        "target": "Future Increment (R_15d - R_obs)",
        "best_epoch": best_epoch,
        "best_val_msle": best_val_msle,
        "test_metrics": test_metrics,
        "naive_baseline_metrics": baseline_metrics,
        "hyperparameters": {
            "batch_size": batch_size,
            "optimizer": "AdamW",
            "weight_decay": weight_decay,
            "initial_lr": lr,
            "early_stopping_patience": early_stop_patience,
            "lr_plateau_patience": lr_patience,
            "max_grad_norm": 1.0,
            "embedding_dim": embedding_dim,
            "hidden_gru": hidden_gru,
            "walk_length": walk_length,
            "n_walks": n_walks,
            "vocab_size": vocab_size,
            "seed": seed,
        },
    }

    with open(out_path / "deepcas_evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(out_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    np.savez_compressed(
        out_path / "test_predictions.npz",
        pred_log2=test_preds,
        true_log2=test_true_log,
        true_raw=test_true_raw,
    )

    print(f"• Saved evaluation report to: {out_path / 'deepcas_evaluation_report.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Train DeepCas on Twitter dataset.")
    parser.add_argument("--data-dir", default="data_twitter/processed_1d", help="Data directory containing .npz files")
    parser.add_argument("--output-dir", default="outputs_deepcas/outputs_1d", help="Output directory")
    parser.add_argument("--epochs", type=int, default=60, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=3, help="LR decay patience")
    parser.add_argument("--early-stop", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default=None, help="Device (cpu or cuda)")
    args = parser.parse_args()

    train_deepcas(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_patience=args.patience,
        early_stop_patience=args.early_stop,
        seed=args.seed,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
