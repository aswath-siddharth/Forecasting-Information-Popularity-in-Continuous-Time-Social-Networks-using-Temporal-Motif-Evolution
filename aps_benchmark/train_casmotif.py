"""Training harness for CasMotif on APS citation cascade benchmark.

Zero leakage contract:
  - SequenceScaler is fitted strictly on the training split only.
  - Test predictions and evaluation report are saved for paired comparison.
  - Validation and test loaders use strict sequential evaluation (shuffle=False).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data_casmotif import (
    APSCascadeDataset,
    BucketedBatchSampler,
    SequenceScaler,
    collate_variable_length,
)
from .model_casmotif import CascadeAttentionGRURegressor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_metrics(
    preds_log: np.ndarray,
    trues_log: np.ndarray,
    trues_raw: np.ndarray,
) -> dict[str, float]:
    msle_log2 = float(np.mean((preds_log - trues_log) ** 2))
    # MSLE natural log: (log2(x))^2 = (ln(x) / ln(2))^2 -> MSLE_ln = MSLE_log2 * (ln(2))^2
    msle_ln = float(msle_log2 * (np.log(2.0) ** 2))
    rmsle_log2 = float(np.sqrt(msle_log2))

    var_true = float(np.var(trues_log))
    r2_log2 = float(1.0 - msle_log2 / var_true) if var_true > 1e-9 else 0.0

    std_p = float(np.std(preds_log))
    std_t = float(np.std(trues_log))
    if std_p > 1e-9 and std_t > 1e-9:
        cov = float(np.mean((preds_log - preds_log.mean()) * (trues_log - trues_log.mean())))
        pcc_log2 = cov / (std_p * std_t)
    else:
        pcc_log2 = 0.0

    mape_log2 = float(np.mean(np.abs(preds_log - trues_log) / np.maximum(trues_log, 1.0)))

    # Raw count scale
    preds_raw = np.maximum(0.0, np.exp2(preds_log) - 1.0)
    mae_count = float(np.mean(np.abs(preds_raw - trues_raw)))
    medae_count = float(np.median(np.abs(preds_raw - trues_raw)))
    rmse_count = float(np.sqrt(np.mean((preds_raw - trues_raw) ** 2)))
    mape_count = float(np.mean(np.abs(preds_raw - trues_raw) / np.maximum(trues_raw, 1.0)))

    return {
        "MSLE_log2": msle_log2,
        "MSLE_ln": msle_ln,
        "RMSLE_log2": rmsle_log2,
        "R2_log2": r2_log2,
        "PCC_log2": pcc_log2,
        "MAPE_log2": mape_log2,
        "MAE_count": mae_count,
        "MedAE_count": medae_count,
        "RMSE_count": rmse_count,
        "MAPE_count": mape_count,
        "n_samples": len(preds_log),
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
        for x, mask, y_log, y_raw in loader:
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            pred = model(x, mask)
            preds.append(pred.squeeze(1).cpu().numpy())
            trues_log.append(y_log.numpy())
            trues_raw.append(y_raw.numpy())

    infer_sec = time.time() - t0
    p = np.concatenate(preds, axis=0)
    tl = np.concatenate(trues_log, axis=0)
    tr = np.concatenate(trues_raw, axis=0)
    msle = float(np.mean((p - tl) ** 2))
    return msle, p, tl, tr, infer_sec


def train_casmotif(
    data_dir: Path,
    output_dir: Path,
    observation_days: int = 1095,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 3,
    early_stop: int = 8,
    epochs: int = 60,
    seed: int = 42,
    device_name: str = "cuda",
) -> dict:
    set_seed(seed)
    device = pick_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[CasMotif] Window={observation_days}d | Seed={seed} | Device={device}", flush=True)

    # 1. Fit scaler strictly on train split only
    train_raw = np.load(data_dir / "aps_train.npz")
    scaler = SequenceScaler().fit(train_raw["X_flat"])
    (output_dir / "scaler.json").write_text(json.dumps(scaler.state_dict(), indent=2), encoding="utf-8")

    ds_train = APSCascadeDataset(data_dir / "aps_train.npz", scaler=scaler)
    ds_val = APSCascadeDataset(data_dir / "aps_val.npz", scaler=scaler)
    ds_test = APSCascadeDataset(data_dir / "aps_test.npz", scaler=scaler)

    train_sampler = BucketedBatchSampler(ds_train.lengths, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(ds_train, batch_sampler=train_sampler, collate_fn=collate_variable_length)

    # Strict sequential evaluation for 1-to-1 alignment with DeepHawkes
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collate_variable_length)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, collate_fn=collate_variable_length)

    mean_target = float(ds_train.y_log.mean())
    model = CascadeAttentionGRURegressor(input_dim=15, hidden_size=64, attention_dim=64, dropout=0.3).to(device)
    model.init_target_bias(mean_target)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"• Model instantiated: AttentionGRU ({n_params:,} parameters) | Target mean={mean_target:.4f}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience)
    criterion = nn.MSELoss()

    best_val_msle = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    epoch_times: list[float] = []

    t_train_start = time.time()
    ckpt_path = output_dir / "best_attention_gru.pt"

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        train_loss_acc = 0.0
        n_batches = 0

        for x, mask, y_log, _ in train_loader:
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y_log = y_log.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x, mask)
            loss = criterion(pred, y_log)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_acc += loss.item()
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
        print(f"[CasMotif s={seed}] ep {epoch:02d}/{epochs:02d} | train_MSLE={tr_msle:.4f} | val_MSLE={val_msle:.4f}{is_best} | lr={curr_lr:.2e} ({ep_duration:.1f}s)", flush=True)

        if val_msle < best_val_msle - 1e-5:
            best_val_msle = val_msle
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            stale_epochs += 1
            if stale_epochs >= early_stop:
                print(f"[CasMotif s={seed}] Early stopping at epoch {epoch} (best epoch {best_epoch}, val MSLE={best_val_msle:.4f})", flush=True)
                break

    train_wall_seconds = time.time() - t_train_start

    # Final evaluation using restored best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_msle, test_pred, test_true_log, test_true_raw, test_infer_sec = evaluate(model, test_loader, device)
    metrics = compute_metrics(test_pred, test_true_log, test_true_raw)

    # Save test predictions for paired comparison
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        cascade_id=ds_test.cascade_id,
        pred_log2=test_pred.astype(np.float32),
        true_log2=test_true_log.astype(np.float32),
        true_raw=test_true_raw.astype(np.float32),
    )

    report = {
        "model": "CasMotif (AttentionGRU)",
        "parameters": n_params,
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

    (output_dir / "casmotif_evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(
        f"• [CasMotif s={seed}] Complete: Test MSLE={metrics['MSLE_log2']:.4f} | "
        f"R2={metrics['R2_log2']:.4f} | Train Time={train_wall_seconds:.1f}s | Infer Time={test_infer_sec:.2f}s",
        flush=True,
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Train CasMotif on APS benchmark.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--observation-time", type=int, default=1095)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--early-stop", type=int, default=8)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    train_casmotif(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        observation_days=args.observation_time,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        early_stop=args.early_stop,
        epochs=args.epochs,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
