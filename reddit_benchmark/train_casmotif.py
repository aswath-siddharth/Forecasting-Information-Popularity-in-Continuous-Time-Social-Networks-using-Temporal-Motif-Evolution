"""Train and evaluate CasMotif on the novel Reddit dataset under the controlled protocol.

Reuses the exact unmodified CasMotif architecture (``pipeline_w.model.build_model``) and
train-split-only ``SequenceScaler`` (``pipeline_w.data``), with the exact same optimization
harness as DeepHawkes (batch 64, AdamW lr=1e-3 wd=1e-4, ReduceLROnPlateau 0.5/3, grad clip
1.0, early stop patience 8 / max 60 epochs, best val checkpoint).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from deephawkes.train import compute_full_metrics
from pipeline_c.data import LengthGroupedBatchSampler
from pipeline_w.data import SequenceScaler, WeiboCascadeDataset, pad_collate_fn
from pipeline_w.model import build_model
from pipeline_w.train_weibo import collect_predictions, run_epoch
from pipeline_w.weibo_parser import FEATURE_DIM


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


def main() -> None:
    p = argparse.ArgumentParser(description="Train CasMotif on the controlled Reddit benchmark.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--observation-time", type=int, required=True)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--attention-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
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

    train_ds = WeiboCascadeDataset(data_dir / "reddit_train.npz", target="increment")
    scaler = SequenceScaler().fit(train_ds.X_flat)  # fitted strictly on train split ONLY
    train_ds.scaler = scaler
    val_ds = WeiboCascadeDataset(data_dir / "reddit_val.npz", scaler=scaler, target="increment")
    test_ds = WeiboCascadeDataset(data_dir / "reddit_test.npz", scaler=scaler, target="increment")

    mean_log = float(train_ds.log_targets().mean())
    model = build_model(
        "attention_gru", FEATURE_DIM, args.hidden_size, args.num_layers,
        args.attention_dim, args.dropout,
    ).to(device)
    model.init_target_bias(mean_log)
    n_params = sum(x.numel() for x in model.parameters())

    train_loader = DataLoader(
        train_ds,
        batch_sampler=LengthGroupedBatchSampler(
            lengths=train_ds.lengths.tolist(), batch_size=args.batch_size, shuffle=True,
        ),
        collate_fn=pad_collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=LengthGroupedBatchSampler(
            lengths=val_ds.lengths.tolist(), batch_size=args.batch_size, shuffle=False,
        ),
        collate_fn=pad_collate_fn,
        num_workers=0,
    )
    # Index-ordered test loader so predictions align 1-to-1 with test_ds.cascade_id
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=pad_collate_fn, num_workers=0)

    print("=" * 72)
    print(f" CasMotif (Reddit Novel Dataset) | window={args.observation_time}s | seed={args.seed}")
    print("=" * 72)
    print(f"• Device: {device} | cascades: train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")
    print(f"• Params: {n_params:,} | mean train log2(y+1)={mean_log:.4f}", flush=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience,
    )

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    ckpt = out_dir / "best_attention_gru.pt"
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        ep_t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, optimizer)
        va = run_epoch(model, val_loader, criterion, device)
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
            f"[casmotif s={args.seed}] ep {ep:02d}/{args.epochs:02d}  "
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
                print(f"[casmotif s={args.seed}] early stop at epoch {ep} (best ep={best_epoch}, val={best_val:.4f})", flush=True)
                break

    train_secs = time.time() - t0
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    pred_test, true_log, true_raw = collect_predictions(model, test_loader, device)
    metrics = compute_full_metrics(pred_test, true_log, true_raw)
    metrics["best_val_MSLE_log2"] = float(best_val)
    metrics["best_epoch"] = int(best_epoch)
    metrics["train_wall_seconds"] = float(train_secs)

    np.savez(
        out_dir / "test_predictions.npz",
        cascade_id=np.asarray([str(c) for c in test_ds.cascade_id.tolist()]),
        pred_log2=pred_test.astype(np.float32),
        true_log2=true_log.astype(np.float32),
        true_raw=true_raw.astype(np.float32),
    )

    report = {
        "model": "CasMotif (15-D Causal Temporal Motifs + Attentive Bi-GRU)",
        "dataset": "Novel Multi-Subreddit Reddit Dataset (data/)",
        "params": int(n_params),
        "protocol": {
            "observation_window_seconds": args.observation_time,
            "prediction_horizon_seconds": 86400,
            "split": "strictly chronological 70/15/15 by post_time",
            "target": "future increment Delta P = R_24h - R_{T_obs}, log2(Delta P + 1)",
            "loss": "MSLE (MSE in log2 space)",
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": f"ReduceLROnPlateau(factor=0.5, patience={args.patience})",
            "grad_clip_max_norm": 1.0,
            "early_stop_patience": args.early_stop,
            "max_epochs": args.epochs,
            "checkpoint": "best validation MSLE epoch",
            "seed": args.seed,
            "device": str(device),
        },
        "data": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "metrics": metrics,
    }
    (out_dir / "casmotif_evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "scaler.json").write_text(json.dumps(scaler.state_dict(), indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f" CasMotif REDDIT TEST RESULTS ({args.observation_time}s, seed={args.seed})")
    print("=" * 72)
    for k in ("MSLE_log2", "MSLE_ln", "R2_log2", "PCC_log2", "MAPE_log2", "MAE_count", "RMSE_count", "MedAE_count"):
        print(f"  {k:<14} {metrics[k]:.4f}")
    print(f"• Report -> {out_dir / 'casmotif_evaluation_report.json'}", flush=True)


if __name__ == "__main__":
    main()
