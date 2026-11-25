"""Train CasFlow on Reddit cascades under the controlled benchmark protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import Cascades, collate, prepare_tensors
from .model import CasFlow
from .data_reddit import load_reddit_casflow_splits
from .train import regression_metrics, predict_casflow, set_seed


def main() -> None:
    p = argparse.ArgumentParser(description="Train CasFlow on Reddit cascades.")
    p.add_argument("--cache-dir", default="reddit_benchmark/cache")
    p.add_argument("--output-dir", default="casflow/runs_reddit_3600/s42")
    p.add_argument("--observation-time", type=int, default=3600)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-flows", type=int, default=4)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--early-stop", type=int, default=8)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-cascades", type=int, default=0, help="Optional cap for quick testing.")
    args = p.parse_args()

    set_seed(args.seed)
    dev = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f" CASFLOW REDDIT BENCHMARK | window={args.observation_time}s | seed={args.seed} | device={dev}")
    print("=" * 76, flush=True)

    tensors_cache = Path(args.cache_dir) / f"casflow_{args.observation_time}" / "prepared_tensors.pt"
    if tensors_cache.exists() and args.max_cascades == 0:
        print(f"• Loading cached preprocessed CasFlow tensors from {tensors_cache}...", flush=True)
        prepared = torch.load(tensors_cache, weights_only=False)
    else:
        splits = load_reddit_casflow_splits(args.cache_dir, args.observation_time, max_cascades=args.max_cascades)
        prepared = prepare_tensors(splits, observation_time=args.observation_time, device=args.device, batch_size=args.batch_size)
        if args.max_cascades == 0:
            tensors_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(prepared, tensors_cache)
            print(f"• Saved preprocessed CasFlow tensors -> {tensors_cache}", flush=True)

    train_ds = Cascades(prepared["train"])
    val_ds = Cascades(prepared["val"])
    test_ds = Cascades(prepared["test"])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, generator=torch.Generator().manual_seed(args.seed),
        pin_memory=(dev.type == "cuda"),
    )
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate, pin_memory=(dev.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate, pin_memory=(dev.type == "cuda"))

    model = CasFlow(
        input_dim=80,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_flows=args.n_flows,
    ).to(dev)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"• CasFlow model parameters: {total_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.patience)

    val_true_raw = prepared["val"]["labels"].numpy()
    val_true_log2 = np.log2(val_true_raw + 1.0)
    test_true_raw = prepared["test"]["labels"].numpy()
    test_true_log2 = np.log2(test_true_raw + 1.0)

    best_val_msle = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_weights_path = out_dir / "best_casflow.pt"
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        n_train = 0

        for x, lengths, labels in train_loader:
            x, labels = x.to(dev), labels.to(dev)
            optimizer.zero_grad(set_to_none=True)
            pred, terms = model(x, lengths, regularize=True)

            pred_log2 = torch.log2(pred + 1.0)
            true_log2 = torch.log2(labels + 1.0)
            pred_loss = F.mse_loss(pred_log2, true_log2)

            loss = pred_loss + sum(terms.values())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            bs = len(labels)
            train_loss_total += float(loss.item()) * bs
            n_train += bs

        train_loss = train_loss_total / max(1, n_train)

        val_pred_raw = predict_casflow(model, val_loader, dev, seed=args.seed + 1000, samples=args.samples)
        val_pred_log2 = np.log2(np.clip(val_pred_raw, 0.0, None) + 1.0)
        val_metrics = regression_metrics(val_pred_log2, val_true_log2, val_true_raw)
        val_msle = val_metrics["MSLE_log2"]

        scheduler.step(val_msle)
        lr_curr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val MSLE_log2: {val_msle:.4f} (R2: {val_metrics['R2_log2']:+.3f}) | lr: {lr_curr:.2e}",
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

    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    test_pred_raw = predict_casflow(model, test_loader, dev, seed=args.seed + 2000, samples=args.samples)
    test_pred_log2 = np.log2(np.clip(test_pred_raw, 0.0, None) + 1.0)
    test_metrics = regression_metrics(test_pred_log2, test_true_log2, test_true_raw)
    train_time = time.time() - t_start

    print("-" * 76)
    print(f"CasFlow Test Results (Seed {args.seed}) | Window: {args.observation_time}s | Time: {train_time:.1f}s")
    for k, v in test_metrics.items():
        print(f"  {k:<15}: {v:.4f}")
    print("-" * 76, flush=True)

    np.savez_compressed(
        out_dir / "test_predictions.npz",
        pred_log2=test_pred_log2,
        true_log2=test_true_log2,
        true_raw=test_true_raw,
        cascade_id=np.array(prepared["test"]["ids"]),
    )

    report = {
        "model": "CasFlow",
        "dataset": "Reddit",
        "observation_time": args.observation_time,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "train_time_sec": train_time,
        "params": total_params,
        "metrics": test_metrics,
    }
    (out_dir / "casflow_evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"• Saved test report -> {out_dir / 'casflow_evaluation_report.json'}", flush=True)


if __name__ == "__main__":
    main()
