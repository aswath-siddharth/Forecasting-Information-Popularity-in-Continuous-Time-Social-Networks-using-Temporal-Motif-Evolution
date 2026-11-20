"""Train CasFlow across fixed seeds (42, 43, 44) under strict controlled protocol."""

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

from .data import Cascades, collate, load_split_cascades, prepare_tensors
from .model import CasFlow


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def regression_metrics(pred_log2: np.ndarray, true_log2: np.ndarray, true_raw: np.ndarray) -> dict[str, float]:
    pred_log2 = np.asarray(pred_log2, dtype=np.float64).ravel()
    true_log2 = np.asarray(true_log2, dtype=np.float64).ravel()
    true_raw = np.asarray(true_raw, dtype=np.float64).ravel()

    residual = pred_log2 - true_log2
    msle_log2 = float(np.mean(residual ** 2))

    pred_raw = np.clip(np.exp2(np.clip(pred_log2, -30, 30)) - 1.0, 0.0, None)
    err = pred_raw - true_raw

    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((true_log2 - true_log2.mean()) ** 2))
    pcc = float(np.corrcoef(pred_log2, true_log2)[0, 1]) if len(pred_log2) > 1 and pred_log2.std() > 0 else float("nan")

    nz = true_log2 > 0
    mape_log2 = float(np.mean(np.abs((true_log2[nz] - pred_log2[nz]) / true_log2[nz]))) if np.any(nz) else float("nan")
    male_log2 = float(np.mean(np.abs(residual)))

    return {
        "MSLE_log2": msle_log2,
        "MSLE_ln": msle_log2 * (math.log(2.0) ** 2),
        "RMSLE_log2": float(math.sqrt(msle_log2)),
        "MAE_count": float(np.mean(np.abs(err))),
        "RMSE_count": float(math.sqrt(np.mean(err ** 2))),
        "MedAE_count": float(np.median(np.abs(err))),
        "R2_log2": float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan"),
        "PCC_log2": pcc,
        "MAPE_log2": mape_log2,
        "MALE_log2": male_log2,
        "n_samples": len(pred_log2),
    }


@torch.no_grad()
def predict_casflow(model: nn.Module, loader: DataLoader, device: torch.device, seed: int = 42, samples: int = 8) -> np.ndarray:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    all_preds = []
    for x, lengths, _ in loader:
        pred, _ = model(x.to(device), lengths, generator=generator, samples=samples)
        all_preds.append(pred.cpu().numpy().ravel())
    return np.concatenate(all_preds)


def train_seed(seed: int, prepared: dict, out_dir: Path, args: argparse.Namespace) -> dict:
    run_dir = out_dir / f"s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = torch.device(args.device)

    train_ds = Cascades(prepared["train"])
    val_ds = Cascades(prepared["val"])
    test_ds = Cascades(prepared["test"])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, generator=torch.Generator().manual_seed(seed),
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=128, shuffle=False,
        collate_fn=collate, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=128, shuffle=False,
        collate_fn=collate, pin_memory=(device.type == "cuda")
    )

    model = CasFlow(
        input_dim=80,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_flows=args.n_flows
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Seed {seed}] Initialized CasFlow with {total_params:,} parameters on {device}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    val_true_raw = prepared["val"]["labels"].numpy()
    val_true_log2 = np.log2(val_true_raw + 1.0)
    test_true_raw = prepared["test"]["labels"].numpy()
    test_true_log2 = np.log2(test_true_raw + 1.0)

    best_val_msle = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    t_start = time.perf_counter()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_pred_loss_total = 0.0
        n_train = 0

        for x, lengths, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
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
            train_pred_loss_total += float(pred_loss.item()) * bs
            n_train += bs

        train_loss = train_loss_total / n_train
        train_pred_msle = train_pred_loss_total / n_train

        # Validation evaluation
        val_pred_raw = predict_casflow(model, val_loader, device, seed=seed + 1000, samples=args.samples)
        val_pred_log2 = np.log2(np.clip(val_pred_raw, 0.0, None) + 1.0)
        val_metrics = regression_metrics(val_pred_log2, val_true_log2, val_true_raw)
        val_msle = val_metrics["MSLE_log2"]

        scheduler.step(val_msle)
        lr_curr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_msle_log2": train_pred_msle,
            "val_msle_log2": val_msle,
            "val_r2": val_metrics["R2_log2"],
            "lr": lr_curr,
            "elapsed_s": time.perf_counter() - t_start,
        }
        history.append(row)

        print(
            f"[Seed {seed} | Epoch {epoch:02d}] Train Loss: {train_loss:.4f} | "
            f"Train MSLE: {train_pred_msle:.4f} | Val MSLE: {val_msle:.4f} | "
            f"Val R²: {val_metrics['R2_log2']:.4f} | LR: {lr_curr:.1e}",
            flush=True,
        )

        if val_msle < best_val_msle:
            best_val_msle = val_msle
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), run_dir / "best_casflow.pt")
        else:
            stale += 1
            if stale >= args.early_stop:
                print(f"[Seed {seed}] Early stopping triggered at epoch {epoch} (best epoch: {best_epoch})", flush=True)
                break

    # Load best checkpoint and evaluate on Test split
    print(f"[Seed {seed}] Loading best checkpoint from epoch {best_epoch} (Val MSLE: {best_val_msle:.4f})...", flush=True)
    model.load_state_dict(torch.load(run_dir / "best_casflow.pt", map_location=device))
    test_pred_raw = predict_casflow(model, test_loader, device, seed=seed + 2000, samples=args.samples)
    test_pred_log2 = np.log2(np.clip(test_pred_raw, 0.0, None) + 1.0)
    test_metrics = regression_metrics(test_pred_log2, test_true_log2, test_true_raw)
    elapsed_total = time.perf_counter() - t_start
    test_metrics["best_epoch"] = best_epoch
    test_metrics["best_val_MSLE_log2"] = best_val_msle
    test_metrics["parameters"] = total_params
    test_metrics["train_time_sec"] = elapsed_total
    test_metrics["train_time_min"] = elapsed_total / 60.0

    # Save outputs
    (run_dir / "casflow_evaluation_report.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    (run_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    np.savez_compressed(
        run_dir / "test_predictions.npz",
        pred_log2=test_pred_log2.astype(np.float32),
        pred_raw=test_pred_raw.astype(np.float32),
        true_log2=test_true_log2.astype(np.float32),
        true_raw=test_true_raw.astype(np.float32),
        cascade_id=np.asarray(prepared["test"]["ids"], dtype=object),
    )

    print(
        f"[Seed {seed} DONE] Time: {elapsed_total:.1f}s ({elapsed_total/60:.2f} min) | "
        f"Test MSLE (log2): {test_metrics['MSLE_log2']:.4f} | "
        f"Test MSLE (ln): {test_metrics['MSLE_ln']:.4f} | Test R²: {test_metrics['R2_log2']:.4f} | "
        f"Test MAPE: {test_metrics['MAPE_log2']:.4f}",
        flush=True,
    )
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train CasFlow baseline on Sina Weibo under controlled protocol.")
    parser.add_argument("--weibo-txt", type=Path, default=Path("weibo_data/dataset_weibo.txt"))
    parser.add_argument("--splits-dir", type=Path, default=Path("weibo_data/processed_3600"))
    parser.add_argument("--cache-file", type=Path, default=Path("weibo_data/casflow_prepared_3600.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("casflow/runs_3600/casflow"))
    parser.add_argument("--observation-time", type=int, default=3600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-flows", type=int, default=8)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load or prepare tensors
    if args.cache_file.exists():
        print(f"Loading cached CasFlow tensors from {args.cache_file}...", flush=True)
        prepared = torch.load(args.cache_file, weights_only=False)
    else:
        print(f"Preparing CasFlow tensors from {args.weibo_txt} and {args.splits_dir} (T_obs={args.observation_time}s)...", flush=True)
        splits = load_split_cascades(args.weibo_txt, args.splits_dir, observation_time=args.observation_time)
        prepared = prepare_tensors(splits, observation_time=args.observation_time, device=args.device, batch_size=args.batch_size)
        args.cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(prepared, args.cache_file)
        print(f"Saved cached CasFlow tensors to {args.cache_file}", flush=True)

    # 2. Train across seeds
    all_metrics = {}
    for seed in args.seeds:
        metrics = train_seed(seed, prepared, args.out_dir, args)
        all_metrics[f"seed_{seed}"] = metrics

    # 3. Compute multi-seed summary if more than 1 seed
    if len(args.seeds) > 1:
        msle_vals = [all_metrics[f"seed_{s}"]["MSLE_log2"] for s in args.seeds]
        r2_vals = [all_metrics[f"seed_{s}"]["R2_log2"] for s in args.seeds]
        print("\n=======================================================", flush=True)
        print(f"CasFlow Multi-Seed Mean MSLE (log2): {np.mean(msle_vals):.4f} ± {np.std(msle_vals):.4f}", flush=True)
        print(f"CasFlow Multi-Seed Mean R² (log2)  : {np.mean(r2_vals):.4f} ± {np.std(r2_vals):.4f}", flush=True)
        print("=======================================================\n", flush=True)
    else:
        s = args.seeds[0]
        m = all_metrics[f"seed_{s}"]
        print("\n=======================================================", flush=True)
        print(f"CasFlow Seed {s} Test MSLE (log2): {m['MSLE_log2']:.4f} | R²: {m['R2_log2']:.4f} | Time: {m['train_time_min']:.2f} min", flush=True)
        print("=======================================================\n", flush=True)


if __name__ == "__main__":
    main()
