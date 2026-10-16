"""Train and evaluate the temporal-motif BiGRU+attention model on the DeepHawkes
Sina Weibo benchmark under a 15-minute early-observation window.

Primary task is regression on log2(increment + 1), matching the published
DeepHawkes objective, so the reported MSLE sits in the same units as the numbers
in that literature. An optional top-k% classification mode is our own adaptation.
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
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from pipeline_c.data import LengthGroupedBatchSampler

from .data import SIDE_DIM, SequenceScaler, WeiboCascadeDataset, pad_collate_fn
from .model import MeanPredictor, build_model
from .weibo_parser import FEATURE_DIM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(requested: str | None) -> torch.device:
    """CPU is the default on Apple silicon *on purpose*.

    PyTorch's MPS backend has no native kernel for packed variable-length GRUs, so
    it falls back and thrashes: measured on this dataset at batch 64, MPS runs
    ~2640 ms/batch against ~420 ms/batch on CPU -- 6.3x slower. Pass --device mps
    to override if a future PyTorch fixes that.
    """
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(pred_log2: np.ndarray, true_log2: np.ndarray, true_raw: np.ndarray) -> dict[str, float]:
    """All inputs in log2(y+1) space except ``true_raw`` which is the count.

    MSLE here is mean squared error in log2 space -- exactly the DeepHawkes
    objective. The natural-log variant is also reported because some papers use
    ln instead of log2, and the two differ by a constant factor of ln(2)^2.
    """
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

    return {
        "MSLE_log2": msle_log2,
        "MSLE_ln": msle_log2 * (math.log(2.0) ** 2),
        "RMSLE_log2": float(math.sqrt(msle_log2)),
        "MAE_count": float(np.mean(np.abs(err))),
        "RMSE_count": float(math.sqrt(np.mean(err ** 2))),
        "MedAE_count": float(np.median(np.abs(err))),
        "R2_log2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "PCC_log2": pcc,
        "n_samples": int(len(true_raw)),
    }


def classification_metrics(
    val_pred_log2: np.ndarray,
    val_true_raw: np.ndarray,
    test_pred_log2: np.ndarray,
    test_true_raw: np.ndarray,
    cutoff: float,
) -> dict[str, float]:
    """Our own adaptation: is this cascade in the top-k% by final popularity?

    The model is a regressor trained on MSLE, which shrinks its predictions toward
    the mean -- so comparing the predicted *count* against the raw cutoff almost
    never fires and yields degenerate precision/recall. Instead we keep the
    regressor's score as the ranking signal and pick the decision threshold that
    maximises F1 on the validation split, then apply that fixed threshold to test.
    ROC-AUC and PR-AUC are threshold-free and unaffected by this choice.
    """
    from sklearn.metrics import (
        accuracy_score, average_precision_score, f1_score,
        precision_score, recall_score, roc_auc_score,
    )

    val_pred = np.asarray(val_pred_log2, dtype=np.float64).ravel()
    test_pred = np.asarray(test_pred_log2, dtype=np.float64).ravel()
    y_val = (np.asarray(val_true_raw, dtype=np.float64).ravel() >= cutoff).astype(int)
    y_test = (np.asarray(test_true_raw, dtype=np.float64).ravel() >= cutoff).astype(int)

    # Sweep candidate thresholds over the validation score distribution.
    best_thr, best_f1 = float(np.median(val_pred)), -1.0
    if y_val.sum() > 0:
        for thr in np.unique(np.quantile(val_pred, np.linspace(0.50, 0.999, 200))):
            f1 = f1_score(y_val, (val_pred >= thr).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)

    y_pred = (test_pred >= best_thr).astype(int)
    both = len(np.unique(y_test)) == 2
    return {
        "cutoff_count": float(cutoff),
        "decision_threshold_log2": float(best_thr),
        "val_f1_at_threshold": float(best_f1),
        "positive_rate": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_pred)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y_test, test_pred)) if both else float("nan"),
    }


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, device, optimizer=None, ema=None) -> float:
    training = optimizer is not None
    model.train(training)
    total, batches = 0.0, 0

    for x, mask, y_log, _, *side in loader:
        x, mask, y_log = x.to(device), mask.to(device), y_log.to(device)
        side = [t.to(device) for t in side]
        with torch.set_grad_enabled(training):
            pred = model(x, mask, *side)
            loss = criterion(pred, y_log)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if ema is not None:
                    ema.update_parameters(model)
        total += float(loss.item())
        batches += 1

    return total / max(batches, 1)


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, trues_log, trues_raw = [], [], []
    for x, mask, y_log, y_raw, *side in loader:
        out = model(x.to(device), mask.to(device), *[t.to(device) for t in side])
        preds.append(out.cpu().numpy().ravel())
        trues_log.append(y_log.numpy().ravel())
        trues_raw.append(y_raw.numpy().ravel())
    return np.concatenate(preds), np.concatenate(trues_log), np.concatenate(trues_raw)


def train_one(name, args, loaders, mean_log_target, device, out_dir) -> dict:
    train_loader, val_loader, test_loader = loaders
    side_dim = SIDE_DIM if (args.side_features and name == "attention_gru") else 0
    model = build_model(name, FEATURE_DIM, args.hidden_size, args.num_layers,
                        args.attention_dim, args.dropout, side_dim).to(device)
    model.init_target_bias(mean_log_target)

    criterion = nn.MSELoss()          # MSE on log2(y+1) == MSLE
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=args.patience, factor=0.5
        )
    # An exponential moving average of the weights is what gets validated and saved:
    # it smooths the epoch-to-epoch jitter in val MSLE that made early stopping fire early.
    ema = (AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay))
           if args.ema_decay > 0 else None)
    eval_model = ema.module if ema is not None else model

    best_val, stale = float("inf"), 0
    ckpt = out_dir / f"best_{name}.pt"
    history = []

    print(f"\n--- Training {name} ({sum(p.numel() for p in model.parameters()):,} params) ---", flush=True)
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, device, optimizer, ema)
        va = run_epoch(eval_model, val_loader, criterion, device)
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(va)
        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"model": name, "epoch": epoch, "train_msle": tr, "val_msle": va, "lr": lr_now})
        print(f"[{name}] epoch {epoch:03d}/{args.epochs:03d}  train_MSLE={tr:.4f}  val_MSLE={va:.4f}  lr={lr_now:.2e}", flush=True)

        if va < best_val - 1e-5:
            best_val, stale = va, 0
            torch.save(eval_model.state_dict(), ckpt)
        else:
            stale += 1
            if stale >= args.early_stop:
                print(f"[{name}] early stop at epoch {epoch}", flush=True)
                break

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    val_pred, val_true_log, val_true_raw = collect_predictions(model, val_loader, device)
    if test_loader is None:
        # Tuning mode: model selection must never see the test split.
        pred, true_log, true_raw = val_pred, val_true_log, val_true_raw
    else:
        pred, true_log, true_raw = collect_predictions(model, test_loader, device)
    metrics = regression_metrics(pred, true_log, true_raw)
    metrics["evaluated_on"] = "val" if test_loader is None else "test"
    metrics["n_params"] = sum(p.numel() for p in model.parameters())
    metrics["best_val_MSLE_log2"] = float(best_val)
    return {
        "metrics": metrics,
        "history": history,
        "predictions": (pred, true_log, true_raw),
        "val_predictions": (val_pred, val_true_raw),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    if device.type == "cpu" and args.threads > 0:
        torch.set_num_threads(args.threads)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print("=" * 70)
    print(f" PIPELINE W: {args.observation_window_note // 60}-MINUTE OBSERVATION ON THE DEEPHAWKES WEIBO BENCHMARK")
    print("=" * 70)
    print(f"• Device: {device}  (threads={torch.get_num_threads()}, bucketed_batches={args.bucket})", flush=True)

    side_window = args.observation_window_note if args.side_features else None
    train_ds = WeiboCascadeDataset(data_dir / "weibo_train.npz", target=args.target, side_window=side_window)
    scaler = SequenceScaler().fit(train_ds.X_flat)
    train_ds.scaler = scaler
    val_ds = WeiboCascadeDataset(data_dir / "weibo_val.npz", scaler=scaler, target=args.target,
                                 side_window=side_window)
    test_ds = WeiboCascadeDataset(data_dir / "weibo_test.npz", scaler=scaler, target=args.target,
                                  side_window=side_window)
    if args.side_features:
        t_max = float(train_ds.X_flat[:, 8].max())
        if t_max >= args.observation_window_note:
            raise SystemExit(f"--observation-window-note {args.observation_window_note} is shorter "
                             f"than the data's events ({t_max:.0f}s); the side features would be wrong.")
        side_mean = train_ds.side.mean(axis=0)
        side_std = np.maximum(train_ds.side.std(axis=0), 1e-6)
        for ds in (train_ds, val_ds, test_ds):
            ds.set_side_scaling(side_mean, side_std)

    mean_log_target = float(train_ds.log_targets().mean())
    print(f"• Cascades: train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}", flush=True)
    print(f"• Events/cascade: mean={train_ds.lengths.mean():.1f}  max={train_ds.lengths.max()}", flush=True)
    print(f"• Target = {args.target}; mean log2(y+1) = {mean_log_target:.4f}", flush=True)

    def loader(ds, shuffle):
        if args.bucket:
            return DataLoader(
                ds,
                batch_sampler=LengthGroupedBatchSampler(
                    lengths=ds.lengths.tolist(), batch_size=args.batch_size, shuffle=shuffle
                ),
                collate_fn=pad_collate_fn,
                num_workers=args.num_workers,
            )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=pad_collate_fn, num_workers=args.num_workers)

    loaders = (loader(train_ds, True), loader(val_ds, False),
               None if args.skip_test else loader(test_ds, False))

    # Non-learned reference: predict the training mean for every cascade.
    test_true_log = test_ds.log_targets()
    baseline = MeanPredictor(mean_log_target)
    results = {} if args.skip_test else {
        "mean_baseline": {
            "metrics": regression_metrics(
                baseline.predict(len(test_ds)).numpy().ravel(), test_true_log, test_ds.y_raw
            )
        }
    }

    all_history = []
    best_preds = None
    best_val_preds = None
    for name in args.models:
        res = train_one(name, args, loaders, mean_log_target, device, out_dir)
        all_history.extend(res["history"])
        if name == "attention_gru":
            best_preds = res["predictions"]
            best_val_preds = res["val_predictions"]
        results[name] = {"metrics": res["metrics"]}

    # Optional top-k% classification framing (our adaptation, not the DeepHawkes task).
    if args.classify_top_pct > 0 and best_preds is not None and not args.skip_test:
        # Cutoff comes from the training split only -- deriving it from test would
        # leak the test label distribution into the task definition.
        cutoff = float(np.percentile(train_ds.y_raw, 100.0 - args.classify_top_pct))
        pred, _, true_raw = best_preds
        val_pred, val_true_raw = best_val_preds
        results["attention_gru"]["classification"] = classification_metrics(
            val_pred, val_true_raw, pred, true_raw, cutoff
        )

    report = {
        "setting": {
            "dataset": "Sina Weibo (released with DeepHawkes, Cao et al., CIKM 2017)",
            "observation_window_seconds": args.observation_window_note,
            "prediction_horizon_hours": 24,
            "target": f"{args.target} retweets (DeepHawkes target = increment after the window)",
            "loss": "MSE on log2(y+1), i.e. MSLE in log2 units",
            "split": "chronological by publish time, 70/15/15 (DeepHawkes protocol)",
            "feature_dim": FEATURE_DIM,
            "note": (
                "DeepHawkes and CasCN report results at 1h/2h/3h observation windows. "
                + ("This run uses the 1-hour window, directly comparable in setting."
                   if args.observation_window_note == 3600 else
                   f"This run uses a {args.observation_window_note // 60}-minute window, which is "
                   "NOT directly comparable to their published numbers.")
            ),
            "training": {
                "seed": args.seed, "scheduler": args.scheduler, "ema_decay": args.ema_decay,
                "early_stop": args.early_stop, "epochs": args.epochs, "lr": args.lr,
                "hidden_size": args.hidden_size, "num_layers": args.num_layers,
                "side_features": args.side_features, "evaluated_on": "val" if args.skip_test else "test",
            },
        },
        "data": {
            "train_cascades": len(train_ds),
            "val_cascades": len(val_ds),
            "test_cascades": len(test_ds),
            "mean_events_per_cascade": float(train_ds.lengths.mean()),
        },
        "results": {k: v for k, v in results.items()},
    }

    (out_dir / "weibo_evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "training_history.json").write_text(json.dumps(all_history, indent=2), encoding="utf-8")
    (out_dir / "scaler.json").write_text(json.dumps(scaler.state_dict(), indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    split_name = "VALIDATION (tuning mode)" if args.skip_test else "TEST-SET"
    print(f" {split_name} RESULTS ({args.observation_window_note}s observation -> 24-hour popularity)")
    print("=" * 70)
    header = f"{'model':<18}{'MSLE(log2)':>12}{'MAE':>12}{'RMSE':>12}{'R2':>9}{'PCC':>8}"
    print(header)
    print("-" * len(header))
    for name, payload in results.items():
        m = payload["metrics"]
        print(f"{name:<18}{m['MSLE_log2']:>12.4f}{m['MAE_count']:>12.2f}"
              f"{m['RMSE_count']:>12.2f}{m['R2_log2']:>9.4f}{m['PCC_log2']:>8.4f}")

    if "classification" in results.get("attention_gru", {}):
        c = results["attention_gru"]["classification"]
        print(f"\nTop-{args.classify_top_pct:g}% classification (our adaptation), "
              f"cutoff={c['cutoff_count']:.0f} retweets, positives={c['positive_rate']:.2%}")
        print(f"  Accuracy={c['accuracy']:.4f}  Precision={c['precision']:.4f}  "
              f"Recall={c['recall']:.4f}  F1={c['f1']:.4f}  ROC-AUC={c['roc_auc']:.4f}  PR-AUC={c['pr_auc']:.4f}")

    print(f"\n• Report written to {out_dir / 'weibo_evaluation_report.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the cascade model on the Weibo benchmark.")
    p.add_argument("--data-dir", default="weibo_data/processed")
    p.add_argument("--output-dir", default="pipeline_w/outputs")
    p.add_argument("--target", choices=["increment", "total"], default="increment",
                   help="'increment' matches the DeepHawkes objective (growth after the window).")
    p.add_argument("--models", nargs="+", default=["attention_gru", "mean_pool", "max_pool"])
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--attention-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=3, help="LR-plateau patience.")
    p.add_argument("--early-stop", type=int, default=8, help="Early-stopping patience.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None,
                   help="Defaults to CPU: MPS is ~6x slower for packed GRUs (see pick_device).")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = leave default).")
    p.add_argument("--bucket", action="store_true", default=True,
                   help="Length-bucketed batching; cuts padding waste ~10x on this dataset.")
    p.add_argument("--no-bucket", action="store_false", dest="bucket")
    p.add_argument("--classify-top-pct", type=float, default=5.0,
                   help="Also report top-k%% classification metrics (0 to disable).")
    p.add_argument("--observation-window-note", type=int, default=900,
                   help="Observation window in seconds; must match the parser setting. "
                        "Also used to build the side features.")
    p.add_argument("--scheduler", choices=["plateau", "cosine"], default="plateau")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="Validate and save an EMA of the weights (0 = off, e.g. 0.999).")
    p.add_argument("--side-features", action="store_true",
                   help="Feed publish hour and momentum summaries to the attention_gru head.")
    p.add_argument("--skip-test", action="store_true",
                   help="Tuning mode: never touch the test split; report validation metrics.")
    return p.parse_args()


if __name__ == "__main__":
    main()
