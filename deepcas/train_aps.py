"""Prepare aligned APS random walks and train DeepCas across seeds.

Reads the exact chronological splits and observed paths from ``aps_data/processed_{obs_days}/``
so cascade populations, 20-year future increment targets (R_7300d - R_{obs_days}), and
splits match CasMotif, DeepHawkes, CasCN, CasFlow, and CasSubTS 1-to-1.
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

from .data import TwitterDeepCasDataset, collate_deepcas
from .model import DeepCasModel
from .train import compute_metrics, evaluate_model, run_epoch, set_seed
from .train_weibo import _sample_walks_for_paths, aggregate_reports


def prepare_aps_deepcas_cache(
    aligned_dir: Path,
    cache_dir: Path,
    obs_days: int,
    horizon_days: int = 7300,
    n_walks: int = 32,
    walk_length: int = 10,
    preprocess_seed: int = 42,
    force: bool = False,
) -> dict:
    """Build or load cached DeepCas walk tensors aligned with APS processed splits."""
    meta_file = cache_dir / "vocab_meta.json"
    split_files = [cache_dir / f"aps_{s}.npz" for s in ("train", "val", "test")]
    if not force and meta_file.exists() and all(p.exists() for p in split_files):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if (
            meta.get("obs_days") == obs_days
            and meta.get("n_walks") == n_walks
            and meta.get("walk_length") == walk_length
        ):
            print(f"• Using cached DeepCas APS walks in {cache_dir}", flush=True)
            return meta

    cache_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, dict] = {}
    cascade_paths: dict[str, list[list[str]]] = {}

    t0 = time.time()
    for split in ("train", "val", "test"):
        src = np.load(aligned_dir / f"aps_{split}.npz", allow_pickle=True)
        cids = np.array([str(c) for c in src["cascade_id"]], dtype=object)
        splits[split] = {
            "cascade_id": cids,
            "y_increment": src["y_increment"].astype(np.float32),
            "y_total": src["y_total"].astype(np.float32),
            "observed_size": src["observed_size"].astype(np.float32),
            "publish_time": np.array([str(x) for x in src["publish_time"]], dtype=object),
        }

        raw_map = json.loads((aligned_dir / f"raw_paths_{split}.json").read_text(encoding="utf-8"))
        for cid in cids:
            entry = raw_map[str(cid)]
            cascade_paths[str(cid)] = [
                [str(u) for u in p]
                for p, t in zip(entry["paths"], entry["times"])
                if float(t) <= float(obs_days)
            ]

    print(
        f"• Loaded {len(cascade_paths):,} aligned APS cascades from {aligned_dir} in {time.time() - t0:.1f}s",
        flush=True,
    )

    vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for cid in splits["train"]["cascade_id"]:
        for path in cascade_paths[str(cid)]:
            for u in path:
                if u not in vocab:
                    vocab[u] = len(vocab)

    vocab_size = len(vocab)
    print(f"• Built train-only APS paper vocabulary: {vocab_size:,} tokens", flush=True)

    rng = random.Random(preprocess_seed)
    split_counts: dict[str, int] = {}
    unk_tokens = 0
    total_tokens = 0

    for split in ("train", "val", "test"):
        cids = splits[split]["cascade_id"]
        n = len(cids)
        split_counts[split] = n
        walks_arr = np.zeros((n, n_walks, walk_length), dtype=np.int64)

        for idx, cid in enumerate(cids):
            raw_walks = _sample_walks_for_paths(
                cascade_paths[str(cid)],
                n_walks=n_walks,
                walk_length=walk_length,
                rng=rng,
            )
            for w_idx, walk in enumerate(raw_walks):
                for step_idx, node_name in enumerate(walk):
                    if node_name == "<PAD>":
                        tid = 0
                    else:
                        tid = vocab.get(node_name, 1)
                        total_tokens += 1
                        if tid == 1:
                            unk_tokens += 1
                    walks_arr[idx, w_idx, step_idx] = tid

        np.savez_compressed(
            cache_dir / f"aps_{split}.npz",
            walks=walks_arr,
            observed_size=splits[split]["observed_size"],
            publish_time=splits[split]["publish_time"],
            cascade_id=splits[split]["cascade_id"],
            y_increment=splits[split]["y_increment"],
            y_total=splits[split]["y_total"],
        )

    unk_rate = (unk_tokens / max(1, total_tokens)) * 100.0
    meta = {
        "obs_days": obs_days,
        "t_obs": obs_days,
        "horizon_days": horizon_days,
        "vocab_size": vocab_size,
        "n_walks": n_walks,
        "walk_length": walk_length,
        "preprocess_seed": preprocess_seed,
        "split_counts": split_counts,
        "unk_token_rate_pct": round(unk_rate, 3),
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"• Cached DeepCas APS tensors to {cache_dir} "
        f"(train={split_counts['train']:,}, val={split_counts['val']:,}, test={split_counts['test']:,}, "
        f"UNK rate={unk_rate:.2f}%)",
        flush=True,
    )
    return meta


def train_single_seed_aps(
    cache_dir: Path,
    out_dir: Path,
    meta: dict,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    lr_patience: int,
    early_stop: int,
    embedding_dim: int,
    hidden_gru: int,
    dropout: float,
    device: torch.device,
) -> dict:
    """Train DeepCas for one seed on APS and save evaluation artifacts."""
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TwitterDeepCasDataset(cache_dir / "aps_train.npz")
    val_ds = TwitterDeepCasDataset(cache_dir / "aps_val.npz")
    test_ds = TwitterDeepCasDataset(cache_dir / "aps_test.npz")

    mean_train_log = float(train_ds.y_log2.mean())
    mean_preds = np.full(len(test_ds), mean_train_log, dtype=np.float64)
    baseline_metrics = compute_metrics(mean_preds, test_ds.y_log2, test_ds.y_raw)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_deepcas, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_deepcas, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_deepcas, num_workers=0)

    model = DeepCasModel(
        vocab_size=meta["vocab_size"],
        embedding_dim=embedding_dim,
        hidden_gru=hidden_gru,
        walk_length=meta["walk_length"],
        dropout=dropout,
    ).to(device)
    model.init_target_bias(mean_train_log)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"• Model params: {n_params:,} | Naive baseline Test MSLE (log2): {baseline_metrics['MSLE_log2']:.4f}", flush=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=lr_patience)

    best_val_msle = float("inf")
    best_epoch = 0
    stale_epochs = 0
    ckpt_path = out_dir / "best_deepcas.pt"
    history = []
    t_start = time.time()

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

        improved = va_loss < best_val_msle
        if improved:
            best_val_msle = va_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
            marker = " *"
        else:
            stale_epochs += 1
            marker = ""

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train MSLE: {tr_loss:.4f} | "
            f"Val MSLE: {va_loss:.4f} | LR: {lr_current:.2e}{marker}",
            flush=True,
        )

        if stale_epochs >= early_stop:
            print(f"• Early stopping at epoch {epoch} (best epoch: {best_epoch})", flush=True)
            break

    elapsed = time.time() - t_start
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_preds, test_true_log, test_true_raw = evaluate_model(model, test_loader, device)
    test_metrics = compute_metrics(test_preds, test_true_log, test_true_raw)

    report = {
        "model": "DeepCas (Li et al., WWW 2017)",
        "dataset": "APS Citations (cascades/aps.parquet)",
        "observation_window_days": meta["obs_days"],
        "observation_window_years": meta["obs_days"] / 365.0,
        "prediction_horizon_days": meta["horizon_days"],
        "prediction_horizon_years": meta["horizon_days"] / 365.0,
        "target": f"Future Increment (R_{meta['horizon_days']}d - R_{meta['obs_days']}d)",
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_msle": best_val_msle,
        "train_time_seconds": round(elapsed, 2),
        "n_params": n_params,
        "split_counts": meta["split_counts"],
        "test_metrics": test_metrics,
        "naive_baseline_metrics": baseline_metrics,
        "hyperparameters": {
            "batch_size": batch_size,
            "optimizer": "AdamW",
            "weight_decay": weight_decay,
            "initial_lr": lr,
            "early_stopping_patience": early_stop,
            "lr_plateau_patience": lr_patience,
            "max_grad_norm": 1.0,
            "embedding_dim": embedding_dim,
            "hidden_gru": hidden_gru,
            "dropout": dropout,
            "walk_length": meta["walk_length"],
            "n_walks": meta["n_walks"],
            "vocab_size": meta["vocab_size"],
        },
    }

    (out_dir / "deepcas_evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "test_predictions.npz",
        pred_log2=test_preds,
        true_log2=test_true_log,
        true_raw=test_true_raw,
        cascade_id=test_ds.cascade_id,
    )

    print(
        f"• Seed {seed} finished in {elapsed:.1f}s | Test MSLE(log2)={test_metrics['MSLE_log2']:.4f} | "
        f"RMSLE(log2)={test_metrics['RMSLE_log2']:.4f} | R2={test_metrics['R2_log2']:.4f} | "
        f"PCC={test_metrics['PCC_log2']:.4f} | MAE={test_metrics['MAE_count']:.2f} | MedAE={test_metrics['MedAE_count']:.2f}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepCas on APS Citation dataset under controlled protocol.")
    parser.add_argument("--obs-days", type=int, default=1095, choices=[1095, 1825])
    parser.add_argument("--horizon-days", type=int, default=7300)
    parser.add_argument("--aligned-dir", default=None, help="Defaults to aps_data/processed_{obs_days}")
    parser.add_argument("--cache-dir", default=None, help="Defaults to aps_data/deepcas_{obs_days}d")
    parser.add_argument("--output-dir", default=None, help="Defaults to deepcas/runs_aps_{obs_days}d")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--early-stop", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=50)
    parser.add_argument("--hidden-gru", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-walks", type=int, default=32)
    parser.add_argument("--walk-length", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()

    obs_days = args.obs_days
    aligned_dir = Path(args.aligned_dir or f"aps_data/processed_{obs_days}")
    cache_dir = Path(args.cache_dir or f"aps_data/deepcas_{obs_days}d")
    out_root = Path(args.output_dir or f"deepcas/runs_aps_{obs_days}d")
    device = torch.device(args.device)

    print("=" * 76)
    print(
        f" DEEPCAS APS BENCHMARK | T_obs={obs_days}d ({obs_days // 365}y) | "
        f"T_pred={args.horizon_days}d ({args.horizon_days // 365}y) | seeds={args.seeds} | device={device}"
    )
    print("=" * 76, flush=True)

    meta = prepare_aps_deepcas_cache(
        aligned_dir=aligned_dir,
        cache_dir=cache_dir,
        obs_days=obs_days,
        horizon_days=args.horizon_days,
        n_walks=args.n_walks,
        walk_length=args.walk_length,
        force=args.force_preprocess,
    )

    reports = []
    for seed in args.seeds:
        seed_dir = out_root / f"s{seed}"
        report_file = seed_dir / "deepcas_evaluation_report.json"
        if not args.force_train and report_file.exists():
            print(f"• Loading existing report for seed {seed}: {report_file}", flush=True)
            reports.append(json.loads(report_file.read_text(encoding="utf-8")))
            continue

        print(f"\n--- Training DeepCas APS ({obs_days}d) | seed={seed} ---", flush=True)
        report = train_single_seed_aps(
            cache_dir=cache_dir,
            out_dir=seed_dir,
            meta=meta,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lr_patience=args.patience,
            early_stop=args.early_stop,
            embedding_dim=args.embedding_dim,
            hidden_gru=args.hidden_gru,
            dropout=args.dropout,
            device=device,
        )
        reports.append(report)

    summary = aggregate_reports(reports, out_root, meta)
    summary["dataset"] = "APS Citations (cascades/aps.parquet)"
    summary["observation_window_days"] = obs_days
    summary["observation_window_years"] = obs_days / 365.0
    summary["prediction_horizon_days"] = args.horizon_days
    summary["prediction_horizon_years"] = args.horizon_days / 365.0
    summary["target"] = f"Future Increment (R_{args.horizon_days}d - R_{obs_days}d)"
    summary.pop("observation_window_seconds", None)
    summary.pop("observation_window_hours", None)
    summary.pop("prediction_horizon_hours", None)
    (out_root / "summary_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print(f" DEEPCAS APS {obs_days}d ({obs_days // 365}y) 3-SEED SUMMARY ({summary['seeds']})")
    print("=" * 76)
    for k in ("MSLE_log2", "RMSLE_log2", "MSLE_ln", "R2_log2", "PCC_log2", "MAPE", "MAE_count", "MedAE_count"):
        m = summary["mean_metrics"][k]
        s = summary["std_metrics"][k]
        print(f"  {k:<15}: {m:.4f} ± {s:.4f}")
    print(f"• Summary saved to: {out_root / 'summary_report.json'}", flush=True)


if __name__ == "__main__":
    main()
