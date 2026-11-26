"""Prepare aligned Weibo random walks and train DeepCas across seeds.

Reads the exact chronological splits from ``weibo_data/processed_{T_obs}/weibo_{train,val,test}.npz``
so cascade populations, target increments (R_24h - R_{T_obs}), and splits match CasMotif,
DeepHawkes, CasCN, CasFlow, and CasSubTS 1-to-1.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import TwitterDeepCasDataset, collate_deepcas
from .model import DeepCasModel
from .train import compute_metrics, evaluate_model, run_epoch, set_seed


def _extract_weibo_graphs(
    raw_file: Path,
    needed_cids: set[str],
    observation_time: int,
) -> dict[str, list[list[str]]]:
    """Stream dataset_weibo.txt once and retain < T_obs diffusion paths for target cascades."""
    cascade_paths: dict[str, list[list[str]]] = {}
    t0 = time.time()
    n_lines = 0

    with raw_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            n_lines += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 5:
                continue
            cid = parts[0]
            if cid not in needed_cids:
                continue

            paths: list[list[str]] = []
            for entry in parts[4].split(" "):
                if not entry:
                    continue
                chain_str, _, t_str = entry.rpartition(":")
                if not chain_str:
                    continue
                nodes = chain_str.split("/")
                if any(n == "-1" for n in nodes):
                    continue
                try:
                    t_now = float(t_str)
                except ValueError:
                    continue
                if t_now < observation_time:
                    paths.append(nodes)

            cascade_paths[cid] = paths
            if len(cascade_paths) == len(needed_cids):
                break

    print(
        f"• Parsed {len(cascade_paths):,}/{len(needed_cids):,} target cascades "
        f"from {n_lines:,} lines in {time.time() - t0:.1f}s",
        flush=True,
    )
    return cascade_paths


def _sample_walks_for_paths(
    paths: list[list[str]],
    n_walks: int,
    walk_length: int,
    rng: random.Random,
) -> list[list[str]]:
    """Construct undirected local graph from observed paths and sample random walks."""
    adj: dict[str, list[str]] = defaultdict(list)
    nodes: set[str] = set()
    node_out_degree: dict[str, int] = defaultdict(int)

    for path in paths:
        if len(path) >= 2:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                adj[u].append(v)
                adj[v].append(u)
                node_out_degree[u] += 1
                nodes.add(u)
                nodes.add(v)
        elif len(path) == 1:
            nodes.add(path[0])

    if not nodes:
        return [["<PAD>"] * walk_length for _ in range(n_walks)]

    node_list = sorted(nodes)
    weights = [node_out_degree.get(n, 0) + 1 for n in node_list]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    walks: list[list[str]] = []
    for _ in range(n_walks):
        curr = rng.choices(node_list, weights=probs, k=1)[0]
        walk = [curr]
        for _ in range(walk_length - 1):
            nbrs = adj.get(curr)
            if nbrs:
                curr = rng.choice(nbrs)
            walk.append(curr)
        walks.append(walk)
    return walks


def prepare_weibo_deepcas_cache(
    raw_file: Path,
    aligned_dir: Path,
    cache_dir: Path,
    observation_time: int,
    n_walks: int = 32,
    walk_length: int = 10,
    preprocess_seed: int = 42,
    force: bool = False,
) -> dict:
    """Build or load cached DeepCas walk tensors aligned with Weibo processed splits."""
    meta_file = cache_dir / "vocab_meta.json"
    split_files = [cache_dir / f"weibo_{s}.npz" for s in ("train", "val", "test")]
    if not force and meta_file.exists() and all(p.exists() for p in split_files):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if (
            meta.get("t_obs") == observation_time
            and meta.get("n_walks") == n_walks
            and meta.get("walk_length") == walk_length
        ):
            print(f"• Using cached DeepCas Weibo walks in {cache_dir}", flush=True)
            return meta

    cache_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, dict[str, np.ndarray]] = {}
    needed_cids: set[str] = set()

    for split in ("train", "val", "test"):
        src = np.load(aligned_dir / f"weibo_{split}.npz", allow_pickle=True)
        cids = np.array([str(c) for c in src["cascade_id"]], dtype=object)
        splits[split] = {
            "cascade_id": cids,
            "y_increment": src["y_increment"].astype(np.float32),
            "y_total": src["y_total"].astype(np.float32),
            "observed_size": src["observed_size"].astype(np.float32),
            "publish_time": src["publish_time"].astype(np.int64),
        }
        needed_cids.update(cids.tolist())

    cascade_paths = _extract_weibo_graphs(raw_file, needed_cids, observation_time)
    missing = len(needed_cids) - len(cascade_paths)
    if missing > 0:
        raise RuntimeError(f"Missing {missing} cascades in {raw_file} for aligned splits in {aligned_dir}")

    # Build train-only user vocabulary (<PAD>=0, <UNK>=1)
    vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for cid in splits["train"]["cascade_id"]:
        for path in cascade_paths[str(cid)]:
            for u in path:
                if u not in vocab:
                    vocab[u] = len(vocab)

    vocab_size = len(vocab)
    print(f"• Built train-only Weibo user vocabulary: {vocab_size:,} tokens", flush=True)

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
            cache_dir / f"weibo_{split}.npz",
            walks=walks_arr,
            observed_size=splits[split]["observed_size"],
            publish_time=splits[split]["publish_time"],
            cascade_id=splits[split]["cascade_id"],
            y_increment=splits[split]["y_increment"],
            y_total=splits[split]["y_total"],
        )

    unk_rate = (unk_tokens / max(1, total_tokens)) * 100.0
    meta = {
        "t_obs": observation_time,
        "prediction_horizon_seconds": 86400,
        "vocab_size": vocab_size,
        "n_walks": n_walks,
        "walk_length": walk_length,
        "preprocess_seed": preprocess_seed,
        "split_counts": split_counts,
        "unk_token_rate_pct": round(unk_rate, 3),
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"• Cached DeepCas Weibo tensors to {cache_dir} "
        f"(train={split_counts['train']:,}, val={split_counts['val']:,}, test={split_counts['test']:,}, "
        f"UNK rate={unk_rate:.2f}%)",
        flush=True,
    )
    return meta


def train_single_seed(
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
    """Train DeepCas for one seed and save evaluation artifacts."""
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TwitterDeepCasDataset(cache_dir / "weibo_train.npz")
    val_ds = TwitterDeepCasDataset(cache_dir / "weibo_val.npz")
    test_ds = TwitterDeepCasDataset(cache_dir / "weibo_test.npz")

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
        "dataset": "Sina Weibo (dataset_weibo.txt)",
        "observation_window_seconds": meta["t_obs"],
        "observation_window_hours": meta["t_obs"] / 3600.0,
        "prediction_horizon_hours": 24,
        "target": "Future Increment (R_24h - R_obs)",
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


def aggregate_reports(reports: list[dict], out_root: Path, meta: dict) -> dict:
    """Compute mean/std summary across seeds and write summary_report.json."""
    metric_keys = [
        "MSLE_log2",
        "RMSLE_log2",
        "MSLE_ln",
        "R2_log2",
        "PCC_log2",
        "MAPE",
        "MAE_count",
        "MedAE_count",
    ]
    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}
    for k in metric_keys:
        vals = [float(r["test_metrics"][k]) for r in reports]
        mean_metrics[k] = float(np.mean(vals))
        std_metrics[k] = float(np.std(vals, ddof=0))

    summary = {
        "model": "DeepCas (Li et al., WWW 2017)",
        "dataset": "Sina Weibo (dataset_weibo.txt)",
        "observation_window_seconds": meta["t_obs"],
        "observation_window_hours": meta["t_obs"] / 3600.0,
        "prediction_horizon_hours": 24,
        "target": "Future Increment (R_24h - R_obs)",
        "seeds": [r["seed"] for r in reports],
        "split_counts": meta["split_counts"],
        "vocab_size": meta["vocab_size"],
        "n_params": reports[0]["n_params"],
        "mean_train_time_seconds": float(np.mean([r["train_time_seconds"] for r in reports])),
        "naive_baseline_metrics": reports[0]["naive_baseline_metrics"],
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "per_seed": [
            {
                "seed": r["seed"],
                "best_epoch": r["best_epoch"],
                "best_val_msle": r["best_val_msle"],
                "train_time_seconds": r["train_time_seconds"],
                "test_metrics": r["test_metrics"],
            }
            for r in reports
        ],
    }

    (out_root / "summary_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepCas on Sina Weibo under controlled protocol.")
    parser.add_argument("--raw-file", default="weibo_data/dataset_weibo.txt")
    parser.add_argument("--observation-time", type=int, default=3600, choices=[3600, 7200])
    parser.add_argument("--aligned-dir", default=None, help="Defaults to weibo_data/processed_{observation_time}")
    parser.add_argument("--cache-dir", default=None, help="Defaults to weibo_data/deepcas_{observation_time}")
    parser.add_argument("--output-dir", default=None, help="Defaults to deepcas/runs_weibo_{observation_time}")
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

    w = args.observation_time
    aligned_dir = Path(args.aligned_dir or f"weibo_data/processed_{w}")
    cache_dir = Path(args.cache_dir or f"weibo_data/deepcas_{w}")
    out_root = Path(args.output_dir or f"deepcas/runs_weibo_{w}")
    raw_file = Path(args.raw_file)
    device = torch.device(args.device)

    print("=" * 76)
    print(f" DEEPCAS WEIBO BENCHMARK | T_obs={w}s ({w/3600:.1f}h) | seeds={args.seeds} | device={device}")
    print("=" * 76, flush=True)

    meta = prepare_weibo_deepcas_cache(
        raw_file=raw_file,
        aligned_dir=aligned_dir,
        cache_dir=cache_dir,
        observation_time=w,
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

        print(f"\n--- Training DeepCas Weibo ({w}s) | seed={seed} ---", flush=True)
        report = train_single_seed(
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
    print("\n" + "=" * 76)
    print(f" DEEPCAS WEIBO {w}s ({w/3600:.1f}h) 3-SEED SUMMARY ({summary['seeds']})")
    print("=" * 76)
    for k in ("MSLE_log2", "RMSLE_log2", "MSLE_ln", "R2_log2", "PCC_log2", "MAPE", "MAE_count", "MedAE_count"):
        m = summary["mean_metrics"][k]
        s = summary["std_metrics"][k]
        print(f"  {k:<15}: {m:.4f} ± {s:.4f}")
    print(f"• Summary saved to: {out_root / 'summary_report.json'}", flush=True)


if __name__ == "__main__":
    main()
