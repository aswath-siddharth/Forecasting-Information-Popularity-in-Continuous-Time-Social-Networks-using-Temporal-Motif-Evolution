"""Train CasFlow on Twitter dataset under the Daily Benchmark Protocol.

Supports CLI flags to switch observation windows (1 day = 86,400s or 2 days = 172,800s).

Protocol (1-to-1 aligned with CasMotif, DeepHawkes, CasCN, CasSubTS, DeepCas Twitter benchmarks):
- Dataset: cascades/twitter.parquet
- Observation Window: 1 day (86,400s) or 2 days (172,800s) via --obs-days
- Prediction Horizon: 15 days (1,296,000s) via --horizon-days
- Filtering: 10 <= N_obs <= 1,000, no hour-of-day filter
- Target: Future increment only: Delta P = max(0, R_15d - R_obs) (evaluated as log2(Delta P + 1))
- Split: Strictly chronological 70% Train / 15% Val / 15% Test by publish timestamp
- Zero Leakage: Global SVD graph embeddings fitted strictly on Train split ONLY
- Optimization Harness: Batch 64, AdamW(lr=1e-3, wd=1e-4), ReduceLROnPlateau(factor=0.5, patience=3),
  grad clip max_norm=1.0, early stop patience=8 (max 60 epochs), latent_dim=64, hidden_dim=128, n_flows=8, samples=8
- Seeds: 42, 43, 44
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

try:
    from .data import Cascade, prepare_tensors
    from .train import train_seed
except ImportError:
    from casflow.data import Cascade, prepare_tensors
    from casflow.train import train_seed


def load_twitter_casflow_splits(
    parquet_path: Path,
    obs_sec: int,
    horizon_sec: int,
) -> dict[str, list[Cascade]]:
    """Extracts and chronologically splits Twitter cascades 1-to-1 with Twitter benchmarks."""
    print(
        f"• Reading {parquet_path} for Twitter CasFlow extraction "
        f"(T_obs={obs_sec:,}s, T_pred={horizon_sec:,}s)...",
        flush=True,
    )
    t0 = time.time()
    table = pq.read_table(parquet_path, columns=["cascade_id", "published", "events"])
    pydict = table.to_pydict()
    n_raw = len(pydict["cascade_id"])

    candidates = []
    for i in range(n_raw):
        cid = str(pydict["cascade_id"][i])
        pub = int(pydict["published"][i])
        ev_list = pydict["events"][i]

        n_obs = 0
        n_horizon = 0
        obs_events: list[tuple[tuple[str, ...], float]] = []
        for ev in ev_list:
            t = float(ev["elapsed"])
            if t <= horizon_sec:
                n_horizon += 1
            if t <= obs_sec:
                n_obs += 1
                obs_events.append((tuple(str(u) for u in ev["path"]), t))

        if 10 <= n_obs <= 1000:
            obs_events.sort(key=lambda it: it[1])
            inc = max(0, n_horizon - n_obs)
            candidates.append({
                "cascade_id": cid,
                "published": pub,
                "events": obs_events[:100],
                "label": float(inc),
            })

    print(
        f"• Scanned {n_raw:,} raw cascades in {time.time() - t0:.1f}s -> "
        f"{len(candidates):,} cascades eligible (10 <= N_obs <= 1,000)",
        flush=True,
    )

    candidates.sort(key=lambda it: it["published"])
    n_total = len(candidates)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    raw_splits = {
        "train": candidates[:n_train],
        "val": candidates[n_train : n_train + n_val],
        "test": candidates[n_train + n_val :],
    }

    splits = {}
    for name, items in raw_splits.items():
        splits[name] = [Cascade(id=it["cascade_id"], events=it["events"], label=it["label"]) for it in items]
        print(f"  Split {name:<5}: {len(splits[name]):,} cascades", flush=True)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CasFlow baseline on Twitter under Controlled Protocol.")
    parser.add_argument("--parquet", type=Path, default=Path("cascades/twitter.parquet"))
    parser.add_argument("--obs-days", type=int, choices=[1, 2], default=1)
    parser.add_argument("--horizon-days", type=int, default=15)
    parser.add_argument("--cache-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
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

    obs_sec = args.obs_days * 86400
    horizon_sec = args.horizon_days * 86400

    if args.cache_file is None:
        args.cache_file = Path(f"twitter_data/casflow_prepared_{args.obs_days}d.pt")
    if args.out_dir is None:
        args.out_dir = Path(f"casflow/runs_twitter_{args.obs_days}d")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_global_start = time.perf_counter()

    print("=" * 72, flush=True)
    print(
        f" CasFlow Twitter Benchmark: {args.obs_days}-Day ({obs_sec:,}s) Window -> "
        f"{args.horizon_days}-Day ({horizon_sec:,}s) Horizon",
        flush=True,
    )
    print(f" Cache File: {args.cache_file}", flush=True)
    print(f" Output Dir: {args.out_dir}", flush=True)
    print(f" Seeds: {args.seeds} | Device: {args.device}", flush=True)
    print("=" * 72, flush=True)

    if args.cache_file.exists():
        print(f"• Loading cached CasFlow tensors from {args.cache_file}...", flush=True)
        prepared = torch.load(args.cache_file, weights_only=False)
    else:
        splits = load_twitter_casflow_splits(
            parquet_path=args.parquet,
            obs_sec=obs_sec,
            horizon_sec=horizon_sec,
        )
        prepared = prepare_tensors(
            splits,
            observation_time=obs_sec,
            device=args.device,
            batch_size=args.batch_size,
        )
        args.cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(prepared, args.cache_file)
        print(f"• Saved cached CasFlow tensors -> {args.cache_file}", flush=True)

    all_metrics = {}
    for seed in args.seeds:
        metrics = train_seed(seed, prepared, args.out_dir, args)
        all_metrics[f"s{seed}"] = metrics

    total_sec = time.perf_counter() - t_global_start

    msle_log2_list = [all_metrics[f"s{s}"]["MSLE_log2"] for s in args.seeds]
    msle_ln_list = [all_metrics[f"s{s}"]["MSLE_ln"] for s in args.seeds]
    r2_list = [all_metrics[f"s{s}"]["R2_log2"] for s in args.seeds]
    pcc_list = [all_metrics[f"s{s}"]["PCC_log2"] for s in args.seeds]
    mape_list = [all_metrics[f"s{s}"]["MAPE_log2"] for s in args.seeds]
    mae_list = [all_metrics[f"s{s}"]["MAE_count"] for s in args.seeds]
    medae_list = [all_metrics[f"s{s}"]["MedAE_count"] for s in args.seeds]
    rmse_list = [all_metrics[f"s{s}"]["RMSE_count"] for s in args.seeds]

    summary = {
        "dataset": "Twitter",
        "obs_days": args.obs_days,
        "obs_seconds": obs_sec,
        "horizon_days": args.horizon_days,
        "horizon_seconds": horizon_sec,
        "protocol": f"Twitter (T_obs = {args.obs_days}d / {obs_sec}s -> T_pred = {args.horizon_days}d / {horizon_sec}s)",
        "target": f"Future Increment Delta P = R_{args.horizon_days}d - R_{args.obs_days}d",
        "split_counts": {
            "train": len(prepared["train"]["labels"]),
            "val": len(prepared["val"]["labels"]),
            "test": len(prepared["test"]["labels"]),
            "total": int(sum(len(prepared[k]["labels"]) for k in ("train", "val", "test"))),
        },
        "model": "CasFlow",
        "parameters": all_metrics[f"s{args.seeds[0]}"]["parameters"],
        "seeds": args.seeds,
        "timing": {
            "total_sec": total_sec,
            "total_min": total_sec / 60.0,
        },
        "metrics_summary": {
            "MSLE_log2": {"mean": float(np.mean(msle_log2_list)), "std": float(np.std(msle_log2_list))},
            "MSLE_ln": {"mean": float(np.mean(msle_ln_list)), "std": float(np.std(msle_ln_list))},
            "R2_log2": {"mean": float(np.mean(r2_list)), "std": float(np.std(r2_list))},
            "PCC_log2": {"mean": float(np.mean(pcc_list)), "std": float(np.std(pcc_list))},
            "MAPE_log2": {"mean": float(np.mean(mape_list)), "std": float(np.std(mape_list))},
            "MAE_count": {"mean": float(np.mean(mae_list)), "std": float(np.std(mae_list))},
            "MedAE_count": {"mean": float(np.mean(medae_list)), "std": float(np.std(medae_list))},
            "RMSE_count": {"mean": float(np.mean(rmse_list)), "std": float(np.std(rmse_list))},
        },
        "per_seed_results": all_metrics,
    }

    (args.out_dir / "summary_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 72, flush=True)
    print(f"        CASFLOW TWITTER {args.obs_days}-DAY BENCHMARK RESULTS", flush=True)
    print("=" * 72, flush=True)
    print(f"Dataset: Twitter (T_obs = {args.obs_days}d / {obs_sec:,}s -> T_pred = {args.horizon_days}d / {horizon_sec:,}s)")
    print(
        f"Splits : Train = {summary['split_counts']['train']:,}, "
        f"Val = {summary['split_counts']['val']:,}, Test = {summary['split_counts']['test']:,}"
    )
    print(f"Params : {summary['parameters']:,}")
    print("-" * 72)
    for s in args.seeds:
        m = all_metrics[f"s{s}"]
        print(
            f"Seed {s} -> MSLE (log2): {m['MSLE_log2']:.4f} | R2: {m['R2_log2']:.4f} | "
            f"PCC: {m['PCC_log2']:.4f} | MedAE: {m['MedAE_count']:.2f} | "
            f"Time: {m['train_time_sec']:.1f}s ({m['train_time_min']:.2f} min)"
        )
    print("-" * 72)
    print(f"3-Seed Mean MSLE (log2) : {np.mean(msle_log2_list):.4f} +/- {np.std(msle_log2_list):.4f}")
    print(f"3-Seed Mean MSLE (ln)   : {np.mean(msle_ln_list):.4f} +/- {np.std(msle_ln_list):.4f}")
    print(f"3-Seed Mean R2          : {np.mean(r2_list):.4f} +/- {np.std(r2_list):.4f}")
    print(f"3-Seed Mean PCC         : {np.mean(pcc_list):.4f} +/- {np.std(pcc_list):.4f}")
    print(f"3-Seed Mean MAPE (log2) : {np.mean(mape_list):.4f} +/- {np.std(mape_list):.4f}")
    print(f"3-Seed Mean MedAE       : {np.mean(medae_list):.2f} +/- {np.std(medae_list):.2f}")
    print(f"3-Seed Mean MAE         : {np.mean(mae_list):.2f} +/- {np.std(mae_list):.2f}")
    print("-" * 72)
    print(f"Total Runtime           : {total_sec:.1f}s ({total_sec/60:.2f} min)")
    print("=" * 72 + "\n", flush=True)


if __name__ == "__main__":
    main()
