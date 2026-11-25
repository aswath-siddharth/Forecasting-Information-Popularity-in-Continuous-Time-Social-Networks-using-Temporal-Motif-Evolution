"""Train CasFlow on APS Citation dataset under the Controlled Benchmark Protocol.

Supports CLI flags to switch observation windows (e.g. 3 years = 1095 days or 5 years = 1825 days).

Protocol (1-to-1 aligned with CasMotif, DeepHawkes, and CasCN APS benchmarks):
- Dataset: cascades/aps.parquet (or aligned aps_data/processed_{obs_days} splits)
- Observation Window: 3 years (1,095 days) or 5 years (1,825 days) via --obs-years
- Prediction Horizon: 20 years (7,300 days) via --horizon-years
- Pre-1997 Cutoff: published < '1997-01-01' (un-truncated 20-year citation horizon)
- Target: Future increment only: Delta P = max(0, R_20y - R_obs) (evaluated as log2(Delta P + 1))
- Filtering: 10 <= N_obs <= 1,000
- Split: Strictly chronological 70% Train / 15% Val / 15% Test by (published, cascade_id)
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
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

try:
    from .data import Cascade, prepare_tensors
    from .train import train_seed
except ImportError:
    from casflow.data import Cascade, prepare_tensors
    from casflow.train import train_seed


def load_aps_casflow_splits(
    parquet_path: Path,
    splits_dir: Path,
    obs_days: int,
    horizon_days: int,
) -> dict[str, list[Cascade]]:
    """Loads APS cascades aligned 1-to-1 to CasMotif/DeepHawkes splits if present, else parses parquet."""
    if splits_dir.exists() and (splits_dir / "raw_paths_test.json").exists() and (splits_dir / "aps_test.npz").exists():
        print(f"• Loading pre-aligned APS splits from {splits_dir}...", flush=True)
        splits: dict[str, list[Cascade]] = {}
        for name in ("train", "val", "test"):
            npz = np.load(splits_dir / f"aps_{name}.npz", allow_pickle=True)
            cids = [str(cid) for cid in npz["cascade_id"]]
            labels = npz["y_increment"].astype(np.float32)
            with open(splits_dir / f"raw_paths_{name}.json", "r", encoding="utf-8") as fh:
                raw_map = json.load(fh)

            items: list[Cascade] = []
            for cid, lbl in zip(cids, labels):
                entry = raw_map[cid]
                events = [
                    (tuple(str(u) for u in p), float(t))
                    for p, t in zip(entry["paths"], entry["times"])
                    if float(t) <= float(obs_days)
                ]
                events.sort(key=lambda it: it[1])
                items.append(Cascade(id=cid, events=events[:100], label=float(lbl)))
            splits[name] = items
            print(f"  Loaded {name:<5}: {len(items):,} cascades", flush=True)
        return splits

    print(
        f"• Reading {parquet_path} for APS CasFlow extraction "
        f"(T_obs={obs_days}d, T_pred={horizon_days}d)...",
        flush=True,
    )
    t0 = time.time()
    pf = pq.ParquetFile(parquet_path)
    candidates = []
    total_scanned = 0

    for batch in pf.iter_batches(columns=["cascade_id", "published", "events"], batch_size=8192):
        total_scanned += batch.num_rows
        mask = pc.less(batch["published"], "1997-01-01")
        filtered = batch.filter(mask)
        if filtered.num_rows == 0:
            continue

        pydict = filtered.to_pydict()
        cids = pydict["cascade_id"]
        pubs = pydict["published"]
        evs_col = pydict["events"]

        for i in range(filtered.num_rows):
            ev_list = evs_col[i]
            n_obs = 0
            n_horizon = 0
            obs_events: list[tuple[tuple[str, ...], float]] = []
            for ev in ev_list:
                t = float(ev["elapsed"])
                if t <= horizon_days:
                    n_horizon += 1
                if t <= obs_days:
                    n_obs += 1
                    obs_events.append((tuple(str(u) for u in ev["path"]), t))

            if 10 <= n_obs <= 1000:
                obs_events.sort(key=lambda it: it[1])
                inc = max(0, n_horizon - n_obs)
                candidates.append({
                    "cascade_id": str(cids[i]),
                    "published": str(pubs[i]),
                    "events": obs_events[:100],
                    "label": float(inc),
                })

    print(
        f"• Scanned {total_scanned:,} raw cascades in {time.time() - t0:.1f}s -> "
        f"{len(candidates):,} cascades eligible",
        flush=True,
    )

    candidates.sort(key=lambda it: (it["published"], it["cascade_id"]))
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
    parser = argparse.ArgumentParser(description="Train CasFlow baseline on APS under Controlled Protocol.")
    parser.add_argument("--parquet", type=Path, default=Path("cascades/aps.parquet"))
    parser.add_argument("--obs-years", type=int, choices=[3, 5], default=3)
    parser.add_argument("--horizon-years", type=int, default=20)
    parser.add_argument("--splits-dir", type=Path, default=None)
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

    obs_days = args.obs_years * 365
    horizon_days = args.horizon_years * 365

    if args.splits_dir is None:
        args.splits_dir = Path(f"aps_data/processed_{obs_days}")
    if args.cache_file is None:
        args.cache_file = Path(f"aps_data/casflow_prepared_{obs_days}d.pt")
    if args.out_dir is None:
        args.out_dir = Path(f"casflow/runs_aps_{obs_days}d")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_global_start = time.perf_counter()

    print("=" * 72, flush=True)
    print(
        f" CasFlow APS Benchmark: {args.obs_years}-Year ({obs_days}d) Window -> "
        f"{args.horizon_years}-Year ({horizon_days}d) Horizon",
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
        splits = load_aps_casflow_splits(
            parquet_path=args.parquet,
            splits_dir=args.splits_dir,
            obs_days=obs_days,
            horizon_days=horizon_days,
        )
        prepared = prepare_tensors(
            splits,
            observation_time=obs_days,
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
        "dataset": "APS",
        "obs_years": args.obs_years,
        "obs_days": obs_days,
        "horizon_years": args.horizon_years,
        "horizon_days": horizon_days,
        "protocol": f"APS Citation (T_obs = {args.obs_years}y / {obs_days}d -> T_pred = {args.horizon_years}y / {horizon_days}d, published < 1997-01-01)",
        "target": f"Future Increment Delta P = R_{args.horizon_years}y - R_{args.obs_years}y",
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
    print(f"        CASFLOW APS {args.obs_years}-YEAR BENCHMARK RESULTS", flush=True)
    print("=" * 72, flush=True)
    print(f"Dataset: APS (T_obs = {args.obs_years}y / {obs_days}d -> T_pred = {args.horizon_years}y / {horizon_days}d)")
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
