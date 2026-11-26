"""Train CasSubTS baseline on APS Citation dataset under the Controlled Protocol.

Supports CLI flags to switch observation windows (3 years = 1095 days or 5 years = 1825 days).

Protocol (1-to-1 aligned with CasMotif, DeepHawkes, CasCN, and CasFlow APS benchmarks):
- Dataset: cascades/aps.parquet (or pre-aligned aps_data/processed_{obs_days} splits)
- Observation Window: 3 years (1,095 days) or 5 years (1,825 days) via --obs-years
- Prediction Horizon: 20 years (7,300 days) via --horizon-years
- Pre-1997 Cutoff: published < '1997-01-01' (un-truncated 20-year horizon)
- Target: Future increment only: Delta P = max(0, R_20y - R_obs) (evaluated as log2(Delta P + 1))
- Filtering: 10 <= N_obs <= 1,000
- Split: Strictly chronological 70% Train / 15% Val / 15% Test by (published, cascade_id)
- Optimization Harness: Batch 64, AdamW(lr=2e-3, wd=1e-4), ReduceLROnPlateau(factor=0.5, patience=3),
  grad clip max_norm=1.0, early stop patience=8 (max 60 epochs), n_subgraphs=6
- Seeds: 42, 43, 44
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

try:
    from pipeline_w.train_twitter import (
        TwitterCascadeGraphDataset as APSCascadeGraphDataset,
        train_single_seed,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pipeline_w.train_twitter import (
        TwitterCascadeGraphDataset as APSCascadeGraphDataset,
        train_single_seed,
    )


def _paths_to_graph(cid: str, paths_and_times: list[tuple[list[str | int], float]]) -> dict:
    """Builds a local node/edge temporal graph compatible with CasSubTS make_collate."""
    node_map: dict[str, int] = {}
    first_time: dict[int, float] = {}
    depth: dict[int, float] = {}
    src: list[int] = []
    dst: list[int] = []
    ets: list[float] = []

    for path, t in paths_and_times:
        nodes = [str(x) for x in path]
        for d_idx, u in enumerate(nodes):
            if u not in node_map:
                nid = len(node_map)
                node_map[u] = nid
                first_time[nid] = float(t)
                depth[nid] = float(d_idx)
        if len(nodes) >= 2:
            u, v = nodes[-2], nodes[-1]
            if u != v:
                src.append(node_map[u])
                dst.append(node_map[v])
                ets.append(float(t))

    n_nodes = max(1, len(node_map))
    return {
        "edge_src": np.array(src, dtype=np.int64),
        "edge_dst": np.array(dst, dtype=np.int64),
        "edge_time": np.array(ets, dtype=np.float32),
        "n_nodes": n_nodes,
        "node_first_time": np.array([first_time.get(j, 0.0) for j in range(n_nodes)], dtype=np.float32),
        "node_depth": np.array([depth.get(j, 0.0) for j in range(n_nodes)], dtype=np.float32),
    }


def extract_aps_cassubts_data(
    parquet_path: Path,
    splits_dir: Path,
    out_dir: Path,
    obs_days: int,
    horizon_days: int,
) -> tuple[dict, dict, dict]:
    """Extracts, splits, and caches APS cascade subtrees aligned 1-to-1 to benchmark splits."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_pkl = out_dir / "cassubts_train.pkl"
    val_pkl = out_dir / "cassubts_val.pkl"
    test_pkl = out_dir / "cassubts_test.pkl"

    if train_pkl.exists() and val_pkl.exists() and test_pkl.exists():
        print(f"• Loading preprocessed APS CasSubTS data from {out_dir}...", flush=True)
        with open(train_pkl, "rb") as f:
            train_data = pickle.load(f)
        with open(val_pkl, "rb") as f:
            val_data = pickle.load(f)
        with open(test_pkl, "rb") as f:
            test_data = pickle.load(f)
        return train_data, val_data, test_data

    processed: dict[str, dict] = {}
    if splits_dir.exists() and (splits_dir / "raw_paths_test.json").exists() and (splits_dir / "aps_test.npz").exists():
        print(f"• Loading pre-aligned APS splits from {splits_dir}...", flush=True)
        for s_name in ("train", "val", "test"):
            t_s = time.time()
            npz = np.load(splits_dir / f"aps_{s_name}.npz", allow_pickle=True)
            cids = [str(cid) for cid in npz["cascade_id"]]
            y_inc = npz["y_increment"].astype(np.float32)
            with open(splits_dir / f"raw_paths_{s_name}.json", "r", encoding="utf-8") as fh:
                raw_map = json.load(fh)

            graphs = []
            for cid in cids:
                entry = raw_map[cid]
                ev_pairs = [
                    (p, float(t))
                    for p, t in zip(entry["paths"], entry["times"])
                    if float(t) <= float(obs_days)
                ]
                ev_pairs.sort(key=lambda it: it[1])
                graphs.append(_paths_to_graph(cid, ev_pairs))

            processed[s_name] = {
                "graphs": graphs,
                "y_increment": y_inc,
                "cascade_id": cids,
            }
            print(f"  Processed {s_name:<5} ({len(graphs):,} cascades) in {time.time() - t_s:.1f}s", flush=True)
    else:
        print(
            f"• Reading {parquet_path} for APS CasSubTS extraction "
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
                obs_pairs = []
                for ev in ev_list:
                    t = float(ev["elapsed"])
                    if t <= horizon_days:
                        n_horizon += 1
                    if t <= obs_days:
                        n_obs += 1
                        obs_pairs.append((ev["path"], t))

                if 10 <= n_obs <= 1000:
                    obs_pairs.sort(key=lambda it: it[1])
                    inc = max(0, n_horizon - n_obs)
                    candidates.append({
                        "cascade_id": str(cids[i]),
                        "published": str(pubs[i]),
                        "obs_pairs": obs_pairs,
                        "y_inc": float(inc),
                    })

        print(
            f"• Scanned {total_scanned:,} raw cascades in {time.time() - t0:.1f}s -> "
            f"{len(candidates):,} eligible",
            flush=True,
        )
        candidates.sort(key=lambda it: (it["published"], it["cascade_id"]))

        n_total = len(candidates)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        split_items = {
            "train": candidates[:n_train],
            "val": candidates[n_train : n_train + n_val],
            "test": candidates[n_train + n_val :],
        }

        for s_name, items in split_items.items():
            t_s = time.time()
            graphs = [_paths_to_graph(it["cascade_id"], it["obs_pairs"]) for it in items]
            processed[s_name] = {
                "graphs": graphs,
                "y_increment": np.array([it["y_inc"] for it in items], dtype=np.float32),
                "cascade_id": [it["cascade_id"] for it in items],
            }
            print(f"  Processed {s_name:<5} ({len(graphs):,} cascades) in {time.time() - t_s:.1f}s", flush=True)

    with open(train_pkl, "wb") as f:
        pickle.dump(processed["train"], f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(val_pkl, "wb") as f:
        pickle.dump(processed["val"], f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(test_pkl, "wb") as f:
        pickle.dump(processed["test"], f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"• Cached preprocessed APS CasSubTS splits -> {out_dir}", flush=True)

    return processed["train"], processed["val"], processed["test"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CasSubTS baseline on APS under Controlled Protocol.")
    parser.add_argument("--parquet", type=Path, default=Path("cascades/aps.parquet"))
    parser.add_argument("--obs-years", type=int, choices=[3, 5], default=3)
    parser.add_argument("--horizon-years", type=int, default=20)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--n-subgraphs", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    obs_days = args.obs_years * 365
    horizon_days = args.horizon_years * 365

    if args.splits_dir is None:
        args.splits_dir = Path(f"aps_data/processed_{obs_days}")
    if args.cache_dir is None:
        args.cache_dir = Path(f"aps_data/cassubts_{obs_days}d")
    if args.out_dir is None:
        args.out_dir = Path(f"cassubts/runs_aps_{obs_days}d")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t_global_start = time.perf_counter()

    print("=" * 72, flush=True)
    print(
        f" CasSubTS APS Benchmark: {args.obs_years}-Year ({obs_days}d) Window -> "
        f"{args.horizon_years}-Year ({horizon_days}d) Horizon",
        flush=True,
    )
    print(f" Cache Dir: {args.cache_dir}", flush=True)
    print(f" Output Dir: {args.out_dir}", flush=True)
    print(f" Seeds: {args.seeds} | Device: {device}", flush=True)
    print("=" * 72, flush=True)

    train_d, val_d, test_d = extract_aps_cassubts_data(
        parquet_path=args.parquet,
        splits_dir=args.splits_dir,
        out_dir=args.cache_dir,
        obs_days=obs_days,
        horizon_days=horizon_days,
    )

    train_ds = APSCascadeGraphDataset(train_d)
    val_ds = APSCascadeGraphDataset(val_d)
    test_ds = APSCascadeGraphDataset(test_d)

    all_metrics = {}
    seed_times = {}

    for seed in args.seeds:
        metrics = train_single_seed(
            seed=seed,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            obs_window=float(obs_days),
            out_dir=args.out_dir,
            device=device,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            early_stop_patience=args.early_stop,
            max_epochs=args.max_epochs,
            n_subgraphs=args.n_subgraphs,
        )
        all_metrics[f"s{seed}"] = metrics
        seed_times[f"seed_{seed}"] = {
            "sec": metrics["train_time_sec"],
            "min": metrics["train_time_min"],
        }

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
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
            "total": len(train_ds) + len(val_ds) + len(test_ds),
        },
        "model": "CasSubTS",
        "parameters": all_metrics[f"s{args.seeds[0]}"]["parameters"],
        "seeds": args.seeds,
        "timing": {
            "per_seed": seed_times,
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
    print(f"        CASSUBTS APS {args.obs_years}-YEAR BENCHMARK RESULTS", flush=True)
    print("=" * 72, flush=True)
    print(f"Dataset: APS (T_obs = {args.obs_years}y / {obs_days}d -> T_pred = {args.horizon_years}y / {horizon_days}d)")
    print(f"Splits : Train = {len(train_ds):,}, Val = {len(val_ds):,}, Test = {len(test_ds):,}")
    print(f"Params : {summary['parameters']:,}")
    print("-" * 72)
    for s in args.seeds:
        m = all_metrics[f"s{s}"]
        t_m = seed_times[f"seed_{s}"]
        print(
            f"Seed {s} -> MSLE (log2): {m['MSLE_log2']:.4f} | R2: {m['R2_log2']:.4f} | "
            f"PCC: {m['PCC_log2']:.4f} | MedAE: {m['MedAE_count']:.2f} | "
            f"Time: {t_m['sec']:.1f}s ({t_m['min']:.2f} min)"
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
