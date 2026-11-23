"""Train CasCN on Twitter dataset under the Daily Benchmark Protocol.

Supports CLI flags to switch observation windows (e.g. 1 day or 2 days).

Protocol:
- Observation Window: 1 day (86,400s) or 2 days (172,800s) via --obs-days
- Prediction Horizon: 15 days (1,296,000s) via --horizon-days
- Target: Future increment only: Delta P = R_15d - R_obs (evaluated as log2(Delta P + 1))
- Filtering: 10 <= N_obs <= 1,000, no hour-of-day filter
- Split: Strictly chronological 70% Train / 15% Val / 15% Test
- Feature Scaler: Fit strictly on Train split ONLY (zero data snooping)
- Seeds: 42, 43, 44 (reporting individual and 3-seed mean +- std, plus training times)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn

try:
    from .model import CasCNModel
    from .data import compute_scaled_directed_laplacian
    from .train import prepare_batches, evaluate, compute_full_metrics
except ImportError:
    from cascn.model import CasCNModel
    from cascn.data import compute_scaled_directed_laplacian
    from cascn.train import prepare_batches, evaluate, compute_full_metrics


MAX_NODES = 100
MAX_STEPS = 100
N_TIME_INTERVAL = 6


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_twitter_cascn_data(
    parquet_path: Path,
    out_dir: Path,
    obs_window: float,
    horizon_window: float,
    n_time_interval: int = 6,
):
    """Extracts, filters, splits, and computes Laplacians for Twitter cascades."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_pkl = out_dir / "cascn_train.pkl"
    val_pkl = out_dir / "cascn_val.pkl"
    test_pkl = out_dir / "cascn_test.pkl"
    scaler_pkl = out_dir / "train_scaler.pkl"

    time_interval = obs_window / n_time_interval

    if train_pkl.exists() and val_pkl.exists() and test_pkl.exists() and scaler_pkl.exists():
        print(f"• Loading preprocessed Twitter CasCN data from {out_dir}...", flush=True)
        with open(train_pkl, "rb") as f:
            train_data = pickle.load(f)
        with open(val_pkl, "rb") as f:
            val_data = pickle.load(f)
        with open(test_pkl, "rb") as f:
            test_data = pickle.load(f)
        with open(scaler_pkl, "rb") as f:
            scaler = pickle.load(f)
        return train_data, val_data, test_data, scaler

    print(f"• Reading {parquet_path} for Twitter cascade extraction (T_obs={obs_window:.0f}s, T_pred={horizon_window:.0f}s)...", flush=True)
    t0 = time.time()
    table = pq.read_table(parquet_path, columns=["cascade_id", "published", "events"])
    pydict = table.to_pydict()
    n_raw = len(pydict["cascade_id"])
    print(f"• Read {n_raw:,} raw Twitter cascades in {time.time() - t0:.2f}s", flush=True)

    candidates = []
    for i in range(n_raw):
        cid = str(pydict["cascade_id"][i])
        pub = int(pydict["published"][i])
        ev_list = pydict["events"][i]

        obs_edges = []
        n_obs = 0
        n_horizon = 0
        for ev in ev_list:
            t = ev["elapsed"]
            if t < horizon_window:
                n_horizon += 1
            if t < obs_window:
                n_obs += 1
                path = ev["path"]
                if len(path) > 1 and path[-1] != path[-2]:
                    obs_edges.append((int(path[-2]), int(path[-1]), float(t)))

        if 10 <= n_obs <= 1000:
            obs_edges.sort(key=lambda it: it[2])
            inc = max(0, n_horizon - n_obs)
            candidates.append({
                "cascade_id": cid,
                "published": pub,
                "edges": obs_edges,
                "y_inc": float(inc),
                "n_obs": n_obs,
            })

    print(f"• {len(candidates):,} cascades passed filtering (10 <= N_obs <= 1,000)", flush=True)

    # Sort strictly chronological by publish timestamp
    candidates.sort(key=lambda it: it["published"])

    n_total = len(candidates)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val

    split_items = {
        "train": candidates[:n_train],
        "val": candidates[n_train:n_train + n_val],
        "test": candidates[n_train + n_val:],
    }

    print(f"• Splits: Train={len(split_items['train']):,}, Val={len(split_items['val']):,}, Test={len(split_items['test']):,}", flush=True)

    # Process graphs into CasCN format
    processed = {}
    for s_name, items in split_items.items():
        print(f"• Processing graphs & Laplacians for {s_name} split ({len(items):,} cascades)...", flush=True)
        t_s = time.time()
        results = []
        for it in items:
            cid = it["cascade_id"]
            y_inc = it["y_inc"]
            edges = it["edges"]

            node_map = {0: 0}
            u_list = [0]
            v_list = [0]
            m_list = [1.0]
            t_list = [0.0]
            seen_edges = {(0, 0)}

            for s, d, t in edges:
                if len(u_list) >= MAX_STEPS:
                    break
                if s not in node_map:
                    if len(node_map) < MAX_NODES:
                        node_map[s] = len(node_map)
                    else:
                        continue
                if d not in node_map:
                    if len(node_map) < MAX_NODES:
                        node_map[d] = len(node_map)
                    else:
                        continue
                u_mapped = node_map[s]
                v_mapped = node_map[d]
                if (u_mapped, v_mapped) not in seen_edges:
                    seen_edges.add((u_mapped, v_mapped))
                    m_list.append(1.0)
                else:
                    m_list.append(0.0)
                u_list.append(u_mapped)
                v_list.append(v_mapped)
                t_list.append(t)

            n_nodes = len(node_map)
            n_steps = len(u_list)
            u_arr = np.array(u_list, dtype=np.int16)
            v_arr = np.array(v_list, dtype=np.int16)
            m_arr = np.array(m_list, dtype=np.float32)
            t_arr = np.array(t_list, dtype=np.float32)
            bin_arr = np.minimum(np.floor(t_arr / time_interval).astype(np.int16), n_time_interval - 1)

            L = compute_scaled_directed_laplacian(n_nodes, u_arr, v_arr)
            y_log2 = np.float32(np.log2(y_inc + 1.0))

            results.append({
                "cascade_id": cid,
                "n_nodes": np.int16(n_nodes),
                "n_steps": np.int16(n_steps),
                "u": u_arr,
                "v": v_arr,
                "m": m_arr,
                "time": t_arr,
                "time_bin": bin_arr,
                "L": L,
                "y_inc": np.float32(y_inc),
                "y_log2": y_log2,
            })

        processed[s_name] = results
        with open(out_dir / f"cascn_{s_name}.pkl", "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Done {s_name} in {time.time() - t_s:.1f}s", flush=True)

    # Fit feature scaler strictly on Train split ONLY (zero data snooping)
    all_train_times = np.concatenate([r["time"] for r in processed["train"]])
    all_train_sizes = np.array([r["n_steps"] for r in processed["train"]], dtype=np.float32)
    scaler = {
        "train_count": len(processed["train"]),
        "time_mean": float(np.mean(all_train_times)),
        "time_std": float(np.std(all_train_times) + 1e-8),
        "size_mean": float(np.mean(all_train_sizes)),
        "size_std": float(np.std(all_train_sizes) + 1e-8),
    }
    with open(scaler_pkl, "wb") as f:
        pickle.dump(scaler, f)
    print(f"• Fit feature scaler strictly on Train split ONLY: {scaler}", flush=True)

    return processed["train"], processed["val"], processed["test"], scaler


def train_single_seed_twitter(
    seed: int,
    train_data: list,
    val_data: list,
    test_data: list,
    scaler: dict,
    out_dir: Path,
    batch_size: int = 64,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    patience_es: int = 8,
    max_epochs: int = 60,
    lr_patience: int = 3,
    lr_factor: float = 0.5,
    device: str = "cuda",
) -> dict:
    t_seed_start = time.perf_counter()
    seed_dir = out_dir / f"s{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = seed_dir / "best_cascn.pt"

    print(f"\n" + "=" * 65, flush=True)
    print(f"Starting CasCN Twitter Training | Seed = {seed} | Device = {device}", flush=True)
    print(f"Harness: Batch={batch_size}, AdamW(lr={lr}, wd={weight_decay}), GradClip={max_grad_norm}, ES={patience_es}, MaxEpochs={max_epochs}", flush=True)
    print("=" * 65, flush=True)

    set_seed(seed)
    rng = np.random.RandomState(seed)

    model = CasCNModel(
        max_nodes=MAX_NODES,
        max_steps=MAX_STEPS,
        num_hidden=32,
        num_kernel=2,
        n_time_interval=N_TIME_INTERVAL,
        n_hidden_dense1=32,
        n_hidden_dense2=16,
        stddev=0.01,
        use_scaled_time=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"• CasCN Architecture Initialized: {total_params:,} parameters", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience
    )
    criterion = nn.MSELoss()

    val_batches = prepare_batches(val_data, batch_size, scaler, shuffle=False)
    test_batches = prepare_batches(test_data, batch_size, scaler, shuffle=False)

    best_val_msle = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        t_ep_start = time.perf_counter()
        model.train()
        train_batches = prepare_batches(train_data, batch_size, scaler, shuffle=True, rng=rng)

        total_loss = 0.0
        total_count = 0

        for b in train_batches:
            L = b["L"].to(device)
            u = b["u"].to(device)
            v = b["v"].to(device)
            m = b["m"].to(device)
            tbin = b["tbin"].to(device)
            mask = b["mask"].to(device)
            nnodes = b["nnodes"].to(device)
            tscaled = b["tscaled"].to(device)
            y = b["y"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(L, u, v, m, tbin, mask, nnodes, tscaled)
            loss = criterion(pred, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            bs = len(y)
            total_loss += float(loss.item()) * bs
            total_count += bs

        train_msle = total_loss / total_count

        # Validation evaluation
        val_msle, val_male, val_mape, _, _ = evaluate(model, val_batches, len(val_data), device)
        scheduler.step(val_msle)
        lr_curr = optimizer.param_groups[0]["lr"]
        ep_sec = time.perf_counter() - t_ep_start

        row = {
            "epoch": epoch,
            "train_msle_log2": train_msle,
            "val_msle_log2": val_msle,
            "val_male_log2": val_male,
            "val_mape": val_mape,
            "lr": lr_curr,
            "epoch_sec": ep_sec,
        }
        history.append(row)

        print(
            f"[Seed {seed} | Epoch {epoch:02d}] Train MSLE: {train_msle:.4f} | "
            f"Val MSLE: {val_msle:.4f} | LR: {lr_curr:.1e} | Time: {ep_sec:.1f}s",
            flush=True,
        )

        if val_msle < best_val_msle:
            best_val_msle = val_msle
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_es:
                print(f"[Seed {seed}] Early stopping at epoch {epoch} (best epoch: {best_epoch}, best Val MSLE: {best_val_msle:.4f})", flush=True)
                break

    # Load best checkpoint and evaluate on test set
    print(f"\n[Seed {seed}] Evaluating best checkpoint (Epoch {best_epoch}) on Test Set...", flush=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, _, _, test_preds_log2, test_targets_log2 = evaluate(model, test_batches, len(test_data), device)

    test_targets_raw = np.array([d["y_inc"] for d in test_data], dtype=np.float32)
    test_metrics = compute_full_metrics(test_preds_log2, test_targets_log2, test_targets_raw)

    elapsed_total = time.perf_counter() - t_seed_start
    test_metrics["best_epoch"] = best_epoch
    test_metrics["best_val_MSLE_log2"] = best_val_msle
    test_metrics["parameters"] = total_params
    test_metrics["train_time_sec"] = elapsed_total
    test_metrics["train_time_min"] = elapsed_total / 60.0

    # Save reports
    with open(seed_dir / "cascn_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    with open(seed_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    np.savez_compressed(
        seed_dir / "test_predictions.npz",
        pred_log2=test_preds_log2.astype(np.float32),
        true_log2=test_targets_log2.astype(np.float32),
        true_raw=test_targets_raw.astype(np.float32),
        cascade_id=np.asarray([d["cascade_id"] for d in test_data], dtype=object),
    )

    print(
        f"[Seed {seed} DONE] Time: {elapsed_total:.1f}s ({elapsed_total/60:.2f} min) | "
        f"Test MSLE (log2): {test_metrics['MSLE_log2']:.4f} | "
        f"Test MSLE (ln): {test_metrics['MSLE_ln']:.4f} | R²: {test_metrics['R2_log2']:.4f} | "
        f"MAPE: {test_metrics['MAPE_log2']:.4f} | MedAE: {test_metrics['MedAE_count']:.2f}",
        flush=True,
    )
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train CasCN baseline on Twitter under the Daily Protocol.")
    parser.add_argument("--parquet", type=Path, default=Path("cascades/twitter.parquet"),
                        help="Path to twitter.parquet dataset.")
    parser.add_argument("--obs-days", type=int, choices=[1, 2], default=1,
                        help="Observation window in days (1 or 2). Default: 1.")
    parser.add_argument("--horizon-days", type=int, default=15,
                        help="Prediction horizon in days. Default: 15.")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Cache directory for extracted split data. Defaults to twitter_data/cascn_{obs_days}d.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory for runs and reports. Defaults to cascn/runs_twitter_{obs_days}d.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="Random seeds to evaluate across. Default: 42 43 44.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    obs_window = float(args.obs_days * 86400.0)
    horizon_window = float(args.horizon_days * 86400.0)

    if args.cache_dir is None:
        args.cache_dir = Path(f"twitter_data/cascn_{args.obs_days}d")
    if args.out_dir is None:
        args.out_dir = Path(f"cascn/runs_twitter_{args.obs_days}d")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_global_start = time.perf_counter()

    print("=" * 70, flush=True)
    print(f" CasCN Twitter Benchmark: {args.obs_days}-Day Observation Window -> {args.horizon_days}-Day Horizon", flush=True)
    print(f" Cache Directory: {args.cache_dir}", flush=True)
    print(f" Output Directory: {args.out_dir}", flush=True)
    print(f" Seeds: {args.seeds} | Device: {args.device}", flush=True)
    print("=" * 70, flush=True)

    # 1. Prepare / Load Data
    train_data, val_data, test_data, scaler = extract_twitter_cascn_data(
        parquet_path=args.parquet,
        out_dir=args.cache_dir,
        obs_window=obs_window,
        horizon_window=horizon_window,
        n_time_interval=N_TIME_INTERVAL,
    )

    # 2. Train across seeds (42, 43, 44)
    all_metrics = {}
    seed_times = {}

    for seed in args.seeds:
        metrics = train_single_seed_twitter(
            seed=seed,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            scaler=scaler,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.grad_clip,
            patience_es=args.early_stop,
            max_epochs=args.max_epochs,
            device=args.device,
        )
        all_metrics[f"s{seed}"] = metrics
        seed_times[f"seed_{seed}"] = {
            "sec": metrics["train_time_sec"],
            "min": metrics["train_time_min"],
        }

    total_training_sec = time.perf_counter() - t_global_start
    total_training_min = total_training_sec / 60.0

    # 3. Compute multi-seed statistics
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
        "horizon_days": args.horizon_days,
        "protocol": f"Daily (T_obs = {args.obs_days}d / {int(obs_window)}s -> T_pred = {args.horizon_days}d / {int(horizon_window)}s)",
        "target": f"Future Increment Delta P = R_{args.horizon_days}d - R_{args.obs_days}d",
        "split_counts": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
            "total": len(train_data) + len(val_data) + len(test_data),
        },
        "model": "CasCN",
        "parameters": all_metrics[f"s{args.seeds[0]}"]["parameters"],
        "seeds": args.seeds,
        "timing": {
            "per_seed": seed_times,
            "total_sec": total_training_sec,
            "total_min": total_training_min,
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

    with open(args.out_dir / "summary_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print(f"        CASCN TWITTER {args.obs_days}-DAY BENCHMARK RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"Dataset: Twitter (T_obs = {args.obs_days}d / {int(obs_window)}s -> T_pred = {args.horizon_days}d / {int(horizon_window)}s)")
    print(f"Splits : Train = {len(train_data):,}, Val = {len(val_data):,}, Test = {len(test_data):,}")
    print(f"Params : {summary['parameters']:,}")
    print("-" * 70)
    for s in args.seeds:
        m = all_metrics[f"s{s}"]
        t_m = seed_times[f"seed_{s}"]
        print(f"Seed {s} -> MSLE (log2): {m['MSLE_log2']:.4f} | R²: {m['R2_log2']:.4f} | MAPE: {m['MAPE_log2']:.4f} | MedAE: {m['MedAE_count']:.2f} | Time: {t_m['sec']:.1f}s ({t_m['min']:.2f} min)")
    print("-" * 70)
    print(f"3-Seed Mean MSLE (log2) : {np.mean(msle_log2_list):.4f} ± {np.std(msle_log2_list):.4f}")
    print(f"3-Seed Mean MSLE (ln)   : {np.mean(msle_ln_list):.4f} ± {np.std(msle_ln_list):.4f}")
    print(f"3-Seed Mean R²          : {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
    print(f"3-Seed Mean PCC         : {np.mean(pcc_list):.4f} ± {np.std(pcc_list):.4f}")
    print(f"3-Seed Mean MAPE (log2) : {np.mean(mape_list):.4f} ± {np.std(mape_list):.4f}")
    print(f"3-Seed Mean MedAE       : {np.mean(medae_list):.2f} ± {np.std(medae_list):.2f}")
    print(f"3-Seed Mean MAE         : {np.mean(mae_list):.2f} ± {np.std(mae_list):.2f}")
    print("-" * 70)
    print(f"Total Training Time     : {total_training_sec:.1f}s ({total_training_min:.2f} min)")
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    main()
