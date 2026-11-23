"""Train CasSubTS baseline on Twitter dataset under the Daily Benchmark Protocol.

Supports CLI flags to switch observation windows (1 day or 2 days).

Protocol:
- Observation Window: 1 day (86,400s) or 2 days (172,800s) via --obs-days
- Prediction Horizon: 15 days (1,296,000s) via --horizon-days
- Target: Future increment only: Delta P = R_15d - R_obs (evaluated as log2(Delta P + 1))
- Filtering: 10 <= N_obs <= 1,000, no hour-of-day filter
- Split: Strictly chronological 70% Train / 15% Val / 15% Test by t_0
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
from torch.utils.data import DataLoader, Dataset

try:
    from .cassubts_model import CasSubTS
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pipeline_w.cassubts_model import CasSubTS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TwitterCascadeGraphDataset(Dataset):
    def __init__(self, data_dict: dict) -> None:
        self.graphs = data_dict["graphs"]
        self.y_raw = data_dict["y_increment"]
        self.cascade_id = data_dict["cascade_id"]

    def __len__(self) -> int:
        return len(self.graphs)

    def log_targets(self) -> np.ndarray:
        return (np.log(np.maximum(self.y_raw, 0.0) + 1.0) / np.log(2.0)).astype(np.float32)

    def __getitem__(self, i: int):
        y_r = float(self.y_raw[i])
        y_l = np.float32(math.log2(max(y_r, 0.0) + 1.0))
        return self.graphs[i], y_l, y_r


def make_collate(n_subgraphs: int, observation_time: float):
    """Batch cascades as one disjoint graph; emit one edge set per cumulative snapshot."""
    thresholds = [observation_time * (k + 1) / n_subgraphs for k in range(n_subgraphs)]

    def collate(items):
        graphs, y_log, y_raw = zip(*items)
        x_base, node_batch, offs, off = [], [], [], 0
        for bi, g in enumerate(graphs):
            n = g["n_nodes"]
            x_base.append(np.stack([
                g["node_depth"],
                g["node_first_time"] / max(observation_time, 1.0),
            ], axis=1))
            node_batch.append(np.full(n, bi, dtype=np.int64))
            offs.append(off)
            off += n

        edge_src_k, edge_dst_k = [], []
        for thr in thresholds:
            s_all, d_all = [], []
            for g, o in zip(graphs, offs):
                m = g["edge_time"] <= thr
                if m.any():
                    s_all.append(g["edge_src"][m].astype(np.int64) + o)
                    d_all.append(g["edge_dst"][m].astype(np.int64) + o)
            edge_src_k.append(torch.from_numpy(np.concatenate(s_all)) if s_all else torch.empty(0, dtype=torch.long))
            edge_dst_k.append(torch.from_numpy(np.concatenate(d_all)) if d_all else torch.empty(0, dtype=torch.long))

        return {
            "x_base": torch.from_numpy(np.concatenate(x_base).astype(np.float32)),
            "node_batch": torch.from_numpy(np.concatenate(node_batch)),
            "n_graphs": len(graphs),
            "edge_src": edge_src_k,
            "edge_dst": edge_dst_k,
        }, torch.tensor(y_log, dtype=torch.float32).unsqueeze(1), torch.tensor(np.asarray(y_raw, dtype=np.float32)).unsqueeze(1)

    return collate


def to_device(b: dict, dev: torch.device) -> dict:
    return {
        "x_base": b["x_base"].to(dev),
        "node_batch": b["node_batch"].to(dev),
        "n_graphs": b["n_graphs"],
        "edge_src": [e.to(dev) for e in b["edge_src"]],
        "edge_dst": [e.to(dev) for e in b["edge_dst"]],
    }


def extract_twitter_cassubts_data(
    parquet_path: Path,
    out_dir: Path,
    obs_window: float,
    horizon_window: float,
):
    """Extracts, filters, splits, and formats Twitter cascade graphs for CasSubTS."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_pkl = out_dir / "cassubts_train.pkl"
    val_pkl = out_dir / "cassubts_val.pkl"
    test_pkl = out_dir / "cassubts_test.pkl"

    if train_pkl.exists() and val_pkl.exists() and test_pkl.exists():
        print(f"• Loading preprocessed Twitter CasSubTS data from {out_dir}...", flush=True)
        with open(train_pkl, "rb") as f:
            train_data = pickle.load(f)
        with open(val_pkl, "rb") as f:
            val_data = pickle.load(f)
        with open(test_pkl, "rb") as f:
            test_data = pickle.load(f)
        return train_data, val_data, test_data

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

        obs_events = []
        n_obs = 0
        n_horizon = 0
        for ev in ev_list:
            t = ev["elapsed"]
            if t < horizon_window:
                n_horizon += 1
            if t < obs_window:
                n_obs += 1
                obs_events.append(ev)

        if 10 <= n_obs <= 1000:
            inc = max(0, n_horizon - n_obs)
            candidates.append({
                "cascade_id": cid,
                "published": pub,
                "obs_events": obs_events,
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

    # Process graphs into CasSubTS format
    processed = {}
    for s_name, items in split_items.items():
        print(f"• Processing graphs for {s_name} split ({len(items):,} cascades)...", flush=True)
        t_s = time.time()
        graphs = []
        y_inc_list = []
        cid_list = []

        for it in items:
            cid = it["cascade_id"]
            y_inc = it["y_inc"]
            evs = it["obs_events"]

            node_map = {-1: 0}
            first_time = {0: 0.0}
            depth = {0: 0.0}
            src, dst, ets = [], [], []

            for e in evs:
                p = e["path"]
                t = float(e["elapsed"])
                nodes = [int(x) for x in p]
                for d_idx, u in enumerate(nodes):
                    if u not in node_map:
                        nid = len(node_map)
                        node_map[u] = nid
                        first_time[nid] = t
                        depth[nid] = float(d_idx)
                if len(nodes) >= 2:
                    u, v = nodes[-2], nodes[-1]
                    if u != v:
                        src.append(node_map[u])
                        dst.append(node_map[v])
                        ets.append(t)

            n_nodes = len(node_map)
            graphs.append({
                "edge_src": np.array(src, dtype=np.int64),
                "edge_dst": np.array(dst, dtype=np.int64),
                "edge_time": np.array(ets, dtype=np.float32),
                "n_nodes": n_nodes,
                "node_first_time": np.array([first_time[j] for j in range(n_nodes)], dtype=np.float32),
                "node_depth": np.array([depth[j] for j in range(n_nodes)], dtype=np.float32),
            })
            y_inc_list.append(y_inc)
            cid_list.append(cid)

        processed[s_name] = {
            "graphs": graphs,
            "y_increment": np.array(y_inc_list, dtype=np.float32),
            "cascade_id": cid_list,
        }
        print(f"  Done {s_name} in {time.time() - t_s:.1f}s", flush=True)

    # Cache preprocessed data
    print(f"• Caching preprocessed CasSubTS splits to {out_dir}...", flush=True)
    with open(train_pkl, "wb") as f:
        pickle.dump(processed["train"], f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(val_pkl, "wb") as f:
        pickle.dump(processed["val"], f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(test_pkl, "wb") as f:
        pickle.dump(processed["test"], f, protocol=pickle.HIGHEST_PROTOCOL)

    return processed["train"], processed["val"], processed["test"]


def compute_full_metrics(pred_log2: np.ndarray, true_log2: np.ndarray, true_raw: np.ndarray) -> dict[str, float]:
    """Computes comprehensive benchmark metrics matching literature standards."""
    pred_log2 = np.asarray(pred_log2, dtype=np.float64).ravel()
    true_log2 = np.asarray(true_log2, dtype=np.float64).ravel()
    true_raw = np.asarray(true_raw, dtype=np.float64).ravel()

    residual = pred_log2 - true_log2
    msle_log2 = float(np.mean(residual ** 2))
    msle_ln = msle_log2 * (math.log(2.0) ** 2)
    rmsle_log2 = float(math.sqrt(msle_log2))

    pred_raw = np.clip(np.exp2(np.clip(pred_log2, -30, 30)) - 1.0, 0.0, None)
    err = pred_raw - true_raw

    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((true_log2 - true_log2.mean()) ** 2))
    r2_log2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    if len(pred_log2) > 1 and pred_log2.std() > 0 and true_log2.std() > 0:
        pcc_log2 = float(np.corrcoef(pred_log2, true_log2)[0, 1])
    else:
        pcc_log2 = float("nan")

    pos = true_log2 > 0
    if np.any(pos):
        mape_log2 = float(np.mean(np.abs(residual[pos]) / true_log2[pos]))
    else:
        mape_log2 = float("nan")

    mae_count = float(np.mean(np.abs(err)))
    rmse_count = float(math.sqrt(np.mean(err ** 2)))
    medae_count = float(np.median(np.abs(err)))

    return {
        "MSLE_log2": msle_log2,
        "MSLE_ln": msle_ln,
        "RMSLE_log2": rmsle_log2,
        "MAE_count": mae_count,
        "RMSE_count": rmse_count,
        "MedAE_count": medae_count,
        "R2_log2": r2_log2,
        "PCC_log2": pcc_log2,
        "MAPE_log2": mape_log2,
        "n_samples": int(len(true_raw)),
    }


def run_epoch(model: nn.Module, loader: DataLoader, crit: nn.Module, dev: torch.device, opt: torch.optim.Optimizer | None = None, grad_clip: float = 1.0) -> float:
    train = opt is not None
    model.train(train)
    tot, nb = 0.0, 0
    for b, y, _ in loader:
        b, y = to_device(b, dev), y.to(dev)
        with torch.set_grad_enabled(train):
            pred = model(b)
            loss = crit(pred, y)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
        tot += float(loss.item())
        nb += 1
    return tot / max(nb, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, dev: torch.device):
    model.eval()
    P, L, R = [], [], []
    for b, y, yr in loader:
        b = to_device(b, dev)
        out = model(b).cpu().numpy().ravel()
        P.append(out)
        L.append(y.numpy().ravel())
        R.append(yr.numpy().ravel())
    return np.concatenate(P), np.concatenate(L), np.concatenate(R)


def train_single_seed(
    seed: int,
    train_ds: TwitterCascadeGraphDataset,
    val_ds: TwitterCascadeGraphDataset,
    test_ds: TwitterCascadeGraphDataset,
    obs_window: float,
    out_dir: Path,
    device: torch.device,
    batch_size: int = 64,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    early_stop_patience: int = 8,
    max_epochs: int = 60,
    n_subgraphs: int = 6,
    gat_dim: int = 32,
    gat_heads: int = 2,
    gat_layers: int = 2,
    gru_hidden: int = 64,
    gru_layers: int = 2,
    dropout: float = 0.2,
) -> dict:
    set_seed(seed)
    seed_out = out_dir / f"s{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)

    coll = make_collate(n_subgraphs, obs_window)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)

    mean_log_target = float(train_ds.log_targets().mean())

    model = CasSubTS(
        base_node_dim=2,
        degree_dim=8,
        gat_dim=gat_dim,
        gat_heads=gat_heads,
        gat_layers=gat_layers,
        gru_hidden=gru_hidden,
        gru_layers=gru_layers,
        n_subgraphs=n_subgraphs,
        dropout=dropout,
    ).to(device)
    model.init_target_bias(mean_log_target)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 65)
    print(f"Starting CasSubTS Twitter Training | Seed = {seed} | Device = {device}")
    print(f"Harness: Batch={batch_size}, AdamW(lr={lr}, wd={weight_decay}), GradClip={grad_clip}, ES={early_stop_patience}, MaxEpochs={max_epochs}")
    print(f"• CasSubTS Architecture Initialized: {n_params:,} parameters (target bias init: {mean_log_target:.4f})")
    print("=" * 65)

    crit = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val = float("inf")
    stale_epochs = 0
    best_epoch = 0
    history = []
    ckpt_path = seed_out / "best_cassubts.pt"

    t_start = time.time()
    for ep in range(1, max_epochs + 1):
        t_ep = time.time()
        train_loss = run_epoch(model, train_loader, crit, device, opt=optimizer, grad_clip=grad_clip)
        val_loss = run_epoch(model, val_loader, crit, device, opt=None)
        scheduler.step(val_loss)

        ep_time = time.time() - t_ep
        cur_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": ep,
            "train_msle": train_loss,
            "val_msle": val_loss,
            "lr": cur_lr,
            "time_sec": ep_time,
        })

        is_best = val_loss < best_val - 1e-5
        mark = " *" if is_best else ""
        print(f"[Seed {seed} | Epoch {ep:02d}] Train MSLE: {train_loss:.4f} | Val MSLE: {val_loss:.4f} | LR: {cur_lr:.1e} | Time: {ep_time:.1f}s{mark}", flush=True)

        if is_best:
            best_val = val_loss
            best_epoch = ep
            stale_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            stale_epochs += 1
            if stale_epochs >= early_stop_patience:
                print(f"[Seed {seed}] Early stopping at epoch {ep} (best epoch: {best_epoch}, best Val MSLE: {best_val:.4f})", flush=True)
                break

    train_time = time.time() - t_start

    # Load best checkpoint and evaluate on Test set
    print(f"\n[Seed {seed}] Evaluating best checkpoint (Epoch {best_epoch}) on Test Set...", flush=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    preds, y_log_true, y_raw_true = predict(model, test_loader, device)
    metrics = compute_full_metrics(preds, y_log_true, y_raw_true)
    metrics["best_epoch"] = best_epoch
    metrics["best_val_MSLE_log2"] = best_val
    metrics["parameters"] = n_params
    metrics["train_time_sec"] = train_time
    metrics["train_time_min"] = train_time / 60.0

    print(
        f"[Seed {seed} DONE] Time: {train_time:.1f}s ({train_time/60:.2f} min) | "
        f"Test MSLE (log2): {metrics['MSLE_log2']:.4f} | Test MSLE (ln): {metrics['MSLE_ln']:.4f} | "
        f"R²: {metrics['R2_log2']:.4f} | PCC: {metrics['PCC_log2']:.4f} | "
        f"MAPE: {metrics['MAPE_log2']:.4f} | MedAE: {metrics['MedAE_count']:.2f}",
        flush=True
    )

    # Save artifacts
    np.savez_compressed(
        seed_out / "test_predictions.npz",
        cascade_id=np.array(test_ds.cascade_id),
        pred_log2=preds.astype(np.float32),
        true_log2=y_log_true.astype(np.float32),
        true_raw=y_raw_true.astype(np.float32),
    )

    with open(seed_out / "cassubts_evaluation_report.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(seed_out / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CasSubTS baseline on Twitter under the Daily Protocol.")
    parser.add_argument("--parquet", type=str, default="cascades/twitter.parquet", help="Path to twitter.parquet dataset.")
    parser.add_argument("--obs-days", type=int, choices=[1, 2], default=1, help="Observation window in days (1 or 2). Default: 1.")
    parser.add_argument("--horizon-days", type=int, default=15, help="Prediction horizon in days. Default: 15.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory for extracted split data.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for runs and reports.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds to evaluate across. Default: 42 43 44.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    obs_window = float(args.obs_days * 86400)
    horizon_window = float(args.horizon_days * 86400)

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(f"twitter_data/cassubts_{args.obs_days}d")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"pipeline_w/runs_twitter_{args.obs_days}d")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f" CasSubTS Twitter Benchmark: {args.obs_days}-Day Observation Window -> {args.horizon_days}-Day Horizon")
    print(f" Cache Directory: {cache_dir}")
    print(f" Output Directory: {out_dir}")
    print(f" Seeds: {args.seeds} | Device: {args.device}")
    print("=" * 70, flush=True)

    train_data, val_data, test_data = extract_twitter_cassubts_data(
        parquet_path=Path(args.parquet),
        out_dir=cache_dir,
        obs_window=obs_window,
        horizon_window=horizon_window,
    )

    train_ds = TwitterCascadeGraphDataset(train_data)
    val_ds = TwitterCascadeGraphDataset(val_data)
    test_ds = TwitterCascadeGraphDataset(test_data)

    dev = torch.device(args.device)

    all_seed_results = {}
    timing_records = {}
    total_start = time.time()

    for seed in args.seeds:
        t0 = time.time()
        res = train_single_seed(
            seed=seed,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            obs_window=obs_window,
            out_dir=out_dir,
            device=dev,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            early_stop_patience=args.early_stop,
            max_epochs=args.max_epochs,
        )
        dt = time.time() - t0
        all_seed_results[f"s{seed}"] = res
        timing_records[f"seed_{seed}"] = {"sec": dt, "min": dt / 60.0}

    total_time = time.time() - total_start

    # Compute aggregate statistics
    agg_metrics = {}
    metric_keys = ["MSLE_log2", "MSLE_ln", "R2_log2", "PCC_log2", "MAPE_log2", "MAE_count", "MedAE_count", "RMSE_count"]
    for k in metric_keys:
        vals = [all_seed_results[s][k] for s in all_seed_results]
        agg_metrics[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    summary_report = {
        "dataset": "Twitter",
        "obs_days": args.obs_days,
        "horizon_days": args.horizon_days,
        "protocol": f"Daily (T_obs = {args.obs_days}d / {obs_window:.0f}s -> T_pred = {args.horizon_days}d / {horizon_window:.0f}s)",
        "target": f"Future Increment Delta P = R_{args.horizon_days}d - R_{args.obs_days}d",
        "split_counts": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
            "total": len(train_ds) + len(val_ds) + len(test_ds),
        },
        "model": "CasSubTS",
        "parameters": all_seed_results["s42"]["parameters"],
        "seeds": args.seeds,
        "timing": {
            "per_seed": timing_records,
            "total_sec": total_time,
            "total_min": total_time / 60.0,
        },
        "metrics_summary": agg_metrics,
        "per_seed_results": all_seed_results,
    }

    with open(out_dir / "summary_report.json", "w") as f:
        json.dump(summary_report, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print(f"        CASSUBTS TWITTER {args.obs_days}-DAY BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Dataset: Twitter (T_obs = {args.obs_days}d / {obs_window:.0f}s -> T_pred = {args.horizon_days}d / {horizon_window:.0f}s)")
    print(f"Splits : Train = {len(train_ds):,}, Val = {len(val_ds):,}, Test = {len(test_ds):,}")
    print(f"Params : {all_seed_results['s42']['parameters']:,}")
    print("-" * 70)
    for s in args.seeds:
        m = all_seed_results[f"s{s}"]
        t = timing_records[f"seed_{s}"]
        print(f"Seed {s} -> MSLE (log2): {m['MSLE_log2']:.4f} | R²: {m['R2_log2']:.4f} | PCC: {m['PCC_log2']:.4f} | MAPE: {m['MAPE_log2']:.4f} | MedAE: {m['MedAE_count']:.2f} | Time: {t['sec']:.1f}s ({t['min']:.2f} min)")
    print("-" * 70)
    print(f"3-Seed Mean MSLE (log2) : {agg_metrics['MSLE_log2']['mean']:.4f} ± {agg_metrics['MSLE_log2']['std']:.4f}")
    print(f"3-Seed Mean MSLE (ln)   : {agg_metrics['MSLE_ln']['mean']:.4f} ± {agg_metrics['MSLE_ln']['std']:.4f}")
    print(f"3-Seed Mean R²          : {agg_metrics['R2_log2']['mean']:.4f} ± {agg_metrics['R2_log2']['std']:.4f}")
    print(f"3-Seed Mean PCC         : {agg_metrics['PCC_log2']['mean']:.4f} ± {agg_metrics['PCC_log2']['std']:.4f}")
    print(f"3-Seed Mean MAPE (log2) : {agg_metrics['MAPE_log2']['mean']:.4f} ± {agg_metrics['MAPE_log2']['std']:.4f}")
    print(f"3-Seed Mean MedAE       : {agg_metrics['MedAE_count']['mean']:.2f} ± {agg_metrics['MedAE_count']['std']:.2f}")
    print(f"3-Seed Mean MAE         : {agg_metrics['MAE_count']['mean']:.2f} ± {agg_metrics['MAE_count']['std']:.2f}")
    print("-" * 70)
    print(f"Total Training Time     : {total_time:.1f}s ({total_time/60:.2f} min)")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
