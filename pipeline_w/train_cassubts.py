"""Train the CasSubTS reimplementation on the same splits, target and metric as
pipeline_w/train_weibo.py, so the two models differ only in architecture."""

from __future__ import annotations

import argparse, json, math, pickle, random, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Dataset

from .cassubts_model import CasSubTS
from .train_weibo import regression_metrics


class CascadeGraphDataset(Dataset):
    def __init__(self, pkl_path: Path) -> None:
        with open(pkl_path, "rb") as fh:
            d = pickle.load(fh)
        self.graphs = d["graphs"]
        self.y_raw = d["y_increment"]
        self.cascade_id = d["cascade_id"]

    def __len__(self) -> int:
        return len(self.graphs)

    def log_targets(self) -> np.ndarray:
        return (np.log(self.y_raw + 1.0) / np.log(2.0)).astype(np.float32)

    def __getitem__(self, i: int):
        return self.graphs[i], np.float32(np.log(self.y_raw[i] + 1.0) / np.log(2.0)), self.y_raw[i]


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
        }, torch.tensor(y_log).unsqueeze(1), torch.tensor(np.asarray(y_raw)).unsqueeze(1)

    return collate


def to_device(b: dict, dev: torch.device) -> dict:
    return {
        "x_base": b["x_base"].to(dev), "node_batch": b["node_batch"].to(dev),
        "n_graphs": b["n_graphs"],
        "edge_src": [e.to(dev) for e in b["edge_src"]],
        "edge_dst": [e.to(dev) for e in b["edge_dst"]],
    }


def run_epoch(model, loader, crit, dev, opt=None, ema=None) -> float:
    train = opt is not None
    model.train(train)
    tot, nb = 0.0, 0
    for b, y, _ in loader:
        b, y = to_device(b, dev), y.to(dev)
        with torch.set_grad_enabled(train):
            loss = crit(model(b), y)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if ema is not None:
                    ema.update_parameters(model)
        tot += float(loss.item()); nb += 1
    return tot / max(nb, 1)


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    P, L, R = [], [], []
    for b, y, yr in loader:
        P.append(model(to_device(b, dev)).cpu().numpy().ravel())
        L.append(y.numpy().ravel()); R.append(yr.numpy().ravel())
    return np.concatenate(P), np.concatenate(L), np.concatenate(R)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--observation-time", type=int, required=True)
    p.add_argument("--n-subgraphs", type=int, default=6)
    p.add_argument("--gat-dim", type=int, default=32)
    p.add_argument("--gat-heads", type=int, default=2)
    p.add_argument("--gat-layers", type=int, default=2)
    p.add_argument("--gru-hidden", type=int, default=64)
    p.add_argument("--gru-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--early-stop", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.002)          # Table 2
    p.add_argument("--weight-decay", type=float, default=1e-4) # eq 28 L_R
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Same training-recipe options as train_weibo.py, so both models can be given
    # identical treatment; the defaults reproduce the original run.
    p.add_argument("--scheduler", choices=["plateau", "cosine"], default="plateau")
    p.add_argument("--ema-decay", type=float, default=0.0)
    p.add_argument("--init-from", default=None,
                   help="Warm-start weights from a checkpoint (optimizer state is NOT restored, "
                        "so the LR schedule restarts; recorded in the report when used).")
    p.add_argument("--resumed-from-epoch", type=int, default=0,
                   help="Epochs already completed before this warm start, for the record.")
    a = p.parse_args()

    seeds = a.seeds if a.seeds is not None else [a.seed]
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    dd = Path(a.data_dir)

    ds = {s: CascadeGraphDataset(dd / f"cassubts_{s}.pkl") for s in ("train", "val", "test")}
    coll = make_collate(a.n_subgraphs, float(a.observation_time))
    dl = {s: DataLoader(ds[s], batch_size=a.batch_size, shuffle=(s == "train"), collate_fn=coll)
          for s in ds}

    mean_log = float(ds["train"].log_targets().mean())
    all_results = []

    for seed in seeds:
        t_seed_start = time.time()
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        seed_out = out / f"s{seed}" if len(seeds) > 1 else out
        seed_out.mkdir(parents=True, exist_ok=True)

        ckpt = seed_out / "best_cassubts.pt"
        report_file = seed_out / "cassubts_evaluation_report.json"
        if ckpt.exists() and report_file.exists():
            print(f"• Seed {seed} already completed. Loading existing results from {report_file}", flush=True)
            with open(report_file, "r", encoding="utf-8") as fh:
                rep = json.load(fh)
            all_results.append({"seed": seed, "metrics": rep["metrics"]})
            m = rep["metrics"]
            continue

        model = CasSubTS(2, 8, a.gat_dim, a.gat_heads, a.gat_layers,
                         a.gru_hidden, a.gru_layers, a.n_subgraphs, a.dropout).to(dev)
        model.init_target_bias(mean_log)
        if a.init_from:
            model.load_state_dict(torch.load(a.init_from, map_location=dev, weights_only=True))
            print(f"• warm-started from {a.init_from} (after {a.resumed_from_epoch} prior epochs)", flush=True)

        print("=" * 70)
        print(f" CasSubTS Training | Seed = {seed} | Device = {dev}")
        print("=" * 70)
        print(f"• cascades  train={len(ds['train']):,} val={len(ds['val']):,} test={len(ds['test']):,}")
        print(f"• window={a.observation_time}s  subgraphs={a.n_subgraphs}  params={sum(x.numel() for x in model.parameters()):,}")
        print(f"• mean log2(y+1)={mean_log:.4f}", flush=True)

        crit = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        if a.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.05)
        else:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5)
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(a.ema_decay)) if a.ema_decay > 0 else None
        eval_model = ema.module if ema is not None else model

        best, stale, hist = float("inf"), 0, []
        ckpt = seed_out / "best_cassubts.pt"
        for ep in range(1, a.epochs + 1):
            t_ep = time.time()
            tr = run_epoch(model, dl["train"], crit, dev, opt, ema)
            va = run_epoch(eval_model, dl["val"], crit, dev)
            if a.scheduler == "cosine":
                sched.step()
            else:
                sched.step(va)
            dt_ep = time.time() - t_ep
            hist.append({"epoch": ep, "train_msle": tr, "val_msle": va, "lr": opt.param_groups[0]["lr"], "time_sec": dt_ep})
            mark = " *" if va < best - 1e-5 else ""
            print(f"[cassubts s{seed}] epoch {ep:03d}/{a.epochs:03d} [{dt_ep:5.1f}s]  train_MSLE={tr:.4f}  val_MSLE={va:.4f}  lr={opt.param_groups[0]['lr']:.2e}{mark}", flush=True)
            if va < best - 1e-5:
                best, stale = va, 0
                torch.save(eval_model.state_dict(), ckpt)
                if len(seeds) > 1:
                    torch.save(eval_model.state_dict(), out / f"best_cassubts_s{seed}.pt")
            else:
                stale += 1
                if stale >= a.early_stop:
                    print(f"[cassubts s{seed}] early stop at epoch {ep}", flush=True); break

        model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
        pr, lg, rw = predict(model, dl["test"], dev)
        m = regression_metrics(pr, lg, rw)
        m["best_val_MSLE_log2"] = float(best)
        dt_seed = time.time() - t_seed_start
        m["train_time_sec"] = dt_seed
        m["train_time_min"] = dt_seed / 60.0

        # Save per-sample test predictions
        np.savez_compressed(
            seed_out / "test_predictions.npz",
            cascade_id=np.array(ds["test"].cascade_id),
            pred_log2=pr.astype(np.float32),
            true_log2=lg.astype(np.float32),
            true_raw=rw.astype(np.float32),
        )

        report = {
            "model": "CasSubTS (reimplementation from Sci Rep 15:6801, 2025 — no official code released)",
            "setting": {
                "observation_window_seconds": a.observation_time,
                "splits": "identical to pipeline_w/train_weibo.py (chronological 70/15/15)",
                "target": "log2(increment+1), MSE loss — same as our model",
                "n_subgraphs": a.n_subgraphs,
                "warm_started_from_checkpoint": bool(a.init_from),
                "prior_epochs_before_warm_start": a.resumed_from_epoch,
                "training": {"seed": seed, "scheduler": a.scheduler, "ema_decay": a.ema_decay,
                             "epochs": a.epochs, "early_stop": a.early_stop, "lr": a.lr},
            },
            "data": {s: len(ds[s]) for s in ds},
            "metrics": m,
        }
        (seed_out / "cassubts_evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (seed_out / "training_history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")

        print(f"\n[Seed {seed} DONE] Time: {dt_seed:.1f}s ({dt_seed/60:.2f} min) | Test MSLE (log2): {m['MSLE_log2']:.4f} | R2: {m['R2_log2']:.4f} | MAE: {m['MAE_count']:.2f}", flush=True)
        all_results.append({"seed": seed, "metrics": m})

    if len(seeds) > 1:
        metrics_keys = ["MSLE_log2", "MSLE_ln", "MAE_count", "RMSE_count", "MedAE_count", "R2_log2", "PCC_log2", "best_val_MSLE_log2", "train_time_min"]
        summary = {
            "window_seconds": a.observation_time,
            "seeds": seeds,
            "seeds_results": all_results,
        }
        for k in metrics_keys:
            vals = [r["metrics"][k] for r in all_results if k in r["metrics"]]
            if vals:
                summary[f"mean_{k}"] = float(np.mean(vals))
                summary[f"std_{k}"] = float(np.std(vals))
        summary["mean_msle_log2"] = summary.get("mean_MSLE_log2", 0.0)
        summary["std_msle_log2"] = summary.get("std_MSLE_log2", 0.0)
        summary["mean_r2_log2"] = summary.get("mean_R2_log2", 0.0)
        summary["std_r2_log2"] = summary.get("std_R2_log2", 0.0)

        (out / "cassubts_metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\n" + "=" * 70, flush=True)
        print(f" CasSubTS Multi-Seed Summary (Seeds: {seeds})", flush=True)
        print("=" * 70, flush=True)
        print(f"  Test MSLE (log2): {summary['mean_msle_log2']:.4f} ± {summary['std_msle_log2']:.4f}", flush=True)
        print(f"  Test R2   (log2): {summary['mean_r2_log2']:.4f} ± {summary['std_r2_log2']:.4f}", flush=True)

    print("\n" + "=" * 70)
    print(f" CasSubTS TEST RESULTS ({a.observation_time}s observation)")
    print("=" * 70)
    for k in ("MSLE_log2", "MSLE_ln", "MAE_count", "RMSE_count", "MedAE_count", "R2_log2", "PCC_log2"):
        print(f"  {k:<14} {m[k]:.4f}")
    print(f"\n• Report -> {out/'cassubts_evaluation_report.json'}", flush=True)


if __name__ == "__main__":
    main()
