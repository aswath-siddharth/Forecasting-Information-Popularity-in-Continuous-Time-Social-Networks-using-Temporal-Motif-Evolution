"""Controlled multi-seed comparison: CasMotif vs CasCN on identical Weibo splits.

Verifies exact cascade-id and target equality across every seed of both models before
computing summary tables and paired statistical significance tests (paired t-test,
Wilcoxon signed-rank test, and 10,000-sample bootstrap 95% CI).

Usage:
    python3 -m cascn.compare --window 3600 \
        --ours-root cascn/runs_3600/casmotif \
        --cascn-root cascn/runs_3600/cascn \
        --output cascn/runs_3600/comparison_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from pipeline_w.data import SequenceScaler, WeiboCascadeDataset, pad_collate_fn
from pipeline_w.model import build_model
from pipeline_w.train_weibo import regression_metrics
from pipeline_w.weibo_parser import FEATURE_DIM
from .model import CasCNModel


def compute_full_metrics(pred_log2: np.ndarray, true_log2: np.ndarray, true_raw: np.ndarray) -> dict[str, float]:
    m = regression_metrics(pred_log2, true_log2, true_raw)
    nz = true_log2 > 0
    m["MAPE_log2"] = float(np.mean(np.abs((true_log2[nz] - pred_log2[nz]) / true_log2[nz])))
    pred_raw = np.clip(np.exp2(np.clip(pred_log2, -20, 20)) - 1.0, 0.0, None)
    pos = true_raw > 0
    m["MAPE_count"] = float(np.mean(np.abs((true_raw[pos] - pred_raw[pos]) / true_raw[pos])))
    m["MALE_log2"] = float(np.mean(np.abs(pred_log2 - true_log2)))
    return m


@torch.no_grad()
def predict_casmotif(splits_dir: Path, run_dir: Path, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    train = WeiboCascadeDataset(splits_dir / "weibo_train.npz")
    scaler = SequenceScaler().fit(train.X_flat)
    test = WeiboCascadeDataset(splits_dir / "weibo_test.npz", scaler=scaler)

    model = build_model("attention_gru", FEATURE_DIM, 64, 1, 64, 0.3).to(device)
    ckpt = run_dir / "best_attention_gru.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    preds, t_log, t_raw = [], [], []
    loader = DataLoader(test, batch_size=128, shuffle=False, collate_fn=pad_collate_fn)
    for x, mask, y_log, y_raw in loader:
        out = model(x.to(device), mask.to(device))
        preds.append(out.cpu().numpy().ravel())
        t_log.append(y_log.numpy().ravel())
        t_raw.append(y_raw.numpy().ravel())

    ids = [str(c) for c in test.cascade_id.tolist()]
    return np.concatenate(preds), np.concatenate(t_log), np.concatenate(t_raw), ids


def load_cascn_preds(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    d = np.load(run_dir / "test_predictions.npz", allow_pickle=True)
    ids = [str(c) for c in d["cascade_id"].tolist()]
    return (
        d["pred_log2"].astype(np.float64).ravel(),
        d["true_log2"].astype(np.float64).ravel(),
        d["true_raw"].astype(np.float64).ravel(),
        ids,
    )


def paired_test(err_ours: np.ndarray, err_base: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    diff = err_ours - err_base
    t_stat, t_p = stats.ttest_rel(err_ours, err_base)
    _, w_p = stats.wilcoxon(err_ours, err_base)

    rng = np.random.default_rng(seed)
    n = len(diff)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boot[i] = diff[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])

    msle_ours = float(err_ours.mean())
    msle_base = float(err_base.mean())
    rel_reduction = float((msle_base - msle_ours) / msle_base) if msle_base > 0 else float("nan")

    return {
        "n_test_cascades": int(n),
        "msle_ours": msle_ours,
        "msle_cascn": msle_base,
        "difference_ours_minus_cascn": float(diff.mean()),
        "relative_msle_reduction": rel_reduction,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "paired_t_stat": float(t_stat),
        "paired_t_p": float(t_p),
        "wilcoxon_p": float(w_p),
        "ours_better_fraction": float((diff < 0).mean()),
        "significant_at_05": bool(t_p < 0.05 and lo * hi > 0),
    }


def summarize_seeds(per_seed_metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = ("MSLE_log2", "MSLE_ln", "RMSLE_log2", "MALE_log2", "R2_log2", "PCC_log2", "MAPE_log2", "MAE_count", "RMSE_count", "MedAE_count")
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in per_seed_metrics.values()], dtype=np.float64)
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Controlled multi-seed comparison: CasMotif vs CasCN.")
    p.add_argument("--window", type=int, default=3600)
    p.add_argument("--splits-dir", type=str, default="weibo_data/processed_3600")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--ours-root", default="cascn/runs_3600/casmotif")
    p.add_argument("--cascn-root", default="cascn/runs_3600/cascn")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--output", default="cascn/runs_3600/comparison_report.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir = Path(args.splits_dir)
    ours_root, cascn_root = Path(args.ours_root), Path(args.cascn_root)

    ours_preds, cascn_preds = {}, {}
    ours_metrics, cascn_metrics = {}, {}
    ref_ids, ref_tlog, ref_traw = None, None, None

    for s in args.seeds:
        p_o, tl_o, tr_o, id_o = predict_casmotif(splits_dir, ours_root / f"s{s}", device)
        p_c, tl_c, tr_c, id_c = load_cascn_preds(cascn_root / f"s{s}")

        if id_o != id_c:
            raise SystemExit(f"Alignment failure on seed {s}: cascade IDs differ between CasMotif and CasCN!")
        if not np.allclose(tl_o, tl_c, atol=1e-5):
            raise SystemExit(f"Alignment failure on seed {s}: log2 targets differ!")

        ref_ids, ref_tlog, ref_traw = id_o, tl_o, tr_o
        ours_preds[f"s{s}"] = p_o
        cascn_preds[f"s{s}"] = p_c
        ours_metrics[f"s{s}"] = compute_full_metrics(p_o, tl_o, tr_o)
        cascn_metrics[f"s{s}"] = compute_full_metrics(p_c, tl_c, tr_c)

    train_ds = WeiboCascadeDataset(splits_dir / "weibo_train.npz")
    mean_pred = np.full_like(ref_tlog, float(train_ds.log_targets().mean()))
    mean_baseline_metrics = compute_full_metrics(mean_pred, ref_tlog, ref_traw)

    ours_ens_pred = np.mean(list(ours_preds.values()), axis=0)
    cascn_ens_pred = np.mean(list(cascn_preds.values()), axis=0)
    ours_ens_metrics = compute_full_metrics(ours_ens_pred, ref_tlog, ref_traw)
    cascn_ens_metrics = compute_full_metrics(cascn_ens_pred, ref_tlog, ref_traw)

    ours_mean_err = np.mean([(v - ref_tlog) ** 2 for v in ours_preds.values()], axis=0)
    cascn_mean_err = np.mean([(v - ref_tlog) ** 2 for v in cascn_preds.values()], axis=0)

    test_seed_mean = paired_test(ours_mean_err, cascn_mean_err, n_boot=args.bootstrap, seed=0)
    test_ensemble = paired_test((ours_ens_pred - ref_tlog) ** 2, (cascn_ens_pred - ref_tlog) ** 2, n_boot=args.bootstrap, seed=0)
    test_seed42 = paired_test((ours_preds[f"s{args.seeds[0]}"] - ref_tlog) ** 2, (cascn_preds[f"s{args.seeds[0]}"] - ref_tlog) ** 2, n_boot=args.bootstrap, seed=0)

    ours_params = sum(x.numel() for x in build_model("attention_gru", FEATURE_DIM, 64, 1, 64, 0.3).parameters())
    cascn_params = sum(x.numel() for x in CasCNModel().parameters())

    report = {
        "window_seconds": args.window,
        "seeds": args.seeds,
        "n_test_cascades": len(ref_ids),
        "alignment_verified": True,
        "params": {
            "casmotif_single": int(ours_params),
            "casmotif_ensemble": int(ours_params * len(args.seeds)),
            "cascn_single": int(cascn_params),
            "cascn_ensemble": int(cascn_params * len(args.seeds)),
        },
        "mean_baseline": mean_baseline_metrics,
        "casmotif": {
            "per_seed": ours_metrics,
            "summary": summarize_seeds(ours_metrics),
            "ensemble": ours_ens_metrics,
        },
        "cascn": {
            "per_seed": cascn_metrics,
            "summary": summarize_seeds(cascn_metrics),
            "ensemble": cascn_ens_metrics,
        },
        "paired_tests": {
            "seed_mean_error": test_seed_mean,
            "ensemble_vs_ensemble": test_ensemble,
            "seed_first": test_seed42,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote controlled comparison report to {out_path}")


if __name__ == "__main__":
    main()
