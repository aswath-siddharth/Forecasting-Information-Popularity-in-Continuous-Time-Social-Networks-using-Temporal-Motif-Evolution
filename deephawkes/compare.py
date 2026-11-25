"""Controlled multi-seed comparison: CasMotif vs DeepHawkes on identical Weibo splits.

Verifies exact cascade-id and target equality across every seed of both models before
computing summary tables and paired statistical significance tests (paired t-test,
Wilcoxon signed-rank test, and 10,000-sample bootstrap 95% CI).

Usage:
    python -m deephawkes.compare --window 3600 \
        --ours-root deephawkes/runs_3600/casmotif \
        --dh-root deephawkes/runs_3600/deephawkes \
        --output deephawkes/runs_3600/comparison_report.json
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
from pipeline_w.weibo_parser import FEATURE_DIM

from .train import compute_full_metrics


@torch.no_grad()
def predict_casmotif(window: int, run_dir: Path, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    splits_dir = Path(f"weibo_data/processed_{window}")
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


def load_deephawkes_preds(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    d = np.load(run_dir / "test_predictions.npz", allow_pickle=True)
    ids = [str(c) for c in d["cascade_id"].tolist()]
    return (
        d["pred_log2"].astype(np.float64).ravel(),
        d["true_log2"].astype(np.float64).ravel(),
        d["true_raw"].astype(np.float64).ravel(),
        ids,
    )


def paired_test(err_ours: np.ndarray, err_base: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    """Compare per-cascade squared log2 errors: diff = ours - baseline (negative favours ours)."""
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
        "msle_deephawkes": msle_base,
        "difference_ours_minus_dh": float(diff.mean()),
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
    keys = ("MSLE_log2", "MSLE_ln", "RMSLE_log2", "R2_log2", "PCC_log2", "MAPE_log2", "MAE_count", "RMSE_count", "MedAE_count")
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
    p = argparse.ArgumentParser(description="Controlled multi-seed comparison: CasMotif vs DeepHawkes.")
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--ours-root", required=True, help="Directory containing s42, s43, s44 subdirs for CasMotif.")
    p.add_argument("--dh-root", required=True, help="Directory containing s42, s43, s44 subdirs for DeepHawkes.")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ours_root, dh_root = Path(args.ours_root), Path(args.dh_root)

    ours_preds, dh_preds = {}, {}
    ours_metrics, dh_metrics = {}, {}
    ref_ids = None
    ref_tlog = None
    ref_traw = None

    for s in args.seeds:
        p_o, tl_o, tr_o, id_o = predict_casmotif(args.window, ours_root / f"s{s}", device)
        p_d, tl_d, tr_d, id_d = load_deephawkes_preds(dh_root / f"s{s}")

        if id_o != id_d:
            raise SystemExit(f"Alignment failure on seed {s}: cascade IDs differ between CasMotif and DeepHawkes!")
        if not np.allclose(tl_o, tl_d, atol=1e-5):
            raise SystemExit(f"Alignment failure on seed {s}: log2 targets differ!")
        if ref_ids is None:
            ref_ids, ref_tlog, ref_traw = id_o, tl_o, tr_o
        else:
            if id_o != ref_ids or not np.allclose(tl_o, ref_tlog, atol=1e-5):
                raise SystemExit(f"Cross-seed alignment failure on seed {s}!")

        ours_preds[f"s{s}"] = p_o
        dh_preds[f"s{s}"] = p_d
        ours_metrics[f"s{s}"] = compute_full_metrics(p_o, tl_o, tr_o)
        dh_metrics[f"s{s}"] = compute_full_metrics(p_d, tl_d, tr_d)

    # Mean baseline (non-learned sanity floor)
    train_ds = WeiboCascadeDataset(f"weibo_data/processed_{args.window}/weibo_train.npz")
    mean_pred = np.full_like(ref_tlog, float(train_ds.log_targets().mean()))
    mean_baseline_metrics = compute_full_metrics(mean_pred, ref_tlog, ref_traw)

    # 3-model ensembles
    ours_ens_pred = np.mean(list(ours_preds.values()), axis=0)
    dh_ens_pred = np.mean(list(dh_preds.values()), axis=0)
    ours_ens_metrics = compute_full_metrics(ours_ens_pred, ref_tlog, ref_traw)
    dh_ens_metrics = compute_full_metrics(dh_ens_pred, ref_tlog, ref_traw)

    # Per-cascade squared log2 error tests
    ours_mean_err = np.mean([(v - ref_tlog) ** 2 for v in ours_preds.values()], axis=0)
    dh_mean_err = np.mean([(v - ref_tlog) ** 2 for v in dh_preds.values()], axis=0)

    test_seed_mean = paired_test(ours_mean_err, dh_mean_err, n_boot=args.bootstrap, seed=0)
    test_ensemble = paired_test((ours_ens_pred - ref_tlog) ** 2, (dh_ens_pred - ref_tlog) ** 2, n_boot=args.bootstrap, seed=0)
    test_seed42 = paired_test(
        (ours_preds[f"s{args.seeds[0]}"] - ref_tlog) ** 2,
        (dh_preds[f"s{args.seeds[0]}"] - ref_tlog) ** 2,
        n_boot=args.bootstrap, seed=0,
    )

    # Pull parameter counts from DeepHawkes s42 report
    dh_rep = json.loads((dh_root / f"s{args.seeds[0]}" / "deephawkes_evaluation_report.json").read_text(encoding="utf-8"))
    ours_model = build_model("attention_gru", FEATURE_DIM, 64, 1, 64, 0.3)
    ours_params = sum(x.numel() for x in ours_model.parameters())

    report = {
        "window_seconds": args.window,
        "seeds": args.seeds,
        "n_test_cascades": len(ref_ids),
        "alignment_verified": True,
        "params": {
            "casmotif_single": int(ours_params),
            "casmotif_ensemble": int(ours_params * len(args.seeds)),
            "deephawkes_single": dh_rep["architecture"]["params"],
        },
        "mean_baseline": mean_baseline_metrics,
        "casmotif": {
            "per_seed": ours_metrics,
            "summary": summarize_seeds(ours_metrics),
            "ensemble": ours_ens_metrics,
        },
        "deephawkes": {
            "per_seed": dh_metrics,
            "summary": summarize_seeds(dh_metrics),
            "ensemble": dh_ens_metrics,
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

    print("=" * 76)
    print(f" CONTROLLED COMPARISON: CasMotif vs DeepHawkes (Weibo {args.window}s, {len(ref_ids):,} test cascades)")
    print("=" * 76)
    print(f"  Mean Baseline       MSLE={mean_baseline_metrics['MSLE_log2']:.4f}")
    os_m = report["casmotif"]["summary"]
    dh_m = report["deephawkes"]["summary"]
    print(f"  DeepHawkes (3-seed) MSLE={dh_m['MSLE_log2']['mean']:.4f} +/- {dh_m['MSLE_log2']['std']:.4f} | "
          f"R2={dh_m['R2_log2']['mean']:.4f} | MAPE={dh_m['MAPE_log2']['mean']:.4f} | MAE={dh_m['MAE_count']['mean']:.2f}")
    print(f"  CasMotif   (3-seed) MSLE={os_m['MSLE_log2']['mean']:.4f} +/- {os_m['MSLE_log2']['std']:.4f} | "
          f"R2={os_m['R2_log2']['mean']:.4f} | MAPE={os_m['MAPE_log2']['mean']:.4f} | MAE={os_m['MAE_count']['mean']:.2f}")
    print(f"  DeepHawkes (ens)    MSLE={dh_ens_metrics['MSLE_log2']:.4f} | R2={dh_ens_metrics['R2_log2']:.4f} | "
          f"MAPE={dh_ens_metrics['MAPE_log2']:.4f} | MAE={dh_ens_metrics['MAE_count']:.2f}")
    print(f"  CasMotif   (ens)    MSLE={ours_ens_metrics['MSLE_log2']:.4f} | R2={ours_ens_metrics['R2_log2']:.4f} | "
          f"MAPE={ours_ens_metrics['MAPE_log2']:.4f} | MAE={ours_ens_metrics['MAE_count']:.2f}")
    print("-" * 76)
    print(f"  Paired test (3-seed mean): diff={test_seed_mean['difference_ours_minus_dh']:+.4f} "
          f"({test_seed_mean['relative_msle_reduction']*100:+.2f}% reduction) | "
          f"95% CI [{test_seed_mean['ci95_low']:+.4f}, {test_seed_mean['ci95_high']:+.4f}] | "
          f"t p={test_seed_mean['paired_t_p']:.4g} | Wilcoxon p={test_seed_mean['wilcoxon_p']:.4g} | "
          f"Ours better on {test_seed_mean['ours_better_fraction']:.1%}")
    print(f"  Paired test (ensemble):    diff={test_ensemble['difference_ours_minus_dh']:+.4f} "
          f"({test_ensemble['relative_msle_reduction']*100:+.2f}% reduction) | "
          f"95% CI [{test_ensemble['ci95_low']:+.4f}, {test_ensemble['ci95_high']:+.4f}] | "
          f"t p={test_ensemble['paired_t_p']:.4g} | Wilcoxon p={test_ensemble['wilcoxon_p']:.4g} | "
          f"Ours better on {test_ensemble['ours_better_fraction']:.1%}")
    print(f"• Report written to {out_path}", flush=True)


if __name__ == "__main__":
    main()

