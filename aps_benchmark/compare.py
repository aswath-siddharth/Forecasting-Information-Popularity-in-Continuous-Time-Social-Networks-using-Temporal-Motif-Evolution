"""Controlled statistical comparison between CasMotif and DeepHawkes on APS citation cascades.

Performs:
  1. Cascade-ID and target equality verification.
  2. Per-seed, 3-seed mean +/- std, and 3-model ensemble metrics.
  3. Paired Student's t-test and Wilcoxon signed-rank test.
  4. 10,000-sample bootstrap 95% confidence interval of paired error difference.
  5. Detailed Runtime & Parameter footprint efficiency comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from .train_casmotif import compute_metrics


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[indices].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, 100.0 * alpha))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return low, high


def run_paired_tests(err_ours: np.ndarray, err_other: np.ndarray) -> dict:
    diff = err_ours - err_other
    mean_diff = float(np.mean(diff))
    mean_ours = float(np.mean(err_ours))
    mean_other = float(np.mean(err_other))
    rel_reduction = float((mean_other - mean_ours) / mean_other) if mean_other > 0 else 0.0

    t_stat, t_p = stats.ttest_rel(err_ours, err_other)
    try:
        w_res = stats.wilcoxon(err_ours, err_other)
        w_p = float(w_res.pvalue)
    except Exception:
        w_p = float("nan")

    low, high = bootstrap_ci(diff)
    better_frac = float(np.mean(err_ours < err_other))

    return {
        "n_test_cascades": int(len(diff)),
        "msle_ours": mean_ours,
        "msle_deephawkes": mean_other,
        "difference_ours_minus_deephawkes": mean_diff,
        "relative_msle_reduction": rel_reduction,
        "ci95_low": low,
        "ci95_high": high,
        "paired_t_stat": float(t_stat),
        "paired_t_p": float(t_p),
        "wilcoxon_p": w_p,
        "ours_better_fraction": better_frac,
        "significant_at_05": bool(t_p < 0.05 and high < 0.0),
    }


def compare_models(
    window: int,
    seeds: list[int],
    ours_root: Path,
    dh_root: Path,
    output_path: Path,
) -> dict:
    ours_preds: dict[int, np.ndarray] = {}
    dh_preds: dict[int, np.ndarray] = {}
    ours_reports: dict[int, dict] = {}
    dh_reports: dict[int, dict] = {}

    true_log_ref: np.ndarray | None = None
    true_raw_ref: np.ndarray | None = None
    ids_ref: np.ndarray | None = None

    for s in seeds:
        ours_p = ours_root / f"s{s}" / "test_predictions.npz"
        dh_p = dh_root / f"s{s}" / "test_predictions.npz"
        if not ours_p.exists() or not dh_p.exists():
            raise FileNotFoundError(f"Missing predictions for seed {s}: {ours_p} or {dh_p}")

        d_ours = np.load(ours_p, allow_pickle=True)
        d_dh = np.load(dh_p, allow_pickle=True)

        ours_true_log = d_ours["true_log2"].ravel()
        dh_true_log = d_dh["true_log2"].ravel()
        ours_true_raw = d_ours["true_raw"].ravel()
        dh_true_raw = d_dh["true_raw"].ravel()
        ours_pred = d_ours["pred_log2"].ravel()
        dh_pred = d_dh["pred_log2"].ravel()

        # Assert exact alignment
        assert np.array_equal(d_ours["cascade_id"].ravel(), d_dh["cascade_id"].ravel()), f"Cascade IDs misaligned for seed {s}!"
        assert np.allclose(ours_true_log, dh_true_log, atol=1e-5), f"Targets misaligned for seed {s}!"

        if true_log_ref is None:
            true_log_ref = ours_true_log
            true_raw_ref = ours_true_raw
            ids_ref = d_ours["cascade_id"].ravel()

        ours_preds[s] = ours_pred
        dh_preds[s] = dh_pred

        ours_reports[s] = json.loads((ours_root / f"s{s}" / "casmotif_evaluation_report.json").read_text(encoding="utf-8"))
        dh_reports[s] = json.loads((dh_root / f"s{s}" / "deephawkes_evaluation_report.json").read_text(encoding="utf-8"))

    # Mean Baseline
    mean_val = float(np.mean(true_log_ref))
    mean_pred = np.full_like(true_log_ref, mean_val)
    mean_metrics = compute_metrics(mean_pred, true_log_ref, true_raw_ref)

    # Per-seed metrics
    def summarize(preds_dict, reports_dict):
        per_seed = {}
        for s in seeds:
            per_seed[f"s{s}"] = compute_metrics(preds_dict[s], true_log_ref, true_raw_ref)

        keys = list(per_seed[f"s{seeds[0]}"].keys())
        summary = {}
        for k in keys:
            if k == "n_samples":
                continue
            vals = [per_seed[f"s{s}"][k] for s in seeds]
            summary[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

        # Ensemble
        ens_pred = np.mean([preds_dict[s] for s in seeds], axis=0)
        ens_metrics = compute_metrics(ens_pred, true_log_ref, true_raw_ref)

        return {"per_seed": per_seed, "summary": summary, "ensemble": ens_metrics}

    ours_summary = summarize(ours_preds, ours_reports)
    dh_summary = summarize(dh_preds, dh_reports)

    # Statistical significance on seed-mean squared log error
    ours_seed_mean_pred = np.mean([ours_preds[s] for s in seeds], axis=0)
    dh_seed_mean_pred = np.mean([dh_preds[s] for s in seeds], axis=0)
    err_ours_mean = (ours_seed_mean_pred - true_log_ref) ** 2
    err_dh_mean = (dh_seed_mean_pred - true_log_ref) ** 2
    paired_seed_mean = run_paired_tests(err_ours_mean, err_dh_mean)

    # Ensemble vs Ensemble paired test
    paired_ensemble = run_paired_tests(
        (ours_summary["ensemble"]["MSLE_log2"] != 0) * ((np.mean([ours_preds[s] for s in seeds], axis=0) - true_log_ref) ** 2),
        (dh_summary["ensemble"]["MSLE_log2"] != 0) * ((np.mean([dh_preds[s] for s in seeds], axis=0) - true_log_ref) ** 2),
    )

    # First-seed paired test
    paired_s0 = run_paired_tests(
        (ours_preds[seeds[0]] - true_log_ref) ** 2,
        (dh_preds[seeds[0]] - true_log_ref) ** 2,
    )

    # Timing analysis
    t_train_ours = [ours_reports[s]["timing"]["train_wall_seconds"] for s in seeds]
    t_train_dh = [dh_reports[s]["timing"]["train_wall_seconds"] for s in seeds]
    t_infer_ours = [ours_reports[s]["timing"]["test_inference_seconds"] for s in seeds]
    t_infer_dh = [dh_reports[s]["timing"]["test_inference_seconds"] for s in seeds]

    timing_summary = {
        "casmotif": {
            "avg_train_wall_seconds": float(np.mean(t_train_ours)),
            "std_train_wall_seconds": float(np.std(t_train_ours)),
            "total_train_wall_seconds_3seeds": float(np.sum(t_train_ours)),
            "avg_inference_seconds": float(np.mean(t_infer_ours)),
            "test_throughput_cascades_per_sec": float(len(true_log_ref) / max(1e-6, np.mean(t_infer_ours))),
        },
        "deephawkes": {
            "avg_train_wall_seconds": float(np.mean(t_train_dh)),
            "std_train_wall_seconds": float(np.std(t_train_dh)),
            "total_train_wall_seconds_3seeds": float(np.sum(t_train_dh)),
            "avg_inference_seconds": float(np.mean(t_infer_dh)),
            "test_throughput_cascades_per_sec": float(len(true_log_ref) / max(1e-6, np.mean(t_infer_dh))),
        },
        "training_speedup_ratio": float(np.mean(t_train_dh) / max(1e-6, np.mean(t_train_ours))),
        "inference_speedup_ratio": float(np.mean(t_infer_dh) / max(1e-6, np.mean(t_infer_ours))),
    }

    full_report = {
        "window_days": window,
        "seeds": seeds,
        "n_test_cascades": len(true_log_ref),
        "alignment_verified": True,
        "params": {
            "casmotif_single": ours_reports[seeds[0]]["parameters"],
            "deephawkes_single": dh_reports[seeds[0]]["parameters"],
        },
        "mean_baseline": mean_metrics,
        "casmotif": ours_summary,
        "deephawkes": dh_summary,
        "timing": timing_summary,
        "paired_tests": {
            "seed_mean_error": paired_seed_mean,
            "ensemble_vs_ensemble": paired_ensemble,
            "seed_first": paired_s0,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print(f" CONTROLLED APS COMPARISON: CasMotif vs DeepHawkes (Window={window}d, {len(true_log_ref):,} cascades)")
    print("=" * 76)
    print(f"  Mean Baseline         MSLE={mean_metrics['MSLE_log2']:.4f}")
    print(
        f"  DeepHawkes (3-seed)   MSLE={dh_summary['summary']['MSLE_log2']['mean']:.4f} +/- "
        f"{dh_summary['summary']['MSLE_log2']['std']:.4f} | R2={dh_summary['summary']['R2_log2']['mean']:.4f} | "
        f"Train Time={timing_summary['deephawkes']['avg_train_wall_seconds']:.1f}s"
    )
    print(
        f"  CasMotif   (3-seed)   MSLE={ours_summary['summary']['MSLE_log2']['mean']:.4f} +/- "
        f"{ours_summary['summary']['MSLE_log2']['std']:.4f} | R2={ours_summary['summary']['R2_log2']['mean']:.4f} | "
        f"Train Time={timing_summary['casmotif']['avg_train_wall_seconds']:.1f}s"
    )
    print(f"  DeepHawkes (ensemble) MSLE={dh_summary['ensemble']['MSLE_log2']:.4f} | R2={dh_summary['ensemble']['R2_log2']:.4f}")
    print(f"  CasMotif   (ensemble) MSLE={ours_summary['ensemble']['MSLE_log2']:.4f} | R2={ours_summary['ensemble']['R2_log2']:.4f}")
    print("-" * 76)
    diff_val = paired_seed_mean['difference_ours_minus_deephawkes']
    rel_red = paired_seed_mean['relative_msle_reduction'] * 100.0
    print(
        f"  Paired t-test: diff={diff_val:.4f} ({rel_red:+.2f}% MSLE) | "
        f"t={paired_seed_mean['paired_t_stat']:.3f} | p={paired_seed_mean['paired_t_p']:.3e}"
    )
    print(
        f"  Wilcoxon p={paired_seed_mean['wilcoxon_p']:.3e} | "
        f"95% CI [{paired_seed_mean['ci95_low']:.4f}, {paired_seed_mean['ci95_high']:.4f}] | "
        f"CasMotif better on {paired_seed_mean['ours_better_fraction']*100:.1f}%"
    )
    print(
        f"  Training Speedup: {timing_summary['training_speedup_ratio']:.2f}x faster | "
        f"Inference Speedup: {timing_summary['inference_speedup_ratio']:.2f}x faster"
    )
    print(f"• Report written -> {output_path}\n", flush=True)

    return full_report


def main() -> None:
    p = argparse.ArgumentParser(description="Statistical comparison between CasMotif and DeepHawkes on APS benchmark.")
    p.add_argument("--window", type=int, required=True, help="Observation window in days (1095 or 1825)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--ours-root", type=Path, required=True)
    p.add_argument("--dh-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    compare_models(
        window=args.window,
        seeds=args.seeds,
        ours_root=args.ours_root,
        dh_root=args.dh_root,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
