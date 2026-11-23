"""Controlled statistical comparison between CasMotif and DeepHawkes on Twitter cascades.

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

        # Assert exact alignment
        assert np.array_equal(d_ours["cascade_id"], d_dh["cascade_id"]), f"Cascade IDs misaligned for seed {s}!"
        assert np.allclose(d_ours["true_log2"], d_dh["true_log2"], atol=1e-5), f"Targets misaligned for seed {s}!"

        if true_log_ref is None:
            true_log_ref = d_ours["true_log2"]
            true_raw_ref = d_ours["true_raw"]
            ids_ref = d_ours["cascade_id"]

        ours_preds[s] = d_ours["pred_log2"]
        dh_preds[s] = d_dh["pred_log2"]

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

        ens_pred = np.mean([preds_dict[s] for s in seeds], axis=0)
        ensemble_metrics = compute_metrics(ens_pred, true_log_ref, true_raw_ref)
        return per_seed, summary, ensemble_metrics

    ours_per_seed, ours_summary, ours_ensemble = summarize(ours_preds, ours_reports)
    dh_per_seed, dh_summary, dh_ensemble = summarize(dh_preds, dh_reports)

    # Paired statistical tests
    sq_err_ours_mean = np.mean([(ours_preds[s] - true_log_ref) ** 2 for s in seeds], axis=0)
    sq_err_dh_mean = np.mean([(dh_preds[s] - true_log_ref) ** 2 for s in seeds], axis=0)
    paired_seed_mean = run_paired_tests(sq_err_ours_mean, sq_err_dh_mean)

    ens_err_ours = (np.mean([ours_preds[s] for s in seeds], axis=0) - true_log_ref) ** 2
    ens_err_dh = (np.mean([dh_preds[s] for s in seeds], axis=0) - true_log_ref) ** 2
    paired_ensemble = run_paired_tests(ens_err_ours, ens_err_dh)

    s1 = seeds[0]
    sq_err_ours_s1 = (ours_preds[s1] - true_log_ref) ** 2
    sq_err_dh_s1 = (dh_preds[s1] - true_log_ref) ** 2
    paired_s1 = run_paired_tests(sq_err_ours_s1, sq_err_dh_s1)

    # Timing analysis
    ours_train_times = [ours_reports[s]["timing"]["train_wall_seconds"] for s in seeds]
    dh_train_times = [dh_reports[s]["timing"]["train_wall_seconds"] for s in seeds]
    ours_infer_times = [ours_reports[s]["timing"]["test_inference_seconds"] for s in seeds]
    dh_infer_times = [dh_reports[s]["timing"]["test_inference_seconds"] for s in seeds]

    n_test = len(true_log_ref)
    timing_comparison = {
        "casmotif": {
            "avg_train_wall_seconds": float(np.mean(ours_train_times)),
            "std_train_wall_seconds": float(np.std(ours_train_times)),
            "total_train_wall_seconds_3seeds": float(np.sum(ours_train_times)),
            "avg_inference_seconds": float(np.mean(ours_infer_times)),
            "test_throughput_cascades_per_sec": float(n_test / np.mean(ours_infer_times)),
        },
        "deephawkes": {
            "avg_train_wall_seconds": float(np.mean(dh_train_times)),
            "std_train_wall_seconds": float(np.std(dh_train_times)),
            "total_train_wall_seconds_3seeds": float(np.sum(dh_train_times)),
            "avg_inference_seconds": float(np.mean(dh_infer_times)),
            "test_throughput_cascades_per_sec": float(n_test / np.mean(dh_infer_times)),
        },
        "training_speedup_ratio": float(np.mean(dh_train_times) / np.mean(ours_train_times)),
        "inference_speedup_ratio": float(np.mean(dh_infer_times) / np.mean(ours_infer_times)),
    }

    report = {
        "window_seconds": window,
        "seeds": seeds,
        "n_test_cascades": n_test,
        "alignment_verified": True,
        "params": {
            "casmotif_single": ours_reports[seeds[0]]["parameters"],
            "deephawkes_single": dh_reports[seeds[0]]["parameters"],
        },
        "mean_baseline": mean_metrics,
        "casmotif": {
            "per_seed": ours_per_seed,
            "summary": ours_summary,
            "ensemble": ours_ensemble,
        },
        "deephawkes": {
            "per_seed": dh_per_seed,
            "summary": dh_summary,
            "ensemble": dh_ensemble,
        },
        "timing": timing_comparison,
        "paired_tests": {
            "seed_mean_error": paired_seed_mean,
            "ensemble_vs_ensemble": paired_ensemble,
            "seed_first": paired_s1,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print(f" CONTROLLED TWITTER COMPARISON: CasMotif vs DeepHawkes (Window={window}s, {n_test:,} cascades)")
    print("=" * 76)
    print(f"  Mean Baseline         MSLE={mean_metrics['MSLE_log2']:.4f}")
    print(f"  DeepHawkes (3-seed)   MSLE={dh_summary['MSLE_log2']['mean']:.4f} +/- {dh_summary['MSLE_log2']['std']:.4f} | R2={dh_summary['R2_log2']['mean']:.4f} | Train Time={timing_comparison['deephawkes']['avg_train_wall_seconds']:.1f}s")
    print(f"  CasMotif   (3-seed)   MSLE={ours_summary['MSLE_log2']['mean']:.4f} +/- {ours_summary['MSLE_log2']['std']:.4f} | R2={ours_summary['R2_log2']['mean']:.4f} | Train Time={timing_comparison['casmotif']['avg_train_wall_seconds']:.1f}s")
    print(f"  DeepHawkes (ensemble) MSLE={dh_ensemble['MSLE_log2']:.4f} | R2={dh_ensemble['R2_log2']:.4f}")
    print(f"  CasMotif   (ensemble) MSLE={ours_ensemble['MSLE_log2']:.4f} | R2={ours_ensemble['R2_log2']:.4f}")
    print("-" * 76)
    diff = paired_seed_mean["difference_ours_minus_deephawkes"]
    red = paired_seed_mean["relative_msle_reduction"] * 100
    print(f"  Paired t-test: diff={diff:.4f} ({red:+.2f}% MSLE) | t={paired_seed_mean['paired_t_stat']:.3f} | p={paired_seed_mean['paired_t_p']:.3e}")
    print(f"  Wilcoxon p={paired_seed_mean['wilcoxon_p']:.3e} | 95% CI [{paired_seed_mean['ci95_low']:.4f}, {paired_seed_mean['ci95_high']:.4f}] | CasMotif better on {paired_seed_mean['ours_better_fraction']*100:.1f}%")
    print(f"  Training Speedup: {timing_comparison['training_speedup_ratio']:.2f}x faster | Inference Speedup: {timing_comparison['inference_speedup_ratio']:.2f}x faster")
    print(f"• Report written -> {output_path}\n", flush=True)

    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Paired statistical comparison between CasMotif and DeepHawkes.")
    p.add_argument("--window", type=int, required=True)
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
