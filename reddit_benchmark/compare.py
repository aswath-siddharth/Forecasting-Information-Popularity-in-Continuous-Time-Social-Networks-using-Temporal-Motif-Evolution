"""Controlled multi-seed comparison: CasMotif vs DeepHawkes on the novel Reddit dataset.

Reads saved ``test_predictions.npz`` from both models across seeds 42, 43, 44, asserts
strict cascade-ID and target equality, computes summary statistics and paired significance
tests (paired t-test, Wilcoxon signed-rank test, 10,000-sample bootstrap 95% CI), and
writes ``comparison_report.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deephawkes.compare import paired_test, summarize_seeds
from deephawkes.train import compute_full_metrics
from pipeline_w.data import WeiboCascadeDataset
from pipeline_w.model import build_model
from pipeline_w.weibo_parser import FEATURE_DIM


def load_preds(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    d = np.load(run_dir / "test_predictions.npz", allow_pickle=True)
    ids = [str(c) for c in d["cascade_id"].tolist()]
    return (
        d["pred_log2"].astype(np.float64).ravel(),
        d["true_log2"].astype(np.float64).ravel(),
        d["true_raw"].astype(np.float64).ravel(),
        ids,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Controlled multi-seed comparison on Reddit.")
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--cache-root", default="reddit_benchmark/cache")
    p.add_argument("--ours-root", required=True)
    p.add_argument("--dh-root", required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    ours_root, dh_root = Path(args.ours_root), Path(args.dh_root)
    ours_preds, dh_preds = {}, {}
    ours_metrics, dh_metrics = {}, {}
    ref_ids = None
    ref_tlog = None
    ref_traw = None

    for s in args.seeds:
        p_o, tl_o, tr_o, id_o = load_preds(ours_root / f"s{s}")
        p_d, tl_d, tr_d, id_d = load_preds(dh_root / f"s{s}")

        if id_o != id_d:
            raise SystemExit(f"Alignment failure on seed {s}: cascade IDs differ!")
        if not np.allclose(tl_o, tl_d, atol=1e-5):
            raise SystemExit(f"Alignment failure on seed {s}: log2 targets differ!")
        if ref_ids is None:
            ref_ids, ref_tlog, ref_traw = id_o, tl_o, tr_o
        elif id_o != ref_ids or not np.allclose(tl_o, ref_tlog, atol=1e-5):
            raise SystemExit(f"Cross-seed alignment failure on seed {s}!")

        ours_preds[f"s{s}"] = p_o
        dh_preds[f"s{s}"] = p_d
        ours_metrics[f"s{s}"] = compute_full_metrics(p_o, tl_o, tr_o)
        dh_metrics[f"s{s}"] = compute_full_metrics(p_d, tl_d, tr_d)

    train_ds = WeiboCascadeDataset(Path(args.cache_root) / f"casmotif_{args.window}" / "reddit_train.npz")
    mean_pred = np.full_like(ref_tlog, float(train_ds.log_targets().mean()))
    mean_baseline_metrics = compute_full_metrics(mean_pred, ref_tlog, ref_traw)

    ours_ens_pred = np.mean(list(ours_preds.values()), axis=0)
    dh_ens_pred = np.mean(list(dh_preds.values()), axis=0)
    ours_ens_metrics = compute_full_metrics(ours_ens_pred, ref_tlog, ref_traw)
    dh_ens_metrics = compute_full_metrics(dh_ens_pred, ref_tlog, ref_traw)

    ours_mean_err = np.mean([(v - ref_tlog) ** 2 for v in ours_preds.values()], axis=0)
    dh_mean_err = np.mean([(v - ref_tlog) ** 2 for v in dh_preds.values()], axis=0)

    test_seed_mean = paired_test(ours_mean_err, dh_mean_err, n_boot=args.bootstrap, seed=0)
    test_ensemble = paired_test((ours_ens_pred - ref_tlog) ** 2, (dh_ens_pred - ref_tlog) ** 2, n_boot=args.bootstrap, seed=0)
    test_seed_first = paired_test(
        (ours_preds[f"s{args.seeds[0]}"] - ref_tlog) ** 2,
        (dh_preds[f"s{args.seeds[0]}"] - ref_tlog) ** 2,
        n_boot=args.bootstrap, seed=0,
    )

    dh_rep = json.loads((dh_root / f"s{args.seeds[0]}" / "deephawkes_evaluation_report.json").read_text(encoding="utf-8"))
    meta = json.loads((Path(args.cache_root) / f"casmotif_{args.window}" / "metadata.json").read_text(encoding="utf-8"))
    ours_params = sum(x.numel() for x in build_model("attention_gru", FEATURE_DIM, 64, 1, 64, 0.3).parameters())

    report = {
        "dataset": "Novel Multi-Subreddit Reddit Dataset (data/)",
        "window_seconds": args.window,
        "seeds": args.seeds,
        "n_test_cascades": len(ref_ids),
        "total_cascades": int(meta["total_cascades_kept"]),
        "alignment_verified": True,
        "motif_rates_train": meta["casmotif_splits"]["train"]["motif_rates"],
        "known_token_fraction": {s: meta["splits"][s]["known_token_fraction"] for s in ("train", "val", "test")},
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
            "seed_first": test_seed_first,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 76)
    print(f" REDDIT CONTROLLED COMPARISON: CasMotif vs DeepHawkes ({args.window}s, {len(ref_ids):,} test cascades)")
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
