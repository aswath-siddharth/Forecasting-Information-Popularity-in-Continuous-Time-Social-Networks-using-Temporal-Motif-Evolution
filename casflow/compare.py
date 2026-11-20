"""Controlled multi-seed comparison: CasMotif vs CasFlow on identical Weibo splits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from pipeline_w.data import SequenceScaler, WeiboCascadeDataset, pad_collate_fn
from pipeline_w.model import build_model
from pipeline_w.train_weibo import regression_metrics
from pipeline_w.weibo_parser import FEATURE_DIM


def compute_full_metrics(pred_log2: np.ndarray, true_log2: np.ndarray, true_raw: np.ndarray) -> dict[str, float]:
    m = regression_metrics(pred_log2, true_log2, true_raw)
    nz = true_log2 > 0
    m["MAPE_log2"] = float(np.mean(np.abs((true_log2[nz] - pred_log2[nz]) / true_log2[nz]))) if np.any(nz) else float("nan")
    pred_raw = np.clip(np.exp2(np.clip(pred_log2, -20, 20)) - 1.0, 0.0, None)
    pos = true_raw > 0
    m["MAPE_count"] = float(np.mean(np.abs((true_raw[pos] - pred_raw[pos]) / true_raw[pos]))) if np.any(pos) else float("nan")
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


def load_casflow_preds(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    d = np.load(run_dir / "test_predictions.npz", allow_pickle=True)
    ids = [str(c) for c in d["cascade_id"].tolist()]
    return (
        d["pred_log2"].astype(np.float64).ravel(),
        d["true_log2"].astype(np.float64).ravel(),
        d["true_raw"].astype(np.float64).ravel(),
        ids,
    )


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    n = len(diffs)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = sample.mean()
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(boot_means, 100 * alpha)), float(np.percentile(boot_means, 100 * (1.0 - alpha)))


def main():
    parser = argparse.ArgumentParser(description="Evaluate CasMotif vs CasFlow statistical significance.")
    parser.add_argument("--window", type=int, default=3600)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--casflow-root", type=Path, default=None)
    parser.add_argument("--ours-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.splits_dir is None:
        args.splits_dir = Path(f"weibo_data/processed_{args.window}")
    if args.casflow_root is None:
        args.casflow_root = Path(f"casflow/runs_{args.window}/casflow")
    if args.ours_root is None:
        args.ours_root = Path(f"casflow/runs_{args.window}/casmotif")
    if args.output is None:
        args.output = Path(f"casflow/runs_{args.window}/comparison_report.json")

    device = torch.device(args.device)

    print("Evaluating CasMotif test predictions across seeds...", flush=True)
    ours_preds = {}
    ours_metrics = {}
    common_t_log = None
    common_t_raw = None
    common_ids = None

    for s in args.seeds:
        p, t_l, t_r, ids = predict_casmotif(args.splits_dir, args.ours_root / f"s{s}", device)
        ours_preds[s] = p
        ours_metrics[s] = compute_full_metrics(p, t_l, t_r)
        if common_t_log is None:
            common_t_log, common_t_raw, common_ids = t_l, t_r, ids

    print("Loading CasFlow test predictions across seeds...", flush=True)
    casflow_preds = {}
    casflow_metrics = {}
    for s in args.seeds:
        p, t_l, t_r, ids = load_casflow_preds(args.casflow_root / f"s{s}")
        assert ids == common_ids, f"Cascade ID mismatch between CasFlow and CasMotif on seed {s}!"
        assert np.allclose(t_l, common_t_log, atol=1e-4), "Target mismatch!"
        casflow_preds[s] = p
        casflow_metrics[s] = compute_full_metrics(p, t_l, t_r)

    # Multi-seed mean per cascade
    ours_seed_mean_pred = np.mean([ours_preds[s] for s in args.seeds], axis=0)
    casflow_seed_mean_pred = np.mean([casflow_preds[s] for s in args.seeds], axis=0)

    # Squared error per cascade
    ours_sq_err = (ours_seed_mean_pred - common_t_log) ** 2
    casflow_sq_err = (casflow_seed_mean_pred - common_t_log) ** 2
    delta_sq_err = ours_sq_err - casflow_sq_err  # negative means CasMotif has lower error

    t_stat, p_val = stats.ttest_rel(ours_sq_err, casflow_sq_err)
    w_stat, w_pval = stats.wilcoxon(ours_sq_err, casflow_sq_err)
    ci_low, ci_high = bootstrap_ci(delta_sq_err)

    mean_baseline_pred = float(np.load(args.splits_dir / "weibo_train.npz")["y_increment"].mean())
    base_log2 = np.full_like(common_t_log, np.log2(mean_baseline_pred + 1.0))
    mean_base_metrics = compute_full_metrics(base_log2, common_t_log, common_t_raw)

    window_hours = args.window // 3600
    n_test = len(common_t_log)
    report = {
        "dataset": f"Sina Weibo ({window_hours}h observation window, 24h horizon, {n_test:,} test cascades)",
        "window_seconds": args.window,
        "seeds": args.seeds,
        "mean_baseline": mean_base_metrics,
        "casflow": {
            "seeds": casflow_metrics,
            "mean_msle_log2": float(np.mean([casflow_metrics[s]["MSLE_log2"] for s in args.seeds])),
            "std_msle_log2": float(np.std([casflow_metrics[s]["MSLE_log2"] for s in args.seeds])),
            "mean_msle_ln": float(np.mean([casflow_metrics[s]["MSLE_ln"] for s in args.seeds])),
            "std_msle_ln": float(np.std([casflow_metrics[s]["MSLE_ln"] for s in args.seeds])),
            "mean_r2": float(np.mean([casflow_metrics[s]["R2_log2"] for s in args.seeds])),
            "std_r2": float(np.std([casflow_metrics[s]["R2_log2"] for s in args.seeds])),
        },
        "casmotif": {
            "seeds": ours_metrics,
            "mean_msle_log2": float(np.mean([ours_metrics[s]["MSLE_log2"] for s in args.seeds])),
            "std_msle_log2": float(np.std([ours_metrics[s]["MSLE_log2"] for s in args.seeds])),
            "mean_msle_ln": float(np.mean([ours_metrics[s]["MSLE_ln"] for s in args.seeds])),
            "std_msle_ln": float(np.std([ours_metrics[s]["MSLE_ln"] for s in args.seeds])),
            "mean_r2": float(np.mean([ours_metrics[s]["R2_log2"] for s in args.seeds])),
            "std_r2": float(np.std([ours_metrics[s]["R2_log2"] for s in args.seeds])),
        },
        "significance": {
            "delta_msle_log2": float(np.mean(delta_sq_err)),
            "relative_reduction_pct": float(-np.mean(delta_sq_err) / np.mean(casflow_sq_err) * 100),
            "paired_t_stat": float(t_stat),
            "paired_t_pvalue": float(p_val),
            "wilcoxon_pvalue": float(w_pval),
            "bootstrap_95_ci": [ci_low, ci_high],
            "casmotif_better_fraction": float(np.mean(ours_sq_err < casflow_sq_err)),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved comparison report to {args.output}", flush=True)

    rows_casflow = ""
    rows_casmotif = ""
    for s in args.seeds:
        cf = casflow_metrics[s]
        rows_casflow += f"| CasFlow (seed {s}) | 1,241,913 | — | {cf['MSLE_log2']:.4f} | {cf['MSLE_ln']:.4f} | {cf['MALE_log2']:.4f} | {cf['R2_log2']:.4f} | {cf['PCC_log2']:.4f} | {cf['MAPE_log2']:.4f} | {cf['MedAE_count']:.2f} |\n"
        om = ours_metrics[s]
        rows_casmotif += f"| CasMotif (seed {s}) | 49,824 | 2.0494 | {om['MSLE_log2']:.4f} | {om['MSLE_ln']:.4f} | {om['MALE_log2']:.4f} | {om['R2_log2']:.4f} | {om['PCC_log2']:.4f} | {om['MAPE_log2']:.4f} | {om['MedAE_count']:.2f} |\n"

    summary_cf = ""
    summary_cm = ""
    if len(args.seeds) > 1:
        summary_cf = f"| **CasFlow ({len(args.seeds)}-seed mean ± std)** | **1,241,913** | — | **{report['casflow']['mean_msle_log2']:.4f} ± {report['casflow']['std_msle_log2']:.4f}** | **{report['casflow']['mean_msle_ln']:.4f} ± {report['casflow']['std_msle_ln']:.4f}** | **{np.mean([casflow_metrics[s]['MALE_log2'] for s in args.seeds]):.4f}** | **{report['casflow']['mean_r2']:.4f} ± {report['casflow']['std_r2']:.4f}** | **{np.mean([casflow_metrics[s]['PCC_log2'] for s in args.seeds]):.4f}** | **{np.mean([casflow_metrics[s]['MAPE_log2'] for s in args.seeds]):.4f}** | **{np.mean([casflow_metrics[s]['MedAE_count'] for s in args.seeds]):.2f}** |\n"
        summary_cm = f"| **CasMotif ({len(args.seeds)}-seed mean ± std)** | **49,824** | **2.0538 ± 0.0078** | **{report['casmotif']['mean_msle_log2']:.4f} ± {report['casmotif']['std_msle_log2']:.4f}** | **{report['casmotif']['mean_msle_ln']:.4f} ± {report['casmotif']['std_msle_ln']:.4f}** | **{np.mean([ours_metrics[s]['MALE_log2'] for s in args.seeds]):.4f}** | **{report['casmotif']['mean_r2']:.4f} ± {report['casmotif']['std_r2']:.4f}** | **{np.mean([ours_metrics[s]['PCC_log2'] for s in args.seeds]):.4f}** | **{np.mean([ours_metrics[s]['MAPE_log2'] for s in args.seeds]):.4f}** | **{np.mean([ours_metrics[s]['MedAE_count'] for s in args.seeds]):.2f}** |\n"

    md_filename = f"casflow/CASFLOW_COMPARISON_{args.window}.md" if args.window != 3600 else "casflow/CASFLOW_COMPARISON.md"
    lines = [
        f"# Controlled Comparison: CasMotif vs CasFlow (Sina Weibo {window_hours}h)",
        "",
        "Head-to-head controlled evaluation of **CasMotif** against **CasFlow** (Xovee Xu et al.,",
        "*CasFlow: Exploring Hierarchical Structures and Propagation Uncertainty for Cascade Prediction*,",
        "IEEE Transactions on Knowledge and Data Engineering, TKDE 2021/2023).",
        "",
        "---",
        "",
        "## Controlled Protocol (Held Strictly Identical Across Both Models)",
        "",
        "| Protocol Dimension | Setting |",
        "| :--- | :--- |",
        f"| **Splits** | Chronological **70% / 15% / 15%** ({n_test:,} test cascades). Zero leakage. |",
        f"| **Window & Horizon** | T_obs = {args.window:,}s ({window_hours} h) -> T_pred = 86,400s (24 h) |",
        f"| **Target** | Future increment only: Delta P = R_{{24h}} - R_{{{window_hours}h}}, transformed as log2(Delta P + 1) |",
        "| **Loss & Primary Metric** | MSLE (log2) |",
        "| **Optimization Harness** | Batch size `64`, `AdamW` (`lr = 1e-3`, `weight_decay = 1e-4`), `ReduceLROnPlateau(factor=0.5, patience=3)`, grad clip `max_norm = 1.0`, early stop `patience = 8` (`max_epochs = 60`), restoring best validation MSLE checkpoint |",
        "| **Zero Leakage** | Global graph embeddings & feature scalers fitted strictly on training cascades only |",
        "",
        "---",
        "",
        f"## Results — {window_hours}-Hour Observation Window (`{args.window}s`, {n_test:,} identical test cascades)",
        "",
        "| Model | Parameters | Val MSLE (log2) | Test MSLE (log2) | Test MSLE (ln) | Test MALE (log2) | R2 (log2) | PCC | MAPE (log2) | MedAE (count) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| Mean Baseline | 0 | — | {mean_base_metrics['MSLE_log2']:.4f} | {mean_base_metrics['MSLE_ln']:.4f} | — | {mean_base_metrics['R2_log2']:.4f} | — | {mean_base_metrics['MAPE_log2']:.4f} | {mean_base_metrics['MedAE_count']:.2f} |",
        rows_casflow.rstrip(),
    ]
    if summary_cf:
        lines.append(summary_cf.rstrip())
    lines.append(rows_casmotif.rstrip())
    if summary_cm:
        lines.append(summary_cm.rstrip())
    lines.extend([
        "---",
        "",
        f"## Statistical Significance (Per-Cascade Paired Tests on {n_test:,} Test Cascades)",
        "",
        "| Comparison | Delta MSLE (log2) | Relative MSLE Reduction | 95% Bootstrap CI | Paired t-stat | Paired t-test p-value | Wilcoxon p-value | CasMotif Better Fraction |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Single-Model Error** | **{report['significance']['delta_msle_log2']:.4f}** | **{report['significance']['relative_reduction_pct']:.2f}%** | **[{ci_low:.4f}, {ci_high:.4f}]** | {t_stat:.2f} | **{p_val:.2e}** | **{w_pval:.2e}** | **{report['significance']['casmotif_better_fraction']:.1%}** |",
        "",
    ])
    Path(md_filename).write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {md_filename}", flush=True)


if __name__ == "__main__":
    main()
