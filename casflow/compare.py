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
    parser.add_argument("--splits-dir", type=Path, default=Path("weibo_data/processed_3600"))
    parser.add_argument("--casflow-root", type=Path, default=Path("casflow/runs_3600/casflow"))
    parser.add_argument("--ours-root", type=Path, default=Path("casflow/runs_3600/casmotif"))
    parser.add_argument("--output", type=Path, default=Path("casflow/runs_3600/comparison_report.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

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

    # 3-model ensembles (average in linear count space then log2)
    ours_ens_raw = np.mean([np.clip(np.exp2(ours_preds[s]) - 1.0, 0.0, None) for s in args.seeds], axis=0)
    ours_ens_pred = np.log2(ours_ens_raw + 1.0)

    casflow_ens_raw = np.mean([np.clip(np.exp2(casflow_preds[s]) - 1.0, 0.0, None) for s in args.seeds], axis=0)
    casflow_ens_pred = np.log2(casflow_ens_raw + 1.0)

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

    report = {
        "dataset": "Sina Weibo (1h observation window, 24h horizon, 6,328 test cascades)",
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
            "ensemble": compute_full_metrics(casflow_ens_pred, common_t_log, common_t_raw),
        },
        "casmotif": {
            "seeds": ours_metrics,
            "mean_msle_log2": float(np.mean([ours_metrics[s]["MSLE_log2"] for s in args.seeds])),
            "std_msle_log2": float(np.std([ours_metrics[s]["MSLE_log2"] for s in args.seeds])),
            "mean_msle_ln": float(np.mean([ours_metrics[s]["MSLE_ln"] for s in args.seeds])),
            "std_msle_ln": float(np.std([ours_metrics[s]["MSLE_ln"] for s in args.seeds])),
            "mean_r2": float(np.mean([ours_metrics[s]["R2_log2"] for s in args.seeds])),
            "std_r2": float(np.std([ours_metrics[s]["R2_log2"] for s in args.seeds])),
            "ensemble": compute_full_metrics(ours_ens_pred, common_t_log, common_t_raw),
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

    # Write Markdown summary table
    md_content = f"""# Controlled Multi-Seed Comparison: CasMotif vs CasFlow (Sina Weibo)

Head-to-head controlled evaluation of **CasMotif** against **CasFlow** (Xovee Xu et al.,
*CasFlow: Exploring Hierarchical Structures and Propagation Uncertainty for Cascade Prediction*,
IEEE Transactions on Knowledge and Data Engineering, TKDE 2021/2023).

---

## Controlled Protocol (Held Strictly Identical Across Both Models)

| Protocol Dimension | Setting |
| :--- | :--- |
| **Splits** | Chronological **70% / 15% / 15%** (`29,530 / 6,328 / 6,328` = `42,186` total). Zero leakage. |
| **Window & Horizon** | $T_\\text{{obs}} = 3,600\\text{{s (1 h)}} \\to T_\\text{{pred}} = 86,400\\text{{s (24 h)}}$ |
| **Target** | Future increment only: $\\Delta P = R_{{24\\text{{h}}}} - R_{{1\\text{{h}}}}$, transformed as $\\log_2(\\Delta P + 1)$ |
| **Loss & Primary Metric** | $\\text{{MSLE}}_{{\\log_2}} = \\frac{{1}}{{N}}\\sum (\\log_2(\\widehat{{\\Delta P}}+1) - \\log_2(\\Delta P+1))^2$ |
| **Optimization Harness** | Batch size `64`, `AdamW` (`lr = 1e-3`, `weight_decay = 1e-4`), `ReduceLROnPlateau(factor=0.5, patience=3)`, grad clip `max_norm = 1.0`, early stop `patience = 8` (`max_epochs = 60`), restoring best validation MSLE checkpoint |
| **Seeds & Zero Leakage** | Seeds `42, 43, 44`; global graph embeddings & feature scalers fitted strictly on the `29,530` training cascades only |

---

## Results — 1-Hour Observation Window (`3600s`, 6,328 identical test cascades)

| Model | Parameters | Val MSLE ($\\log_2$) $\\downarrow$ | Test MSLE ($\\log_2$) $\\downarrow$ | Test MSLE ($\\ln$) $\\downarrow$ | Test MALE ($\\log_2$) $\\downarrow$ | $R^2$ ($\\log_2$) $\\uparrow$ | PCC $\\uparrow$ | MAPE ($\\log_2$) $\\downarrow$ | MedAE (count) $\\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Mean Baseline | 0 | — | {mean_base_metrics['MSLE_log2']:.4f} | {mean_base_metrics['MSLE_ln']:.4f} | — | {mean_base_metrics['R2_log2']:.4f} | — | {mean_base_metrics['MAPE_log2']:.4f} | {mean_base_metrics['MedAE_count']:.2f} |
| CasFlow (seed 42) | 1,241,913 | — | {casflow_metrics[42]['MSLE_log2']:.4f} | {casflow_metrics[42]['MSLE_ln']:.4f} | {casflow_metrics[42]['MALE_log2']:.4f} | {casflow_metrics[42]['R2_log2']:.4f} | {casflow_metrics[42]['PCC_log2']:.4f} | {casflow_metrics[42]['MAPE_log2']:.4f} | {casflow_metrics[42]['MedAE_count']:.2f} |
| CasFlow (seed 43) | 1,241,913 | — | {casflow_metrics[43]['MSLE_log2']:.4f} | {casflow_metrics[43]['MSLE_ln']:.4f} | {casflow_metrics[43]['MALE_log2']:.4f} | {casflow_metrics[43]['R2_log2']:.4f} | {casflow_metrics[43]['PCC_log2']:.4f} | {casflow_metrics[43]['MAPE_log2']:.4f} | {casflow_metrics[43]['MedAE_count']:.2f} |
| CasFlow (seed 44) | 1,241,913 | — | {casflow_metrics[44]['MSLE_log2']:.4f} | {casflow_metrics[44]['MSLE_ln']:.4f} | {casflow_metrics[44]['MALE_log2']:.4f} | {casflow_metrics[44]['R2_log2']:.4f} | {casflow_metrics[44]['PCC_log2']:.4f} | {casflow_metrics[44]['MAPE_log2']:.4f} | {casflow_metrics[44]['MedAE_count']:.2f} |
| **CasFlow (3-seed mean ± std)** | **1,241,913** | — | **{report['casflow']['mean_msle_log2']:.4f} ± {report['casflow']['std_msle_log2']:.4f}** | **{report['casflow']['mean_msle_ln']:.4f} ± {report['casflow']['std_msle_ln']:.4f}** | **{np.mean([casflow_metrics[s]['MALE_log2'] for s in args.seeds]):.4f}** | **{report['casflow']['mean_r2']:.4f} ± {report['casflow']['std_r2']:.4f}** | **{np.mean([casflow_metrics[s]['PCC_log2'] for s in args.seeds]):.4f}** | **{np.mean([casflow_metrics[s]['MAPE_log2'] for s in args.seeds]):.4f}** | **{np.mean([casflow_metrics[s]['MedAE_count'] for s in args.seeds]):.2f}** |
| CasMotif (seed 42) | 49,824 | 2.0494 | {ours_metrics[42]['MSLE_log2']:.4f} | {ours_metrics[42]['MSLE_ln']:.4f} | {ours_metrics[42]['MALE_log2']:.4f} | {ours_metrics[42]['R2_log2']:.4f} | {ours_metrics[42]['PCC_log2']:.4f} | {ours_metrics[42]['MAPE_log2']:.4f} | {ours_metrics[42]['MedAE_count']:.2f} |
| CasMotif (seed 43) | 49,824 | 2.0628 | {ours_metrics[43]['MSLE_log2']:.4f} | {ours_metrics[43]['MSLE_ln']:.4f} | {ours_metrics[43]['MALE_log2']:.4f} | {ours_metrics[43]['R2_log2']:.4f} | {ours_metrics[43]['PCC_log2']:.4f} | {ours_metrics[43]['MAPE_log2']:.4f} | {ours_metrics[43]['MedAE_count']:.2f} |
| CasMotif (seed 44) | 49,824 | 2.0491 | {ours_metrics[44]['MSLE_log2']:.4f} | {ours_metrics[44]['MSLE_ln']:.4f} | {ours_metrics[44]['MALE_log2']:.4f} | {ours_metrics[44]['R2_log2']:.4f} | {ours_metrics[44]['PCC_log2']:.4f} | {ours_metrics[44]['MAPE_log2']:.4f} | {ours_metrics[44]['MedAE_count']:.2f} |
| **CasMotif (3-seed mean ± std)** | **49,824** | **2.0538 ± 0.0078** | **{report['casmotif']['mean_msle_log2']:.4f} ± {report['casmotif']['std_msle_log2']:.4f}** | **{report['casmotif']['mean_msle_ln']:.4f} ± {report['casmotif']['std_msle_ln']:.4f}** | **{np.mean([ours_metrics[s]['MALE_log2'] for s in args.seeds]):.4f}** | **{report['casmotif']['mean_r2']:.4f} ± {report['casmotif']['std_r2']:.4f}** | **{np.mean([ours_metrics[s]['PCC_log2'] for s in args.seeds]):.4f}** | **{np.mean([ours_metrics[s]['MAPE_log2'] for s in args.seeds]):.4f}** | **{np.mean([ours_metrics[s]['MedAE_count'] for s in args.seeds]):.2f}** |
| CasFlow (3-model ensemble) | 3,725,739 | — | {report['casflow']['ensemble']['MSLE_log2']:.4f} | {report['casflow']['ensemble']['MSLE_ln']:.4f} | {report['casflow']['ensemble']['MALE_log2']:.4f} | {report['casflow']['ensemble']['R2_log2']:.4f} | {report['casflow']['ensemble']['PCC_log2']:.4f} | {report['casflow']['ensemble']['MAPE_log2']:.4f} | {report['casflow']['ensemble']['MedAE_count']:.2f} |
| **CasMotif (3-model ensemble)** | **149,472** | — | **{report['casmotif']['ensemble']['MSLE_log2']:.4f}** | **{report['casmotif']['ensemble']['MSLE_ln']:.4f}** | **{report['casmotif']['ensemble']['MALE_log2']:.4f}** | **{report['casmotif']['ensemble']['R2_log2']:.4f}** | **{report['casmotif']['ensemble']['PCC_log2']:.4f}** | **{report['casmotif']['ensemble']['MAPE_log2']:.4f}** | **{report['casmotif']['ensemble']['MedAE_count']:.2f}** |

---

## Statistical Significance (Per-Cascade Paired Tests on 6,328 Test Cascades)

| Comparison | $\\Delta\\text{{MSLE}}_{{\\log_2}}$ (CasMotif $-$ CasFlow) | Relative MSLE Reduction | 95% Bootstrap CI | Paired $t$-stat | Paired $t$-test $p$-value | Wilcoxon $p$-value | CasMotif Better Fraction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3-Seed Mean Error** | **{report['significance']['delta_msle_log2']:.4f}** | **{report['significance']['relative_reduction_pct']:.2f}%** | **[{ci_low:.4f}, {ci_high:.4f}]** | {t_stat:.2f} | **{p_val:.2e}** | **{w_pval:.2e}** | **{report['significance']['casmotif_better_fraction']:.1%}** |
"""
    Path("casflow/CASFLOW_COMPARISON.md").write_text(md_content, encoding="utf-8")
    print("Generated casflow/CASFLOW_COMPARISON.md", flush=True)


if __name__ == "__main__":
    main()
