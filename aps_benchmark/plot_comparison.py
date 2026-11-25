"""Generate publication-quality 3-panel model comparison chart for the APS citation benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p1095 = Path("aps_benchmark/runs_1095/comparison_report.json")
    if not p1095.exists():
        raise SystemExit(f"Missing {p1095}; run the 3-year APS benchmark first.")
    r1095 = json.loads(p1095.read_text(encoding="utf-8"))

    p1825 = Path("aps_benchmark/runs_1825/comparison_report.json")
    r1825 = json.loads(p1825.read_text(encoding="utf-8")) if p1825.exists() else None

    # Extract 3-year numbers
    dh_3y_msle = r1095["deephawkes"]["summary"]["MSLE_log2"]["mean"]
    dh_3y_std = r1095["deephawkes"]["summary"]["MSLE_log2"]["std"]
    dh_3y_r2 = r1095["deephawkes"]["summary"]["R2_log2"]["mean"]
    dh_3y_params = r1095["params"]["deephawkes_single"]["total"]
    dh_3y_time = r1095["timing"]["deephawkes"]["avg_train_wall_seconds"]

    cm_3y_msle = r1095["casmotif"]["summary"]["MSLE_log2"]["mean"]
    cm_3y_std = r1095["casmotif"]["summary"]["MSLE_log2"]["std"]
    cm_3y_r2 = r1095["casmotif"]["summary"]["R2_log2"]["mean"]
    cm_3y_params = r1095["params"]["casmotif_single"]
    cm_3y_time = r1095["timing"]["casmotif"]["avg_train_wall_seconds"]

    mean_3y_msle = r1095["mean_baseline"]["MSLE_log2"]
    red_3y = r1095["paired_tests"]["seed_mean_error"]["relative_msle_reduction"] * 100.0
    p_3y = r1095["paired_tests"]["seed_mean_error"]["paired_t_p"]
    speedup_3y = r1095["timing"]["training_speedup_ratio"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2), dpi=220)
    fig.patch.set_facecolor("white")

    c_mean = "#94a3b8"
    c_dh = "#e11d48"
    c_cm = "#0f766e"

    # ------------------------------------------------------------------
    # Panel A: 3-Year Window (1,095 days)
    # ------------------------------------------------------------------
    ax = axes[0]
    labels_a = ["Mean\nBaseline", "DeepHawkes\n(CIKM '17)", "CasMotif\n(Ours)"]
    vals_a = [mean_3y_msle, dh_3y_msle, cm_3y_msle]
    errs_a = [0.0, dh_3y_std, cm_3y_std]
    r2_a = [0.0, dh_3y_r2, cm_3y_r2]
    x_a = np.arange(len(labels_a))

    bars_a = ax.bar(x_a, vals_a, yerr=errs_a, capsize=4, color=[c_mean, c_dh, c_cm], width=0.52, edgecolor="#1e293b", lw=0.7)
    ax.set_xticks(x_a)
    ax.set_xticklabels(labels_a, fontsize=9.2)
    ax.set_ylabel("MSLE (log₂)  ↓ lower is better", fontweight="bold")
    ax.set_title(f"(a) APS 3-Year Window ({r1095['n_test_cascades']:,} test papers)", fontweight="bold", pad=12)
    ax.set_ylim(0, max(vals_a) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for b, v, e, r2 in zip(bars_a, vals_a, errs_a, r2_a):
        txt = f"{v:.3f}\n(R²=0.000)" if e == 0 else f"{v:.3f} ±{e:.3f}\n(R²={r2:.3f})"
        ax.text(b.get_x() + b.get_width() / 2, v + e + 0.12, txt, ha="center", va="bottom", fontsize=8.4, fontweight="bold")

    ax.annotate(
        f"−{red_3y:.1f}% vs DeepHawkes\n(p = {p_3y:.1e})",
        xy=(2.0, cm_3y_msle + cm_3y_std + 0.55),
        xytext=(1.45, max(vals_a) * 0.95),
        ha="center", fontsize=8.4, fontweight="bold", color=c_cm,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdfa", edgecolor=c_cm, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=c_cm, lw=1.2, connectionstyle="arc3,rad=-0.12"),
    )

    # ------------------------------------------------------------------
    # Panel B: 5-Year Window (1,825 days)
    # ------------------------------------------------------------------
    ax = axes[1]
    if r1825 is not None:
        dh_5y_msle = r1825["deephawkes"]["summary"]["MSLE_log2"]["mean"]
        dh_5y_std = r1825["deephawkes"]["summary"]["MSLE_log2"]["std"]
        dh_5y_r2 = r1825["deephawkes"]["summary"]["R2_log2"]["mean"]
        cm_5y_msle = r1825["casmotif"]["summary"]["MSLE_log2"]["mean"]
        cm_5y_std = r1825["casmotif"]["summary"]["MSLE_log2"]["std"]
        cm_5y_r2 = r1825["casmotif"]["summary"]["R2_log2"]["mean"]
        mean_5y_msle = r1825["mean_baseline"]["MSLE_log2"]
        red_5y = r1825["paired_tests"]["seed_mean_error"]["relative_msle_reduction"] * 100.0
        p_5y = r1825["paired_tests"]["seed_mean_error"]["paired_t_p"]

        labels_b = ["Mean\nBaseline", "DeepHawkes\n(CIKM '17)", "CasMotif\n(Ours)"]
        vals_b = [mean_5y_msle, dh_5y_msle, cm_5y_msle]
        errs_b = [0.0, dh_5y_std, cm_5y_std]
        r2_b = [0.0, dh_5y_r2, cm_5y_r2]
        x_b = np.arange(len(labels_b))

        bars_b = ax.bar(x_b, vals_b, yerr=errs_b, capsize=4, color=[c_mean, c_dh, c_cm], width=0.52, edgecolor="#1e293b", lw=0.7)
        ax.set_xticks(x_b)
        ax.set_xticklabels(labels_b, fontsize=9.2)
        ax.set_ylabel("MSLE (log₂)  ↓ lower is better", fontweight="bold")
        ax.set_title(f"(b) APS 5-Year Window ({r1825['n_test_cascades']:,} test papers)", fontweight="bold", pad=12)
        ax.set_ylim(0, max(vals_b) * 1.25)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

        for b, v, e, r2 in zip(bars_b, vals_b, errs_b, r2_b):
            txt = f"{v:.3f}\n(R²=0.000)" if e == 0 else f"{v:.3f} ±{e:.3f}\n(R²={r2:.3f})"
            ax.text(b.get_x() + b.get_width() / 2, v + e + 0.12, txt, ha="center", va="bottom", fontsize=8.4, fontweight="bold")

        ax.annotate(
            f"−{red_5y:.1f}% vs DeepHawkes\n(p = {p_5y:.1e})",
            xy=(2.0, cm_5y_msle + cm_5y_std + 0.55),
            xytext=(1.45, max(vals_b) * 0.95),
            ha="center", fontsize=8.4, fontweight="bold", color=c_cm,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdfa", edgecolor=c_cm, lw=1.0),
            arrowprops=dict(arrowstyle="->", color=c_cm, lw=1.2, connectionstyle="arc3,rad=-0.12"),
        )
    else:
        ax.text(0.5, 0.5, "5-Year Benchmark in progress...", ha="center", va="center", fontsize=11, color="#64748b")
        ax.set_title("(b) APS 5-Year Window", fontweight="bold", pad=12)

    # ------------------------------------------------------------------
    # Panel C: Parameter Footprint & Efficiency
    # ------------------------------------------------------------------
    ax = axes[2]
    models = ["DeepHawkes\n(3-Year)", "DeepHawkes\n(5-Year)" if r1825 else "DeepHawkes", "CasMotif\n(Ours)"]
    dh_5y_params = r1825["params"]["deephawkes_single"]["total"] if r1825 else dh_3y_params
    p_vals = [dh_3y_params, dh_5y_params, cm_3y_params]
    c_list = [c_dh, c_dh, c_cm]
    x_c = np.arange(len(models))

    bars_c = ax.bar(x_c, p_vals, color=c_list, width=0.52, edgecolor="#1e293b", lw=0.7)
    ax.set_yscale("log")
    ax.set_xticks(x_c)
    ax.set_xticklabels(models, fontsize=9.2)
    ax.set_ylabel("Parameters (log scale)  ↓ lower is lighter", fontweight="bold")
    ax.set_title("(c) Model Parameter Footprint", fontweight="bold", pad=12)
    ax.set_ylim(1e4, max(p_vals) * 8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    ratio_3y = dh_3y_params / cm_3y_params
    for b, pv in zip(bars_c, p_vals):
        txt = f"{pv:,}"
        ax.text(b.get_x() + b.get_width() / 2, pv * 1.25, txt, ha="center", va="bottom", fontsize=8.4, fontweight="bold")

    ax.annotate(
        f"{ratio_3y:.1f}× smaller footprint\n(Motifs vs Node Embeddings)",
        xy=(2.0, cm_3y_params * 1.5),
        xytext=(1.4, cm_3y_params * 25.0),
        ha="center", fontsize=8.4, fontweight="bold", color=c_cm,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdfa", edgecolor=c_cm, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=c_cm, lw=1.2, connectionstyle="arc3,rad=-0.12"),
    )

    plt.tight_layout()
    out_chart = Path("aps_benchmark/aps_comparison_chart.png")
    fig.savefig(out_chart, dpi=220)
    plt.close(fig)
    print(f"• Saved chart -> {out_chart}", flush=True)


if __name__ == "__main__":
    main()
