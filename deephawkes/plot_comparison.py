"""Generate publication-quality model comparison charts from saved JSON reports.

Reads:
  - deephawkes/runs_3600/comparison_report.json
  - deephawkes/runs_900/comparison_report.json
and includes the controlled 3-seed CasSubTS 1h result from the
``cassubts-baseline-comparison`` branch (identical 42,186-cascade Weibo 1h splits).

Writes:
  - deephawkes/model_comparison_chart.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    r3600 = json.loads(Path("deephawkes/runs_3600/comparison_report.json").read_text(encoding="utf-8"))
    r900 = json.loads(Path("deephawkes/runs_900/comparison_report.json").read_text(encoding="utf-8"))

    # Controlled 3-seed CasSubTS numbers on the exact same 42,186-cascade Weibo 1h splits
    # (origin/cassubts-baseline-comparison: pipeline_w/MULTISEED_RESULTS.md)
    cassubts_1h_msle_mean = 1.9864
    cassubts_1h_msle_std = 0.0258
    cassubts_1h_r2 = 0.6238
    cassubts_1h_params = 153_970

    # Extract DeepHawkes & CasMotif numbers directly from saved JSON reports
    dh_1h_msle = r3600["deephawkes"]["summary"]["MSLE_log2"]["mean"]
    dh_1h_std = r3600["deephawkes"]["summary"]["MSLE_log2"]["std"]
    dh_1h_r2 = r3600["deephawkes"]["summary"]["R2_log2"]["mean"]
    dh_1h_params = r3600["params"]["deephawkes_single"]["total"]

    cm_1h_msle = r3600["casmotif"]["summary"]["MSLE_log2"]["mean"]
    cm_1h_std = r3600["casmotif"]["summary"]["MSLE_log2"]["std"]
    cm_1h_r2 = r3600["casmotif"]["summary"]["R2_log2"]["mean"]
    cm_1h_params = r3600["params"]["casmotif_single"]
    mean_1h_msle = r3600["mean_baseline"]["MSLE_log2"]

    dh_15_msle = r900["deephawkes"]["summary"]["MSLE_log2"]["mean"]
    dh_15_std = r900["deephawkes"]["summary"]["MSLE_log2"]["std"]
    dh_15_r2 = r900["deephawkes"]["summary"]["R2_log2"]["mean"]

    cm_15_msle = r900["casmotif"]["summary"]["MSLE_log2"]["mean"]
    cm_15_std = r900["casmotif"]["summary"]["MSLE_log2"]["std"]
    cm_15_r2 = r900["casmotif"]["summary"]["R2_log2"]["mean"]
    mean_15_msle = r900["mean_baseline"]["MSLE_log2"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=220)
    fig.patch.set_facecolor("white")

    c_mean = "#94a3b8"
    c_dh = "#e11d48"
    c_cs = "#f59e0b"
    c_cm = "#0f766e"

    # ------------------------------------------------------------------
    # Panel A: 1-Hour Observation Window (MSLE log2, lower is better)
    # ------------------------------------------------------------------
    ax = axes[0]
    labels_a = ["Mean\nBaseline", "DeepHawkes\n(CIKM '17)", "CasSubTS\n(Sci. Rep. '25)", "CasMotif\n(Ours)"]
    vals_a = [mean_1h_msle, dh_1h_msle, cassubts_1h_msle_mean, cm_1h_msle]
    errs_a = [0.0, dh_1h_std, cassubts_1h_msle_std, cm_1h_std]
    cols_a = [c_mean, c_dh, c_cs, c_cm]

    x_a = np.arange(len(labels_a))
    bars_a = ax.bar(x_a, vals_a, yerr=errs_a, capsize=4, color=cols_a, width=0.58, edgecolor="#1e293b", linewidth=0.7)
    ax.set_xticks(x_a)
    ax.set_xticklabels(labels_a, fontsize=9.5)
    ax.set_ylabel("MSLE (log₂)  ↓ lower is better", fontweight="bold")
    ax.set_title("(a) Standard 1-Hour Window (42,186 cascades)", fontweight="bold", pad=12)
    ax.set_ylim(0, 6.3)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for b, v, e in zip(bars_a, vals_a, errs_a):
        txt = f"{v:.3f}" if e == 0 else f"{v:.3f}\n±{e:.3f}"
        ax.text(b.get_x() + b.get_width() / 2, v + e + 0.14, txt, ha="center", va="bottom", fontsize=8.8, fontweight="bold")

    ax.annotate(
        "−25.7% vs DeepHawkes\n(p = 2.7e-29)",
        xy=(3.0, cm_1h_msle + 0.58),
        xytext=(1.85, 4.45),
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        color=c_cm,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdfa", edgecolor=c_cm, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=c_cm, lw=1.2, connectionstyle="arc3,rad=-0.15"),
    )

    # ------------------------------------------------------------------
    # Panel B: 15-Minute Ultra-Early Window (MSLE log2 + R² callout)
    # ------------------------------------------------------------------
    ax = axes[1]
    labels_b = ["Mean\nBaseline", "DeepHawkes\n(CIKM '17)", "CasMotif\n(Ours)"]
    vals_b = [mean_15_msle, dh_15_msle, cm_15_msle]
    errs_b = [0.0, dh_15_std, cm_15_std]
    r2_b = [0.000, dh_15_r2, cm_15_r2]
    cols_b = [c_mean, c_dh, c_cm]

    x_b = np.arange(len(labels_b))
    bars_b = ax.bar(x_b, vals_b, yerr=errs_b, capsize=4, color=cols_b, width=0.52, edgecolor="#1e293b", linewidth=0.7)
    ax.set_xticks(x_b)
    ax.set_xticklabels(labels_b, fontsize=9.5)
    ax.set_ylabel("MSLE (log₂)  ↓ lower is better", fontweight="bold")
    ax.set_title("(b) Ultra-Early 15-Min Window (23,585 cascades)", fontweight="bold", pad=12)
    ax.set_ylim(0, 6.8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for b, v, e, r2 in zip(bars_b, vals_b, errs_b, r2_b):
        txt = f"{v:.3f}\n(R²={r2:.3f})" if e == 0 else f"{v:.3f} ±{e:.3f}\n(R²={r2:.3f})"
        ax.text(b.get_x() + b.get_width() / 2, v + e + 0.14, txt, ha="center", va="bottom", fontsize=8.6, fontweight="bold")

    ax.annotate(
        "−42.7% vs DeepHawkes\n(p = 1.4e-27)",
        xy=(2.0, cm_15_msle + 0.65),
        xytext=(1.55, 5.55),
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        color=c_cm,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdfa", edgecolor=c_cm, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=c_cm, lw=1.2, connectionstyle="arc3,rad=-0.12"),
    )

    # ------------------------------------------------------------------
    # Panel C: Efficiency Frontier (Params vs MSLE on 1h Weibo)
    # ------------------------------------------------------------------
    ax = axes[2]
    pts = [
        ("CasMotif (Ours)\n49.8k params\nMSLE 1.945 | R² 0.632", cm_1h_params, cm_1h_msle, c_cm, (-10, 62)),
        ("CasSubTS\n154.0k params\nMSLE 1.986 | R² 0.624", cassubts_1h_params, cassubts_1h_msle_mean, c_cs, (42, 2)),
        ("DeepHawkes\n14.94M params\nMSLE 2.617 | R² 0.505", dh_1h_params, dh_1h_msle, c_dh, (-125, -36)),
    ]
    for name, prm, msle, col, offset in pts:
        ax.scatter([prm], [msle], s=190, color=col, edgecolor="#0f172a", linewidth=1.2, zorder=5)
        ax.annotate(
            name,
            xy=(prm, msle),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.4,
            fontweight="bold",
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=col, lw=1.0, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=col, lw=1.1),
        )

    ax.set_xscale("log")
    ax.set_xlim(2e4, 4e7)
    ax.set_ylim(1.75, 2.85)
    ax.set_xlabel("Total Parameters (log scale)  ← smaller is better", fontweight="bold")
    ax.set_ylabel("1-Hour MSLE (log₂)  ↓ lower is better", fontweight="bold")
    ax.set_title("(c) Accuracy vs. Parameter Footprint (1h)", fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.text(
        0.04, 0.05, "★ Ideal: bottom-left\n(lowest error + fewest params)",
        transform=ax.transAxes, fontsize=8.2, color=c_cm, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#f0fdfa", edgecolor=c_cm, lw=0.8),
    )

    fig.suptitle(
        "Sina Weibo Controlled Benchmark (Chronological 70/15/15 Split, 3-Seed Mean ± Std)",
        fontsize=13, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = Path("deephawkes/model_comparison_chart.png")
    fig.savefig(out_path, dpi=220)
    print(f"• Saved chart -> {out_path}")


if __name__ == "__main__":
    main()
