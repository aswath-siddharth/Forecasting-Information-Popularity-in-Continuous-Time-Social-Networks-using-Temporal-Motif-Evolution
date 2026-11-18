"""Generate 3-panel model comparison chart for the novel Reddit benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p3600 = Path("reddit_benchmark/runs_3600/comparison_report.json")
    p900 = Path("reddit_benchmark/runs_900/comparison_report.json")
    if not p3600.exists() or not p900.exists():
        raise SystemExit("Both runs_3600 and runs_900 comparison_report.json files are required.")

    r3600 = json.loads(p3600.read_text(encoding="utf-8"))
    r900 = json.loads(p900.read_text(encoding="utf-8"))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=220)
    fig.patch.set_facecolor("white")

    c_mean, c_dh, c_cm = "#94a3b8", "#e11d48", "#0f766e"

    # Panel A: 1-Hour Window
    ax = axes[0]
    m1 = r3600["mean_baseline"]["MSLE_log2"]
    dh1 = r3600["deephawkes"]["summary"]["MSLE_log2"]["mean"]
    dh1_s = r3600["deephawkes"]["summary"]["MSLE_log2"]["std"]
    dh1_r2 = r3600["deephawkes"]["summary"]["R2_log2"]["mean"]
    cm1 = r3600["casmotif"]["summary"]["MSLE_log2"]["mean"]
    cm1_s = r3600["casmotif"]["summary"]["MSLE_log2"]["std"]
    cm1_r2 = r3600["casmotif"]["summary"]["R2_log2"]["mean"]
    red1 = r3600["paired_tests"]["seed_mean_error"]["relative_msle_reduction"] * 100
    p1 = r3600["paired_tests"]["seed_mean_error"]["paired_t_p"]

    vals_a = [m1, dh1, cm1]
    errs_a = [0.0, dh1_s, cm1_s]
    r2_a = [0.0, dh1_r2, cm1_r2]
    x_a = np.arange(3)
    bars_a = ax.bar(x_a, vals_a, yerr=errs_a, capsize=4, color=[c_mean, c_dh, c_cm], width=0.52, edgecolor="#1e293b", lw=0.7)
    ax.set_xticks(x_a)
    ax.set_xticklabels(["Mean\nBaseline", "DeepHawkes\n(CIKM '17)", "CasMotif\n(Ours)"])
    ax.set_ylabel("MSLE (log₂)  ↓ lower is better", fontweight="bold")
    ax.set_title(f"(a) Reddit 1-Hour Window ({r3600['total_cascades']:,} cascades)", fontweight="bold", pad=12)
    ax.set_ylim(0, max(vals_a) * 1.24)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for b, v, e, r2 in zip(bars_a, vals_a, errs_a, r2_a):
        txt = f"{v:.3f}\n(R²={r2:.3f})" if e == 0 else f"{v:.3f} ±{e:.3f}\n(R²={r2:.3f})"
        ax.text(b.get_x() + b.get_width() / 2, v + e + max(vals_a) * 0.025, txt, ha="center", va="bottom", fontsize=8.6, fontweight="bold")
    ax.annotate(
        f"{-red1:+.1f}% vs DeepHawkes\n(p = {p1:.2e})",
        xy=(2.0, cm1 + cm1_s + max(vals_a) * 0.13),
        xytext=(1.45, max(vals_a) * 0.98),
        ha="center", fontsize=8.5, fontweight="bold", color=c_cm,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdfa", edgecolor=c_cm, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=c_cm, lw=1.2, connectionstyle="arc3,rad=-0.12"),
    )

    # Panel B: 15-Minute Window
    ax = axes[1]
    m2 = r900["mean_baseline"]["MSLE_log2"]
    dh2 = r900["deephawkes"]["summary"]["MSLE_log2"]["mean"]
    dh2_s = r900["deephawkes"]["summary"]["MSLE_log2"]["std"]
    dh2_r2 = r900["deephawkes"]["summary"]["R2_log2"]["mean"]
    cm2 = r900["casmotif"]["summary"]["MSLE_log2"]["mean"]
    cm2_s = r900["casmotif"]["summary"]["MSLE_log2"]["std"]
    cm2_r2 = r900["casmotif"]["summary"]["R2_log2"]["mean"]
    red2 = r900["paired_tests"]["seed_mean_error"]["relative_msle_reduction"] * 100
    p2 = r900["paired_tests"]["seed_mean_error"]["paired_t_p"]

    vals_b = [m2, dh2, cm2]
    errs_b = [0.0, dh2_s, cm2_s]
    r2_b = [0.0, dh2_r2, cm2_r2]
    x_b = np.arange(3)
    bars_b = ax.bar(x_b, vals_b, yerr=errs_b, capsize=4, color=[c_mean, c_dh, c_cm], width=0.52, edgecolor="#1e293b", lw=0.7)
    ax.set_xticks(x_b)
    ax.set_xticklabels(["Mean\nBaseline", "DeepHawkes\n(CIKM '17)", "CasMotif\n(Ours)"])
    ax.set_ylabel("MSLE (log₂)  ↓ lower is better", fontweight="bold")
    ax.set_title(f"(b) Reddit 15-Min Window ({r900['total_cascades']:,} cascades)", fontweight="bold", pad=12)
    ax.set_ylim(0, max(vals_b) * 1.24)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for b, v, e, r2 in zip(bars_b, vals_b, errs_b, r2_b):
        txt = f"{v:.3f}\n(R²={r2:.3f})" if e == 0 else f"{v:.3f} ±{e:.3f}\n(R²={r2:.3f})"
        ax.text(b.get_x() + b.get_width() / 2, v + e + max(vals_b) * 0.025, txt, ha="center", va="bottom", fontsize=8.6, fontweight="bold")
    ax.annotate(
        f"{-red2:+.1f}% vs DeepHawkes\n(p = {p2:.2e})",
        xy=(2.0, cm2 + cm2_s + max(vals_b) * 0.13),
        xytext=(1.45, max(vals_b) * 0.98),
        ha="center", fontsize=8.5, fontweight="bold", color=c_cm,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdfa", edgecolor=c_cm, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=c_cm, lw=1.2, connectionstyle="arc3,rad=-0.12"),
    )

    # Panel C: Efficiency Frontier (Params vs 1h MSLE)
    ax = axes[2]
    cm_p = r3600["params"]["casmotif_single"]
    dh_p = r3600["params"]["deephawkes_single"]["total"]
    y_lo = min(cm1, dh1) * 0.85
    y_hi = max(cm1, dh1) * 1.15
    ax.scatter([cm_p], [cm1], s=210, color=c_cm, edgecolor="#0f172a", lw=1.2, zorder=5)
    ax.scatter([dh_p], [dh1], s=210, color=c_dh, edgecolor="#0f172a", lw=1.2, zorder=5)
    ax.annotate(
        f"CasMotif (Ours)\n{cm_p/1e3:.1f}k params\nMSLE {cm1:.3f} | R² {cm1_r2:.3f}",
        xy=(cm_p, cm1), xytext=(22, 35), textcoords="offset points",
        fontsize=8.6, fontweight="bold", color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=c_cm, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=c_cm, lw=1.2),
    )
    ax.annotate(
        f"DeepHawkes\n{dh_p/1e6:.2f}M params\nMSLE {dh1:.3f} | R² {dh1_r2:.3f}",
        xy=(dh_p, dh1), xytext=(-135, -40), textcoords="offset points",
        fontsize=8.6, fontweight="bold", color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=c_dh, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=c_dh, lw=1.2),
    )
    ax.set_xscale("log")
    ax.set_xlim(2e4, max(dh_p * 3.0, 5e7))
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Total Parameters (log scale)  ← smaller is better", fontweight="bold")
    ax.set_ylabel("1-Hour MSLE (log₂)  ↓ lower is better", fontweight="bold")
    ax.set_title("(c) Accuracy vs. Parameter Footprint (1h)", fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Novel Multi-Subreddit Reddit Benchmark (Chronological 70/15/15 Split, 3-Seed Mean ± Std)",
        fontsize=13, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = Path("reddit_benchmark/reddit_comparison_chart.png")
    fig.savefig(out_path, dpi=220)
    print(f"• Saved chart -> {out_path}")


if __name__ == "__main__":
    main()
