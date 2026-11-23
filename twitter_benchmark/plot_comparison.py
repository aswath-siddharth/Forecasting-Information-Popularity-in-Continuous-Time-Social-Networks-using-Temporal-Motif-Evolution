"""Generate publication-quality 3-panel model comparison chart for the Twitter cascade benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p86400 = Path("twitter_benchmark/runs_86400/comparison_report.json")
    if not p86400.exists():
        raise SystemExit(f"Missing {p86400}; run the 1-day Twitter benchmark first.")
    r86400 = json.loads(p86400.read_text(encoding="utf-8"))

    p172800 = Path("twitter_benchmark/runs_172800/comparison_report.json")
    r172800 = json.loads(p172800.read_text(encoding="utf-8")) if p172800.exists() else None

    # Extract 1-day numbers
    dh_1d_msle = r86400["deephawkes"]["summary"]["MSLE_log2"]["mean"]
    dh_1d_std = r86400["deephawkes"]["summary"]["MSLE_log2"]["std"]
    dh_1d_r2 = r86400["deephawkes"]["summary"]["R2_log2"]["mean"]
    dh_1d_params = r86400["params"]["deephawkes_single"]["total"]
    dh_1d_time = r86400["timing"]["deephawkes"]["avg_train_wall_seconds"]

    cm_1d_msle = r86400["casmotif"]["summary"]["MSLE_log2"]["mean"]
    cm_1d_std = r86400["casmotif"]["summary"]["MSLE_log2"]["std"]
    cm_1d_r2 = r86400["casmotif"]["summary"]["R2_log2"]["mean"]
    cm_1d_params = r86400["params"]["casmotif_single"]
    cm_1d_time = r86400["timing"]["casmotif"]["avg_train_wall_seconds"]

    mean_1d_msle = r86400["mean_baseline"]["MSLE_log2"]
    red_1d = r86400["paired_tests"]["seed_mean_error"]["relative_msle_reduction"] * 100
    p_1d = r86400["paired_tests"]["seed_mean_error"]["paired_t_p"]
    speedup_1d = r86400["timing"]["training_speedup_ratio"]

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
    # Panel A: 1-Day Window (86,400s)
    # ------------------------------------------------------------------
    ax = axes[0]
    labels_a = ["Mean\nBaseline", "DeepHawkes\n(CIKM '17)", "CasMotif\n(Ours)"]
    vals_a = [mean_1d_msle, dh_1d_msle, cm_1d_msle]
    errs_a = [0.0, dh_1d_std, cm_1d_std]
    r2_a = [0.0, dh_1d_r2, cm_1d_r2]
    x_a = np.arange(len(labels_a))

    bars_a = ax.bar(x_a, vals_a, yerr=errs_a, capsize=4, color=[c_mean, c_dh, c_cm], width=0.52, edgecolor="#1e293b", lw=0.7)
    ax.set_xticks(x_a)
    ax.set_xticklabels(labels_a, fontsize=9.2)
    ax.set_ylabel("MSLE (log₂)  ↓ lower is better", fontweight="bold")
    ax.set_title(f"(a) Twitter 1-Day Window ({r86400['n_test_cascades']:,} test cascades)", fontweight="bold", pad=12)
    ax.set_ylim(0, max(vals_a) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for b, v, e, r2 in zip(bars_a, vals_a, errs_a, r2_a):
        txt = f"{v:.3f}\n(R²=0.000)" if e == 0 else f"{v:.3f} ±{e:.3f}\n(R²={r2:.3f})"
        ax.text(b.get_x() + b.get_width() / 2, v + e + 0.12, txt, ha="center", va="bottom", fontsize=8.4, fontweight="bold")

    ax.annotate(
        f"−{red_1d:.1f}% vs DeepHawkes\n(p = {p_1d:.1e})",
        xy=(2.0, cm_1d_msle + cm_1d_std + 0.55),
        xytext=(1.45, max(vals_a) * 0.95),
        ha="center", fontsize=8.4, fontweight="bold", color=c_cm,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdfa", edgecolor=c_cm, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=c_cm, lw=1.2, connectionstyle="arc3,rad=-0.12"),
    )

    # ------------------------------------------------------------------
    # Panel B: 2-Day Window (172,800s)
    # ------------------------------------------------------------------
    ax = axes[1]
    if r172800 is not None:
        dh_2d_msle = r172800["deephawkes"]["summary"]["MSLE_log2"]["mean"]
        dh_2d_std = r172800["deephawkes"]["summary"]["MSLE_log2"]["std"]
        dh_2d_r2 = r172800["deephawkes"]["summary"]["R2_log2"]["mean"]
        cm_2d_msle = r172800["casmotif"]["summary"]["MSLE_log2"]["mean"]
        cm_2d_std = r172800["casmotif"]["summary"]["MSLE_log2"]["std"]
        cm_2d_r2 = r172800["casmotif"]["summary"]["R2_log2"]["mean"]
        mean_2d_msle = r172800["mean_baseline"]["MSLE_log2"]
        red_2d = r172800["paired_tests"]["seed_mean_error"]["relative_msle_reduction"] * 100
        p_2d = r172800["paired_tests"]["seed_mean_error"]["paired_t_p"]

        labels_b = ["Mean\nBaseline", "DeepHawkes\n(CIKM '17)", "CasMotif\n(Ours)"]
        vals_b = [mean_2d_msle, dh_2d_msle, cm_2d_msle]
        errs_b = [0.0, dh_2d_std, cm_2d_std]
        r2_b = [0.0, dh_2d_r2, cm_2d_r2]
        x_b = np.arange(len(labels_b))

        bars_b = ax.bar(x_b, vals_b, yerr=errs_b, capsize=4, color=[c_mean, c_dh, c_cm], width=0.52, edgecolor="#1e293b", lw=0.7)
        ax.set_xticks(x_b)
        ax.set_xticklabels(labels_b, fontsize=9.2)
        ax.set_ylabel("MSLE (log₂)  ↓ lower is better", fontweight="bold")
        ax.set_title(f"(b) Twitter 2-Day Window ({r172800['n_test_cascades']:,} test cascades)", fontweight="bold", pad=12)
        ax.set_ylim(0, max(vals_b) * 1.25)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

        for b, v, e, r2 in zip(bars_b, vals_b, errs_b, r2_b):
            txt = f"{v:.3f}\n(R²=0.000)" if e == 0 else f"{v:.3f} ±{e:.3f}\n(R²={r2:.3f})"
            ax.text(b.get_x() + b.get_width() / 2, v + e + 0.12, txt, ha="center", va="bottom", fontsize=8.4, fontweight="bold")

        ax.annotate(
            f"−{red_2d:.1f}% vs DeepHawkes\n(p = {p_2d:.1e})",
            xy=(2.0, cm_2d_msle + cm_2d_std + 0.55),
            xytext=(1.45, max(vals_b) * 0.95),
            ha="center", fontsize=8.4, fontweight="bold", color=c_cm,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdfa", edgecolor=c_cm, lw=1.0),
            arrowprops=dict(arrowstyle="->", color=c_cm, lw=1.2, connectionstyle="arc3,rad=-0.12"),
        )
    else:
        ax.text(0.5, 0.5, "2-Day Benchmark in progress...", ha="center", va="center", fontsize=11, color="#64748b")
        ax.set_title("(b) Twitter 2-Day Window", fontweight="bold", pad=12)

    # ------------------------------------------------------------------
    # Panel C: Efficiency Frontier (Parameters & Runtime vs MSLE)
    # ------------------------------------------------------------------
    ax = axes[2]
    pts = [
        (f"CasMotif (Ours)\n{cm_1d_params/1e3:.1f}k params | {cm_1d_time:.1f}s train\nMSLE {cm_1d_msle:.3f} | R² {cm_1d_r2:.3f}", cm_1d_params, cm_1d_msle, c_cm, (-15, 55)),
        (f"DeepHawkes\n{dh_1d_params/1e6:.2f}M params | {dh_1d_time:.1f}s train\nMSLE {dh_1d_msle:.3f} | R² {dh_1d_r2:.3f}", dh_1d_params, dh_1d_msle, c_dh, (-140, -40)),
    ]
    for name, prm, msle, col, offset in pts:
        ax.scatter([prm], [msle], s=200, color=col, edgecolor="#0f172a", linewidth=1.2, zorder=5)
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
    ax.set_xlim(2e4, 3e7)
    y_min = min(cm_1d_msle, dh_1d_msle) * 0.8
    y_max = max(cm_1d_msle, dh_1d_msle) * 1.35
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Total Parameters (log scale)  ← smaller is better", fontweight="bold")
    ax.set_ylabel("1-Day MSLE (log₂)  ↓ lower is better", fontweight="bold")
    ax.set_title("(c) Accuracy vs. Parameter Footprint (1d)", fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    ax.text(
        0.04, 0.05, f"★ Speedup: {speedup_1d:.1f}x faster training\n★ Params: {dh_1d_params/cm_1d_params:.0f}x fewer parameters",
        transform=ax.transAxes, fontsize=8.3, color=c_cm, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#f0fdfa", edgecolor=c_cm, lw=0.8),
    )

    fig.suptitle(
        "Twitter Controlled Cascade Benchmark (Chronological 70/15/15 Split, 3-Seed Mean ± Std)",
        fontsize=13, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = Path("twitter_benchmark/twitter_comparison_chart.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    print(f"• Saved chart -> {out_path}")


if __name__ == "__main__":
    main()
