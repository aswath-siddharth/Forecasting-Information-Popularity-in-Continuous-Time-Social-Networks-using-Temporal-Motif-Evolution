"""Master end-to-end runner for the APS Controlled Benchmark (CasMotif vs DeepHawkes).

Runs both models across seeds 42, 43, 44 on 3-year (1095 days) and 5-year (1825 days) windows,
holding splits, targets, and optimization harness 100% constant, and tracks runtimes.

Usage:
    python -m aps_benchmark.run_benchmark                     # full run (3y + 5y, 3 seeds)
    python -m aps_benchmark.run_benchmark --windows 1095      # 3-year only
    python -m aps_benchmark.run_benchmark --windows 1825      # 5-year only
    python -m aps_benchmark.run_benchmark --seeds 42          # quick smoke test on seed 42
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_step(cmd: list[str], title: str) -> float:
    print("\n" + "#" * 76)
    print(f"# {title}")
    print("# CMD:", " ".join(cmd))
    print("#" * 76, flush=True)
    t0 = time.time()
    res = subprocess.run(cmd)
    duration = time.time() - t0
    if res.returncode != 0:
        raise SystemExit(f"Step failed with exit code {res.returncode}: {title}")
    print(f"• Completed step in {duration:.1f}s\n", flush=True)
    return duration


def main() -> None:
    p = argparse.ArgumentParser(description="Run controlled APS benchmark (CasMotif vs DeepHawkes).")
    p.add_argument("--windows", type=int, nargs="+", default=[1095, 1825], help="Observation windows in days (1095 for 3y, 1825 for 5y)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--device", default="cuda")
    p.add_argument("--input", type=Path, default=Path("cascades.zip"))
    p.add_argument("--force", action="store_true", help="Re-run even if report already exists.")
    args = p.parse_args()

    py = sys.executable
    t_benchmark_start = time.time()

    for w in args.windows:
        w_years = w // 365
        proc_dir = Path(f"aps_data/processed_{w}")
        dh_data_dir = Path(f"aps_data/deephawkes_{w}")
        run_root = Path(f"aps_benchmark/runs_{w}")

        # 1. Parse APS cascades for CasMotif
        if args.force or not (proc_dir / "aps_test.npz").exists():
            run_step(
                [py, "-m", "aps_benchmark.parser",
                 "--input", str(args.input),
                 "--output-dir", str(proc_dir),
                 "--observation-time", str(w)],
                f"Parse APS {w_years}-Year Splits ({w} days)",
            )
        else:
            print(f"• Using cached CasMotif data for {w} days ({proc_dir})", flush=True)

        # 2. Build DeepHawkes flat buffers aligned 1-to-1
        if args.force or not (dh_data_dir / "deephawkes_test.npz").exists():
            run_step(
                [py, "-m", "aps_benchmark.deephawkes_data",
                 "--processed-dir", str(proc_dir),
                 "--output-dir", str(dh_data_dir),
                 "--observation-time", str(w)],
                f"Build DeepHawkes Aligned Buffers ({w} days)",
            )
        else:
            print(f"• Using cached DeepHawkes data for {w} days ({dh_data_dir})", flush=True)

        # 3. Train CasMotif across seeds
        for s in args.seeds:
            out_dir = run_root / "casmotif" / f"s{s}"
            if not args.force and (out_dir / "casmotif_evaluation_report.json").exists():
                print(f"• Skipping CasMotif {w}d seed {s} (already exists: {out_dir})", flush=True)
                continue
            run_step(
                [py, "-m", "aps_benchmark.train_casmotif",
                 "--data-dir", str(proc_dir),
                 "--output-dir", str(out_dir),
                 "--observation-time", str(w),
                 "--batch-size", "64",
                 "--lr", "1e-3",
                 "--weight-decay", "1e-4",
                 "--patience", "3",
                 "--early-stop", "8",
                 "--epochs", "60",
                 "--seed", str(s),
                 "--device", args.device],
                f"Train CasMotif (AttentionGRU) | window={w_years}y ({w}d) | seed={s}",
            )

        # 4. Train DeepHawkes across seeds
        for s in args.seeds:
            out_dir = run_root / "deephawkes" / f"s{s}"
            if not args.force and (out_dir / "deephawkes_evaluation_report.json").exists():
                print(f"• Skipping DeepHawkes {w}d seed {s} (already exists: {out_dir})", flush=True)
                continue
            run_step(
                [py, "-m", "aps_benchmark.train_deephawkes",
                 "--data-dir", str(dh_data_dir),
                 "--output-dir", str(out_dir),
                 "--observation-time", str(w),
                 "--batch-size", "64",
                 "--lr", "1e-3",
                 "--emb-lr", "1e-4",
                 "--weight-decay", "1e-4",
                 "--emb-l2", "0.01",
                 "--patience", "3",
                 "--early-stop", "8",
                 "--epochs", "60",
                 "--seed", str(s),
                 "--device", args.device],
                f"Train DeepHawkes | window={w_years}y ({w}d) | seed={s}",
            )

        # 5. Paired statistical comparison
        run_step(
            [py, "-m", "aps_benchmark.compare",
             "--window", str(w),
             "--seeds", *[str(s) for s in args.seeds],
             "--ours-root", str(run_root / "casmotif"),
             "--dh-root", str(run_root / "deephawkes"),
             "--output", str(run_root / "comparison_report.json")],
            f"Controlled Paired Comparison (CasMotif vs DeepHawkes) | window={w_years}y ({w}d)",
        )

    # 6. Generate 3-panel publication comparison chart
    if Path("aps_benchmark/runs_1095/comparison_report.json").exists():
        run_step(
            [py, "-m", "aps_benchmark.plot_comparison"],
            "Generate APS Comparison Chart",
        )

    total_wall_time = time.time() - t_benchmark_start
    print(f"\n============================================================================")
    print(f" ALL APS BENCHMARKS COMPLETED SUCCESSFULLY in {total_wall_time/60.0:.1f} minutes")
    print(f"============================================================================\n")


if __name__ == "__main__":
    main()
