"""Master end-to-end runner for the Twitter Controlled Benchmark (CasMotif vs DeepHawkes).

Runs both models across seeds 42, 43, 44 on 1-day (86400s) and 2-day (172800s) windows,
holding splits, targets, and optimization harness 100% constant, and tracks runtimes.

Usage:
    python -m twitter_benchmark.run_benchmark                    # full run (1d + 2d, 3 seeds)
    python -m twitter_benchmark.run_benchmark --windows 86400    # 1-day only
    python -m twitter_benchmark.run_benchmark --seeds 42         # quick smoke test on seed 42
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
    p = argparse.ArgumentParser(description="Run controlled Twitter benchmark (CasMotif vs DeepHawkes).")
    p.add_argument("--windows", type=int, nargs="+", default=[86400, 172800])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true", help="Re-run even if report already exists.")
    args = p.parse_args()

    py = sys.executable
    t_benchmark_start = time.time()

    for w in args.windows:
        w_days = w // 86400
        proc_dir = Path(f"twitter_data/processed_{w}")
        dh_data_dir = Path(f"twitter_data/deephawkes_{w}")
        run_root = Path(f"twitter_benchmark/runs_{w}")

        # 1. Parse Twitter cascades for CasMotif
        if args.force or not (proc_dir / "twitter_test.npz").exists():
            run_step(
                [py, "-m", "twitter_benchmark.parser",
                 "--input", "twitter_data/twitter.parquet",
                 "--output-dir", str(proc_dir),
                 "--observation-time", str(w)],
                f"Parse Twitter {w_days}-Day Splits ({w}s)",
            )
        else:
            print(f"• Using cached CasMotif data for {w}s ({proc_dir})", flush=True)

        # 2. Build DeepHawkes flat buffers aligned 1-to-1
        if args.force or not (dh_data_dir / "deephawkes_test.npz").exists():
            run_step(
                [py, "-m", "twitter_benchmark.deephawkes_data",
                 "--processed-dir", str(proc_dir),
                 "--output-dir", str(dh_data_dir),
                 "--observation-time", str(w)],
                f"Build DeepHawkes Aligned Buffers ({w}s)",
            )
        else:
            print(f"• Using cached DeepHawkes data for {w}s ({dh_data_dir})", flush=True)

        # 3. Train CasMotif across seeds
        for s in args.seeds:
            out_dir = run_root / "casmotif" / f"s{s}"
            if not args.force and (out_dir / "casmotif_evaluation_report.json").exists():
                print(f"• Skipping CasMotif {w}s seed {s} (already exists: {out_dir})", flush=True)
                continue
            run_step(
                [py, "-m", "twitter_benchmark.train_casmotif",
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
                f"Train CasMotif (AttentionGRU) | window={w_days}d ({w}s) | seed={s}",
            )

        # 4. Train DeepHawkes across seeds
        for s in args.seeds:
            out_dir = run_root / "deephawkes" / f"s{s}"
            if not args.force and (out_dir / "deephawkes_evaluation_report.json").exists():
                print(f"• Skipping DeepHawkes {w}s seed {s} (already exists: {out_dir})", flush=True)
                continue
            run_step(
                [py, "-m", "twitter_benchmark.train_deephawkes",
                 "--data-dir", str(dh_data_dir),
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
                f"Train DeepHawkes | window={w_days}d ({w}s) | seed={s}",
            )

        # 5. Paired statistical comparison
        run_step(
            [py, "-m", "twitter_benchmark.compare",
             "--window", str(w),
             "--seeds", *[str(s) for s in args.seeds],
             "--ours-root", str(run_root / "casmotif"),
             "--dh-root", str(run_root / "deephawkes"),
             "--output", str(run_root / "comparison_report.json")],
            f"Controlled Paired Comparison (CasMotif vs DeepHawkes) | window={w_days}d ({w}s)",
        )

    # 6. Generate 3-panel publication comparison chart
    if Path("twitter_benchmark/runs_86400/comparison_report.json").exists():
        run_step(
            [py, "-m", "twitter_benchmark.plot_comparison"],
            "Generate Twitter Comparison Chart",
        )

    total_wall_time = time.time() - t_benchmark_start
    print(f"\n============================================================================")
    print(f" ALL BENCHMARKS COMPLETED SUCCESSFULLY in {total_wall_time / 60:.1f} minutes")
    print(f"============================================================================\n", flush=True)


if __name__ == "__main__":
    main()
