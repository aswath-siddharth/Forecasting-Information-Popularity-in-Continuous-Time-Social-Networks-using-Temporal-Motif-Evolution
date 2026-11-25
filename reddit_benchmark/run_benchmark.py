"""End-to-end runner for the controlled DeepHawkes vs CasMotif Reddit benchmark.

Runs on the novel 4,050-subreddit Reddit dataset in ``data/`` across seeds 42, 43, 44
for both 1h (3600s) and 15m (900s) observation windows under the exact controlled
protocol, keeping all caches and outputs inside ``reddit_benchmark/``.

Usage:
    python -m reddit_benchmark.run_benchmark
    python -m reddit_benchmark.run_benchmark --windows 900
    python -m reddit_benchmark.run_benchmark --windows 3600
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list[str], title: str) -> None:
    print("\n" + "#" * 76)
    print(f"# {title}")
    print("# CMD:", " ".join(cmd))
    print("#" * 76, flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise SystemExit(f"Step failed with exit code {res.returncode}: {title}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run controlled Reddit DeepHawkes vs CasMotif benchmark.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--windows", type=int, nargs="+", default=[3600, 7200])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 4) // 2)))
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=60, help="Max training epochs (default: 60).")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    p.add_argument("--max-files", type=int, default=0, help="Optional file cap for quick tests (0 = all 4,050 files).")
    p.add_argument("--force", action="store_true", help="Re-run even if outputs already exist.")
    args = p.parse_args()

    py = sys.executable
    cache_root = Path("reddit_benchmark/cache")

    for w in args.windows:
        cm_cache = cache_root / f"casmotif_{w}"
        dh_cache = cache_root / f"deephawkes_{w}"
        run_root = Path(f"reddit_benchmark/runs_{w}")

        # 1. Parallel parsing (skips if cached)
        if args.force or not (cm_cache / "reddit_test.npz").exists() or not (dh_cache / "deephawkes_test.npz").exists():
            cmd = [
                py, "-m", "reddit_benchmark.parser",
                "--data-dir", args.data_dir,
                "--cache-root", str(cache_root),
                "--observation-time", str(w),
                "--workers", str(args.workers),
            ]
            if args.max_files > 0:
                cmd += ["--max-files", str(args.max_files)]
            run_step(cmd, f"Parse Reddit JSON dataset ({w}s) -> aligned CasMotif + DeepHawkes splits")
        else:
            print(f"• Using cached aligned Reddit splits for {w}s ({cache_root})", flush=True)

        # 2. DeepHawkes (3 seeds)
        for s in args.seeds:
            out_dir = run_root / "deephawkes" / f"s{s}"
            if not args.force and (out_dir / "deephawkes_evaluation_report.json").exists():
                print(f"• Skipping DeepHawkes Reddit {w}s seed {s} (already done)", flush=True)
                continue
            run_step(
                [py, "-m", "deephawkes.train",
                 "--data-dir", str(dh_cache),
                 "--output-dir", str(out_dir),
                 "--observation-time", str(w),
                 "--batch-size", str(args.batch_size),
                 "--lr", "1e-3",
                 "--weight-decay", "1e-4",
                 "--patience", "3",
                 "--early-stop", "8",
                 "--epochs", str(args.epochs),
                 "--seed", str(s),
                 "--device", args.device],
                f"Train DeepHawkes (Reddit) | window={w}s | seed={s}",
            )

        # 3. CasMotif (3 seeds)
        for s in args.seeds:
            out_dir = run_root / "casmotif" / f"s{s}"
            if not args.force and (out_dir / "casmotif_evaluation_report.json").exists():
                print(f"• Skipping CasMotif Reddit {w}s seed {s} (already done)", flush=True)
                continue
            run_step(
                [py, "-m", "reddit_benchmark.train_casmotif",
                 "--data-dir", str(cm_cache),
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
                f"Train CasMotif (Reddit) | window={w}s | seed={s}",
            )

        # 4. Aligned paired statistical comparison
        run_step(
            [py, "-m", "reddit_benchmark.compare",
             "--window", str(w),
             "--seeds", *[str(s) for s in args.seeds],
             "--cache-root", str(cache_root),
             "--ours-root", str(run_root / "casmotif"),
             "--dh-root", str(run_root / "deephawkes"),
             "--output", str(run_root / "comparison_report.json")],
            f"Controlled Paired Comparison (Reddit) | window={w}s",
        )

    # 5. Generate comparison chart if both windows completed
    if (Path("reddit_benchmark/runs_3600/comparison_report.json").exists()
            and Path("reddit_benchmark/runs_900/comparison_report.json").exists()):
        run_step(
            [py, "-m", "reddit_benchmark.plot_comparison"],
            "Generate 3-Panel Reddit Model Comparison Chart",
        )


if __name__ == "__main__":
    main()
