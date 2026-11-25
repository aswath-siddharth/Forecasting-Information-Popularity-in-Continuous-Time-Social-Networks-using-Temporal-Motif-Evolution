"""Runner for DeepHawkes on the Reddit dataset benchmark across observation windows and random seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Run DeepHawkes on Reddit benchmark across windows and seeds.")
    p.add_argument("--cache-dir", default="reddit_benchmark/cache")
    p.add_argument("--windows", type=int, nargs="+", default=[3600, 7200], help="Observation windows in seconds (default: 3600 7200).")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--early-stop", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    py = sys.executable

    for w in args.windows:
        dh_data = Path(args.cache_dir) / f"deephawkes_{w}"
        if not (dh_data / "deephawkes_test.npz").exists():
            print(f"• Building aligned Reddit split cache for {w}s...", flush=True)
            subprocess.run([
                py, "-m", "reddit_benchmark.parser",
                "--data-dir", "data",
                "--cache-root", str(args.cache_dir),
                "--observation-time", str(w),
            ], check=True)

        for s in args.seeds:
            out_dir = Path(f"reddit_benchmark/runs_{w}/deephawkes/s{s}")
            report_file = out_dir / "deephawkes_evaluation_report.json"
            if not args.force and report_file.exists():
                print(f"• DeepHawkes Reddit {w}s seed {s} already completed: {report_file}")
                continue

            cmd = [
                py, "-m", "deephawkes.train",
                "--data-dir", str(dh_data),
                "--output-dir", str(out_dir),
                "--observation-time", str(w),
                "--seed", str(s),
                "--epochs", str(args.epochs),
                "--early-stop", str(args.early_stop),
                "--patience", str(args.patience),
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--weight-decay", str(args.weight_decay),
                "--device", args.device,
            ]

            print("\n" + "=" * 76)
            print(f" RUNNING: DeepHawkes Reddit | window={w}s | seed={s}")
            print("=" * 76, flush=True)

            res = subprocess.run(cmd)
            if res.returncode != 0:
                print(f"ERROR: DeepHawkes run failed for window {w} seed {s}", file=sys.stderr)
                sys.exit(res.returncode)


if __name__ == "__main__":
    main()

