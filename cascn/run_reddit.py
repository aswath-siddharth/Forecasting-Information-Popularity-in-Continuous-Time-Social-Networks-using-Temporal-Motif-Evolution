"""Orchestrator to run CasCN Reddit benchmark across windows and seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .data_reddit import load_reddit_cascn_split


def aggregate_metrics(seed_reports: list[dict]) -> dict:
    metrics_keys = list(seed_reports[0]["metrics"].keys())
    agg = {}
    for k in metrics_keys:
        vals = [r["metrics"][k] for r in seed_reports if k in r["metrics"]]
        if vals:
            agg[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "values": [float(v) for v in vals],
            }
    return agg


def main() -> None:
    p = argparse.ArgumentParser(description="Run CasCN on Reddit benchmark across windows and seeds.")
    p.add_argument("--cache-dir", default="reddit_benchmark/cache")
    p.add_argument("--windows", type=int, nargs="+", default=[7200], help="Observation windows in seconds (default: 7200).")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--early-stop", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-cascades", type=int, default=0, help="Optional cap for quick testing.")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    py = sys.executable

    for w in args.windows:
        # Step 1: Ensure all split caches exist
        cache_path = Path(args.cache_dir) / f"cascn_{w}"
        for split in ("val", "test", "train"):
            split_file = cache_path / f"cascn_{split}.pkl"
            if not split_file.exists():
                print(f"• Pre-extracting CasCN {split} split for {w}s into cache...", flush=True)
                load_reddit_cascn_split(args.cache_dir, split=split, observation_time=w, max_cascades=args.max_cascades)

        seed_reports = []
        for s in args.seeds:
            out_dir = Path(f"cascn/runs_reddit_{w}/s{s}")
            report_file = out_dir / "cascn_evaluation_report.json"
            if not args.force and report_file.exists():
                print(f"• CasCN Reddit {w}s seed {s} already completed: {report_file}")
                with open(report_file, "r") as fh:
                    seed_reports.append(json.load(fh))
                continue

            cmd = [
                py, "-m", "cascn.train_reddit",
                "--cache-dir", args.cache_dir,
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
            if args.max_cascades > 0:
                cmd += ["--max-cascades", str(args.max_cascades)]

            print("\n" + "=" * 76)
            print(f" RUNNING: CasCN Reddit | window={w}s | seed={s}")
            print("=" * 76, flush=True)

            res = subprocess.run(cmd)
            if res.returncode != 0:
                print(f"ERROR: CasCN run failed for window {w} seed {s}")
                sys.exit(res.returncode)

            if report_file.exists():
                with open(report_file, "r") as fh:
                    seed_reports.append(json.load(fh))

        if len(seed_reports) == len(args.seeds):
            summary = {
                "model": "CasCN",
                "dataset": "Reddit",
                "observation_time": w,
                "seeds": args.seeds,
                "summary": aggregate_metrics(seed_reports),
            }
            summary_path = Path(f"cascn/runs_reddit_{w}/summary_report.json")
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"\n• Saved Reddit {w}s 3-seed summary -> {summary_path}")
            msle_m = summary["summary"]["MSLE_log2"]["mean"]
            msle_s = summary["summary"]["MSLE_log2"]["std"]
            print(f"  MSLE_log2: {msle_m:.4f} ± {msle_s:.4f}")


if __name__ == "__main__":
    main()
