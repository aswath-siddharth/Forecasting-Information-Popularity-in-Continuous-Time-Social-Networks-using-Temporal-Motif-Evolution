"""End-to-end runner for the controlled DeepHawkes vs CasMotif benchmark.

Runs both models across seeds 42, 43, 44 on both 1h (3600s) and 15m (900s) Sina Weibo
observation windows under the identical controlled protocol, keeping all outputs inside
``deephawkes/`` and leaving ``pipeline_w/`` completely untouched.

Usage:
    python -m deephawkes.run_benchmark
    python -m deephawkes.run_benchmark --windows 3600 --seeds 42 43 44
    python -m deephawkes.run_benchmark --force
"""

from __future__ import annotations

import argparse
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
    p = argparse.ArgumentParser(description="Run controlled DeepHawkes vs CasMotif multi-seed benchmark.")
    p.add_argument("--windows", type=int, nargs="+", default=[3600, 900])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true", help="Re-run even if output report already exists.")
    args = p.parse_args()

    py = sys.executable

    for w in args.windows:
        proc_dir = Path(f"weibo_data/processed_{w}")
        dh_data_dir = Path(f"weibo_data/deephawkes_{w}")
        run_root = Path(f"deephawkes/runs_{w}")

        if not (proc_dir / "weibo_test.npz").exists():
            run_step(
                [py, "-m", "pipeline_w.weibo_parser",
                 "--input", "weibo_data/dataset_weibo.txt",
                 "--output-dir", str(proc_dir),
                 "--observation-time", str(w)],
                f"Parse Weibo splits ({w}s)",
            )

        if not (dh_data_dir / "deephawkes_test.npz").exists():
            run_step(
                [py, "-m", "deephawkes.data",
                 "--input", "weibo_data/dataset_weibo.txt",
                 "--splits-dir", str(proc_dir),
                 "--output-dir", str(dh_data_dir),
                 "--observation-time", str(w)],
                f"Extract DeepHawkes paths aligned to {w}s splits",
            )

        # 1. DeepHawkes seeds
        for s in args.seeds:
            out_dir = run_root / "deephawkes" / f"s{s}"
            if not args.force and (out_dir / "deephawkes_evaluation_report.json").exists():
                print(f"• Skipping DeepHawkes {w}s seed {s} (already done: {out_dir})", flush=True)
                continue
            run_step(
                [py, "-m", "deephawkes.train",
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
                f"Train DeepHawkes | window={w}s | seed={s}",
            )

        # 2. CasMotif seeds (using unmodified pipeline_w.train_weibo, outputs saved in deephawkes/)
        for s in args.seeds:
            out_dir = run_root / "casmotif" / f"s{s}"
            if not args.force and (out_dir / "weibo_evaluation_report.json").exists():
                print(f"• Skipping CasMotif {w}s seed {s} (already done: {out_dir})", flush=True)
                continue
            run_step(
                [py, "-m", "pipeline_w.train_weibo",
                 "--data-dir", str(proc_dir),
                 "--output-dir", str(out_dir),
                 "--models", "attention_gru",
                 "--batch-size", "64",
                 "--lr", "1e-3",
                 "--weight-decay", "1e-4",
                 "--patience", "3",
                 "--early-stop", "8",
                 "--epochs", "60",
                 "--observation-window-note", str(w),
                 "--seed", str(s),
                 "--device", args.device],
                f"Train CasMotif (attention_gru) | window={w}s | seed={s}",
            )

        # 3. Aligned paired statistical comparison
        run_step(
            [py, "-m", "deephawkes.compare",
             "--window", str(w),
             "--seeds", *[str(s) for s in args.seeds],
             "--ours-root", str(run_root / "casmotif"),
             "--dh-root", str(run_root / "deephawkes"),
             "--output", str(run_root / "comparison_report.json")],
            f"Controlled Paired Comparison | window={w}s",
        )


if __name__ == "__main__":
    main()

