"""Master runner to execute all baseline models and CasMotif on the Reddit dataset.

Sequentially activates each model's dedicated branch, runs the multi-seed benchmark
under the standardized 70/15/15 chronological protocol, aggregates test metrics across
seeds, and generates a unified cross-model comparison leaderboard.

Models supported:
  1. DeepHawkes (deephawkes-all-benchmarks)
  2. DeepCas     (deepcas-all-benchmarks)
  3. CasCN       (cascn-all-benchmarks)
  4. CasFlow     (casflow-all-benchmarks)
  5. CasSubTS    (cassubts-all-benchmarks)
  6. CasMotif    (casmotif-all-benchmarks)

Usage:
  # One command to run all 6 models across 3 seeds (42, 43, 44) on 1h (3600s) and 2h (7200s):
  python run_all_reddit.py

  # Quick test slice (e.g. 200 cascades) across all models:
  python run_all_reddit.py --max-cascades 200 --epochs 2

  # Specific window or models:
  python run_all_reddit.py --windows 3600 7200 --models deephawkes casmotif
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


MODEL_CONFIGS = {
    "deephawkes": {
        "name": "DeepHawkes (CIKM '17)",
        "branch": "deephawkes-all-benchmarks",
        "module": "reddit_benchmark.run_benchmark",
        "report_template": "reddit_benchmark/runs_{w}/deephawkes/s{s}/deephawkes_evaluation_report.json",
        "has_max_cascades": False,
    },
    "deepcas": {
        "name": "DeepCas (WWW '17)",
        "branch": "deepcas-all-benchmarks",
        "module": "pipeline_deepcas.run_reddit",
        "report_template": "reddit_deepcas/runs_{w}/s{s}/deepcas_evaluation_report.json",
        "has_max_cascades": True,
    },
    "cascn": {
        "name": "CasCN (CIKM '19)",
        "branch": "cascn-all-benchmarks",
        "module": "cascn.run_reddit",
        "report_template": "cascn/runs_reddit_{w}/s{s}/cascn_evaluation_report.json",
        "has_max_cascades": True,
    },
    "casflow": {
        "name": "CasFlow (IEEE TKDE '21)",
        "branch": "casflow-all-benchmarks",
        "module": "casflow.run_reddit",
        "report_template": "casflow/runs_reddit_{w}/s{s}/casflow_evaluation_report.json",
        "has_max_cascades": True,
    },
    "cassubts": {
        "name": "CasSubTS (Sci. Rep. '25)",
        "branch": "cassubts-all-benchmarks",
        "module": "cassubts.run_reddit",
        "report_template": "cassubts/runs_reddit_{w}/s{s}/cassubts_evaluation_report.json",
        "has_max_cascades": True,
    },
    "casmotif": {
        "name": "CasMotif (Ours)",
        "branch": "casmotif-all-benchmarks",
        "module": "reddit_benchmark.run_casmotif",
        "report_template": "reddit_benchmark/runs_{w}/casmotif/s{s}/casmotif_evaluation_report.json",
        "has_max_cascades": False,
    },
}


def get_current_branch() -> str:
    res = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
    return res.stdout.strip() or "casmotif-all-benchmarks"


def switch_branch(branch: str) -> bool:
    print(f"\n>>> Switching to branch: {branch} ...", flush=True)
    res = subprocess.run(["git", "checkout", branch], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"ERROR switching to {branch}:\n{res.stderr}", file=sys.stderr)
        return False
    return True


def collect_and_format_summary(windows: list[int], seeds: list[int], models: list[str]) -> dict:
    summary = {}

    for w in windows:
        summary[str(w)] = {}
        for m_key in models:
            cfg = MODEL_CONFIGS[m_key]
            seed_metrics = []
            param_count = None
            train_times = []

            for s in seeds:
                rep_path = Path(cfg["report_template"].format(w=w, s=s))
                if rep_path.exists():
                    try:
                        data = json.loads(rep_path.read_text(encoding="utf-8"))
                        m = data.get("metrics", {})
                        if not m and "MSLE_log2" in data:
                            m = data
                        seed_metrics.append(m)
                        if "params" in data:
                            param_count = data["params"]
                        if "train_time_sec" in data:
                            train_times.append(data["train_time_sec"])
                        elif "elapsed_sec" in data:
                            train_times.append(data["elapsed_sec"])
                    except Exception as ex:
                        print(f"Warning reading {rep_path}: {ex}", flush=True)

            if seed_metrics:
                msle_vals = [m.get("MSLE_log2", float("nan")) for m in seed_metrics]
                rmsle_vals = [m.get("RMSLE_log2", float("nan")) for m in seed_metrics]
                r2_vals = [m.get("R2_log2", float("nan")) for m in seed_metrics]
                mae_vals = [m.get("MAE_count", float("nan")) for m in seed_metrics]
                medae_vals = [m.get("MedAE_count", float("nan")) for m in seed_metrics]

                summary[str(w)][m_key] = {
                    "name": cfg["name"],
                    "n_runs": len(seed_metrics),
                    "params": param_count,
                    "avg_train_time_sec": float(sum(train_times) / len(train_times)) if train_times else None,
                    "MSLE_log2_mean": float(sum(msle_vals) / len(msle_vals)),
                    "MSLE_log2_std": float((sum((x - sum(msle_vals)/len(msle_vals))**2 for x in msle_vals) / len(msle_vals))**0.5) if len(msle_vals) > 1 else 0.0,
                    "RMSLE_mean": float(sum(rmsle_vals) / len(rmsle_vals)),
                    "R2_mean": float(sum(r2_vals) / len(r2_vals)),
                    "MAE_mean": float(sum(mae_vals) / len(mae_vals)),
                    "MedAE_mean": float(sum(medae_vals) / len(medae_vals)),
                }

    return summary


def print_summary_table(summary: dict) -> None:
    for w_str, model_dict in summary.items():
        if not model_dict:
            continue
        print("\n" + "=" * 92)
        print(f" REDDIT BENCHMARK LEADERBOARD | Observation Window = {w_str}s (Horizon = 86,400s)")
        print("=" * 92)
        print(f"{'Model':<26} | {'MSLE (log2)':<16} | {'RMSLE':<10} | {'R2':<8} | {'MedAE':<8} | {'Params':<10} | {'Time/Seed':<8}")
        print("-" * 92)

        sorted_models = sorted(model_dict.items(), key=lambda kv: kv[1]["MSLE_log2_mean"])
        for m_key, info in sorted_models:
            msle_str = f"{info['MSLE_log2_mean']:.4f} ± {info['MSLE_log2_std']:.3f}"
            rmsle_str = f"{info['RMSLE_mean']:.4f}"
            r2_str = f"{info['R2_mean']:+.3f}"
            medae_str = f"{info['MedAE_mean']:.1f}"
            params_str = f"{info['params']:,}" if info['params'] else "—"
            time_str = f"{info['avg_train_time_sec']:.1f}s" if info['avg_train_time_sec'] else "—"
            print(f"{info['name']:<26} | {msle_str:<16} | {rmsle_str:<10} | {r2_str:<8} | {medae_str:<8} | {params_str:<10} | {time_str:<8}")
        print("=" * 92 + "\n", flush=True)


def format_detailed_commit_message(m_key: str, cfg: dict, windows: list[int], seeds: list[int]) -> str:
    """Format rich, conventional commit message with quantitative test metrics."""
    lines = [
        f"docs({m_key}): add Reddit benchmark evaluation results ({', '.join(f'{w}s' for w in windows)})",
        "",
        f"Model: {cfg['name']}",
        "Dataset: Novel Multi-Subreddit Comment-Cascade Dataset (data/)",
        f"Protocol: Chronological 70/15/15 split, target Delta P = R_24h - R_T_obs, MSLE_log2 loss",
        f"Observation Windows: {', '.join(f'{w}s ({w/3600:.1f}h)' for w in windows)}",
        f"Horizon: 86,400s (24 hours)",
        f"Seeds: {seeds}",
        "",
        "Evaluation Results:",
    ]

    for w in windows:
        msle_list = []
        r2_list = []
        medae_list = []
        for s in seeds:
            rep = Path(cfg["report_template"].format(w=w, s=s))
            if rep.exists():
                try:
                    data = json.loads(rep.read_text(encoding="utf-8"))
                    m = data.get("metrics", {}) or data
                    if "MSLE_log2" in m:
                        msle_list.append(m["MSLE_log2"])
                    if "R2_log2" in m:
                        r2_list.append(m["R2_log2"])
                    if "MedAE_count" in m:
                        medae_list.append(m["MedAE_count"])
                except Exception:
                    pass

        if msle_list:
            mean_msle = sum(msle_list) / len(msle_list)
            std_msle = (sum((x - mean_msle)**2 for x in msle_list) / len(msle_list))**0.5 if len(msle_list) > 1 else 0.0
            mean_r2 = sum(r2_list) / len(r2_list) if r2_list else 0.0
            mean_medae = sum(medae_list) / len(medae_list) if medae_list else 0.0
            lines.append(
                f"- Window {w}s ({w/3600:.1f}h): MSLE_log2 = {mean_msle:.4f} ± {std_msle:.3f} | "
                f"R2 = {mean_r2:+.3f} | MedAE = {mean_medae:.1f} ({len(msle_list)} runs)"
            )

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Run all cascade prediction models on the Reddit dataset.")
    p.add_argument(
        "--windows", type=int, nargs="+", default=[3600, 7200],
        help="Observation windows in seconds (default: 3600 7200, representing 1h and 2h windows with 24h horizon)."
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds (default: 42 43 44).")
    p.add_argument(
        "--models", type=str, nargs="+",
        default=["deephawkes", "deepcas", "cascn", "casflow", "cassubts", "casmotif"],
        choices=list(MODEL_CONFIGS.keys()),
        help="Models to execute."
    )
    p.add_argument("--epochs", type=int, default=60, help="Max training epochs (default: 60).")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    p.add_argument("--device", default="cuda", help="Execution device (default: cuda).")
    p.add_argument("--max-cascades", type=int, default=0, help="Optional cascade count cap for quick testing (default: 0 = all).")
    p.add_argument("--force", action="store_true", help="Re-run even if checkpoints and reports already exist.")
    p.add_argument("--commit-results", action="store_true", help="Automatically commit generated JSON evaluation reports to each model branch.")
    p.add_argument("--summary-only", action="store_true", help="Skip model training and display aggregated leaderboard.")
    args = p.parse_args()

    initial_branch = get_current_branch()
    py = sys.executable
    t_benchmark_start = time.time()

    print("=" * 92)
    print(" NOVEL REDDIT COMMENT-CASCADE MULTI-MODEL BENCHMARK RUNNER")
    print("=" * 92)
    print(f"• Active Workspaces: 6 Model Branches")
    print(f"• Models Selected: {', '.join(args.models)}")
    print(f"• Windows: {args.windows}s | Seeds: {args.seeds} | Device: {args.device}")
    if args.max_cascades > 0:
        print(f"• Quick-test mode: max_cascades={args.max_cascades}")
    print("=" * 92, flush=True)

    try:
        if not args.summary_only:
            # Pre-check: Ensure aligned Reddit splits exist for each requested window
            cache_root = Path("reddit_benchmark/cache")
            for w in args.windows:
                cm_test = cache_root / f"casmotif_{w}" / "reddit_test.npz"
                dh_test = cache_root / f"deephawkes_{w}" / "deephawkes_test.npz"
                if not cm_test.exists() or not dh_test.exists():
                    print(f"\n• Aligned Reddit cache for {w}s not found. Building aligned splits...", flush=True)
                    switch_branch("deephawkes-all-benchmarks")
                    parse_cmd = [
                        py, "-m", "reddit_benchmark.parser",
                        "--data-dir", "data",
                        "--cache-root", str(cache_root),
                        "--observation-time", str(w),
                        "--workers", "8",
                    ]
                    if args.max_cascades > 0:
                        parse_cmd += ["--max-files", "100"]
                    p_res = subprocess.run(parse_cmd)
                    if p_res.returncode != 0:
                        print(f"WARNING: Parser for window {w} returned code {p_res.returncode}", file=sys.stderr)

            for m_key in args.models:
                cfg = MODEL_CONFIGS[m_key]
                print(f"\n{'#' * 92}")
                print(f"# MODEL: {cfg['name']} (Branch: {cfg['branch']})")
                print(f"{'#' * 92}", flush=True)

                if not switch_branch(cfg["branch"]):
                    print(f"Skipping {cfg['name']} due to branch switch failure.", file=sys.stderr)
                    continue

                cmd = [
                    py, "-m", cfg["module"],
                    "--windows", *[str(w) for w in args.windows],
                    "--seeds", *[str(s) for s in args.seeds],
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--device", args.device,
                ]
                if args.force:
                    cmd.append("--force")
                if args.max_cascades > 0 and cfg["has_max_cascades"]:
                    cmd += ["--max-cascades", str(args.max_cascades)]

                print(f"CMD: {' '.join(cmd)}", flush=True)
                res = subprocess.run(cmd)
                if res.returncode != 0:
                    print(f"WARNING: {cfg['name']} run returned exit code {res.returncode}", file=sys.stderr)

                if args.commit_results:
                    reports_to_add = []
                    for w in args.windows:
                        for s in args.seeds:
                            rep = Path(cfg["report_template"].format(w=w, s=s))
                            if rep.exists():
                                reports_to_add.append(str(rep))
                                hist = rep.parent / "training_history.json"
                                if hist.exists():
                                    reports_to_add.append(str(hist))
                        rep_dir = Path(cfg["report_template"].format(w=w, s=args.seeds[0])).parent.parent
                        sum_rep = rep_dir / "comparison_report.json"
                        if sum_rep.exists():
                            reports_to_add.append(str(sum_rep))

                    if reports_to_add:
                        subprocess.run(["git", "add", "-f", *reports_to_add])
                        commit_msg = format_detailed_commit_message(m_key, cfg, args.windows, args.seeds)
                        c_res = subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
                        if c_res.returncode == 0:
                            print(f"• Successfully committed {len(reports_to_add)} evaluation files to branch {cfg['branch']}", flush=True)
                        else:
                            print(f"• Results already up to date on branch {cfg['branch']}", flush=True)

    finally:
        switch_branch(initial_branch)

    # Aggregate and present results
    summary = collect_and_format_summary(args.windows, args.seeds, args.models)
    print_summary_table(summary)

    out_summary_file = Path("reddit_benchmark_all_models_summary.json")
    out_summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"• Saved multi-model summary report in {time.time()-t_benchmark_start:.1f}s -> {out_summary_file}", flush=True)


if __name__ == "__main__":
    main()
