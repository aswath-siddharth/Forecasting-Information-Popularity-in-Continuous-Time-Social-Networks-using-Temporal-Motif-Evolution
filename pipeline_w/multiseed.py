"""Unattended multi-seed run of our model against CasSubTS (Weibo, 1 h).

Targets: the CasSubTS paper's published 1 h score (MSLE 1.987, R² 0.561, MAPE 0.299)
and our own CasSubTS run on identical data (``outputs_cassubts_3600``, MSLE 1.951).

1. Wait for the tuning runs (configs B and C, seed 42, ``--skip-test``) to finish, and
   run the extra tuning configs (D, E) the same way. B and C overfit after ~11 epochs
   under a 60-epoch cosine schedule, so D anneals over 20 epochs and E adds stronger
   regularisation on top.
2. Pick the training recipe by **validation** MSLE among A (the original run) and B-E.
   The test split plays no part in this choice.
3. Train two more seeds of ours with the chosen recipe. The 3-model ensemble size is
   fixed here, in advance: 3 x ~50K parameters stays below one CasSubTS (154K), so
   the ensemble is a parameter-matched comparison. Its size is never tuned on test.
4. Score every run once on the test split: each seed, the seed mean, and the
   3-model ensemble, against both targets, with a paired test against our CasSubTS run.
5. Optional robustness phase: CasSubTS seeds trained with the same recipe as ours.

A failed job is logged and skipped; the comparison uses whatever runs completed and
records how many there were.

    python -m pipeline_w.multiseed --wait-pids 88808 88809
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .compare_models import cassubts_predictions
from .paper_metrics import metrics as paper_metrics
from .predict_weibo import predict_run

WINDOW = 3600
DATA = Path(f"weibo_data/processed_{WINDOW}")
GRAPHS = Path(f"weibo_data/cassubts_{WINDOW}")
TUNING = Path(f"pipeline_w/tuning_{WINDOW}")
OUT = Path(f"pipeline_w/multiseed_{WINDOW}")

# Optimiser schedule, EMA, epoch budget and early stopping are always matched between the
# two models. Model-specific regularisation (dropout, weight decay) stays at each model's
# own setting; E's stronger regularisation applies to ours only and is disclosed as such.
_COSINE60 = ["--scheduler", "cosine", "--ema-decay", "0.999", "--epochs", "60", "--early-stop", "15"]
_COSINE20 = ["--scheduler", "cosine", "--ema-decay", "0.999", "--epochs", "20", "--early-stop", "8"]
RECIPES = {
    "A": {"ours": ["--epochs", "60", "--early-stop", "8"], "cassubts": ["--epochs", "60", "--early-stop", "8"]},
    "B": {"ours": _COSINE60, "cassubts": _COSINE60},
    "C": {"ours": _COSINE60 + ["--side-features"], "cassubts": _COSINE60},
    "D": {"ours": _COSINE20 + ["--side-features"], "cassubts": _COSINE20},
    "E": {"ours": _COSINE20 + ["--side-features", "--dropout", "0.5", "--weight-decay", "1e-3"],
          "cassubts": _COSINE20},
}
SEED42_DIR = {"A": Path(f"pipeline_w/outputs_{WINDOW}"),
              **{c: TUNING / f"cfg{c}" for c in "BCDE"}}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_for_tuning(pids: list[int]) -> None:
    while any(alive(p) for p in pids):
        time.sleep(60)
    log("tuning processes have exited")


def val_score(cfg: str) -> float | None:
    if cfg == "A":
        pred, target, _ = predict_run(SEED42_DIR["A"], DATA, split="val")
        return float(np.mean((pred - target) ** 2))
    report = SEED42_DIR[cfg] / "weibo_evaluation_report.json"
    if not report.exists():
        log(f"FAILED tuning config {cfg}: no report at {report}")
        return None
    m = json.loads(report.read_text())["results"]["attention_gru"]["metrics"]
    assert m["evaluated_on"] == "val", f"config {cfg} was not scored on validation"
    return float(m["MSLE_log2"])


def run_jobs(jobs: list[tuple[str, list[str], Path]], slots: int) -> dict[str, bool]:
    """Run (name, argv, logfile) jobs with at most ``slots`` concurrently."""
    pending, running, ok = list(jobs), {}, {}
    while pending or running:
        while pending and len(running) < slots:
            name, argv, logfile = pending.pop(0)
            logfile.parent.mkdir(parents=True, exist_ok=True)
            fh = open(logfile, "w")
            running[name] = (subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT), fh, time.time())
            log(f"started {name}  -> {logfile}")
        time.sleep(30)
        for name in list(running):
            proc, fh, t0 = running[name]
            if proc.poll() is not None:
                fh.close()
                ok[name] = proc.returncode == 0
                status = "finished" if ok[name] else f"FAILED (exit {proc.returncode})"
                log(f"{status} {name} after {(time.time() - t0) / 3600:.2f} h")
                del running[name]
    return ok


def summarise(rows: dict[str, dict]) -> dict:
    keys = ("MSLE_log2", "R2_log2", "MAPE_log2", "MAE_count")
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows.values()])
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "per_seed": [float(x) for x in v]}
    out["n_seeds"] = len(rows)
    return out


def scored(pred: np.ndarray, target: np.ndarray) -> dict:
    m = paper_metrics(pred, target)
    m["MAE_count"] = float(np.mean(np.abs(np.clip(2.0 ** pred - 1, 0, None) - (2.0 ** target - 1))))
    return m


def paired(err_a: np.ndarray, err_b: np.ndarray) -> dict:
    """Per-cascade paired comparison; negative difference favours ``a``."""
    from scipy import stats
    diff = err_a - err_b
    rng = np.random.default_rng(0)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    t_p = float(stats.ttest_rel(err_a, err_b).pvalue)
    return {"msle_a": float(err_a.mean()), "msle_b": float(err_b.mean()),
            "difference": float(diff.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
            "paired_t_p": t_p, "wilcoxon_p": float(stats.wilcoxon(err_a, err_b).pvalue),
            "a_better_fraction": float((diff < 0).mean()),
            "significant_at_05": bool(t_p < 0.05 and lo * hi > 0)}


PAPER = {"MSLE_log2": 1.987, "R2_log2": 0.561, "MAPE_log2": 0.299}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wait-pids", type=int, nargs="*", default=[])
    p.add_argument("--slots", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--configs", nargs="+", default=["A", "B", "C", "D", "E"])
    p.add_argument("--skip-cassubts-seeds", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Exercise selection and scoring on existing checkpoints only; no training.")
    a = p.parse_args()
    report_name = "dryrun_report.json" if a.dry_run else "multiseed_report.json"
    if a.dry_run:
        a.configs, a.seeds, a.skip_cassubts_seeds, a.wait_pids = ["A"], [42], True, []
    OUT.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    # 1-2. Wait for every tuning run, then choose the recipe on validation only.
    wait_for_tuning(a.wait_pids)
    scores = {c: val_score(c) for c in a.configs}
    scores = {c: s for c, s in scores.items() if s is not None}
    chosen = min(scores, key=scores.get)
    log("validation MSLE: " + ", ".join(f"{c}={s:.4f}" for c, s in scores.items()) + f"  -> chosen {chosen}")
    (OUT / ("dryrun_selection.json" if a.dry_run else "selection.json")).write_text(json.dumps(
        {"val_MSLE_log2": scores, "chosen": chosen, "recipe": RECIPES[chosen],
         "rule": "lowest validation MSLE; test split not consulted"}, indent=2))

    # 3. Two more seeds of ours.
    ours_dirs = {42: SEED42_DIR[chosen]}
    jobs = []
    for s in a.seeds:
        if s == 42:
            continue
        ours_dirs[s] = OUT / f"ours_s{s}"
        jobs.append((f"ours_s{s}", [py, "-m", "pipeline_w.train_weibo", "--data-dir", str(DATA),
                     "--output-dir", str(ours_dirs[s]), "--models", "attention_gru",
                     "--observation-window-note", str(WINDOW), "--threads", str(a.threads),
                     "--seed", str(s), *RECIPES[chosen]["ours"]], OUT / "logs" / f"ours_s{s}.log"))
    ok = run_jobs(jobs, a.slots)
    ours_dirs = {s: d for s, d in ours_dirs.items() if s == 42 or ok.get(f"ours_s{s}")}
    if a.dry_run:
        ours_dirs[43] = ours_dirs[42]          # exercises the ensemble path

    # Ensemble behaviour on validation first, so it is on record before test is read.
    val_p = {s: predict_run(d, DATA, split="val") for s, d in ours_dirs.items()}
    vt = next(iter(val_p.values()))[1]
    val_single = {f"ours_s{s}": float(np.mean((v[0] - vt) ** 2)) for s, v in val_p.items()}
    val_ens = float(np.mean((np.mean([v[0] for v in val_p.values()], axis=0) - vt) ** 2))
    log("validation: " + ", ".join(f"{k}={v:.4f}" for k, v in val_single.items())
        + f", ensemble({len(val_p)})={val_ens:.4f}")

    # 4. Test split, scored once.
    log("scoring on the test split")
    cas_pred, t, ids = cassubts_predictions(WINDOW, GRAPHS, Path(f"pipeline_w/outputs_cassubts_{WINDOW}"), 6)
    test_p = {}
    for s, d in ours_dirs.items():
        pr, tt, ii = predict_run(d, DATA, split="test")
        if ii != ids or not np.allclose(tt, t):
            raise SystemExit(f"ours_s{s}: test cascades or targets do not match CasSubTS — refusing to compare")
        test_p[f"ours_s{s}"] = pr
    ens = np.mean(list(test_p.values()), axis=0)

    per_run = {k: scored(v, t) for k, v in test_p.items()}
    per_run["ours_ensemble"] = scored(ens, t)
    per_run["cassubts_our_run"] = scored(cas_pred, t)
    for k, m in per_run.items():
        log(f"  {k:<18} MSLE={m['MSLE_log2']:.4f}  R2={m['R2_log2']:.4f}  MAPE={m['MAPE_log2']:.4f}  MAE={m['MAE_count']:.2f}")

    cas_err = (cas_pred - t) ** 2
    result = {
        "window_seconds": WINDOW, "n_test_cascades": int(len(t)),
        "chosen_recipe": chosen, "val_selection": scores,
        "validation": {"single": val_single, "ensemble": val_ens},
        "test": per_run,
        "ours_single_seed_summary": summarise({k: per_run[k] for k in test_p}),
        "targets": {"cassubts_paper": PAPER, "cassubts_our_run": per_run["cassubts_our_run"]},
        "ensemble_vs_cassubts_our_run": paired((ens - t) ** 2, cas_err),
        "single_seed_mean_error_vs_cassubts_our_run": paired(
            np.mean([(v - t) ** 2 for v in test_p.values()], axis=0), cas_err),
        "params": {"ours_single": int(json.loads((ours_dirs[42] / "weibo_evaluation_report.json").read_text())
                                      ["results"]["attention_gru"]["metrics"].get("n_params", 49824)),
                   "cassubts": 153970},
    }
    result["params"]["ours_ensemble"] = result["params"]["ours_single"] * len(test_p)
    pe = result["ensemble_vs_cassubts_our_run"]
    log(f"ensemble vs our CasSubTS run: {pe['msle_a']:.4f} vs {pe['msle_b']:.4f}  diff {pe['difference']:+.4f} "
        f"CI [{pe['ci95_low']:+.4f}, {pe['ci95_high']:+.4f}]  t p={pe['paired_t_p']:.4f}  Wilcoxon p={pe['wilcoxon_p']:.4g}")
    (OUT / report_name).write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"PRIMARY DONE -> {OUT / report_name}")

    # 5. Optional robustness check: CasSubTS seeds under the same recipe as ours.
    if a.skip_cassubts_seeds:
        return
    cas_jobs, cas_dirs = [], {}
    for s in a.seeds:
        cas_dirs[s] = OUT / f"cassubts_s{s}"
        cas_jobs.append((f"cassubts_s{s}", [py, "-m", "pipeline_w.train_cassubts", "--data-dir", str(GRAPHS),
                         "--output-dir", str(cas_dirs[s]), "--observation-time", str(WINDOW),
                         "--threads", str(a.threads), "--seed", str(s), *RECIPES[chosen]["cassubts"]],
                         OUT / "logs" / f"cassubts_s{s}.log"))
    ok = run_jobs(cas_jobs, a.slots)
    cas_runs = {}
    for s, d in cas_dirs.items():
        if ok.get(f"cassubts_s{s}"):
            pr, tt, ii = cassubts_predictions(WINDOW, GRAPHS, d, 6)
            assert ii == ids and np.allclose(tt, t)
            cas_runs[f"cassubts_s{s}"] = pr
            log(f"  cassubts_s{s:<8} MSLE={np.mean((pr - t) ** 2):.4f}")
    if cas_runs:
        result["robustness_cassubts_same_recipe"] = {
            "summary": summarise({k: scored(v, t) for k, v in cas_runs.items()}),
            "ensemble": scored(np.mean(list(cas_runs.values()), axis=0), t),
            "ours_mean_error_vs_cassubts_mean_error": paired(
                np.mean([(v - t) ** 2 for v in test_p.values()], axis=0),
                np.mean([(v - t) ** 2 for v in cas_runs.values()], axis=0)),
        }
        (OUT / report_name).write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"ALL DONE -> {OUT / report_name}")


if __name__ == "__main__":
    main()
