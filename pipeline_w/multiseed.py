"""Unattended multi-seed comparison of our model against CasSubTS (Weibo, 1 h).

1. Wait for the tuning runs (configs B and C, seed 42, ``--skip-test``) to finish, and
   run the extra tuning configs (D, E) the same way. B and C overfit after ~11 epochs
   under a 60-epoch cosine schedule, so D anneals over 20 epochs and E adds stronger
   regularisation on top.
2. Pick the training recipe by **validation** MSLE among A (the original run) and B-E.
   The test split plays no part in this choice.
3. Train the extra seeds: ours with the chosen recipe, CasSubTS with the *same* recipe
   (scheduler, EMA, epoch budget, early stopping), so neither model gets better training.
4. Score every run on the test split once and compare seed-averaged results.

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wait-pids", type=int, nargs="*", default=[])
    p.add_argument("--slots", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--tune", nargs="*", default=["D", "E"], help="Extra tuning configs to run first.")
    p.add_argument("--tune-threads", type=int, default=3)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    # 1. Extra tuning configs (seed 42, validation only), alongside the B/C runs already going.
    run_jobs([(f"tune_cfg{c}", [py, "-m", "pipeline_w.train_weibo", "--data-dir", str(DATA),
               "--output-dir", str(SEED42_DIR[c]), "--models", "attention_gru",
               "--observation-window-note", str(WINDOW), "--threads", str(a.tune_threads),
               "--seed", "42", "--skip-test", *RECIPES[c]["ours"]], TUNING / "logs" / f"cfg{c}.log")
              for c in a.tune], a.slots)
    wait_for_tuning(a.wait_pids)

    # 2. Choose the recipe on validation only.
    scores = {c: val_score(c) for c in ["A", "B", "C", *a.tune]}
    scores = {c: s for c, s in scores.items() if s is not None}
    chosen = min(scores, key=scores.get)
    log("validation MSLE: " + ", ".join(f"{c}={s:.4f}" for c, s in scores.items()) + f"  -> chosen {chosen}")
    (OUT / "selection.json").write_text(json.dumps(
        {"val_MSLE_log2": scores, "chosen": chosen, "recipe": RECIPES[chosen],
         "rule": "lowest validation MSLE; test split not consulted"}, indent=2))

    # 3. Extra seeds, interleaved so a partial comparison exists if the night runs short.
    ours_dirs = {42: SEED42_DIR[chosen]}
    cas_dirs = {}
    jobs = []
    for s in a.seeds:
        d = OUT / f"cassubts_s{s}"
        cas_dirs[s] = d
        jobs.append((f"cassubts_s{s}", [py, "-m", "pipeline_w.train_cassubts", "--data-dir", str(GRAPHS),
                     "--output-dir", str(d), "--observation-time", str(WINDOW), "--threads", str(a.threads),
                     "--seed", str(s), *RECIPES[chosen]["cassubts"]], OUT / "logs" / f"cassubts_s{s}.log"))
    for s in a.seeds:
        if s in ours_dirs:
            continue
        d = OUT / f"ours_s{s}"
        ours_dirs[s] = d
        jobs.append((f"ours_s{s}", [py, "-m", "pipeline_w.train_weibo", "--data-dir", str(DATA),
                     "--output-dir", str(d), "--models", "attention_gru",
                     "--observation-window-note", str(WINDOW), "--threads", str(a.threads),
                     "--seed", str(s), *RECIPES[chosen]["ours"]], OUT / "logs" / f"ours_s{s}.log"))
    order = ["cassubts_s42", "ours_s43", "cassubts_s43", "ours_s44", "cassubts_s44"]
    jobs.sort(key=lambda j: order.index(j[0]) if j[0] in order else len(order))
    ok = run_jobs(jobs, a.slots)

    # 4. Score every completed run on the test split, once.
    log("scoring all runs on the test split")
    preds, ref_ids, ref_t = {}, None, None

    def add(name, pred, target, ids):
        nonlocal ref_ids, ref_t
        if ref_ids is None:
            ref_ids, ref_t = ids, target
        if ids != ref_ids or not np.allclose(target, ref_t):
            raise SystemExit(f"{name}: test cascades or targets do not match — refusing to compare")
        preds[name] = pred

    for s, d in ours_dirs.items():
        if s == 42 or ok.get(f"ours_s{s}"):
            add(f"ours_s{s}", *predict_run(d, DATA, split="test"))
    for s, d in cas_dirs.items():
        if ok.get(f"cassubts_s{s}"):
            add(f"cassubts_s{s}", *cassubts_predictions(WINDOW, GRAPHS, d, 6))
    add("cassubts_original", *cassubts_predictions(WINDOW, GRAPHS, Path(f"pipeline_w/outputs_cassubts_{WINDOW}"), 6))

    t = ref_t
    per_run = {}
    for name, pr in preds.items():
        m = paper_metrics(pr, t)
        m["MAE_count"] = float(np.mean(np.abs(np.clip(2.0 ** pr - 1, 0, None) - (2.0 ** t - 1))))
        per_run[name] = m
        log(f"  {name:<20} MSLE={m['MSLE_log2']:.4f}  R2={m['R2_log2']:.4f}  MAPE={m['MAPE_log2']:.4f}")

    ours = {k: v for k, v in per_run.items() if k.startswith("ours_")}
    cas = {k: v for k, v in per_run.items() if k.startswith("cassubts_s")}
    result = {"window_seconds": WINDOW, "chosen_recipe": chosen, "val_selection": scores,
              "n_test_cascades": int(len(t)), "per_run": per_run,
              "ours": summarise(ours), "cassubts_same_recipe": summarise(cas) if cas else None}

    if cas:
        from scipy import stats
        err_o = np.mean([(preds[k] - t) ** 2 for k in ours], axis=0)
        err_c = np.mean([(preds[k] - t) ** 2 for k in cas], axis=0)
        diff = err_o - err_c                        # positive favours CasSubTS
        rng = np.random.default_rng(0)
        boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        seed_o = [v["MSLE_log2"] for v in ours.values()]
        seed_c = [v["MSLE_log2"] for v in cas.values()]
        result["paired_per_cascade_seed_averaged"] = {
            "msle_ours": float(err_o.mean()), "msle_cassubts": float(err_c.mean()),
            "difference": float(diff.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
            "paired_t_p": float(stats.ttest_rel(err_o, err_c).pvalue),
            "wilcoxon_p": float(stats.wilcoxon(err_o, err_c).pvalue),
            "ours_better_fraction": float((diff < 0).mean()),
            "significant_at_05": bool(stats.ttest_rel(err_o, err_c).pvalue < 0.05 and lo * hi > 0),
        }
        if len(seed_o) > 1 and len(seed_c) > 1:
            result["welch_t_over_seeds_p"] = float(stats.ttest_ind(seed_o, seed_c, equal_var=False).pvalue)
        pc = result["paired_per_cascade_seed_averaged"]
        log(f"ours {result['ours']['MSLE_log2']['mean']:.4f} ± {result['ours']['MSLE_log2']['std']:.4f}  vs  "
            f"CasSubTS {result['cassubts_same_recipe']['MSLE_log2']['mean']:.4f} ± "
            f"{result['cassubts_same_recipe']['MSLE_log2']['std']:.4f}  | diff {pc['difference']:+.4f} "
            f"CI [{lo:+.4f}, {hi:+.4f}]  t p={pc['paired_t_p']:.4f}  Wilcoxon p={pc['wilcoxon_p']:.4g}")

    (OUT / "multiseed_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"ALL DONE -> {OUT / 'multiseed_report.json'}")


if __name__ == "__main__":
    main()
