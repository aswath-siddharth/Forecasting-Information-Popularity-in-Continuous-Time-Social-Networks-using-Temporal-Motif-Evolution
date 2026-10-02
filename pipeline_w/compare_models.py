"""Paired significance test between our model and the CasSubTS reimplementation.

A difference in mean MSLE means little on its own: squared log-error on this data is
dominated by a handful of enormous tail errors, so two models can differ by a few
percent on the mean while being indistinguishable cascade by cascade. This script
pairs the two models' per-cascade errors on the *same* test rows and reports a
bootstrap CI, a paired t-test and a Wilcoxon signed-rank test.

It asserts that both models were evaluated on identical cascade ids with identical
targets before comparing anything; if the two pipelines ever drift out of alignment
the run fails loudly rather than reporting a meaningless difference.

    python -m pipeline_w.compare_models --window 3600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def our_predictions(window: int, out_dir: Path):
    from .data import SequenceScaler, WeiboCascadeDataset, pad_collate_fn
    from .model import build_model
    from .weibo_parser import FEATURE_DIM

    train = WeiboCascadeDataset(f"weibo_data/processed_{window}/weibo_train.npz")
    scaler = SequenceScaler().fit(train.X_flat)
    test = WeiboCascadeDataset(f"weibo_data/processed_{window}/weibo_test.npz", scaler=scaler)

    model = build_model("attention_gru", FEATURE_DIM, 64, 1, 64, 0.3)
    model.load_state_dict(torch.load(out_dir / "best_attention_gru.pt", map_location="cpu", weights_only=True))
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for x, mask, y_log, _ in DataLoader(test, batch_size=128, collate_fn=pad_collate_fn):
            preds.append(model(x, mask).numpy().ravel())
            targets.append(y_log.numpy().ravel())
    ids = [str(c) for c in test.cascade_id.tolist()]
    return np.concatenate(preds), np.concatenate(targets), ids


def cassubts_predictions(window: int, data_dir: Path, out_dir: Path, n_subgraphs: int):
    from .cassubts_model import CasSubTS
    from .train_cassubts import CascadeGraphDataset, make_collate, to_device

    ds = CascadeGraphDataset(data_dir / "cassubts_test.pkl")
    model = CasSubTS(n_subgraphs=n_subgraphs)
    model.load_state_dict(torch.load(out_dir / "best_cassubts.pt", map_location="cpu", weights_only=True))
    model.eval()

    preds, targets = [], []
    collate = make_collate(n_subgraphs, float(window))
    with torch.no_grad():
        for batch, y_log, _ in DataLoader(ds, batch_size=128, collate_fn=collate):
            preds.append(model(to_device(batch, torch.device("cpu"))).numpy().ravel())
            targets.append(y_log.numpy().ravel())
    return np.concatenate(preds), np.concatenate(targets), list(ds.cascade_id)


def main() -> None:
    p = argparse.ArgumentParser(description="Paired comparison of the two models.")
    p.add_argument("--window", type=int, default=3600)
    p.add_argument("--n-subgraphs", type=int, default=6)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None, help="Optional path to write the result as JSON.")
    a = p.parse_args()

    p_ours, t_ours, id_ours = our_predictions(a.window, Path(f"pipeline_w/outputs_{a.window}"))
    p_cas, t_cas, id_cas = cassubts_predictions(
        a.window, Path(f"weibo_data/cassubts_{a.window}"),
        Path(f"pipeline_w/outputs_cassubts_{a.window}"), a.n_subgraphs,
    )

    if id_ours != id_cas:
        raise SystemExit("Test sets are not aligned — the two pipelines disagree on cascade ids.")
    if not np.allclose(t_ours, t_cas):
        raise SystemExit("Test sets are aligned but targets differ — refusing to compare.")
    print(f"✓ aligned on {len(p_ours):,} identical test cascades with identical targets\n")

    err_ours = (p_ours - t_ours) ** 2
    err_cas = (p_cas - t_cas) ** 2
    diff = err_ours - err_cas          # positive favours CasSubTS

    from scipy import stats
    t_stat, t_p = stats.ttest_rel(err_ours, err_cas)
    _, w_p = stats.wilcoxon(err_ours, err_cas)

    rng = np.random.default_rng(a.seed)
    n = len(diff)
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(a.bootstrap)])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    result = {
        "window_seconds": a.window,
        "n_test_cascades": int(n),
        "msle_ours": float(err_ours.mean()),
        "msle_cassubts": float(err_cas.mean()),
        "difference": float(diff.mean()),
        "ci95_low": float(lo), "ci95_high": float(hi),
        "paired_t_p": float(t_p),
        "wilcoxon_p": float(w_p),
        "cassubts_better_fraction": float((diff > 0).mean()),
        "significant_at_05": bool(t_p < 0.05 and lo * hi > 0),
    }

    print(f"  ours      MSLE = {result['msle_ours']:.4f}")
    print(f"  CasSubTS  MSLE = {result['msle_cassubts']:.4f}")
    print(f"  difference     = {result['difference']:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"\n  paired t-test  p = {t_p:.4f}")
    print(f"  Wilcoxon       p = {w_p:.4g}")
    print(f"  CasSubTS better on {result['cassubts_better_fraction']:.1%} of cascades")
    print("\n  => " + ("SIGNIFICANT at p<0.05" if result["significant_at_05"]
                       else "NOT significant at p<0.05 — the gap is within noise"))

    if a.output:
        Path(a.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n• written to {a.output}")


if __name__ == "__main__":
    main()
