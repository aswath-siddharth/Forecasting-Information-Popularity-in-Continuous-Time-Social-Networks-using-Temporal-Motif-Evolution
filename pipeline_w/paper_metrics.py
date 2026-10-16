"""MSLE, R² and MAPE for both models, matching Table 4 of the CasSubTS paper.

The paper's Eq. 31 defines MAPE without saying which space it is computed in. Its
reported values (~0.3) are implausible on raw retweet counts, so MAPE is reported
both in log2 space and on raw counts. Cascades with a zero target are excluded from
MAPE, since the ratio is undefined there.

    python -m pipeline_w.paper_metrics --window 3600 \
        --output pipeline_w/outputs_cassubts_3600/table4_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .compare_models import cassubts_predictions, our_predictions


def metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    msle = float(np.mean((pred - target) ** 2))
    r2 = float(1 - np.sum((target - pred) ** 2) / np.sum((target - target.mean()) ** 2))

    nz = target > 0                                  # log2(y+1) = 0  <=>  zero increment
    mape_log = float(np.mean(np.abs((target[nz] - pred[nz]) / target[nz])))

    y, y_hat = 2.0 ** target - 1, 2.0 ** np.clip(pred, 0, None) - 1
    pos = y > 0
    mape_count = float(np.mean(np.abs((y[pos] - y_hat[pos]) / y[pos])))

    return {"MSLE_log2": msle, "R2_log2": r2, "MAPE_log2": mape_log,
            "MAPE_count": mape_count, "n": int(len(target)), "n_nonzero": int(nz.sum())}


def main() -> None:
    p = argparse.ArgumentParser(description="Table 4 metrics for both models.")
    p.add_argument("--window", type=int, default=3600)
    p.add_argument("--n-subgraphs", type=int, default=6)
    p.add_argument("--output", default=None, help="Optional path to write the result as JSON.")
    a = p.parse_args()

    p_ours, t_ours, id_ours = our_predictions(a.window, Path(f"pipeline_w/outputs_{a.window}"))
    p_cas, t_cas, id_cas = cassubts_predictions(
        a.window, Path(f"weibo_data/cassubts_{a.window}"),
        Path(f"pipeline_w/outputs_cassubts_{a.window}"), a.n_subgraphs,
    )
    if id_ours != id_cas or not np.allclose(t_ours, t_cas):
        raise SystemExit("Test sets are not aligned — refusing to compare.")

    result = {"window_seconds": a.window,
              "ours": metrics(p_ours, t_ours), "cassubts": metrics(p_cas, t_cas)}
    print(json.dumps(result, indent=2))

    if a.output:
        Path(a.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n• written to {a.output}")


if __name__ == "__main__":
    main()
