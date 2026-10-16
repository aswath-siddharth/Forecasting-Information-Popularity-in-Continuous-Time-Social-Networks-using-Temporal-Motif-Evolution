"""Predictions from a finished ``train_weibo`` run, rebuilt from its own report.

Lets a checkpoint chosen in tuning mode (``--skip-test``) be scored on the test split
exactly once, after model selection, without retraining it. The architecture and
side-feature settings are read from the run's ``weibo_evaluation_report.json``; runs
from before those settings were recorded fall back to the original defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SIDE_DIM, SequenceScaler, WeiboCascadeDataset, pad_collate_fn
from .model import build_model
from .weibo_parser import FEATURE_DIM


def run_config(run_dir: Path) -> dict:
    setting = json.loads((run_dir / "weibo_evaluation_report.json").read_text())["setting"]
    training = setting.get("training", {})
    return {
        "window": int(setting["observation_window_seconds"]),
        "hidden_size": int(training.get("hidden_size", 64)),
        "num_layers": int(training.get("num_layers", 1)),
        "side_features": bool(training.get("side_features", False)),
    }


@torch.no_grad()
def predict_run(run_dir: str | Path, data_dir: str | Path, split: str = "test"):
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    cfg = run_config(run_dir)
    side_window = cfg["window"] if cfg["side_features"] else None

    train = WeiboCascadeDataset(data_dir / "weibo_train.npz", side_window=side_window)
    scaler = SequenceScaler().fit(train.X_flat)
    ds = WeiboCascadeDataset(data_dir / f"weibo_{split}.npz", scaler=scaler, side_window=side_window)
    if side_window is not None:
        ds.set_side_scaling(train.side.mean(axis=0), np.maximum(train.side.std(axis=0), 1e-6))

    model = build_model("attention_gru", FEATURE_DIM, cfg["hidden_size"], cfg["num_layers"],
                        64, 0.3, SIDE_DIM if cfg["side_features"] else 0)
    model.load_state_dict(torch.load(run_dir / "best_attention_gru.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()

    preds, targets = [], []
    for x, mask, y_log, _, *side in DataLoader(ds, batch_size=128, collate_fn=pad_collate_fn):
        preds.append(model(x, mask, *side).numpy().ravel())
        targets.append(y_log.numpy().ravel())
    return np.concatenate(preds), np.concatenate(targets), [str(c) for c in ds.cascade_id.tolist()]
