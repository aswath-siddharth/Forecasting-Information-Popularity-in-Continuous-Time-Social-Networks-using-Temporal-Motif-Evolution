"""Data loader and cascade extraction for CasFlow on Reddit."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import Cascade, Cascades, collate, prepare_tensors


def load_reddit_casflow_splits(
    cache_dir: str | Path,
    observation_time: int = 3600,
    max_cascades: int = 0,
) -> dict[str, list[Cascade]]:
    """Loads Reddit cascade splits aligned to DeepHawkes/CasMotif."""
    cache_dir = Path(cache_dir)
    splits: dict[str, list[Cascade]] = {}

    for split in ("train", "val", "test"):
        dh_file = cache_dir / f"deephawkes_{observation_time}" / f"deephawkes_{split}.npz"
        if not dh_file.exists():
            raise FileNotFoundError(f"DeepHawkes aligned split not found at {dh_file}")

        dh_data = np.load(dh_file, allow_pickle=True)
        c_offsets = dh_data["cascade_path_offsets"]
        p_offsets = dh_data["path_node_offsets"]
        nodes = dh_data["node_ids"]
        times = dh_data["path_times"]
        y_inc = dh_data["y_increment"]
        cids = dh_data["cascade_id"]

        n_cascades = len(c_offsets) - 1
        if max_cascades > 0:
            n_cascades = min(n_cascades, max_cascades)

        items: list[Cascade] = []
        for i in range(n_cascades):
            p_start, p_end = c_offsets[i], c_offsets[i + 1]
            cid = str(cids[i])
            lbl = float(y_inc[i])

            events: list[tuple[tuple[str, ...], float]] = []
            for p_idx in range(p_start, p_end):
                n_start, n_end = p_offsets[p_idx], p_offsets[p_idx + 1]
                p_nodes = tuple(str(int(u)) for u in nodes[n_start:n_end])
                t_val = float(times[p_idx])
                if t_val <= float(observation_time):
                    events.append((p_nodes, t_val))

            # Cap at first 100 events per CasFlow standard
            events.sort(key=lambda it: it[1])
            items.append(Cascade(id=cid, events=events[:100], label=lbl))

        splits[split] = items
        print(f"• Loaded CasFlow {split} split: {len(items):,} cascades", flush=True)

    return splits
