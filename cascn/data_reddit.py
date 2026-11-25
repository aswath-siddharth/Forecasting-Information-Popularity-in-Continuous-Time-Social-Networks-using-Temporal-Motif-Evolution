"""Data processing and cascade graph extraction for CasCN on Reddit."""

from __future__ import annotations

import math
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .data import compute_scaled_directed_laplacian, process_single_cascade, MAX_NODES, MAX_STEPS, N_TIME_INTERVAL


def extract_single_cascade_data(
    args: tuple,
) -> dict[str, Any] | None:
    """Helper to process single cascade graph into CasCN format."""
    cid, y_inc, paths_list, times_list, obs_window, time_interval = args

    srcs, dsts, ets = [], [], []
    for p, t in zip(paths_list, times_list):
        if t > obs_window:
            continue
        if len(p) >= 2:
            srcs.append(int(p[-2]))
            dsts.append(int(p[-1]))
            ets.append(float(t))

    g = {
        'edge_src': np.array(srcs, dtype=np.int32),
        'edge_dst': np.array(dsts, dtype=np.int32),
        'edge_time': np.array(ets, dtype=np.float32),
    }

    return process_single_cascade((cid, y_inc, g), obs_window=obs_window, time_interval=time_interval)


def load_reddit_cascn_split(
    cache_dir: str | Path,
    split: str = "train",
    observation_time: int = 3600,
    max_cascades: int = 0,
    workers: int = 4,
) -> list[dict[str, Any]]:
    """Loads Reddit split and extracts CasCN directed Laplacians and node sequences."""
    cache_dir = Path(cache_dir)
    cascn_cache = cache_dir / f"cascn_{observation_time}"
    cascn_cache.mkdir(parents=True, exist_ok=True)
    cache_file = cascn_cache / f"cascn_{split}.pkl"

    if cache_file.exists() and max_cascades == 0:
        with open(cache_file, "rb") as fh:
            return pickle.load(fh)

    dh_file = cache_dir / f"deephawkes_{observation_time}" / f"deephawkes_{split}.npz"
    if not dh_file.exists():
        raise FileNotFoundError(f"DeepHawkes aligned split not found at {dh_file}")

    t0 = time.time()
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

    time_interval = float(observation_time) / float(N_TIME_INTERVAL)
    items: list[dict[str, Any]] = []

    for i in range(n_cascades):
        p_start, p_end = c_offsets[i], c_offsets[i + 1]
        cid = str(cids[i])
        y = float(y_inc[i])

        paths_list = []
        times_list = []
        for p_idx in range(p_start, p_end):
            n_start, n_end = p_offsets[p_idx], p_offsets[p_idx + 1]
            paths_list.append(nodes[n_start:n_end])
            times_list.append(times[p_idx])

        item = extract_single_cascade_data((cid, y, paths_list, times_list, float(observation_time), time_interval))
        if item is not None:
            items.append(item)

    print(f"• Extracted CasCN graphs for {split} ({len(items):,} cascades) in {time.time()-t0:.1f}s", flush=True)

    if max_cascades == 0:
        with open(cache_file, "wb") as fh:
            pickle.dump(items, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"• Saved cached CasCN data -> {cache_file}")

    return items


def fit_train_scaler(train_dataset: list[dict[str, Any]]) -> dict[str, float]:
    """Fit scaler strictly on the train split (zero leakage)."""
    all_times = []
    all_sizes = []
    for it in train_dataset:
        all_times.extend(it['time'].tolist())
        all_sizes.append(float(it['n_nodes']))

    arr_t = np.array(all_times, dtype=np.float64)
    arr_s = np.array(all_sizes, dtype=np.float64)

    return {
        'time_mean': float(arr_t.mean()) if len(arr_t) > 0 else 0.0,
        'time_std': float(arr_t.std()) if len(arr_t) > 0 else 1.0,
        'size_mean': float(arr_s.mean()) if len(arr_s) > 0 else 1.0,
        'size_std': float(arr_s.std()) if len(arr_s) > 0 else 1.0,
    }
