"""Dataset and walk sampling utilities for DeepCas on Reddit cascades."""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def sample_walks_from_paths(
    cascade_path_offsets: np.ndarray,
    path_node_offsets: np.ndarray,
    node_ids: np.ndarray,
    n_walks: int = 32,
    walk_length: int = 10,
    seed: int = 42,
    max_cascades: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample N_w random walks of length L from sequence of diffusion paths."""
    rng = random.Random(seed)
    n_cascades = len(cascade_path_offsets) - 1
    if max_cascades > 0:
        n_cascades = min(n_cascades, max_cascades)

    walks_arr = np.zeros((n_cascades, n_walks, walk_length), dtype=np.int64)
    obs_sizes = np.zeros(n_cascades, dtype=np.float32)

    for i in range(n_cascades):
        p_start, p_end = cascade_path_offsets[i], cascade_path_offsets[i + 1]
        obs_sizes[i] = float(max(1, p_end - p_start))
        adj = defaultdict(list)
        cascade_nodes = set()
        node_out_deg = defaultdict(int)

        for p_idx in range(p_start, p_end):
            n_start, n_end = path_node_offsets[p_idx], path_node_offsets[p_idx + 1]
            p = node_ids[n_start:n_end]
            if len(p) >= 2:
                for j in range(len(p) - 1):
                    u, v = int(p[j]), int(p[j + 1])
                    adj[u].append(v)
                    adj[v].append(u)
                    node_out_deg[u] += 1
                    cascade_nodes.add(u)
                    cascade_nodes.add(v)
            elif len(p) == 1:
                cascade_nodes.add(int(p[0]))

        if not cascade_nodes:
            continue

        nlist = list(cascade_nodes)
        weights = [node_out_deg[n] + 1 for n in nlist]
        tot = sum(weights)
        probs = [w / tot for w in weights]

        for w_idx in range(n_walks):
            curr = rng.choices(nlist, weights=probs, k=1)[0]
            walk = [curr]
            for _ in range(walk_length - 1):
                nbrs = adj.get(curr)
                if nbrs:
                    curr = rng.choice(nbrs)
                walk.append(curr)
            walks_arr[i, w_idx] = walk

    return walks_arr, obs_sizes


class RedditDeepCasDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        split: str = "train",
        observation_time: int = 3600,
        n_walks: int = 32,
        walk_length: int = 10,
        seed: int = 42,
        max_cascades: int = 0,
    ) -> None:
        cache_dir = Path(cache_dir)
        deepcas_cache = cache_dir / f"deepcas_{observation_time}"
        deepcas_cache.mkdir(parents=True, exist_ok=True)
        deepcas_file = deepcas_cache / f"deepcas_{split}.npz"

        # Check if pre-sampled walks exist
        if deepcas_file.exists() and max_cascades == 0:
            data = np.load(deepcas_file, allow_pickle=True)
            self.walks = data["walks"].astype(np.int64)
            self.observed_size = data["observed_size"].astype(np.float32)
            self.y_raw = data["y_increment"].astype(np.float32)
            self.cascade_id = data["cascade_id"]
        else:
            # Build from aligned deephawkes split
            dh_file = cache_dir / f"deephawkes_{observation_time}" / f"deephawkes_{split}.npz"
            if not dh_file.exists():
                raise FileNotFoundError(f"DeepHawkes aligned split not found at {dh_file}")

            t0 = time.time()
            dh_data = np.load(dh_file, allow_pickle=True)
            y_all = dh_data["y_increment"].astype(np.float32)
            cids_all = dh_data["cascade_id"]

            walks_arr, obs_sizes = sample_walks_from_paths(
                dh_data["cascade_path_offsets"],
                dh_data["path_node_offsets"],
                dh_data["node_ids"],
                n_walks=n_walks,
                walk_length=walk_length,
                seed=seed,
                max_cascades=max_cascades,
            )

            n_samples = len(walks_arr)
            self.walks = walks_arr
            self.observed_size = obs_sizes
            self.y_raw = y_all[:n_samples]
            self.cascade_id = cids_all[:n_samples]

            if max_cascades == 0:
                np.savez_compressed(
                    deepcas_file,
                    walks=self.walks,
                    observed_size=self.observed_size,
                    y_increment=self.y_raw,
                    cascade_id=self.cascade_id,
                )
                print(f"• Precomputed and saved DeepCas walks for {split} split ({n_samples:,} cascades) in {time.time()-t0:.1f}s -> {deepcas_file}")

        self.y_log2 = np.log2(np.clip(self.y_raw, 0.0, None) + 1.0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.walks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        walks_t = torch.from_numpy(self.walks[idx])
        obs_sz_t = torch.tensor(self.observed_size[idx], dtype=torch.float32)
        y_log_t = torch.tensor(self.y_log2[idx], dtype=torch.float32)
        return walks_t, obs_sz_t, y_log_t, float(self.y_raw[idx])


def collate_deepcas(batch):
    walks = torch.stack([item[0] for item in batch], dim=0)
    obs_sz = torch.stack([item[1] for item in batch], dim=0)
    y_log = torch.stack([item[2] for item in batch], dim=0)
    y_raw = np.array([item[3] for item in batch], dtype=np.float64)
    return walks, obs_sz, y_log, y_raw
