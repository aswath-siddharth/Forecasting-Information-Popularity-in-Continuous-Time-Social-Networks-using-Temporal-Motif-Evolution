"""Dataset and cascade subtree extraction for CasSubTS on Reddit."""

from __future__ import annotations

import math
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class RedditCascadeGraphDataset(Dataset):
    def __init__(self, data_dict: dict) -> None:
        self.graphs = data_dict["graphs"]
        self.y_raw = data_dict["y_increment"]
        self.cascade_id = data_dict["cascade_id"]

    def __len__(self) -> int:
        return len(self.graphs)

    def log_targets(self) -> np.ndarray:
        return (np.log(np.maximum(self.y_raw, 0.0) + 1.0) / np.log(2.0)).astype(np.float32)

    def __getitem__(self, i: int):
        y_r = float(self.y_raw[i])
        y_l = np.float32(math.log2(max(y_r, 0.0) + 1.0))
        return self.graphs[i], y_l, y_r


def make_collate(n_subgraphs: int, observation_time: float):
    thresholds = [observation_time * (k + 1) / n_subgraphs for k in range(n_subgraphs)]

    def collate(items):
        graphs, y_log, y_raw = zip(*items)
        x_base, node_batch, offs, off = [], [], [], 0
        for bi, g in enumerate(graphs):
            n = g["n_nodes"]
            x_base.append(np.stack([
                g["node_depth"],
                g["node_first_time"] / max(observation_time, 1.0),
            ], axis=1))
            node_batch.append(np.full(n, bi, dtype=np.int64))
            offs.append(off)
            off += n

        edge_src_k, edge_dst_k = [], []
        for thr in thresholds:
            s_all, d_all = [], []
            for g, o in zip(graphs, offs):
                m = g["edge_time"] <= thr
                if m.any():
                    s_all.append(g["edge_src"][m].astype(np.int64) + o)
                    d_all.append(g["edge_dst"][m].astype(np.int64) + o)
            edge_src_k.append(torch.from_numpy(np.concatenate(s_all)) if s_all else torch.empty(0, dtype=torch.long))
            edge_dst_k.append(torch.from_numpy(np.concatenate(d_all)) if d_all else torch.empty(0, dtype=torch.long))

        return {
            "x_base": torch.from_numpy(np.concatenate(x_base).astype(np.float32)),
            "node_batch": torch.from_numpy(np.concatenate(node_batch)),
            "n_graphs": len(graphs),
            "edge_src": edge_src_k,
            "edge_dst": edge_dst_k,
        }, torch.tensor(y_log, dtype=torch.float32).unsqueeze(1), torch.tensor(np.asarray(y_raw, dtype=np.float32)).unsqueeze(1)

    return collate


def to_device(b: dict, dev: torch.device) -> dict:
    return {
        "x_base": b["x_base"].to(dev),
        "node_batch": b["node_batch"].to(dev),
        "n_graphs": b["n_graphs"],
        "edge_src": [e.to(dev) for e in b["edge_src"]],
        "edge_dst": [e.to(dev) for e in b["edge_dst"]],
    }


def load_reddit_cassubts_splits(
    cache_dir: str | Path,
    observation_time: int = 3600,
    max_cascades: int = 0,
) -> tuple[dict, dict, dict]:
    """Extracts cascade subtrees for train, val, test on Reddit."""
    cache_dir = Path(cache_dir)
    out_dir = cache_dir / f"cassubts_{observation_time}"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_data = {}
    for split in ("train", "val", "test"):
        pkl_path = out_dir / f"cassubts_{split}.pkl"
        if pkl_path.exists() and max_cascades == 0:
            with open(pkl_path, "rb") as fh:
                splits_data[split] = pickle.load(fh)
            continue

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

        graphs = []
        for i in range(n_cascades):
            p_start, p_end = c_offsets[i], c_offsets[i + 1]
            node_id = {}
            first_time = {}
            depth = {}
            src, dst, ets = [], [], []

            for p_idx in range(p_start, p_end):
                n_start, n_end = p_offsets[p_idx], p_offsets[p_idx + 1]
                p = nodes[n_start:n_end]
                t = float(times[p_idx])
                for d_idx, name in enumerate(p):
                    if name not in node_id:
                        nid = len(node_id)
                        node_id[name] = nid
                        first_time[nid] = t
                        depth[nid] = d_idx
                if len(p) >= 2:
                    u = node_id[p[-2]]
                    v = node_id[p[-1]]
                    src.append(u)
                    dst.append(v)
                    ets.append(t)

            n = max(1, len(node_id))
            graphs.append({
                "edge_src": np.array(src, dtype=np.int32),
                "edge_dst": np.array(dst, dtype=np.int32),
                "edge_time": np.array(ets, dtype=np.float32),
                "n_nodes": n,
                "node_first_time": np.array([first_time.get(j, 0.0) for j in range(n)], dtype=np.float32),
                "node_depth": np.array([depth.get(j, 0) for j in range(n)], dtype=np.float32),
            })

        split_dict = {
            "graphs": graphs,
            "y_increment": y_inc[:n_cascades].astype(np.float32),
            "cascade_id": cids[:n_cascades],
        }
        splits_data[split] = split_dict

        if max_cascades == 0:
            with open(pkl_path, "wb") as fh:
                pickle.dump(split_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"• Saved CasSubTS {split} split ({n_cascades:,} cascades) in {time.time()-t0:.1f}s -> {pkl_path}", flush=True)

    return splits_data["train"], splits_data["val"], splits_data["test"]
