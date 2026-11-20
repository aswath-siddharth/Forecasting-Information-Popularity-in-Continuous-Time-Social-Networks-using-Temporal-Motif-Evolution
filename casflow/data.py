"""CasFlow data extraction and preprocessing aligned 1-to-1 to Weibo splits."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .graph import global_embeddings, local_embeddings


@dataclass
class Cascade:
    id: str
    events: list[tuple[tuple[str, ...], float]]
    label: float


def load_split_cascades(
    weibo_txt: Path | str,
    splits_dir: Path | str,
    observation_time: int = 3600,
) -> dict[str, list[Cascade]]:
    """Loads cascades aligned 1-to-1 to weibo_{train,val,test}.npz."""
    weibo_txt = Path(weibo_txt)
    splits_dir = Path(splits_dir)
    split_ids: dict[str, list[str]] = {}
    split_labels: dict[str, np.ndarray] = {}
    for name in ("train", "val", "test"):
        npz = np.load(splits_dir / f"weibo_{name}.npz", allow_pickle=True)
        split_ids[name] = [str(cid) for cid in npz["cascade_id"]]
        split_labels[name] = npz["y_increment"].astype(np.float32)

    wanted = {cid: (split, idx) for split, ids in split_ids.items() for idx, cid in enumerate(ids)}
    raw_events: dict[str, list[tuple[tuple[str, ...], float]]] = {}

    print(f"Reading diffusion chains from {weibo_txt} for {len(wanted):,} cascades...", flush=True)
    t0 = time.time()
    with open(weibo_txt, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            cid = parts[0]
            if cid not in wanted:
                continue

            events: list[tuple[tuple[str, ...], float]] = []
            for entry in parts[4].split(" "):
                if not entry:
                    continue
                chain, _, t_str = entry.rpartition(":")
                if not chain:
                    continue
                nodes = chain.split("/")
                if any(n == "-1" for n in nodes):
                    continue
                try:
                    t_now = float(t_str)
                except ValueError:
                    continue
                if t_now < observation_time:
                    events.append((tuple(nodes), t_now))

            events.sort(key=lambda it: it[1])
            # Cap at first 100 events per CasFlow paper standard
            raw_events[cid] = events[:100]

    print(f"Read {len(raw_events):,} cascades in {time.time() - t0:.1f}s", flush=True)

    splits: dict[str, list[Cascade]] = {}
    for name in ("train", "val", "test"):
        items = []
        for cid, lbl in zip(split_ids[name], split_labels[name]):
            ev = raw_events.get(cid, [])
            items.append(Cascade(id=cid, events=ev, label=float(lbl)))
        splits[name] = items
        print(f"Split {name}: {len(items):,} cascades", flush=True)

    return splits


class Cascades(Dataset):
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.features = tensors["features"]
        self.labels = tensors["labels"]
        self.lengths = tensors["lengths"]
        self.offsets = torch.cat((torch.zeros(1, dtype=torch.long), self.lengths.cumsum(0)))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.features[self.offsets[index] : self.offsets[index + 1]],
            self.labels[index],
        )


def collate(batch):
    features, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in features], dtype=torch.long)
    return pad_sequence(features, batch_first=True), lengths, torch.stack(labels)


def prepare_tensors(
    splits: dict[str, list[Cascade]],
    observation_time: int = 3600,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
) -> dict:
    """Computes global SVD embeddings on Train only and local GraphWave signatures on all."""
    print("Fitting global SVD embeddings on Train split ONLY (zero leakage)...", flush=True)
    t0 = time.time()
    index, vectors, graph_info = global_embeddings(splits["train"], dimension=40, seed=0)
    print(f"Global graph: {graph_info['nodes']:,} nodes, {graph_info['edges']:,} edges in {time.time() - t0:.1f}s", flush=True)

    prepared = {}
    for name, items in splits.items():
        print(f"Computing local GraphWave signatures for {name} ({len(items):,} cascades)...", flush=True)
        t_split = time.time()
        local = local_embeddings(items, observation_time, device=device, batch_size=batch_size)

        global_features = []
        known = 0
        for cascade in items:
            feats = np.zeros((len(cascade.events), 40), dtype=np.float32)
            for i, (path, _) in enumerate(cascade.events):
                if path[-1] in index:
                    feats[i] = vectors[index[path[-1]]]
                    known += 1
            global_features.append(torch.from_numpy(feats))

        lengths = torch.tensor([len(x) for x in local], dtype=torch.long)
        combined_features = torch.cat([torch.cat((cg, gg), -1) for cg, gg in zip(local, global_features)])
        labels = torch.tensor([c.label for c in items], dtype=torch.float32)

        prepared[name] = {
            "features": combined_features,
            "lengths": lengths,
            "labels": labels,
            "ids": [c.id for c in items],
        }
        cov = known / max(1, int(lengths.sum()))
        print(f"Done {name} in {time.time() - t_split:.1f}s (coverage: {cov:.1%})", flush=True)

    return prepared
