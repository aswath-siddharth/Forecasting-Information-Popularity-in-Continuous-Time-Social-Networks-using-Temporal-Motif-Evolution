"""Dataset and DataLoader utilities for DeepCas on Twitter."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TwitterDeepCasDataset(Dataset):
    def __init__(self, npz_path: str | Path, target: str = "increment") -> None:
        path = Path(npz_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        data = np.load(path, allow_pickle=True)
        self.walks = data["walks"].astype(np.int64)  # (N, n_walks, walk_length)
        self.observed_size = data["observed_size"].astype(np.float32)
        self.publish_time = data["publish_time"]
        self.cascade_id = data["cascade_id"]

        if target == "increment":
            self.y_raw = data["y_increment"].astype(np.float32)
        elif target == "total":
            self.y_raw = data["y_total"].astype(np.float32)
        else:
            raise ValueError(f"Unknown target: {target}")

        self.y_log2 = np.log2(np.clip(self.y_raw, 0.0, None) + 1.0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.walks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Returns (walks, observed_size, y_log2, y_raw)."""
        walks_t = torch.from_numpy(self.walks[idx])  # (n_walks, walk_length)
        obs_sz_t = torch.tensor(self.observed_size[idx], dtype=torch.float32)
        y_log_t = torch.tensor(self.y_log2[idx], dtype=torch.float32)
        return walks_t, obs_sz_t, y_log_t, float(self.y_raw[idx])


def collate_deepcas(batch):
    walks = torch.stack([item[0] for item in batch], dim=0)       # (B, n_walks, walk_length)
    obs_sz = torch.stack([item[1] for item in batch], dim=0)     # (B,)
    y_log = torch.stack([item[2] for item in batch], dim=0)      # (B,)
    y_raw = np.array([item[3] for item in batch], dtype=np.float64)
    return walks, obs_sz, y_log, y_raw
