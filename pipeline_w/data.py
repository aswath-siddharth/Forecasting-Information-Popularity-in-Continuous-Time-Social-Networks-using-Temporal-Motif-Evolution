"""Dataset plumbing for the Weibo benchmark splits produced by ``weibo_parser.py``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .weibo_parser import FEATURE_DIM, UNBOUNDED_COLS


class SequenceScaler:
    """log1p + standardisation over the unbounded feature columns.

    Fitted on the training split only -- the val/test splits are strictly later in
    time, so fitting on them would leak future information into the model.
    """

    def __init__(self, unbounded_cols: tuple[int, ...] = UNBOUNDED_COLS) -> None:
        self.cols = list(unbounded_cols)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, flat: np.ndarray) -> "SequenceScaler":
        vals = np.log1p(np.maximum(0.0, flat[:, self.cols].astype(np.float64)))
        self.mean = vals.mean(axis=0).astype(np.float32)
        var = np.maximum(vals.var(axis=0), 1e-7)
        self.scale = np.sqrt(var).astype(np.float32)
        return self

    def transform(self, seq: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            return seq
        out = seq.copy()
        vals = np.log1p(np.maximum(0.0, out[:, self.cols]))
        out[:, self.cols] = (vals - self.mean) / self.scale
        return out

    def state_dict(self) -> dict:
        return {"cols": self.cols, "mean": self.mean.tolist(), "scale": self.scale.tolist()}


class WeiboCascadeDataset(Dataset):
    """One item per cascade: the full 15-minute event sequence plus its target."""

    def __init__(
        self,
        npz_path: str | Path,
        scaler: SequenceScaler | None = None,
        target: str = "increment",
        log_base_2: bool = True,
    ) -> None:
        data = np.load(npz_path, allow_pickle=True)
        self.X_flat: np.ndarray = data["X_flat"]
        self.offsets: np.ndarray = data["offsets"]
        self.y_increment: np.ndarray = data["y_increment"]
        self.y_total: np.ndarray = data["y_total"]
        self.observed_size: np.ndarray = data["observed_size"]
        self.cascade_id: np.ndarray = data["cascade_id"]
        self.scaler = scaler
        self.log_base_2 = log_base_2

        if target == "increment":
            self.y_raw = self.y_increment
        elif target == "total":
            self.y_raw = self.y_total
        else:
            raise ValueError("target must be 'increment' or 'total'")

        if self.X_flat.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected {FEATURE_DIM} feature columns, got {self.X_flat.shape[1]}")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def log_targets(self) -> np.ndarray:
        """DeepHawkes' training target: log2(y + 1)."""
        base = np.log(2.0) if self.log_base_2 else 1.0
        return (np.log(self.y_raw + 1.0) / base).astype(np.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start, end = int(self.offsets[idx]), int(self.offsets[idx + 1])
        seq = self.X_flat[start:end]
        if self.scaler is not None:
            seq = self.scaler.transform(seq)
        base = np.log(2.0) if self.log_base_2 else 1.0
        y_log = np.float32(np.log(self.y_raw[idx] + 1.0) / base)
        return (
            torch.from_numpy(np.ascontiguousarray(seq)),
            torch.tensor([y_log], dtype=torch.float32),
            torch.tensor([self.y_raw[idx]], dtype=torch.float32),
        )


def pad_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad to the batch max and emit a True-for-padding mask for the attention layer."""
    sequences, y_logs, y_raws = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    positions = torch.arange(padded.shape[1]).unsqueeze(0)
    padding_mask = positions >= lengths.unsqueeze(1)
    return padded, padding_mask, torch.stack(y_logs), torch.stack(y_raws)
