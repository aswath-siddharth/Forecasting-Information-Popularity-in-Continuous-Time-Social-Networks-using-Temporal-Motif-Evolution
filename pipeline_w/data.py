"""Dataset plumbing for the Weibo benchmark splits produced by ``weibo_parser.py``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .weibo_parser import CHINA_UTC_OFFSET_SECONDS, FEATURE_DIM, FEATURE_NAMES, UNBOUNDED_COLS

T_FROM_ROOT_COL = FEATURE_NAMES.index("t_from_root")
SIDE_FEATURE_NAMES = ("hour_sin", "hour_cos", "log2_observed_size", "recent_frac", "log_idle")
SIDE_DIM = len(SIDE_FEATURE_NAMES)


def cascade_side_features(X_flat: np.ndarray, offsets: np.ndarray, observed_size: np.ndarray,
                          publish_time: np.ndarray, window: float) -> np.ndarray:
    """Cascade-level context that the per-event sequence does not carry directly.

    Everything here is known at the end of the observation window, so none of it
    leaks the target: publish hour (Beijing, cyclic) sets how many active hours the
    24h horizon still contains; observed size, the share of events in the last
    quarter of the window, and the silence since the last event describe momentum.
    """
    n = len(offsets) - 1
    out = np.zeros((n, SIDE_DIM), dtype=np.float32)
    beijing = (publish_time + CHINA_UTC_OFFSET_SECONDS) % 86400
    angle = 2.0 * np.pi * beijing / 86400.0
    out[:, 0], out[:, 1] = np.sin(angle), np.cos(angle)
    out[:, 2] = np.log2(observed_size + 1.0)
    t = X_flat[:, T_FROM_ROOT_COL]
    for i in range(n):
        ti = t[offsets[i]:offsets[i + 1]]
        if len(ti) == 0:
            out[i, 3], out[i, 4] = 0.0, np.log1p(window)
            continue
        out[i, 3] = float(np.mean(ti >= 0.75 * window))
        out[i, 4] = np.log1p(max(0.0, window - float(ti.max())))
    return out


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
        side_window: float | None = None,
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

        # Optional cascade-level side features; standardise with set_side_scaling()
        # using statistics from the training split only.
        self.side: np.ndarray | None = None
        if side_window is not None:
            self.side = cascade_side_features(self.X_flat, self.offsets, self.observed_size,
                                              data["publish_time"], float(side_window))

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

    def set_side_scaling(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.side = ((self.side - mean) / std).astype(np.float32)

    def log_targets(self) -> np.ndarray:
        """DeepHawkes' training target: log2(y + 1)."""
        base = np.log(2.0) if self.log_base_2 else 1.0
        return (np.log(self.y_raw + 1.0) / base).astype(np.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        start, end = int(self.offsets[idx]), int(self.offsets[idx + 1])
        seq = self.X_flat[start:end]
        if self.scaler is not None:
            seq = self.scaler.transform(seq)
        base = np.log(2.0) if self.log_base_2 else 1.0
        y_log = np.float32(np.log(self.y_raw[idx] + 1.0) / base)
        item = (
            torch.from_numpy(np.ascontiguousarray(seq)),
            torch.tensor([y_log], dtype=torch.float32),
            torch.tensor([self.y_raw[idx]], dtype=torch.float32),
        )
        if self.side is not None:
            item += (torch.from_numpy(self.side[idx]),)
        return item


def pad_collate_fn(batch: list[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
    """Pad to the batch max and emit a True-for-padding mask for the attention layer.

    Side features, when the dataset carries them, are appended as a fifth tensor.
    """
    sequences, y_logs, y_raws, *side = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    positions = torch.arange(padded.shape[1]).unsqueeze(0)
    padding_mask = positions >= lengths.unsqueeze(1)
    out = (padded, padding_mask, torch.stack(y_logs), torch.stack(y_raws))
    if side:
        out += (torch.stack(side[0]),)
    return out
