"""Dataset loading and batch collation for CasMotif on Twitter cascade benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

FEATURE_DIM = 15
UNBOUNDED_COLS = (5, 6, 7, 8, 10, 11, 12, 13, 14)


class SequenceScaler:
    """log1p + standardisation over unbounded feature columns.
    Fitted exclusively on the training split to guarantee zero data leakage.
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
        return {
            "cols": self.cols,
            "mean": self.mean.tolist() if self.mean is not None else [],
            "scale": self.scale.tolist() if self.scale is not None else [],
        }

    def load_state_dict(self, d: dict) -> None:
        self.cols = d["cols"]
        self.mean = np.array(d["mean"], dtype=np.float32)
        self.scale = np.array(d["scale"], dtype=np.float32)


class TwitterCascadeDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path,
        scaler: SequenceScaler | None = None,
    ) -> None:
        data = np.load(npz_path, allow_pickle=True)
        self.X_flat: np.ndarray = data["X_flat"]
        self.offsets: np.ndarray = data["offsets"]
        self.y_raw: np.ndarray = data["y_increment"].astype(np.float32)
        self.cascade_id: np.ndarray = data["cascade_id"]
        self.scaler = scaler
        self.y_log = (np.log(self.y_raw + 1.0) / np.log(2.0)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start, end = int(self.offsets[idx]), int(self.offsets[idx + 1])
        seq = self.X_flat[start:end]
        if self.scaler is not None:
            seq = self.scaler.transform(seq)
        return (
            torch.from_numpy(np.ascontiguousarray(seq)),
            torch.tensor([self.y_log[idx]], dtype=torch.float32),
            torch.tensor([self.y_raw[idx]], dtype=torch.float32),
        )


def collate_variable_length(
    items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs, y_logs, y_raws = zip(*items)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    b, max_len, _ = padded.shape
    idx = torch.arange(max_len).unsqueeze(0).expand(b, max_len)
    padding_mask = idx >= lengths.unsqueeze(1)
    y_log_t = torch.stack(y_logs, dim=0)
    y_raw_t = torch.stack(y_raws, dim=0)
    return padded, padding_mask, y_log_t, y_raw_t


class BucketedBatchSampler(Sampler[list[int]]):
    def __init__(self, lengths: np.ndarray, batch_size: int = 64, shuffle: bool = True) -> None:
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sorted_indices = np.argsort(self.lengths)

    def __iter__(self):
        chunks = [
            self.sorted_indices[i : i + self.batch_size].tolist()
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        if self.shuffle:
            np.random.shuffle(chunks)
        yield from chunks

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size
