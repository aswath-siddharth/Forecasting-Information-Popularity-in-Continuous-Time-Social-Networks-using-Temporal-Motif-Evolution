from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass(frozen=True)
class JoinStats:
    feature_rows: int
    label_rows: int
    joined_rows: int
    dropped_feature_rows: int
    dropped_label_rows: int


def load_and_join_data(
    features_path: str | Path,
    labels_path: str | Path,
    sequence_column: str = "X_seq_json",
    target_column: str = "y_viral",
) -> tuple[pd.DataFrame, JoinStats]:
    """Load Pipeline A features and Pipeline B labels, then inner-join on post_id."""
    features_path = Path(features_path)
    labels_path = Path(labels_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {labels_path}. Add pipeline_b_labels.csv or pass --labels."
        )

    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    required_feature_cols = {"post_id", sequence_column}
    required_label_cols = {"post_id", target_column}
    missing_features = required_feature_cols - set(features_df.columns)
    missing_labels = required_label_cols - set(labels_df.columns)
    if missing_features:
        raise ValueError(f"Missing feature columns in {features_path}: {sorted(missing_features)}")
    if missing_labels:
        raise ValueError(f"Missing label columns in {labels_path}: {sorted(missing_labels)}")

    merged = pd.merge(features_df, labels_df[["post_id", target_column]], on="post_id", how="inner")
    feature_post_ids = set(features_df["post_id"])
    label_post_ids = set(labels_df["post_id"])
    stats = JoinStats(
        feature_rows=len(features_df),
        label_rows=len(labels_df),
        joined_rows=len(merged),
        dropped_feature_rows=int((~features_df["post_id"].isin(label_post_ids)).sum()),
        dropped_label_rows=int((~labels_df["post_id"].isin(feature_post_ids)).sum()),
    )

    if merged.empty:
        raise ValueError("Inner join produced zero rows. Check post_id overlap between inputs.")

    return merged.reset_index(drop=True), stats


def parse_sequence(value: str | Iterable[Iterable[float]], feature_dim: int) -> np.ndarray:
    """Deserialize one JSON sequence and validate it as [K, feature_dim]."""
    parsed = json.loads(value) if isinstance(value, str) else value
    array = np.asarray(parsed, dtype=np.float32)

    if array.ndim != 2:
        raise ValueError(f"Expected a 2D sequence matrix, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("Encountered an empty sequence matrix.")
    if array.shape[1] != feature_dim:
        raise ValueError(f"Expected feature_dim={feature_dim}, got shape {array.shape}")

    return array


class CascadeDataset(Dataset):
    """PyTorch dataset for variable-length cascade feature sequences."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        sequence_column: str = "X_seq_json",
        target_column: str = "y_viral",
        feature_dim: int = 20,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.sequence_column = sequence_column
        self.target_column = target_column
        self.feature_dim = feature_dim

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[idx]
        sequence = parse_sequence(row[self.sequence_column], self.feature_dim)
        label = np.float32(row[self.target_column])
        return torch.from_numpy(sequence), torch.tensor([label], dtype=torch.float32)


def pad_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences and create True-for-padding boolean masks."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([sequence.shape[0] for sequence in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    max_len = padded.shape[1]
    positions = torch.arange(max_len).unsqueeze(0)
    padding_mask = positions >= lengths.unsqueeze(1)
    label_tensor = torch.stack(labels, dim=0)

    return padded, padding_mask, label_tensor
