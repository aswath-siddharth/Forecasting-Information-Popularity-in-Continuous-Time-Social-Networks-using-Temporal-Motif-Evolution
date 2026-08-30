from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset


@dataclass(frozen=True)
class JoinStats:
    feature_rows: int
    label_rows: int
    joined_rows: int
    dropped_feature_rows: int
    dropped_label_rows: int
    files_processed: int = 1


def extract_contract_id(post_dict: dict, duplicate_counts: dict[str, int]) -> str:
    """Derive the canonical Pipeline B contract post_id: {user}_{time}__dupN."""
    user = post_dict.get("post_user") or post_dict.get("author") or "unknown"
    raw_time = post_dict.get("post_time")
    if raw_time is None:
        raw_time = post_dict.get("created_utc", post_dict.get("created", 0.0))
    try:
        time_int = int(float(raw_time))
    except (ValueError, TypeError):
        time_int = 0
    base_id = f"{user}_{time_int}"

    count = duplicate_counts.get(base_id, 0)
    duplicate_counts[base_id] = count + 1
    if count > 0:
        return f"{base_id}__dup{count}"
    return base_id


def load_labels_map(labels_path: str | Path, target_column: str = "y_viral") -> dict[str, float]:
    """Read target virality labels into a fast in-memory dictionary (~70 MB for 1.2M labels)."""
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    labels_map: dict[str, float] = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "post_id" not in reader.fieldnames or target_column not in reader.fieldnames:
            raise ValueError(
                f"Labels file {labels_path} must contain 'post_id' and '{target_column}' columns."
            )
        for row in reader:
            try:
                labels_map[row["post_id"]] = float(row[target_column])
            except (ValueError, TypeError):
                continue
    return labels_map


def discover_file_pairs(
    features_dir: str | Path,
    data_dir: str | Path,
    category: str | None = None,
    max_files: int | None = None,
    seed: int = 42,
) -> tuple[list[tuple[Path, Path | None]], list[tuple[Path, Path | None]], list[tuple[Path, Path | None]]]:
    """Discover feature CSVs and matching raw JSON files, splitting 80/10/10 into train/val/test."""
    features_dir = Path(features_dir)
    data_dir = Path(data_dir)

    all_csvs = sorted(list(features_dir.rglob("*.csv")))
    if category:
        cat_lower = category.strip().lower()
        all_csvs = [f for f in all_csvs if cat_lower in f.parent.name.lower() or cat_lower in str(f).lower()]

    if max_files is not None and max_files > 0:
        all_csvs = all_csvs[:max_files]

    if not all_csvs:
        raise FileNotFoundError(
            f"No feature CSV files found in {features_dir}"
            + (f" matching category '{category}'" if category else "")
        )

    file_pairs: list[tuple[Path, Path | None]] = []
    for feat_file in all_csvs:
        sub_name = feat_file.stem[:-9] if feat_file.stem.endswith("_features") else feat_file.stem
        rel_dir = feat_file.parent.name
        json_file = data_dir / rel_dir / f"{sub_name}.json"
        if not json_file.exists():
            candidates = list(data_dir.rglob(f"{sub_name}.json"))
            json_file = candidates[0] if candidates else None
        file_pairs.append((feat_file, json_file))

    # Shuffle with seed and split 80 / 10 / 10
    rng = random.Random(seed)
    shuffled_pairs = list(file_pairs)
    rng.shuffle(shuffled_pairs)

    n_total = len(shuffled_pairs)
    n_train = max(1, int(0.8 * n_total))
    n_val = max(1, int(0.1 * n_total)) if n_total >= 3 else 0

    train_pairs = shuffled_pairs[:n_train]
    val_pairs = shuffled_pairs[n_train : n_train + n_val] if n_val > 0 else shuffled_pairs[:n_train]
    test_pairs = shuffled_pairs[n_train + n_val :] if (n_total - n_train - n_val) > 0 else shuffled_pairs[:n_train]

    return train_pairs, val_pairs, test_pairs


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


UNBOUNDED_COLS = (5, 6, 7, 8, 14, 15, 16, 17, 18, 19)


class SequenceScaler:
    """Scales unbounded features in 20-D sequence matrices X_seq [K, 20].
    Applies log1p transform followed by StandardScaler on unbounded columns:
        Cols 5-8: CSR Structural Centrality metrics
        Cols 14-15: Idle Time & Reactivated Descendants
        Col 16: Motif Velocity delta_t_M
        Cols 17-18: Window index & cumulative prior comments
        Col 19: Inter-arrival gap delta_t_edge
    Bounded features (One-hot motifs 0-4, VADER sentiment 9-12, I_reactivate 13) are left unchanged.
    """

    def __init__(self, unbounded_cols: tuple[int, ...] = UNBOUNDED_COLS) -> None:
        self.unbounded_cols = list(unbounded_cols)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, sequences: Iterable[np.ndarray]) -> SequenceScaler:
        count = 0
        sum_x = np.zeros(len(self.unbounded_cols), dtype=np.float64)
        sum_sq = np.zeros(len(self.unbounded_cols), dtype=np.float64)

        for seq in sequences:
            if seq.size == 0:
                continue
            unbounded_vals = np.log1p(np.maximum(0.0, seq[:, self.unbounded_cols]))
            count += unbounded_vals.shape[0]
            sum_x += unbounded_vals.sum(axis=0)
            sum_sq += (unbounded_vals ** 2).sum(axis=0)

        if count > 0:
            self.mean = (sum_x / count).astype(np.float32)
            var = (sum_sq / count) - (self.mean ** 2)
            var = np.maximum(var, 1e-7)
            self.scale = np.sqrt(var).astype(np.float32)
        else:
            self.mean = np.zeros(len(self.unbounded_cols), dtype=np.float32)
            self.scale = np.ones(len(self.unbounded_cols), dtype=np.float32)
        return self

    def transform(self, seq: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None or seq.size == 0:
            return seq
        seq_copy = seq.copy()
        unbounded_vals = np.log1p(np.maximum(0.0, seq_copy[:, self.unbounded_cols]))
        seq_copy[:, self.unbounded_cols] = (unbounded_vals - self.mean) / self.scale
        return seq_copy


class StreamingCascadeDataset(IterableDataset):
    """Memory-efficient streaming dataset that loads files on-the-fly, keeping RAM under 1 GB."""

    def __init__(
        self,
        file_pairs: list[tuple[Path, Path | None]],
        labels_map: dict[str, float],
        sequence_column: str = "X_seq_json",
        target_column: str = "y_viral",
        feature_dim: int = 20,
        shuffle_files: bool = True,
        sample_ratio: float = 1.0,
        scaler: SequenceScaler | None = None,
    ) -> None:
        super().__init__()
        self.file_pairs = file_pairs
        self.labels_map = labels_map
        self.sequence_column = sequence_column
        self.target_column = target_column
        self.feature_dim = feature_dim
        self.shuffle_files = shuffle_files
        self.sample_ratio = sample_ratio
        self.scaler = scaler

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            pairs = list(self.file_pairs)
        else:
            pairs = [
                f for idx, f in enumerate(self.file_pairs) if idx % worker_info.num_workers == worker_info.id
            ]

        if self.shuffle_files:
            random.shuffle(pairs)

        for feat_file, json_file in pairs:
            # Map post_k to labels using json metadata
            post_labels: dict[str, float] = {}
            if json_file and json_file.exists():
                try:
                    with open(json_file, "r", encoding="utf-8") as jf:
                        raw_posts = json.load(jf)
                        if not isinstance(raw_posts, list):
                            raw_posts = [raw_posts]
                        dup_tracker: dict[str, int] = {}
                        for idx, post_obj in enumerate(raw_posts):
                            contract_id = extract_contract_id(post_obj, dup_tracker)
                            if contract_id in self.labels_map:
                                post_labels[f"post_{idx}"] = self.labels_map[contract_id]
                                post_labels[contract_id] = self.labels_map[contract_id]
                except Exception:
                    pass

            try:
                with open(feat_file, "r", encoding="utf-8") as ff:
                    reader = csv.DictReader(ff)
                    for row in reader:
                        pid = row.get("post_id")
                        label = post_labels.get(pid, self.labels_map.get(pid))
                        if label is None:
                            continue

                        if 0.0 < self.sample_ratio < 1.0 and random.random() > self.sample_ratio:
                            continue

                        seq_str = row.get(self.sequence_column)
                        if not seq_str:
                            continue

                        seq_array = parse_sequence(seq_str, self.feature_dim)
                        if self.scaler is not None:
                            seq_array = self.scaler.transform(seq_array)
                        yield torch.from_numpy(seq_array), torch.tensor([float(label)], dtype=torch.float32)
            except Exception:
                continue


class CascadeDataset(Dataset):
    """In-memory PyTorch dataset for variable-length cascade feature sequences (single file mode)."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        sequence_column: str = "X_seq_json",
        target_column: str = "y_viral",
        feature_dim: int = 20,
        scaler: SequenceScaler | None = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.sequence_column = sequence_column
        self.target_column = target_column
        self.feature_dim = feature_dim
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[idx]
        sequence = parse_sequence(row[self.sequence_column], self.feature_dim)
        if self.scaler is not None:
            sequence = self.scaler.transform(sequence)
        label = np.float32(row[self.target_column])
        return torch.from_numpy(sequence), torch.tensor([label], dtype=torch.float32)


def load_single_file_data(
    features_path: Path,
    labels_path: Path,
    sequence_column: str = "X_seq_json",
    target_column: str = "y_viral",
    sample_ratio: float = 1.0,
) -> tuple[pd.DataFrame, JoinStats]:
    """Load single CSV files for features and labels."""
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
        files_processed=1,
    )

    if merged.empty:
        raise ValueError("Inner join produced zero rows. Check post_id overlap between inputs.")

    if 0.0 < sample_ratio < 1.0:
        merged = merged.sample(frac=sample_ratio, random_state=42)

    return merged.reset_index(drop=True), stats


def pad_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences and create True-for-padding boolean masks."""
    if not batch:
        return torch.empty(0), torch.empty(0, dtype=torch.bool), torch.empty(0)

    sequences, labels = zip(*batch)
    lengths = torch.tensor([sequence.shape[0] for sequence in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    max_len = padded.shape[1]
    positions = torch.arange(max_len).unsqueeze(0)
    padding_mask = positions >= lengths.unsqueeze(1)
    label_tensor = torch.stack(labels, dim=0)

    return padded, padding_mask, label_tensor
