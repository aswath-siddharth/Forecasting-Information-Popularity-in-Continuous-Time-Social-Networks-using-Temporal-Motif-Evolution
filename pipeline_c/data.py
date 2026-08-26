from __future__ import annotations

import csv
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
    files_processed: int = 1


def _extract_contract_id(post_dict: dict, duplicate_counts: dict[str, int]) -> str:
    """Derive the canonical Pipeline B contract post_id: {user}_{time}__dupN."""
    user = post_dict.get("post_user") or post_dict.get("author") or "unknown"
    raw_time = post_dict.get("post_time")
    if raw_time is None:
        raw_time = post_dict.get("created_utc", post_dict.get("created", 0.0))
    time_int = int(float(raw_time))
    base_id = f"{user}_{time_int}"

    count = duplicate_counts.get(base_id, 0)
    duplicate_counts[base_id] = count + 1
    if count > 0:
        return f"{base_id}__dup{count}"
    return base_id


def _load_labels_map(labels_path: Path, target_column: str = "y_viral") -> dict[str, float]:
    """Read target virality labels into a fast in-memory dictionary."""
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


def _load_directory_data(
    features_dir: Path,
    labels_path: Path,
    data_dir: Path,
    sequence_column: str = "X_seq_json",
    target_column: str = "y_viral",
    category: str | None = None,
    max_files: int | None = None,
    sample_ratio: float = 1.0,
) -> tuple[pd.DataFrame, JoinStats]:
    """Discover feature CSV files, align with raw JSON post IDs, and join labels."""
    labels_map = _load_labels_map(labels_path, target_column=target_column)
    all_feature_files = sorted(list(features_dir.rglob("*.csv")))

    if category:
        cat_lower = category.strip().lower()
        all_feature_files = [
            f for f in all_feature_files if cat_lower in f.parent.name.lower() or cat_lower in str(f).lower()
        ]

    if max_files is not None and max_files > 0:
        all_feature_files = all_feature_files[:max_files]

    if not all_feature_files:
        raise FileNotFoundError(
            f"No feature CSV files found in {features_dir}"
            + (f" matching category '{category}'" if category else "")
        )

    print(f"Aligning and loading features from {len(all_feature_files)} CSV files...")

    collected_dfs: list[pd.DataFrame] = []
    total_feature_rows = 0
    total_joined_rows = 0
    total_dropped_feature_rows = 0
    processed_count = 0

    for feat_file in all_feature_files:
        try:
            feat_df = pd.read_csv(feat_file)
        except Exception:
            continue

        if feat_df.empty or "post_id" not in feat_df.columns or sequence_column not in feat_df.columns:
            continue

        total_feature_rows += len(feat_df)
        sub_name = feat_file.stem[:-9] if feat_file.stem.endswith("_features") else feat_file.stem

        # Locate corresponding raw JSON file in data_dir
        rel_dir = feat_file.parent.name
        json_file = data_dir / rel_dir / f"{sub_name}.json"
        if not json_file.exists():
            candidates = list(data_dir.rglob(f"{sub_name}.json"))
            json_file = candidates[0] if candidates else None

        post_index_to_label: dict[str, float] = {}
        if json_file and json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as jf:
                    raw_posts = json.load(jf)
                    if not isinstance(raw_posts, list):
                        raw_posts = [raw_posts]
                    dup_tracker: dict[str, int] = {}
                    for idx, post_obj in enumerate(raw_posts):
                        contract_id = _extract_contract_id(post_obj, dup_tracker)
                        if contract_id in labels_map:
                            post_index_to_label[f"post_{idx}"] = labels_map[contract_id]
                            post_index_to_label[contract_id] = labels_map[contract_id]
            except Exception:
                pass

        feat_df[target_column] = feat_df["post_id"].map(
            lambda pid: post_index_to_label.get(pid, labels_map.get(pid, np.nan))
        )

        matched_df = feat_df.dropna(subset=[target_column]).copy()
        dropped = len(feat_df) - len(matched_df)
        total_dropped_feature_rows += dropped
        total_joined_rows += len(matched_df)

        if not matched_df.empty:
            if 0.0 < sample_ratio < 1.0:
                matched_df = matched_df.sample(frac=sample_ratio, random_state=42)
            collected_dfs.append(matched_df[["post_id", sequence_column, target_column]])
            processed_count += 1

    if not collected_dfs:
        raise ValueError(
            "Inner join produced zero rows across all feature files. Check post_id overlap."
        )

    merged = pd.concat(collected_dfs, ignore_index=True)
    stats = JoinStats(
        feature_rows=total_feature_rows,
        label_rows=len(labels_map),
        joined_rows=total_joined_rows,
        dropped_feature_rows=total_dropped_feature_rows,
        dropped_label_rows=max(0, len(labels_map) - merged["post_id"].nunique()),
        files_processed=processed_count,
    )

    return merged.reset_index(drop=True), stats


def load_and_join_data(
    features_path: str | Path,
    labels_path: str | Path,
    data_dir: str | Path = "data",
    sequence_column: str = "X_seq_json",
    target_column: str = "y_viral",
    category: str | None = None,
    max_files: int | None = None,
    sample_ratio: float = 1.0,
) -> tuple[pd.DataFrame, JoinStats]:
    """Load Pipeline A features and Pipeline B labels, supporting both single CSV and directory input."""
    features_path = Path(features_path)
    labels_path = Path(labels_path)
    data_dir = Path(data_dir)

    if not features_path.exists():
        raise FileNotFoundError(f"Feature path not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {labels_path}. Add pipeline_b_labels.csv or pass --labels."
        )

    if features_path.is_dir():
        return _load_directory_data(
            features_dir=features_path,
            labels_path=labels_path,
            data_dir=data_dir,
            sequence_column=sequence_column,
            target_column=target_column,
            category=category,
            max_files=max_files,
            sample_ratio=sample_ratio,
        )

    # Single-file mode
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
