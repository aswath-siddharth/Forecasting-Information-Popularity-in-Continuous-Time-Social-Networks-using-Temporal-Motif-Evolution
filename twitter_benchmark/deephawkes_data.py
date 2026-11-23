"""Prepare compact flat diffusion path buffers for DeepHawkes on Twitter cascades,
strictly aligned 1-to-1 to the CasMotif splits.

Zero leakage contract:
  - Vocabulary is built strictly on the training split only.
  - Users with frequency < min_user_freq (default 2) map to <UNK> (id 1).
  - <PAD> is reserved at id 0.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

PAD_ID = 0
UNK_ID = 1


def build_train_vocab(
    raw_train_json: Path,
    min_user_freq: int = 2,
) -> dict[str, int]:
    with open(raw_train_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    counter: Counter[str] = Counter()
    for item in data.values():
        for path in item["paths"]:
            counter.update(path)

    vocab: dict[str, int] = {}
    for user, cnt in counter.items():
        if cnt >= min_user_freq:
            vocab[user] = len(vocab) + 2  # 0=PAD, 1=UNK

    print(f"• User vocabulary built from {len(data):,} training cascades:")
    print(f"  Total unique train users: {len(counter):,}")
    print(f"  Vocab users kept (freq >= {min_user_freq}): {len(vocab):,}")
    print(f"  Singletons mapped to <UNK>: {len(counter) - len(vocab):,}")
    return vocab


def process_split(
    raw_split_json: Path,
    vocab: dict[str, int],
    observation_time: int,
    max_path_len: int = 10,
    n_time_intervals: int = 6,
) -> dict[str, np.ndarray]:
    with open(raw_split_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    cascade_ids = list(data.keys())
    n_cascades = len(cascade_ids)
    time_interval = observation_time / float(n_time_intervals)

    cascade_path_offsets = np.zeros(n_cascades + 1, dtype=np.int64)
    path_node_offsets_list: list[int] = [0]
    node_ids_list: list[int] = []
    path_time_bins_list: list[int] = []
    y_increments: list[float] = []

    total_tokens = 0
    known_tokens = 0

    for i, cid in enumerate(cascade_ids):
        item = data[cid]
        paths = item["paths"]
        times = item["times"]
        y_increments.append(float(item["y_increment"]))
        cascade_path_offsets[i + 1] = cascade_path_offsets[i] + len(paths)

        for p, t in zip(paths, times):
            # Truncate to max_path_len if needed
            if max_path_len > 0 and len(p) > max_path_len:
                p = p[-max_path_len:]

            bin_idx = min(n_time_intervals - 1, max(0, int(t // time_interval)))
            path_time_bins_list.append(bin_idx)

            for u in p:
                total_tokens += 1
                vid = vocab.get(u, UNK_ID)
                if vid != UNK_ID:
                    known_tokens += 1
                node_ids_list.append(vid)

            path_node_offsets_list.append(len(node_ids_list))

    coverage = (known_tokens / total_tokens * 100.0) if total_tokens else 0.0
    print(f"  Vocab token coverage: {known_tokens:,}/{total_tokens:,} ({coverage:.1f}%)")

    return {
        "cascade_id": np.asarray(cascade_ids, dtype=object),
        "y_increment": np.asarray(y_increments, dtype=np.float32),
        "cascade_path_offsets": cascade_path_offsets,
        "path_node_offsets": np.asarray(path_node_offsets_list, dtype=np.int64),
        "node_ids": np.asarray(node_ids_list, dtype=np.int64),
        "path_time_bins": np.asarray(path_time_bins_list, dtype=np.int64),
    }


def prepare_deephawkes_data(
    processed_dir: Path,
    output_dir: Path,
    observation_time: int = 86400,
    min_user_freq: int = 2,
    max_path_len: int = 10,
    n_time_intervals: int = 6,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_train = processed_dir / "raw_paths_train.json"
    if not raw_train.exists():
        raise FileNotFoundError(f"Missing {raw_train}; run twitter_benchmark.parser first.")

    vocab = build_train_vocab(raw_train, min_user_freq=min_user_freq)
    (output_dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")

    for split in ("train", "val", "test"):
        raw_json = processed_dir / f"raw_paths_{split}.json"
        print(f"• Processing DeepHawkes {split} split ({raw_json.name})...")
        buf = process_split(
            raw_json,
            vocab=vocab,
            observation_time=observation_time,
            max_path_len=max_path_len,
            n_time_intervals=n_time_intervals,
        )
        out_npz = output_dir / f"deephawkes_{split}.npz"
        np.savez_compressed(out_npz, **buf)
        print(f"  Saved -> {out_npz.name} ({out_npz.stat().st_size / 1e6:.1f} MB)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare DeepHawkes flat buffers on Twitter splits.")
    p.add_argument("--processed-dir", type=Path, required=True, help="Path to twitter_data/processed_{w}")
    p.add_argument("--output-dir", type=Path, required=True, help="Path to twitter_data/deephawkes_{w}")
    p.add_argument("--observation-time", type=int, default=86400)
    p.add_argument("--min-user-freq", type=int, default=2)
    args = p.parse_args()

    prepare_deephawkes_data(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        observation_time=args.observation_time,
        min_user_freq=args.min_user_freq,
    )


if __name__ == "__main__":
    main()
