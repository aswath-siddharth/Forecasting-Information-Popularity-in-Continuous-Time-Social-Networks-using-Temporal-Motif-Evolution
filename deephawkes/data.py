"""Extract diffusion-path datasets for DeepHawkes, aligned to the existing pipeline_w splits.

DeepHawkes (Cao et al., CIKM 2017) consumes the observed retweet propagation paths
``(u_0, u_1, ..., u_k)`` and their timestamps ``t_j < T_obs``, rather than CasMotif's
15-D causal motif features.

To keep the comparison strictly controlled:
- Splits are NOT re-derived. This script reads the exact ``cascade_id`` sequence and
  ``y_increment`` targets from ``weibo_{train,val,test}.npz`` produced by
  ``pipeline_w.weibo_parser``, guaranteeing identical chronological 70/15/15 partitions
  and identical labels.
- The user vocabulary is constructed strictly on the training split (zero leakage from
  validation or test). Index 0 is ``<PAD>`` and index 1 is ``<UNK>``. Users with fewer
  than ``min_user_freq`` occurrences in training (default 2) are mapped to ``<UNK>`` so
  that the ``<UNK>`` embedding receives real gradient updates during training and
  generalises properly to unseen users in the strictly later chronological val/test splits.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

PAD_ID = 0
UNK_ID = 1


def extract_raw_paths(
    weibo_txt: Path,
    wanted: set[str],
    observation_time: int,
    max_path_len: int = 10,
) -> dict[str, tuple[list[list[str]], list[float]]]:
    """Return ``{cascade_id: (paths, times)}`` for all wanted cascades."""
    out: dict[str, tuple[list[list[str]], list[float]]] = {}

    with open(weibo_txt, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            cid = parts[0]
            if cid not in wanted:
                continue

            observed: list[tuple[list[str], int]] = []
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
                    t_now = int(t_str)
                except ValueError:
                    continue
                if t_now < observation_time:
                    # Keep the last max_path_len users along the diffusion path if longer.
                    if max_path_len > 0 and len(nodes) > max_path_len:
                        nodes = nodes[-max_path_len:]
                    observed.append((nodes, t_now))

            observed.sort(key=lambda it: it[1])
            if not observed:
                continue

            paths = [nodes for nodes, _ in observed]
            times = [float(max(0, t_now)) for _, t_now in observed]
            out[cid] = (paths, times)

    return out


def build_train_vocab(
    train_ids: list[str],
    raw_paths: dict[str, tuple[list[list[str]], list[float]]],
    min_user_freq: int = 2,
) -> tuple[dict[str, int], dict[str, int]]:
    """Build ``user_str -> id`` strictly from the training split."""
    counts: Counter[str] = Counter()
    for cid in train_ids:
        paths, _ = raw_paths[cid]
        for p in paths:
            counts.update(p)

    vocab: dict[str, int] = {}
    for user, cnt in counts.items():
        if cnt >= min_user_freq:
            vocab[user] = len(vocab) + 2  # 0=PAD, 1=UNK

    stats = {
        "unique_train_users": len(counts),
        "vocab_users_kept": len(vocab),
        "vocab_size_with_special": len(vocab) + 2,
        "min_user_freq": min_user_freq,
        "singleton_or_rare_train_users": len(counts) - len(vocab),
    }
    return vocab, stats


def encode_split(
    ids: list[str],
    labels: np.ndarray,
    raw_paths: dict[str, tuple[list[list[str]], list[float]]],
    vocab: dict[str, int],
    observation_time: int,
    n_time_intervals: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Encode a split into compact flat CSR-style numpy arrays for fast GPU batching.

    Layout:
      - ``cascade_path_offsets``: [N + 1] slice into path arrays for cascade i
      - ``path_node_offsets``:    [P + 1] slice into ``node_ids`` for path p
      - ``node_ids``:             [T] int32 user token ids (>=1; 0 reserved for PAD)
      - ``path_time_bins``:       [P] int16 interval index in [0, n_time_intervals - 1]
      - ``path_times``:           [P] float32 elapsed seconds since publish
      - ``y_increment``:          [N] float32 target popularity increment
      - ``cascade_id``:           [N] string array
    """
    cascade_path_offsets = np.zeros(len(ids) + 1, dtype=np.int64)
    path_node_offsets_list: list[int] = [0]
    node_ids_list: list[int] = []
    path_time_bins_list: list[int] = []
    path_times_list: list[float] = []

    total_tokens = 0
    known_tokens = 0
    total_paths = 0
    max_path_len = 0

    for i, cid in enumerate(ids):
        paths, times = raw_paths[cid]
        cascade_path_offsets[i] = total_paths
        for p, t in zip(paths, times):
            plen = len(p)
            if plen > max_path_len:
                max_path_len = plen
            for u in p:
                uid = vocab.get(u, UNK_ID)
                node_ids_list.append(uid)
                total_tokens += 1
                if uid != UNK_ID:
                    known_tokens += 1
            path_node_offsets_list.append(len(node_ids_list))

            # Discretize t in [0, T_obs) into n_time_intervals equal bins.
            bin_idx = int((t / max(float(observation_time), 1.0)) * n_time_intervals)
            if bin_idx < 0:
                bin_idx = 0
            elif bin_idx >= n_time_intervals:
                bin_idx = n_time_intervals - 1
            path_time_bins_list.append(bin_idx)
            path_times_list.append(t)
            total_paths += 1
        cascade_path_offsets[i + 1] = total_paths

    arrays = {
        "cascade_id": np.asarray(ids),
        "y_increment": labels.astype(np.float32),
        "cascade_path_offsets": cascade_path_offsets,
        "path_node_offsets": np.asarray(path_node_offsets_list, dtype=np.int64),
        "node_ids": np.asarray(node_ids_list, dtype=np.int32),
        "path_time_bins": np.asarray(path_time_bins_list, dtype=np.int16),
        "path_times": np.asarray(path_times_list, dtype=np.float32),
    }
    stats = {
        "n_cascades": len(ids),
        "n_paths": total_paths,
        "mean_paths_per_cascade": total_paths / max(len(ids), 1),
        "max_path_len": max_path_len,
        "total_tokens": total_tokens,
        "known_token_fraction": known_tokens / max(total_tokens, 1),
    }
    return arrays, stats


def main() -> None:
    p = argparse.ArgumentParser(description="Build DeepHawkes path datasets aligned to existing Weibo splits.")
    p.add_argument("--input", default="weibo_data/dataset_weibo.txt")
    p.add_argument("--splits-dir", required=True, help="Directory holding weibo_{train,val,test}.npz")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--observation-time", type=int, required=True)
    p.add_argument("--n-time-intervals", type=int, default=6,
                   help="Number of equal discrete time-decay intervals (DeepHawkes uses 6 for 1h).")
    p.add_argument("--max-path-len", type=int, default=10,
                   help="Maximum user hops per diffusion path (DeepHawkes default = 10).")
    p.add_argument("--min-user-freq", type=int, default=2,
                   help="Minimum training-split occurrence count for a dedicated user embedding.")
    args = p.parse_args()

    splits_dir, out_dir = Path(args.splits_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    order: dict[str, list[str]] = {}
    labels: dict[str, np.ndarray] = {}
    wanted: set[str] = set()
    for s in ("train", "val", "test"):
        d = np.load(splits_dir / f"weibo_{s}.npz", allow_pickle=True)
        ids = [str(x) for x in d["cascade_id"].tolist()]
        order[s] = ids
        labels[s] = d["y_increment"].astype(np.float32)
        wanted.update(ids)

    print(f"• Aligning DeepHawkes paths to {len(wanted):,} cascades (window={args.observation_time}s)", flush=True)
    raw_paths = extract_raw_paths(Path(args.input), wanted, args.observation_time, args.max_path_len)
    print(f"• Extracted raw paths for {len(raw_paths):,}/{len(wanted):,} cascades", flush=True)

    missing_total = sum(1 for cid in wanted if cid not in raw_paths)
    if missing_total > 0:
        raise RuntimeError(f"Alignment mismatch: {missing_total} cascades missing from raw file!")

    vocab, vocab_stats = build_train_vocab(order["train"], raw_paths, min_user_freq=args.min_user_freq)
    print(f"• Train vocab: {vocab_stats['vocab_size_with_special']:,} total "
          f"({vocab_stats['vocab_users_kept']:,} users with freq >= {args.min_user_freq} + PAD/UNK; "
          f"{vocab_stats['unique_train_users']:,} unique train users)", flush=True)

    split_stats: dict[str, dict] = {}
    for s in ("train", "val", "test"):
        arrays, st = encode_split(
            order[s], labels[s], raw_paths, vocab,
            args.observation_time, args.n_time_intervals,
        )
        np.savez(out_dir / f"deephawkes_{s}.npz", **arrays)
        split_stats[s] = st
        print(f"  {s:<5} cascades={st['n_cascades']:,}  paths={st['n_paths']:,}  "
              f"mean_paths={st['mean_paths_per_cascade']:.1f}  "
              f"known_tokens={st['known_token_fraction']:.1%}", flush=True)

    meta = {
        "observation_time": args.observation_time,
        "n_time_intervals": args.n_time_intervals,
        "max_path_len": args.max_path_len,
        "vocab": vocab_stats,
        "splits": split_stats,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"• Saved DeepHawkes dataset -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

