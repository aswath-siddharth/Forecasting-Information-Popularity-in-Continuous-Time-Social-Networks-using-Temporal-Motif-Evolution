"""Parse Twitter cascades into DeepCas random walks and chronological benchmark splits.

Follows the benchmark protocol from the literature (DeepCas WWW'17, CasFT, VaCas, DeepHawkes):
1. Observation Window: T_obs = 86,400s (1-day) or 172,800s (2-day).
2. Prediction Horizon: T_pred = 15 days (1,296,000s).
3. Target: Future increment only: Delta_P = R_15d - R_obs.
4. Cascade Size Filter: 10 <= N(T_obs) <= 1,000.
5. No hour filter (Twitter activity is globally distributed across time zones).
6. Split: Strictly Chronological (by publish time t_0): 70% Train / 15% Val / 15% Test.
7. Random Walks: N_w walks of length L=10 sampled on each cascade graph.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def sample_cascade_walks(
    events: list[dict],
    t_obs: int,
    n_walks: int = 32,
    walk_length: int = 10,
    rng: random.Random = None,
) -> list[list[int]]:
    """Construct cascade graph up to t_obs and sample random walks."""
    if rng is None:
        rng = random.Random(42)

    adj = defaultdict(list)
    nodes = set()
    node_out_degree = defaultdict(int)

    for e in events:
        if e["elapsed"] > t_obs:
            continue
        path = e["path"]
        if len(path) >= 2:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                adj[u].append(v)
                adj[v].append(u)  # allow bidirectional graph exploration
                node_out_degree[u] += 1
                nodes.add(u)
                nodes.add(v)
        elif len(path) == 1:
            nodes.add(path[0])

    if not nodes:
        return [[0] * walk_length for _ in range(n_walks)]

    node_list = list(nodes)
    # Prefer sampling start nodes with higher participation
    weights = [node_out_degree.get(n, 1) + 1 for n in node_list]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    walks = []
    for _ in range(n_walks):
        curr = rng.choices(node_list, weights=probs, k=1)[0]
        walk = [curr]
        for _ in range(walk_length - 1):
            nbrs = adj.get(curr)
            if nbrs:
                curr = rng.choice(nbrs)
            walk.append(curr)
        walks.append(walk)

    return walks


def process_twitter_dataset(
    parquet_path: str,
    output_dir: str,
    observation_time: int = 86400,
    horizon_time: int = 1296000,
    min_observed: int = 10,
    max_observed: int = 1000,
    n_walks: int = 32,
    walk_length: int = 10,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    np.random.seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"• Loading {parquet_path}...")
    print(f"  T_obs = {observation_time}s ({observation_time/86400:.1f} days)")
    print(f"  T_pred = {horizon_time}s ({horizon_time/86400:.1f} days)")
    print(f"  Filter: {min_observed} <= N(T_obs) <= {max_observed}, no hour filter")

    pf = pq.ParquetFile(parquet_path)
    cascades_filtered = []
    total_read = 0

    for batch in pf.iter_batches(batch_size=10000, columns=["cascade_id", "published", "events"]):
        batch_items = batch.to_pylist()
        total_read += len(batch_items)
        for c in batch_items:
            events = c["events"]
            n_obs = sum(1 for e in events if e["elapsed"] <= observation_time)
            if min_observed <= n_obs <= max_observed:
                n_pred = sum(1 for e in events if e["elapsed"] <= horizon_time)
                increment = max(0, n_pred - n_obs)
                pub_time = int(c["published"])
                cascades_filtered.append({
                    "cascade_id": str(c["cascade_id"]),
                    "publish_time": pub_time,
                    "observed_size": n_obs,
                    "total_15d": n_pred,
                    "increment": increment,
                    "events": events,
                })

    print(f"• Total cascades read: {total_read:,}")
    print(f"• Cascades surviving filter: {len(cascades_filtered):,} ({len(cascades_filtered)/total_read:.1%})")

    # Strictly chronological sort by publish_time (t_0)
    cascades_filtered.sort(key=lambda x: x["publish_time"])

    # 70% Train / 15% Val / 15% Test partition
    n_total = len(cascades_filtered)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val

    split_cascades = {
        "train": cascades_filtered[:n_train],
        "val": cascades_filtered[n_train : n_train + n_val],
        "test": cascades_filtered[n_train + n_val :],
    }

    print(f"• Partitions: Train={len(split_cascades['train']):,}, Val={len(split_cascades['val']):,}, Test={len(split_cascades['test']):,}")

    # Step 1: Sample walks for all cascades
    print("• Generating random walks for each cascade graph...")
    raw_walks = {}
    for split_name, c_list in split_cascades.items():
        walks_list = []
        for c in c_list:
            w = sample_cascade_walks(
                c["events"],
                t_obs=observation_time,
                n_walks=n_walks,
                walk_length=walk_length,
                rng=rng,
            )
            walks_list.append(w)
        raw_walks[split_name] = walks_list

    # Step 2: Build vocabulary from TRAIN set ONLY (to prevent test leakage)
    print("• Building node vocabulary from training split...")
    node_to_idx = {"<PAD>": 0, "<UNK>": 1}
    for w_list in raw_walks["train"]:
        for walk in w_list:
            for node in walk:
                if node not in node_to_idx:
                    node_to_idx[node] = len(node_to_idx)

    vocab_size = len(node_to_idx)
    print(f"• Vocabulary size (from train): {vocab_size:,} unique nodes (indices 0=PAD, 1=UNK)")

    # Save vocabulary
    with open(out_path / "vocab_meta.json", "w") as f:
        json.dump({
            "vocab_size": vocab_size,
            "pad_idx": 0,
            "unk_idx": 1,
            "n_walks": n_walks,
            "walk_length": walk_length,
            "t_obs": observation_time,
            "t_pred": horizon_time,
        }, f, indent=2)

    # Step 3: Encode walks to integer indices and save as .npz
    for split_name, c_list in split_cascades.items():
        w_list = raw_walks[split_name]
        encoded = np.zeros((len(c_list), n_walks, walk_length), dtype=np.int32)
        for i, c_walks in enumerate(w_list):
            for j, walk in enumerate(c_walks):
                for k, node in enumerate(walk):
                    encoded[i, j, k] = node_to_idx.get(node, 1)

        y_inc = np.asarray([c["increment"] for c in c_list], dtype=np.float32)
        y_tot = np.asarray([c["total_15d"] for c in c_list], dtype=np.float32)
        obs_sz = np.asarray([c["observed_size"] for c in c_list], dtype=np.float32)
        pub_t = np.asarray([c["publish_time"] for c in c_list], dtype=np.int64)
        c_ids = np.asarray([c["cascade_id"] for c in c_list], dtype=object)

        out_file = out_path / f"twitter_{split_name}.npz"
        np.savez_compressed(
            out_file,
            walks=encoded,
            y_increment=y_inc,
            y_total=y_tot,
            observed_size=obs_sz,
            publish_time=pub_t,
            cascade_id=c_ids,
        )
        print(f"  Saved {split_name} ({len(c_list):,} cascades) to {out_file.name}")
        log_inc = np.log2(y_inc + 1.0)
        print(f"    mean log2(inc+1)={log_inc.mean():.4f}, median inc={np.median(y_inc):.1f}, mean obs={obs_sz.mean():.1f}")

    print(f"\n• Successfully prepared all splits in {out_path}!")
    return {
        "n_train": len(split_cascades["train"]),
        "n_val": len(split_cascades["val"]),
        "n_test": len(split_cascades["test"]),
        "vocab_size": vocab_size,
    }


def main():
    parser = argparse.ArgumentParser(description="Parse Twitter cascades for DeepCas.")
    parser.add_argument("--input", default="cascades/twitter.parquet", help="Path to twitter.parquet")
    parser.add_argument("--output-dir", default="data_twitter/processed_1d", help="Output directory")
    parser.add_argument("--observation-time", type=int, default=86400, help="Observation window in seconds (default: 86400 for 1-day)")
    parser.add_argument("--horizon", type=int, default=1296000, help="Prediction horizon in seconds (default: 1296000 for 15-day)")
    parser.add_argument("--min-observed", type=int, default=10, help="Min observed retweets (default: 10)")
    parser.add_argument("--max-observed", type=int, default=1000, help="Max observed retweets (default: 1000)")
    parser.add_argument("--n-walks", type=int, default=32, help="Number of random walks per cascade (default: 32)")
    parser.add_argument("--walk-length", type=int, default=10, help="Length of each walk (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    process_twitter_dataset(
        parquet_path=args.input,
        output_dir=args.output_dir,
        observation_time=args.observation_time,
        horizon_time=args.horizon,
        min_observed=args.min_observed,
        max_observed=args.max_observed,
        n_walks=args.n_walks,
        walk_length=args.walk_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
