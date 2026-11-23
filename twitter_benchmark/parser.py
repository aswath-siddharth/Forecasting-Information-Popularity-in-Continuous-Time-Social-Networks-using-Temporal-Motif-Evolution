"""Parse Twitter cascades into per-event temporal motif features for CasMotif
and raw diffusion paths for DeepHawkes.

Protocol specification (Twitter Cascade Benchmark):
  - Observation window: T_obs in {86400s (1 day), 172800s (2 days)}
  - Prediction horizon: T_pred = 15 days (1,296,000s)
  - Target: Future increment Delta P = R_{15d} - R_{T_obs}, log2(Delta P + 1)
  - Filtering: 10 <= N(T_obs) <= 1,000
  - Hour filter: Disabled (global Twitter activity across all time zones)
  - Split: Strictly chronological by publish time t_0 (70% train / 15% val / 15% test)
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# 15 Causal Motif Features
FEATURE_NAMES = (
    "motif_single",        # 0
    "motif_star",          # 1
    "motif_chain",         # 2
    "motif_dyad",          # 3
    "motif_cycle",         # 4
    "tree_depth",          # 5
    "parent_out_degree",   # 6
    "node_count",          # 7
    "t_from_root",         # 8
    "i_reactivate",        # 9
    "t_idle",              # 10
    "descendant_count",    # 11
    "dt_motif",            # 12
    "cumulative_prior",    # 13
    "dt_edge",             # 14
)
FEATURE_DIM = len(FEATURE_NAMES)
UNBOUNDED_COLS = (5, 6, 7, 8, 10, 11, 12, 13, 14)
REACTIVATION_IDLE_SECONDS = 300.0
CHAIN_DEPTH_THRESHOLD = 3


@dataclass
class CascadeGraph:
    """Directed retweet graph for causal motif detection."""

    node_to_id: dict[str, int] = field(default_factory=dict)
    adj: dict[int, list[int]] = field(default_factory=dict)
    edge_set: set[tuple[int, int]] = field(default_factory=set)
    node_history: dict[int, list[float]] = field(default_factory=dict)
    node_depth: dict[int, int] = field(default_factory=dict)
    parent_map: dict[int, int] = field(default_factory=dict)
    descendant_counts: dict[int, int] = field(default_factory=dict)

    def node_id(self, name: str) -> int:
        nid = self.node_to_id.get(name)
        if nid is None:
            nid = len(self.node_to_id)
            self.node_to_id[name] = nid
            self.adj[nid] = []
        return nid

    def neighbors(self, nid: int) -> list[int]:
        return self.adj.get(nid, [])

    def add_edge(self, u: int, v: int, t: float, v_depth: int) -> None:
        self.adj.setdefault(u, []).append(v)
        self.edge_set.add((u, v))
        self.node_history.setdefault(v, []).append(t)

        if v not in self.node_depth or v_depth < self.node_depth[v]:
            self.node_depth[v] = v_depth
        self.node_depth.setdefault(u, max(0, v_depth - 1))

        if u != v and v not in self.parent_map:
            self.parent_map[v] = u

        curr: int | None = u
        visited: set[int] = set()
        while curr is not None and curr not in visited:
            visited.add(curr)
            self.descendant_counts[curr] = self.descendant_counts.get(curr, 0) + 1
            curr = self.parent_map.get(curr)


def detect_motifs(graph: CascadeGraph, u: int, v: int, parent_depth: int) -> list[float]:
    v_neighbors = graph.neighbors(v)
    is_cycle = 1.0 if any((w, u) in graph.edge_set for w in v_neighbors) else 0.0
    is_dyad = 1.0 if (v, u) in graph.edge_set else 0.0
    is_chain = 1.0 if (parent_depth + 1) >= CHAIN_DEPTH_THRESHOLD else 0.0
    is_star = 1.0 if len(graph.neighbors(u)) >= 1 else 0.0
    is_single = 1.0 if not (is_cycle or is_dyad or is_chain or is_star) else 0.0
    return [is_single, is_star, is_chain, is_dyad, is_cycle]


@dataclass
class ParsedTwitterCascade:
    cascade_id: str
    publish_time: int
    features: np.ndarray       # [K, 15]
    raw_paths: list[list[str]] # diffusion paths for DeepHawkes
    raw_times: list[float]     # event timestamps for DeepHawkes
    observed_size: int
    total_15d: int
    increment: int             # Delta P = max(0, total_15d - observed_size)


def parse_single_cascade(
    row: dict,
    observation_time: int,
    horizon_seconds: int,
    min_observed: int,
    max_observed: int,
) -> ParsedTwitterCascade | None:
    events = row["events"]
    observed_events = [e for e in events if e["elapsed"] <= observation_time]
    n_obs = len(observed_events)
    if n_obs < min_observed or n_obs > max_observed:
        return None

    total_15d = sum(1 for e in events if e["elapsed"] <= horizon_seconds)
    observed_events.sort(key=lambda it: it["elapsed"])

    graph = CascadeGraph()
    root_id = graph.node_id("-1")
    graph.node_depth[root_id] = 0

    rows: list[list[float]] = []
    raw_paths: list[list[str]] = []
    raw_times: list[float] = []
    t_prev = 0.0

    for idx, e in enumerate(observed_events):
        path = [str(x) for x in e["path"]]
        t_f = float(e["elapsed"])
        raw_paths.append(path)
        raw_times.append(t_f)

        src_name = path[-2] if len(path) >= 2 else "-1"
        dst_name = path[-1]

        u = graph.node_id(src_name)
        v = graph.node_id(dst_name)
        v_depth = len(path) - 1
        parent_depth = v_depth - 1

        motifs = detect_motifs(graph, u, v, parent_depth)

        history_u = graph.node_history.get(u)
        dt_motif = t_f - min(history_u) if history_u else 0.0

        prior_v = [t for t in graph.node_history.get(v, []) if t < t_f]
        if prior_v:
            t_idle = t_f - max(prior_v)
            i_reactivate = 1.0 if t_idle > REACTIVATION_IDLE_SECONDS else 0.0
        else:
            t_idle = t_f
            i_reactivate = 0.0

        parent_out_degree = float(len(graph.neighbors(u)))
        descendant_count = float(graph.descendant_counts.get(u, 0))

        graph.add_edge(u, v, t_f, v_depth)

        rows.append([
            *motifs,
            float(v_depth),
            parent_out_degree,
            float(len(graph.node_to_id)),
            t_f,
            i_reactivate,
            t_idle,
            descendant_count,
            dt_motif,
            float(idx),
            t_f - t_prev,
        ])
        t_prev = t_f

    if not rows:
        return None

    publish_time = int(row["published"])
    return ParsedTwitterCascade(
        cascade_id=str(row["cascade_id"]),
        publish_time=publish_time,
        features=np.asarray(rows, dtype=np.float32),
        raw_paths=raw_paths,
        raw_times=raw_times,
        observed_size=n_obs,
        total_15d=total_15d,
        increment=max(0, total_15d - n_obs),
    )


def save_split_npz(path: Path, cascades: list[ParsedTwitterCascade]) -> None:
    offsets = np.zeros(len(cascades) + 1, dtype=np.int64)
    for i, c in enumerate(cascades):
        offsets[i + 1] = offsets[i] + c.features.shape[0]

    flat = np.empty((int(offsets[-1]), FEATURE_DIM), dtype=np.float32)
    for i, c in enumerate(cascades):
        flat[offsets[i]: offsets[i + 1]] = c.features

    np.savez_compressed(
        path,
        X_flat=flat,
        offsets=offsets,
        y_increment=np.asarray([c.increment for c in cascades], dtype=np.float32),
        y_total=np.asarray([c.total_15d for c in cascades], dtype=np.float32),
        observed_size=np.asarray([c.observed_size for c in cascades], dtype=np.float32),
        publish_time=np.asarray([c.publish_time for c in cascades], dtype=np.int64),
        cascade_id=np.asarray([c.cascade_id for c in cascades], dtype=object),
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
    )


def save_raw_paths_json(path: Path, cascades: list[ParsedTwitterCascade]) -> None:
    data = {
        c.cascade_id: {
            "paths": c.raw_paths,
            "times": c.raw_times,
            "y_increment": c.increment,
            "publish_time": c.publish_time,
        }
        for c in cascades
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def parse_twitter_dataset(
    input_path: Path,
    output_dir: Path,
    observation_time: int = 86400,
    horizon_seconds: int = 15 * 86400,
    min_observed: int = 10,
    max_observed: int = 1000,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"• Loading Twitter dataset from {input_path} (T_obs={observation_time}s, T_pred={horizon_seconds}s)...", flush=True)

    if input_path.suffix == ".zip":
        with zipfile.ZipFile(input_path) as z:
            f = io.BytesIO(z.read("cascades/twitter.parquet"))
            pf = pq.ParquetFile(f)
    else:
        pf = pq.ParquetFile(input_path)

    parsed_cascades: list[ParsedTwitterCascade] = []
    total_scanned = 0

    for batch in pf.iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            total_scanned += 1
            c = parse_single_cascade(
                row,
                observation_time=observation_time,
                horizon_seconds=horizon_seconds,
                min_observed=min_observed,
                max_observed=max_observed,
            )
            if c is not None:
                parsed_cascades.append(c)

        if total_scanned % 10000 == 0:
            print(f"  Scanned {total_scanned:,}/{pf.metadata.num_rows:,} cascades | Kept {len(parsed_cascades):,}...", flush=True)

    print(f"• Filtering complete: kept {len(parsed_cascades):,} cascades from {total_scanned:,} ({time.time()-t0:.1f}s)", flush=True)

    # Strictly chronological sort by publish time t_0
    parsed_cascades.sort(key=lambda c: c.publish_time)

    # 70% Train / 15% Val / 15% Test
    n = len(parsed_cascades)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    n_test = n - n_train - n_val

    train_cascades = parsed_cascades[:n_train]
    val_cascades = parsed_cascades[n_train: n_train + n_val]
    test_cascades = parsed_cascades[n_train + n_val:]

    print(f"• Partitioning {n:,} cascades into strictly chronological splits:")
    print(f"  Train: {len(train_cascades):,} (70.0%) [pub {train_cascades[0].publish_time} -> {train_cascades[-1].publish_time}]")
    print(f"  Val:   {len(val_cascades):,} (15.0%) [pub {val_cascades[0].publish_time} -> {val_cascades[-1].publish_time}]")
    print(f"  Test:  {len(test_cascades):,} (15.0%) [pub {test_cascades[0].publish_time} -> {test_cascades[-1].publish_time}]")

    assert train_cascades[-1].publish_time <= val_cascades[0].publish_time, "Train/Val chronological leak!"
    assert val_cascades[-1].publish_time <= test_cascades[0].publish_time, "Val/Test chronological leak!"

    for name, split in (("train", train_cascades), ("val", val_cascades), ("test", test_cascades)):
        npz_p = output_dir / f"twitter_{name}.npz"
        json_p = output_dir / f"raw_paths_{name}.json"
        save_split_npz(npz_p, split)
        save_raw_paths_json(json_p, split)
        print(f"  Saved {name}: {npz_p.name} ({npz_p.stat().st_size / 1e6:.1f} MB), {json_p.name}", flush=True)

    summary = {
        "dataset": "twitter",
        "observation_time_seconds": observation_time,
        "prediction_horizon_seconds": horizon_seconds,
        "min_observed": min_observed,
        "max_observed": max_observed,
        "total_filtered_cascades": n,
        "train_count": len(train_cascades),
        "val_count": len(val_cascades),
        "test_count": len(test_cascades),
        "target": "future increment Delta P = R_15d - R_{T_obs}, log2(Delta P + 1)",
        "feature_dim": FEATURE_DIM,
        "feature_names": list(FEATURE_NAMES),
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"• Done -> {output_dir} ({time.time()-t0:.1f}s)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Parse Twitter cascades into CasMotif & DeepHawkes formats.")
    p.add_argument("--input", type=Path, default=Path("twitter_data/twitter.parquet"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--observation-time", type=int, default=86400, help="Observation window in seconds (default 86400 = 1 day)")
    p.add_argument("--horizon", type=int, default=15 * 86400, help="Prediction horizon in seconds (default 15 days)")
    p.add_argument("--min-observed", type=int, default=10)
    p.add_argument("--max-observed", type=int, default=1000)
    args = p.parse_args()

    parse_twitter_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        observation_time=args.observation_time,
        horizon_seconds=args.horizon,
        min_observed=args.min_observed,
        max_observed=args.max_observed,
    )


if __name__ == "__main__":
    main()
