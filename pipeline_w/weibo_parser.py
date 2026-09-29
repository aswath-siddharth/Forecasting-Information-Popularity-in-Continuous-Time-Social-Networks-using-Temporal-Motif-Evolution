"""Parse the DeepHawkes Sina Weibo cascade file into per-event temporal-motif features.

Dataset contract (dataset_weibo.txt), one cascade per line, tab separated:

    <cascade_id>\t<root_user>\t<publish_unixtime>\t<n_retweets>\t<path> <path> ...

where each ``<path>`` is ``u1/u2/.../uk:t`` -- the full propagation chain from the
root down to the retweeting user ``uk``, and ``t`` is seconds elapsed since publish.
The chain implies one directed edge ``u_{k-1} -> u_k`` at time ``t``; the k==1 entry
is the original post itself and carries no edge.

The observation-window, hour-of-day, malformed-node and cascade-size filters below
mirror ``gen_sequence/gen_shortestpath.py`` from the released DeepHawkes code so that
our numbers sit on the same cascade population as the published baselines. The only
intentional deviation is ``observation_time``, which we set to 900s (15 min) instead
of their 3600s to create the early-prediction setting.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Feature layout -- 15 columns, no sentiment (the released Weibo data carries no text)
# ---------------------------------------------------------------------------
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

# Columns that are unbounded counts/durations and need log1p + standardisation.
# Excluded: 0-4 (motif one-hot) and 9 (binary reactivation flag).
UNBOUNDED_COLS = (5, 6, 7, 8, 10, 11, 12, 13, 14)

CHINA_UTC_OFFSET_SECONDS = 8 * 3600
REACTIVATION_IDLE_SECONDS = 300.0
CHAIN_DEPTH_THRESHOLD = 3


@dataclass
class CascadeGraph:
    """Incrementally built directed retweet graph with the bookkeeping motif
    detection needs. Adjacency is kept both as per-node lists (for traversal) and
    as an edge set (for O(1) existence checks during dyad/cycle detection)."""

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

        # The propagation path gives us ground-truth depth, so set it directly
        # rather than deriving it incrementally as the Reddit pipeline must.
        if v not in self.node_depth or v_depth < self.node_depth[v]:
            self.node_depth[v] = v_depth
        self.node_depth.setdefault(u, max(0, v_depth - 1))

        if u != v and v not in self.parent_map:
            self.parent_map[v] = u

        # Credit every ancestor of u with one more descendant. The visited guard
        # makes this safe even if repeated users introduce a cycle in parent_map.
        curr: int | None = u
        visited: set[int] = set()
        while curr is not None and curr not in visited:
            visited.add(curr)
            self.descendant_counts[curr] = self.descendant_counts.get(curr, 0) + 1
            curr = self.parent_map.get(curr)

    def build_csr(self) -> tuple[np.ndarray, np.ndarray]:
        """Materialise the CSR buffers (indptr, indices) described in the spec.

        Motif detection above runs off the incremental adjacency because it must
        query a graph that is still growing; CSR is the compacted end state.
        """
        n = len(self.node_to_id)
        indptr = np.zeros(n + 1, dtype=np.int32)
        for nid in range(n):
            indptr[nid + 1] = indptr[nid] + len(self.adj.get(nid, []))
        indices = np.empty(int(indptr[-1]), dtype=np.int32)
        for nid in range(n):
            nbrs = self.adj.get(nid, [])
            if nbrs:
                indices[indptr[nid]: indptr[nid + 1]] = nbrs
        return indptr, indices


def detect_motifs(graph: CascadeGraph, u: int, v: int, parent_depth: int) -> list[float]:
    """Classify the motif that edge ``u -> v`` completes, evaluated causally on the
    graph state *before* the edge is inserted."""
    v_neighbors = graph.neighbors(v)

    # Reciprocal 3-cycle: u -> v closes A->B->C->A if some v -> w -> u exists.
    is_cycle = 1.0 if any((w, u) in graph.edge_set for w in v_neighbors) else 0.0
    # Ping-pong dyad: the reverse edge is already present.
    is_dyad = 1.0 if (v, u) in graph.edge_set else 0.0
    # Deep directed chain: this edge lands the child at depth >= K.
    is_chain = 1.0 if (parent_depth + 1) >= CHAIN_DEPTH_THRESHOLD else 0.0
    # Sibling star/fork: the parent already has a child, so this makes >= 2.
    is_star = 1.0 if len(graph.neighbors(u)) >= 1 else 0.0
    # Single direct reply: only when no higher-order motif fired.
    is_single = 1.0 if not (is_cycle or is_dyad or is_chain or is_star) else 0.0

    return [is_single, is_star, is_chain, is_dyad, is_cycle]


@dataclass
class ParsedCascade:
    cascade_id: str
    publish_time: int
    features: np.ndarray   # [K, FEATURE_DIM]
    observed_size: int     # DeepHawkes' len(observation_path), root entry included
    total_24h: int         # retweet entries with t < 24h, root entry included
    increment: int         # total_24h - observed_size  <- the published target


def parse_line(
    line: str,
    observation_time: int,
    horizon_seconds: int,
    min_observed: int,
    max_observed: int,
    hour_filter: bool,
) -> ParsedCascade | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 5:
        return None

    cascade_id, _root_user, publish_raw, _n_nodes, path_blob = parts
    try:
        publish_time = int(publish_raw)
    except ValueError:
        return None

    # DeepHawkes keeps only messages published between 08:00 and 18:59 Beijing time.
    # time.gmtime on a shifted epoch reproduces that without depending on the
    # machine's own TZ setting (their code used the ambient localtime in China).
    if hour_filter:
        hour = time.gmtime(publish_time + CHINA_UTC_OFFSET_SECONDS).tm_hour
        if hour <= 7 or hour >= 19:
            return None

    observed: list[tuple[list[str], int]] = []
    total_24h = 0
    for entry in path_blob.split(" "):
        if not entry:
            continue
        chain_str, _, t_str = entry.rpartition(":")
        if not chain_str:
            continue
        nodes = chain_str.split("/")
        if any(n == "-1" for n in nodes):   # malformed node, dropped upstream too
            continue
        try:
            t_now = int(t_str)
        except ValueError:
            continue
        if t_now < horizon_seconds:
            total_24h += 1
        if t_now < observation_time:
            observed.append((nodes, t_now))

    if len(observed) < min_observed or len(observed) > max_observed:
        return None

    observed.sort(key=lambda item: item[1])

    graph = CascadeGraph()
    rows: list[list[float]] = []
    t_prev = 0.0

    for idx, (nodes, t_now) in enumerate(observed):
        if len(nodes) < 2:
            # The root entry itself: register the author, emit no edge.
            root_id = graph.node_id(nodes[0])
            graph.node_depth.setdefault(root_id, 0)
            graph.node_history.setdefault(root_id, []).append(float(t_now))
            t_prev = float(t_now)
            continue

        src_name, dst_name = nodes[-2], nodes[-1]
        u = graph.node_id(src_name)
        v = graph.node_id(dst_name)
        v_depth = len(nodes) - 1
        parent_depth = v_depth - 1
        t_f = float(t_now)

        motifs = detect_motifs(graph, u, v, parent_depth)

        # Motif completion speed: elapsed time from the parent's first appearance.
        history_u = graph.node_history.get(u)
        dt_motif = t_f - min(history_u) if history_u else 0.0

        # Reactivation / dormancy.
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
            *motifs,                        # 0-4
            float(v_depth),                 # 5
            parent_out_degree,              # 6
            float(len(graph.node_to_id)),   # 7
            t_f,                            # 8
            i_reactivate,                   # 9
            t_idle,                         # 10
            descendant_count,               # 11
            dt_motif,                       # 12
            float(idx),                     # 13
            t_f - t_prev,                   # 14
        ])
        t_prev = t_f

    if not rows:
        return None

    graph.build_csr()   # compact the final adjacency, per the architecture spec

    observed_size = len(observed)
    return ParsedCascade(
        cascade_id=cascade_id,
        publish_time=publish_time,
        features=np.asarray(rows, dtype=np.float32),
        observed_size=observed_size,
        total_24h=total_24h,
        increment=max(0, total_24h - observed_size),
    )


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def save_split(path: Path, cascades: list[ParsedCascade]) -> None:
    """Store variable-length sequences flat with an offset index (CSR-style), so
    the whole split is three contiguous arrays instead of thousands of objects."""
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
        y_total=np.asarray([c.total_24h for c in cascades], dtype=np.float32),
        observed_size=np.asarray([c.observed_size for c in cascades], dtype=np.float32),
        publish_time=np.asarray([c.publish_time for c in cascades], dtype=np.int64),
        cascade_id=np.asarray([c.cascade_id for c in cascades], dtype=object),
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
    )


def build(args: argparse.Namespace) -> None:
    src = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kept: list[ParsedCascade] = []
    n_lines = 0
    n_malformed = 0

    print(f"• Reading {src} (observation window = {args.observation_time}s)", flush=True)
    with open(src, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n_lines += 1
            parsed = parse_line(
                line,
                observation_time=args.observation_time,
                horizon_seconds=args.horizon,
                min_observed=args.min_observed,
                max_observed=args.max_observed,
                hour_filter=not args.no_hour_filter,
            )
            if parsed is None:
                n_malformed += 1
            else:
                kept.append(parsed)
            if n_lines % 20000 == 0:
                print(f"  ... {n_lines:,} lines read, {len(kept):,} cascades kept", flush=True)

    if not kept:
        raise SystemExit(
            "No cascades survived filtering. With a 15-minute window the "
            f"'>= {args.min_observed} observed retweets' rule is strict -- try lowering "
            "--min-observed or check the input format."
        )

    # DeepHawkes splits chronologically by publish time (14/20, 3/20, 3/20), not
    # randomly. That ordering is also what keeps a cascade wholly inside one split.
    kept.sort(key=lambda c: c.publish_time)
    n = len(kept)
    n_train = int(n * 14 / 20)
    n_val = int(n * 17 / 20)
    splits = {
        "train": kept[:n_train],
        "val": kept[n_train:n_val],
        "test": kept[n_val:],
    }

    print(
        f"\n• Lines read        : {n_lines:,}\n"
        f"• Cascades kept     : {n:,}  ({n / max(n_lines, 1):.1%} of file)\n"
        f"• Filtered out      : {n_malformed:,}",
        flush=True,
    )
    for name, group in splits.items():
        if not group:
            continue
        inc = np.asarray([c.increment for c in group], dtype=np.float64)
        obs = np.asarray([c.observed_size for c in group], dtype=np.float64)
        lens = np.asarray([c.features.shape[0] for c in group], dtype=np.float64)
        save_split(out_dir / f"weibo_{name}.npz", group)
        print(
            f"  {name:<5} n={len(group):>6,}  events/cascade mean={lens.mean():6.1f} "
            f"max={int(lens.max()):>5}  observed mean={obs.mean():6.1f}  "
            f"increment mean={inc.mean():8.1f} median={np.median(inc):7.1f} max={int(inc.max()):,}",
            flush=True,
        )

    print(f"\n• Saved to {out_dir}/weibo_{{train,val,test}}.npz", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 15-minute Weibo cascade features from dataset_weibo.txt.")
    p.add_argument("--input", required=True, help="Path to dataset_weibo.txt")
    p.add_argument("--output-dir", default="weibo_data/processed")
    p.add_argument("--observation-time", type=int, default=900,
                   help="Observation window in seconds (default 900 = 15 min; DeepHawkes used 3600).")
    p.add_argument("--horizon", type=int, default=24 * 3600, help="Prediction horizon in seconds.")
    p.add_argument("--min-observed", type=int, default=10,
                   help="Minimum retweets inside the window (DeepHawkes: 10).")
    p.add_argument("--max-observed", type=int, default=1000,
                   help="Maximum retweets inside the window (DeepHawkes: 1000).")
    p.add_argument("--no-hour-filter", action="store_true",
                   help="Disable the DeepHawkes 08:00-18:59 publish-hour filter.")
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
