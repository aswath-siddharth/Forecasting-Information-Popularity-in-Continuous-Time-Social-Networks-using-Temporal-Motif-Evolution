"""Parallel single-pass parser for the novel Reddit JSON dataset (data/).

Streams all subreddit JSON files under ``data/`` across multiple CPU workers and emits
100%-aligned chronological 70/15/15 splits for BOTH:
  - CasMotif (15-D causal temporal motif event sequences evaluated strictly on G_{t-},
    reusing ``CascadeGraph`` and ``detect_motifs`` from ``pipeline_w.weibo_parser``)
  - DeepHawkes (observed propagation paths + train-only user vocabulary, zero leakage)

Protocol alignment (identical to the Sina Weibo controlled benchmark):
  - Keep posts with ``min_observed <= observed_size <= max_observed`` (default 10..1000,
    where ``observed_size`` includes the root post plus comments with ``0 <= dt < T_obs``).
  - Target: future 24-hour increment ``Delta P = R_24h - R_{T_obs}``.
  - Split: strictly chronological by ``post_time`` (t_0), 70% train / 15% val / 15% test.
  - DeepHawkes user vocabulary: built strictly on the training split (``0=PAD``, ``1=UNK``,
    users with training count >= ``min_user_freq`` get dedicated embeddings).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from deephawkes.data import UNK_ID
from pipeline_w.weibo_parser import (
    FEATURE_DIM,
    FEATURE_NAMES,
    REACTIVATION_IDLE_SECONDS,
    CascadeGraph,
    detect_motifs,
)

SPECIAL_ACCOUNTS = {"[deleted]", "[removed]", "automoderator", ""}


def _is_valid_user(u: Any) -> bool:
    if not isinstance(u, str):
        return False
    return u.strip().lower() not in SPECIAL_ACCOUNTS


def _flatten_tree(
    comments: list[dict[str, Any]],
    ancestor_path: list[str],
    post_time: float,
    observation_time: int,
    horizon_seconds: int,
    max_path_len: int,
    observed: list[tuple[list[str], float]],
    total_24h_box: list[int],
) -> None:
    """Iterative-safe recursive walk extracting (path, dt) pairs from nested Reddit replies."""
    for c in comments:
        if not isinstance(c, dict):
            continue
        author = c.get("user")
        ctime = c.get("time")
        replies = c.get("replies")

        next_path = ancestor_path
        if _is_valid_user(author) and ctime is not None:
            try:
                dt = float(ctime) - post_time
            except (TypeError, ValueError):
                dt = -1.0
            if dt >= 0.0:
                full_path = ancestor_path + [author.strip()]
                next_path = full_path
                if dt < horizon_seconds:
                    total_24h_box[0] += 1
                if dt < observation_time:
                    dh_path = full_path[-max_path_len:] if max_path_len > 0 and len(full_path) > max_path_len else full_path
                    observed.append((dh_path, dt))

        if replies and isinstance(replies, list):
            _flatten_tree(
                replies, next_path, post_time, observation_time,
                horizon_seconds, max_path_len, observed, total_24h_box,
            )


def _extract_casmotif_features(observed: list[tuple[list[str], float]]) -> np.ndarray | None:
    """Compute exact 15-D causal motif features on G_{t-} matching pipeline_w.weibo_parser."""
    graph = CascadeGraph()
    rows: list[list[float]] = []
    t_prev = 0.0

    for idx, (nodes, t_f) in enumerate(observed):
        if len(nodes) < 2:
            root_id = graph.node_id(nodes[0])
            graph.node_depth.setdefault(root_id, 0)
            graph.node_history.setdefault(root_id, []).append(t_f)
            t_prev = t_f
            continue

        src_name, dst_name = nodes[-2], nodes[-1]
        u = graph.node_id(src_name)
        v = graph.node_id(dst_name)
        v_depth = len(nodes) - 1
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
            *motifs,                        # 0-4: single, star, chain, dyad, cycle
            float(v_depth),                 # 5: tree_depth
            parent_out_degree,              # 6: parent_out_degree
            float(len(graph.node_to_id)),   # 7: node_count
            t_f,                            # 8: t_from_root
            i_reactivate,                   # 9: i_reactivate
            t_idle,                         # 10: t_idle
            descendant_count,               # 11: descendant_count
            dt_motif,                       # 12: dt_motif
            float(idx),                     # 13: cumulative_prior
            max(0.0, t_f - t_prev),         # 14: dt_edge
        ])
        t_prev = t_f

    if not rows:
        return None
    return np.asarray(rows, dtype=np.float32)


def _process_file_chunk(task: tuple[list[str], int, int, int, int, int]) -> tuple[list[tuple], int]:
    """Worker function: parse a batch of subreddit JSON files."""
    file_paths, obs_time, horizon, min_obs, max_obs, max_path_len = task
    kept: list[tuple] = []
    posts_seen = 0

    for fp_str in file_paths:
        fp = Path(fp_str)
        subreddit = fp.stem
        category = fp.parent.name
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                posts = json.load(fh)
        except Exception:
            continue
        if not isinstance(posts, list):
            continue

        for p_idx, post in enumerate(posts):
            posts_seen += 1
            if not isinstance(post, dict):
                continue
            post_user = post.get("post_user")
            post_time_raw = post.get("post_time")
            if not _is_valid_user(post_user) or post_time_raw is None:
                continue
            try:
                post_time = float(post_time_raw)
            except (TypeError, ValueError):
                continue
            if post_time <= 0:
                continue

            root_u = post_user.strip()
            observed: list[tuple[list[str], float]] = [([root_u], 0.0)]
            total_24h_box = [1]  # includes root post, matching Weibo convention

            comments = post.get("comments")
            if comments and isinstance(comments, list):
                _flatten_tree(
                    comments, [root_u], post_time, obs_time,
                    horizon, max_path_len, observed, total_24h_box,
                )

            obs_size = len(observed)
            if obs_size < min_obs or obs_size > max_obs:
                continue

            observed.sort(key=lambda it: it[1])
            feats = _extract_casmotif_features(observed)
            if feats is None:
                continue

            total_24h = max(total_24h_box[0], obs_size)
            increment = total_24h - obs_size
            cid = f"{category}/{subreddit}_{p_idx}_{int(post_time)}"
            paths = [p for p, _ in observed]
            times = [t for _, t in observed]

            kept.append((post_time, cid, feats, paths, times, obs_size, total_24h, increment))

    return kept, posts_seen


def _save_casmotif_split(path: Path, records: list[tuple]) -> dict[str, Any]:
    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    for i, r in enumerate(records):
        offsets[i + 1] = offsets[i] + r[2].shape[0]

    X_flat = np.empty((int(offsets[-1]), FEATURE_DIM), dtype=np.float32)
    y_inc = np.empty(len(records), dtype=np.float32)
    y_tot = np.empty(len(records), dtype=np.float32)
    obs_sz = np.empty(len(records), dtype=np.int32)
    pub_t = np.empty(len(records), dtype=np.int64)
    cids: list[str] = []

    for i, r in enumerate(records):
        post_time, cid, feats, _paths, _times, obs_size, total_24h, inc = r
        X_flat[offsets[i] : offsets[i + 1]] = feats
        y_inc[i] = float(inc)
        y_tot[i] = float(total_24h)
        obs_sz[i] = int(obs_size)
        pub_t[i] = int(post_time)
        cids.append(cid)

    np.savez(
        path,
        X_flat=X_flat,
        offsets=offsets,
        y_increment=y_inc,
        y_total=y_tot,
        observed_size=obs_sz,
        publish_time=pub_t,
        cascade_id=np.asarray(cids),
    )

    motif_rates = {FEATURE_NAMES[j]: float(X_flat[:, j].mean()) for j in range(5)} if len(X_flat) > 0 else {}
    return {
        "n_cascades": len(records),
        "n_events": int(len(X_flat)),
        "mean_events_per_cascade": float(np.diff(offsets).mean()) if records else 0.0,
        "max_events_per_cascade": int(np.diff(offsets).max()) if records else 0,
        "increment_mean": float(y_inc.mean()) if records else 0.0,
        "increment_median": float(np.median(y_inc)) if records else 0.0,
        "increment_max": float(y_inc.max()) if records else 0.0,
        "motif_rates": motif_rates,
    }


def _save_deephawkes_split(
    path: Path,
    records: list[tuple],
    vocab: dict[str, int],
    observation_time: int,
    n_time_intervals: int,
) -> dict[str, Any]:
    cascade_path_offsets = np.zeros(len(records) + 1, dtype=np.int64)
    path_node_offsets_list: list[int] = [0]
    node_ids_list: list[int] = []
    path_time_bins_list: list[int] = []
    path_times_list: list[float] = []
    y_inc = np.empty(len(records), dtype=np.float32)
    cids: list[str] = []

    total_tokens = 0
    known_tokens = 0
    total_paths = 0
    max_path_len = 0

    for i, r in enumerate(records):
        _post_time, cid, _feats, paths, times, _obs_size, _total_24h, inc = r
        cids.append(cid)
        y_inc[i] = float(inc)
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

            bin_idx = int((t / max(float(observation_time), 1.0)) * n_time_intervals)
            if bin_idx < 0:
                bin_idx = 0
            elif bin_idx >= n_time_intervals:
                bin_idx = n_time_intervals - 1
            path_time_bins_list.append(bin_idx)
            path_times_list.append(t)
            total_paths += 1
        cascade_path_offsets[i + 1] = total_paths

    np.savez(
        path,
        cascade_id=np.asarray(cids),
        y_increment=y_inc,
        cascade_path_offsets=cascade_path_offsets,
        path_node_offsets=np.asarray(path_node_offsets_list, dtype=np.int64),
        node_ids=np.asarray(node_ids_list, dtype=np.int32),
        path_time_bins=np.asarray(path_time_bins_list, dtype=np.int16),
        path_times=np.asarray(path_times_list, dtype=np.float32),
    )

    return {
        "n_cascades": len(records),
        "n_paths": total_paths,
        "mean_paths_per_cascade": total_paths / max(len(records), 1),
        "max_path_len": max_path_len,
        "total_tokens": total_tokens,
        "known_token_fraction": known_tokens / max(total_tokens, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build aligned CasMotif + DeepHawkes splits from data/.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--cache-root", default="reddit_benchmark/cache")
    p.add_argument("--observation-time", type=int, required=True, help="3600 (1h) or 900 (15m).")
    p.add_argument("--horizon-seconds", type=int, default=86400)
    p.add_argument("--min-observed", type=int, default=10)
    p.add_argument("--max-observed", type=int, default=1000)
    p.add_argument("--n-time-intervals", type=int, default=6)
    p.add_argument("--max-path-len", type=int, default=10)
    p.add_argument("--min-user-freq", type=int, default=2)
    p.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 4) // 2)))
    p.add_argument("--chunk-size", type=int, default=25)
    p.add_argument("--max-files", type=int, default=0, help="Optional file cap for quick tests (0 = all).")
    args = p.parse_args()

    t0 = time.time()
    files = sorted(str(f) for f in Path(args.data_dir).rglob("*.json"))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"No JSON files found under {args.data_dir}")

    print("=" * 76)
    print(f" REDDIT PARALLEL PARSER | window={args.observation_time}s | files={len(files):,} | workers={args.workers}")
    print("=" * 76, flush=True)

    chunks = [
        (files[i : i + args.chunk_size], args.observation_time, args.horizon_seconds,
         args.min_observed, args.max_observed, args.max_path_len)
        for i in range(0, len(files), args.chunk_size)
    ]

    all_records: list[tuple] = []
    total_posts = 0
    done_files = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for kept_chunk, n_posts in pool.map(_process_file_chunk, chunks):
            all_records.extend(kept_chunk)
            total_posts += n_posts
            done_files += args.chunk_size
            if done_files % 500 == 0 or done_files >= len(files):
                print(
                    f"  ... {min(done_files, len(files)):,}/{len(files):,} files | "
                    f"posts={total_posts:,} | kept={len(all_records):,} ({time.time()-t0:.1f}s)",
                    flush=True,
                )

    if not all_records:
        raise SystemExit("No cascades passed the filter criteria.")

    # Strictly chronological split by post_time (70 / 15 / 15)
    all_records.sort(key=lambda r: (r[0], r[1]))
    n_total = len(all_records)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    splits = {
        "train": all_records[:n_train],
        "val": all_records[n_train : n_train + n_val],
        "test": all_records[n_train + n_val :],
    }

    print(f"\n• Chronological 70/15/15 split of {n_total:,} cascades: "
          f"train={len(splits['train']):,}  val={len(splits['val']):,}  test={len(splits['test']):,}", flush=True)

    # Build DeepHawkes user vocabulary strictly from the training split ONLY
    user_counts: Counter[str] = Counter()
    for r in splits["train"]:
        for path in r[3]:
            user_counts.update(path)
    vocab: dict[str, int] = {}
    for user, cnt in user_counts.items():
        if cnt >= args.min_user_freq:
            vocab[user] = len(vocab) + 2  # 0=PAD, 1=UNK

    vocab_stats = {
        "unique_train_users": len(user_counts),
        "vocab_users_kept": len(vocab),
        "vocab_size_with_special": len(vocab) + 2,
        "min_user_freq": args.min_user_freq,
        "singleton_or_rare_train_users": len(user_counts) - len(vocab),
    }
    print(f"• Train-only user vocab: {vocab_stats['vocab_size_with_special']:,} "
          f"({vocab_stats['vocab_users_kept']:,} users with freq >= {args.min_user_freq} + PAD/UNK; "
          f"{vocab_stats['unique_train_users']:,} unique train users)", flush=True)

    cache_root = Path(args.cache_root)
    cm_dir = cache_root / f"casmotif_{args.observation_time}"
    dh_dir = cache_root / f"deephawkes_{args.observation_time}"
    cm_dir.mkdir(parents=True, exist_ok=True)
    dh_dir.mkdir(parents=True, exist_ok=True)

    cm_stats, dh_stats = {}, {}
    for s in ("train", "val", "test"):
        cm_stats[s] = _save_casmotif_split(cm_dir / f"reddit_{s}.npz", splits[s])
        dh_stats[s] = _save_deephawkes_split(
            dh_dir / f"deephawkes_{s}.npz", splits[s], vocab,
            args.observation_time, args.n_time_intervals,
        )
        mr = cm_stats[s]["motif_rates"]
        print(
            f"  {s:<5} cascades={cm_stats[s]['n_cascades']:,}  "
            f"mean_ev={cm_stats[s]['mean_events_per_cascade']:.1f}  "
            f"inc_med={cm_stats[s]['increment_median']:.1f}  "
            f"known_tok={dh_stats[s]['known_token_fraction']:.1%}  "
            f"motifs(star={mr.get('motif_star',0):.1%}, chain={mr.get('motif_chain',0):.1%}, "
            f"dyad={mr.get('motif_dyad',0):.1%}, cycle={mr.get('motif_cycle',0):.1%})",
            flush=True,
        )

    dh_meta = {
        "dataset": "Novel Multi-Subreddit Reddit Cascade Dataset (data/)",
        "observation_time": args.observation_time,
        "horizon_seconds": args.horizon_seconds,
        "n_time_intervals": args.n_time_intervals,
        "max_path_len": args.max_path_len,
        "total_posts_scanned": total_posts,
        "total_cascades_kept": n_total,
        "vocab": vocab_stats,
        "splits": dh_stats,
        "casmotif_splits": cm_stats,
    }
    (dh_dir / "metadata.json").write_text(json.dumps(dh_meta, indent=2), encoding="utf-8")
    (cm_dir / "metadata.json").write_text(json.dumps(dh_meta, indent=2), encoding="utf-8")
    print(f"\n• Saved aligned Reddit caches in {time.time()-t0:.1f}s -> {cm_dir} & {dh_dir}", flush=True)


if __name__ == "__main__":
    main()
