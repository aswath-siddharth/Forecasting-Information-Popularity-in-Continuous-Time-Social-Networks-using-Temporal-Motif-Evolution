"""Extract cascade graphs for CasSubTS, aligned to the *existing* pipeline_w splits.

CasSubTS consumes the cascade graph itself (nodes, directed edges, edge times),
not our per-event feature matrix, so the raw structure has to be re-extracted.

To make the comparison controlled rather than merely similar, this does NOT
re-derive the splits. It reads the cascade ids already stored in
``weibo_{train,val,test}.npz`` and emits graphs for exactly those cascades in
exactly that order, so both models see an identical train/val/test partition and
identical labels. Any difference in results is then attributable to the model.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def extract_graphs(
    weibo_txt: Path,
    wanted: dict[str, int],
    observation_time: int,
) -> dict[str, dict]:
    """Return {cascade_id: {edges, edge_times, n_nodes, node_first_time, node_depth}}."""
    out: dict[str, dict] = {}

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
                    observed.append((nodes, t_now))

            observed.sort(key=lambda it: it[1])

            node_id: dict[str, int] = {}
            first_time: dict[int, float] = {}
            depth: dict[int, int] = {}
            src, dst, ets = [], [], []

            for nodes, t_now in observed:
                for d, name in enumerate(nodes):
                    if name not in node_id:
                        nid = len(node_id)
                        node_id[name] = nid
                        first_time[nid] = float(t_now)
                        depth[nid] = d
                if len(nodes) < 2:
                    continue
                u, v = node_id[nodes[-2]], node_id[nodes[-1]]
                src.append(u); dst.append(v); ets.append(float(t_now))

            n = len(node_id)
            if n == 0 or not src:
                continue

            out[cid] = {
                "edge_src": np.asarray(src, dtype=np.int32),
                "edge_dst": np.asarray(dst, dtype=np.int32),
                "edge_time": np.asarray(ets, dtype=np.float32),
                "n_nodes": n,
                "node_first_time": np.asarray([first_time[i] for i in range(n)], dtype=np.float32),
                "node_depth": np.asarray([depth[i] for i in range(n)], dtype=np.float32),
            }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build CasSubTS graph inputs on the existing splits.")
    p.add_argument("--input", default="weibo_data/dataset_weibo.txt")
    p.add_argument("--splits-dir", required=True, help="Directory holding weibo_{train,val,test}.npz")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--observation-time", type=int, required=True)
    args = p.parse_args()

    splits_dir, out_dir = Path(args.splits_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    order: dict[str, list[str]] = {}
    labels: dict[str, np.ndarray] = {}
    wanted: dict[str, int] = {}
    for s in ("train", "val", "test"):
        d = np.load(splits_dir / f"weibo_{s}.npz", allow_pickle=True)
        ids = [str(x) for x in d["cascade_id"].tolist()]
        order[s] = ids
        labels[s] = d["y_increment"].astype(np.float32)
        for i in ids:
            wanted[i] = 1
    print(f"• Need graphs for {len(wanted):,} cascades (window {args.observation_time}s)", flush=True)

    graphs = extract_graphs(Path(args.input), wanted, args.observation_time)
    print(f"• Extracted {len(graphs):,}", flush=True)

    for s in ("train", "val", "test"):
        keep = [(cid, y) for cid, y in zip(order[s], labels[s]) if cid in graphs]
        missing = len(order[s]) - len(keep)
        payload = {
            "cascade_id": [c for c, _ in keep],
            "y_increment": np.asarray([y for _, y in keep], dtype=np.float32),
            "graphs": [graphs[c] for c, _ in keep],
        }
        with open(out_dir / f"cassubts_{s}.pkl", "wb") as fh:
            pickle.dump(payload, fh, protocol=4)
        sizes = np.asarray([g["n_nodes"] for g in payload["graphs"]])
        print(f"  {s:<6} {len(keep):>6,} cascades (dropped {missing})  "
              f"nodes mean={sizes.mean():.1f} max={sizes.max()}", flush=True)


if __name__ == "__main__":
    main()
