import os
import time
import math
import pickle
import numpy as np
from multiprocessing import Pool, cpu_count

MAX_NODES = 100
MAX_STEPS = 100
OBS_WINDOW = 3600.0
N_TIME_INTERVAL = 6
TIME_INTERVAL = math.ceil(OBS_WINDOW / N_TIME_INTERVAL)  # 600.0

def compute_scaled_directed_laplacian(n, src, dst, alpha=0.95):
    """
    Exact implementation of CasCN's caslaplacian.calculate_scaled_laplacian_dir(nx_G, lambda_max=None)
    for a directed graph with n nodes, root self-loop (0, 0), and edges (src, dst).
    """
    A = np.zeros((n, n), dtype=np.float64)
    A[0, 0] = 1.0  # walk_time == 0 adds self-loop on root node 0 in CasCN preprocess_graph_signal.py
    if len(src) > 0:
        A[src, dst] = 1.0

    if n == 1:
        # For a 1-node graph with self-loop, P = [[1]], I - P = [[0]], scaled L = -I = [[-1]]
        return np.array([[-1.0]], dtype=np.float32)

    row_sum = A.sum(axis=1)
    dangling = (row_sum == 0)
    M = A.copy()
    if np.any(dangling):
        M[dangling, :] = 1.0 / n
        row_sum = M.sum(axis=1)
    M = M / row_sum[:, None]

    P = alpha * M + (1.0 - alpha) / n

    # Stationary distribution via power iteration (exact, deterministic, fast for n <= 100)
    p = np.full(n, 1.0 / n, dtype=np.float64)
    PT = P.T
    for _ in range(200):
        p_next = PT @ p
        p_next = p_next / p_next.sum()
        if np.max(np.abs(p_next - p)) < 1e-9:
            p = p_next
            break
        p = p_next

    p = np.maximum(p, 1e-12)
    p = p / p.sum()
    sqrtp = np.sqrt(p)
    I = np.eye(n, dtype=np.float64)
    Q = (sqrtp[:, None] * (I - P)) / sqrtp[None, :]

    # Symmetric part of Q: use Gershgorin circle theorem or 15-step power iteration for largest magnitude eigenvalue
    Q_sym = 0.5 * (Q + Q.T)
    if n <= 20:
        eigvals = np.linalg.eigvalsh(Q_sym)
        lambda_max = float(np.max(np.abs(eigvals)))
    else:
        v = np.ones(n, dtype=np.float64) / np.sqrt(n)
        for _ in range(15):
            v_next = Q_sym @ v
            nrm = np.linalg.norm(v_next)
            if nrm < 1e-9:
                break
            v = v_next / nrm
        lambda_max = float(abs(v @ (Q_sym @ v)))
    if lambda_max < 1e-6:
        lambda_max = 2.0

    L_scaled = (2.0 / lambda_max) * Q - I
    return L_scaled.astype(np.float32)


import argparse
from pathlib import Path


def process_single_cascade(args, obs_window=3600.0, time_interval=600.0):
    cid, y_inc, g = args
    edge_src = g['edge_src']
    edge_dst = g['edge_dst']
    edge_time = g['edge_time']

    # Filter events within observation window t < obs_window
    # Cap sequence steps at MAX_STEPS (1 root event at t=0 + up to 99 retweet events)
    # Remap node IDs in order of appearance up to MAX_NODES
    node_map = {0: 0}
    u_list = [0]
    v_list = [0]
    m_list = [1.0]
    t_list = [0.0]
    seen_edges = {(0, 0)}

    for s, d, t in zip(edge_src, edge_dst, edge_time):
        if t > obs_window:
            continue
        if len(u_list) >= MAX_STEPS:
            break
        s_int = int(s)
        d_int = int(d)
        if s_int not in node_map:
            if len(node_map) < MAX_NODES:
                node_map[s_int] = len(node_map)
            else:
                continue
        if d_int not in node_map:
            if len(node_map) < MAX_NODES:
                node_map[d_int] = len(node_map)
            else:
                continue
        u_mapped = node_map[s_int]
        v_mapped = node_map[d_int]
        if (u_mapped, v_mapped) not in seen_edges:
            seen_edges.add((u_mapped, v_mapped))
            m_list.append(1.0)
        else:
            m_list.append(0.0)
        u_list.append(u_mapped)
        v_list.append(v_mapped)
        t_list.append(float(t))

    n_nodes = len(node_map)
    n_steps = len(u_list)
    u_arr = np.array(u_list, dtype=np.int16)
    v_arr = np.array(v_list, dtype=np.int16)
    m_arr = np.array(m_list, dtype=np.float32)
    t_arr = np.array(t_list, dtype=np.float32)
    # Time interval bin in [0, N_TIME_INTERVAL - 1]
    bin_arr = np.minimum(np.floor(t_arr / time_interval).astype(np.int16), N_TIME_INTERVAL - 1)

    L = compute_scaled_directed_laplacian(n_nodes, u_arr, v_arr)
    y_log2 = np.float32(np.log2(float(y_inc) + 1.0))

    return {
        'cascade_id': str(cid),
        'n_nodes': np.int16(n_nodes),
        'n_steps': np.int16(n_steps),
        'u': u_arr,
        'v': v_arr,
        'm': m_arr,
        'time': t_arr,
        'time_bin': bin_arr,
        'L': L,
        'y_inc': np.float32(y_inc),
        'y_log2': y_log2,
    }


def extract_raw_edges(weibo_txt: Path, wanted_cids: set[str], obs_window: float) -> dict[str, tuple[list[int], list[int], list[float]]]:
    """Return {cascade_id: (edge_src, edge_dst, edge_time)} for wanted cascades."""
    edges: dict[str, tuple[list[int], list[int], list[float]]] = {}
    with open(weibo_txt, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 5:
                continue
            cid = parts[0]
            if cid not in wanted_cids:
                continue
            srcs, dsts, times = [], [], []
            for item in parts[4].split():
                if not item:
                    continue
                chain, _, t_str = item.rpartition(':')
                if not chain:
                    continue
                nodes = chain.split('/')
                if any(n == '-1' for n in nodes):
                    continue
                try:
                    t_val = float(t_str)
                except ValueError:
                    continue
                if t_val > obs_window:
                    continue
                if len(nodes) > 1:
                    srcs.append(int(nodes[-2]))
                    dsts.append(int(nodes[-1]))
                    times.append(t_val)
            if len(srcs) > 1:
                order = np.argsort(times)
                srcs = [srcs[i] for i in order]
                dsts = [dsts[i] for i in order]
                times = [times[i] for i in order]
            edges[cid] = (srcs, dsts, times)
    return edges


def main():
    parser = argparse.ArgumentParser(description="Extract CasCN graph data aligned to Weibo splits.")
    parser.add_argument("--input", default="weibo_data/dataset_weibo.txt")
    parser.add_argument("--splits-dir", default="weibo_data/processed_7200")
    parser.add_argument("--output-dir", default="weibo_data/cascn_7200")
    parser.add_argument("--observation-time", type=float, default=7200.0)
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_window = float(args.observation_time)
    time_interval = obs_window / N_TIME_INTERVAL

    splits = {}
    wanted_cids = set()
    for s in ['train', 'val', 'test']:
        d = np.load(splits_dir / f"weibo_{s}.npz", allow_pickle=True)
        cids = [str(c) for c in d['cascade_id'].tolist()]
        y_incs = d['y_increment'].astype(np.float32)
        splits[s] = (cids, y_incs)
        wanted_cids.update(cids)

    print(f"• Loaded splits: train={len(splits['train'][0])}, val={len(splits['val'][0])}, test={len(splits['test'][0])}")
    print(f"• Extracting cascade edges from {args.input} (window={obs_window}s)...")
    raw_edges = extract_raw_edges(Path(args.input), wanted_cids, obs_window)
    print(f"• Extracted edges for {len(raw_edges)}/{len(wanted_cids)} cascades")

    train_res = None
    for s in ['train', 'val', 'test']:
        cids, y_incs = splits[s]
        results = []
        t0 = time.time()
        for idx, (cid, y_inc) in enumerate(zip(cids, y_incs)):
            srcs, dsts, times = raw_edges.get(cid, ([], [], []))
            g = {'edge_src': srcs, 'edge_dst': dsts, 'edge_time': times}
            res = process_single_cascade((cid, y_inc, g), obs_window=obs_window, time_interval=time_interval)
            results.append(res)
            if (idx + 1) % 10000 == 0:
                print(f"  [{s}] Processed {idx + 1}/{len(cids)} cascades ({time.time() - t0:.1f}s)")

        out_p = out_dir / f"cascn_{s}.pkl"
        with open(out_p, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"• Saved {s} ({len(results)} cascades) -> {out_p} ({time.time() - t0:.1f}s)")
        if s == 'train':
            train_res = results

    # Fit Feature Scaler (Time Scaler) strictly on the Training cascades ONLY (zero leakage)
    all_train_times = np.concatenate([r['time'] for r in train_res])
    all_train_sizes = np.array([r['n_steps'] for r in train_res], dtype=np.float32)
    scaler_stats = {
        'train_count': len(train_res),
        'time_mean': float(np.mean(all_train_times)),
        'time_std': float(np.std(all_train_times) + 1e-8),
        'size_mean': float(np.mean(all_train_sizes)),
        'size_std': float(np.std(all_train_sizes) + 1e-8),
    }
    with open(out_dir / 'train_scaler.pkl', 'wb') as f:
        pickle.dump(scaler_stats, f)
    print("• Fit Feature Scaler strictly on Train split ONLY:", scaler_stats)


if __name__ == '__main__':
    main()
