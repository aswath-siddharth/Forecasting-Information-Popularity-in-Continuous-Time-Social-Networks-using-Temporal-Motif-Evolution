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


def process_single_cascade(args):
    cid, y_inc, g = args
    edge_src = g['edge_src']
    edge_dst = g['edge_dst']
    edge_time = g['edge_time']

    # Filter events within observation window t < 3600s (all in cassubts_3600 are <= 3599s)
    # Cap sequence steps at MAX_STEPS (1 root event at t=0 + up to 99 retweet events)
    # Remap node IDs in order of appearance up to MAX_NODES
    node_map = {0: 0}
    u_list = [0]
    v_list = [0]
    m_list = [1.0]
    t_list = [0.0]
    seen_edges = {(0, 0)}

    for s, d, t in zip(edge_src, edge_dst, edge_time):
        if t > OBS_WINDOW:
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
    bin_arr = np.minimum(np.floor(t_arr / TIME_INTERVAL).astype(np.int16), N_TIME_INTERVAL - 1)

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


def preprocess_split(split_name, in_path, out_path):
    t0 = time.time()
    with open(in_path, 'rb') as f:
        data = pickle.load(f)

    cids = data['cascade_id']
    y_incs = data['y_increment']
    graphs = data['graphs']
    n_samples = len(cids)
    print(f"[{split_name}] Loaded {n_samples} cascades from {in_path}")

    args_iter = list(zip(cids, y_incs, graphs))
    results = []
    for idx, arg in enumerate(args_iter):
        results.append(process_single_cascade(arg))
        if (idx + 1) % 5000 == 0:
            print(f"  [{split_name}] Processed {idx + 1}/{n_samples} cascades ({time.time() - t0:.1f}s)")

    with open(out_path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    dt = time.time() - t0
    avg_nodes = np.mean([r['n_nodes'] for r in results])
    avg_steps = np.mean([r['n_steps'] for r in results])
    print(f"[{split_name}] Saved {len(results)} cascades to {out_path} in {dt:.2f}s (avg_nodes={avg_nodes:.1f}, avg_steps={avg_steps:.1f})")
    return results


if __name__ == '__main__':
    base_dir = '/Users/priya/Desktop/cascn_implement_in_weibo/data'
    out_dir = os.path.join(base_dir, 'cascn_preprocessed')
    os.makedirs(out_dir, exist_ok=True)

    for split in ['train', 'val', 'test']:
        in_p = os.path.join(base_dir, 'cassubts_3600', f'cassubts_{split}.pkl')
        out_p = os.path.join(out_dir, f'cascn_{split}.pkl')
        preprocess_split(split, in_p, out_p)

    # Fit Feature Scaler (Time Scaler) strictly on the 29,530 Training cascades ONLY (Protocol #5)
    with open(os.path.join(out_dir, 'cascn_train.pkl'), 'rb') as f:
        train_res = pickle.load(f)

    all_train_times = np.concatenate([r['time'] for r in train_res])
    all_train_sizes = np.array([r['n_steps'] for r in train_res], dtype=np.float32)
    scaler_stats = {
        'train_count': len(train_res),
        'time_mean': float(np.mean(all_train_times)),
        'time_std': float(np.std(all_train_times) + 1e-8),
        'size_mean': float(np.mean(all_train_sizes)),
        'size_std': float(np.std(all_train_sizes) + 1e-8),
    }
    with open(os.path.join(out_dir, 'train_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler_stats, f)
    print("Fit Feature Scaler strictly on Train split ONLY (29,530 cascades):", scaler_stats)
