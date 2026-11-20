import os
import json
import time
import math
import random
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
try:
    from .model import CasCNModel
except ImportError:
    from cascn.model import CasCNModel

from pipeline_w.train_weibo import regression_metrics


def compute_full_metrics(pred_log2: np.ndarray, true_log2: np.ndarray, true_raw: np.ndarray) -> dict[str, float]:
    m = regression_metrics(pred_log2, true_log2, true_raw)
    nz = true_log2 > 0
    m["MAPE_log2"] = float(np.mean(np.abs((true_log2[nz] - pred_log2[nz]) / true_log2[nz]))) if np.any(nz) else 0.0
    pred_raw = np.clip(np.exp2(np.clip(pred_log2, -20, 20)) - 1.0, 0.0, None)
    pos = true_raw > 0
    m["MAPE_count"] = float(np.mean(np.abs((true_raw[pos] - pred_raw[pos]) / true_raw[pos]))) if np.any(pos) else 0.0
    m["MALE_log2"] = float(np.mean(np.abs(pred_log2 - true_log2)))
    return m

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_batches(dataset, batch_size, scaler, shuffle=True, rng=None):
    """
    Groups cascades into mini-batches of size `batch_size`.
    For training, we sort into size-similar buckets with random jitter so that
    each batch has tight (T_cur, N_cur) padding bounds while still shuffling across epochs.
    Every cascade in the split is included (no dropped cascades on Val/Test).
    """
    n = len(dataset)
    if shuffle:
        # Add random noise to step length to randomize bucket assignment each epoch,
        # then shuffle the order of batches.
        noise = rng.uniform(-8.0, 8.0, size=n)
        keys = np.array([d['n_steps'] for d in dataset], dtype=np.float32) + noise
        order = np.argsort(keys)
        batch_indices = [order[i:i + batch_size] for i in range(0, n, batch_size)]
        rng.shuffle(batch_indices)
    else:
        # Sort by n_steps for fast evaluation, preserving exact index mapping
        order = np.argsort([d['n_steps'] for d in dataset])
        batch_indices = [order[i:i + batch_size] for i in range(0, n, batch_size)]

    t_mean = scaler['time_mean']
    t_std = scaler['time_std']
    s_mean = scaler['size_mean']
    s_std = scaler['size_std']

    batches = []
    for idxs in batch_indices:
        B = len(idxs)
        items = [dataset[i] for i in idxs]
        T_cur = int(max(it['n_steps'] for it in items))
        N_cur = int(max(it['n_nodes'] for it in items))

        L_b = np.zeros((B, N_cur, N_cur), dtype=np.float32)
        u_b = np.zeros((B, T_cur), dtype=np.int64)
        v_b = np.zeros((B, T_cur), dtype=np.int64)
        m_b = np.zeros((B, T_cur), dtype=np.float32)
        tbin_b = np.zeros((B, T_cur), dtype=np.int64)
        mask_b = np.zeros((B, T_cur), dtype=np.float32)
        nnodes_b = np.zeros((B,), dtype=np.int64)
        tscaled_b = np.zeros((B, T_cur, 2), dtype=np.float32)
        y_b = np.zeros((B,), dtype=np.float32)

        for b_i, it in enumerate(items):
            nt = int(it['n_steps'])
            nn_i = int(it['n_nodes'])
            L_b[b_i, :nn_i, :nn_i] = it['L']
            u_b[b_i, :nt] = it['u']
            v_b[b_i, :nt] = it['v']
            m_b[b_i, :nt] = it['m']
            tbin_b[b_i, :nt] = it['time_bin']
            mask_b[b_i, :nt] = 1.0
            nnodes_b[b_i] = nn_i

            # Standardized time features using Train-only scaler
            t_norm = (it['time'] - t_mean) / t_std
            dt = np.diff(it['time'], prepend=0.0)
            dt_norm = (dt - (t_mean / max(s_mean, 1.0))) / (t_std + 1e-8)
            tscaled_b[b_i, :nt, 0] = t_norm
            tscaled_b[b_i, :nt, 1] = dt_norm

            y_b[b_i] = it['y_log2']

        batches.append({
            'idxs': idxs,
            'L': torch.from_numpy(L_b),
            'u': torch.from_numpy(u_b),
            'v': torch.from_numpy(v_b),
            'm': torch.from_numpy(m_b),
            'tbin': torch.from_numpy(tbin_b),
            'mask': torch.from_numpy(mask_b),
            'nnodes': torch.from_numpy(nnodes_b),
            'tscaled': torch.from_numpy(tscaled_b),
            'y': torch.from_numpy(y_b),
        })
    return batches


@torch.no_grad()
def evaluate(model, batches, n_total, device):
    model.eval()
    preds_all = np.zeros(n_total, dtype=np.float32)
    targets_all = np.zeros(n_total, dtype=np.float32)

    for b in batches:
        L = b['L'].to(device)
        u = b['u'].to(device)
        v = b['v'].to(device)
        m = b['m'].to(device)
        tbin = b['tbin'].to(device)
        mask = b['mask'].to(device)
        nnodes = b['nnodes'].to(device)
        tscaled = b['tscaled'].to(device)

        pred = model(L, u, v, m, tbin, mask, nnodes, tscaled)
        # Ensure non-negative prediction in log2(Delta P + 1) space
        pred = torch.clamp(pred, min=0.0)
        preds_all[b['idxs']] = pred.cpu().numpy()
        targets_all[b['idxs']] = b['y'].numpy()

    diffs = preds_all - targets_all
    msle = float(np.mean(diffs ** 2))
    male = float(np.mean(np.abs(diffs)))
    mape = float(np.mean(np.abs(diffs) / np.maximum(targets_all, 1.0)))
    return msle, male, mape, preds_all, targets_all


def train_single_seed(
    seed,
    train_data,
    val_data,
    test_data,
    scaler,
    out_dir,
    observation_window=7200,
    batch_size=64,
    lr=2e-3,
    weight_decay=1e-4,
    max_grad_norm=1.0,
    patience_es=8,
    max_epochs=60,
    lr_patience=3,
    lr_factor=0.5,
    device='cpu',
):
    t_seed_start = time.time()
    seed_dir = os.path.join(out_dir, f"s{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    ckpt_path = os.path.join(seed_dir, "best_cascn.pt")

    print(f"\n=======================================================")
    print(f"Starting CasCN Training | Seed = {seed} | Device = {device}")
    print(f"Protocol: Batch={batch_size}, Optimizer=AdamW(lr={lr}, wd={weight_decay}), Clip={max_grad_norm}, ES_Patience={patience_es}, MaxEpochs={max_epochs}")
    print(f"=======================================================", flush=True)

    set_seed(seed)
    rng = np.random.RandomState(seed)

    model = CasCNModel(
        max_nodes=100,
        max_steps=100,
        num_hidden=32,
        num_kernel=2,
        n_time_interval=6,
        n_hidden_dense1=32,
        n_hidden_dense2=16,
        stddev=0.01,
        use_scaled_time=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=lr_patience
    )
    criterion = nn.MSELoss()  # Applied on log2(Delta P + 1), which is exact MSLE in base 2

    val_batches = prepare_batches(val_data, batch_size, scaler, shuffle=False)
    test_batches = prepare_batches(test_data, batch_size, scaler, shuffle=False)

    best_val_msle = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        t_ep = time.time()
        model.train()
        train_batches = prepare_batches(train_data, batch_size, scaler, shuffle=True, rng=rng)

        total_loss = 0.0
        total_count = 0

        for step_i, b in enumerate(train_batches):
            L = b['L'].to(device)
            u = b['u'].to(device)
            v = b['v'].to(device)
            m = b['m'].to(device)
            tbin = b['tbin'].to(device)
            mask = b['mask'].to(device)
            nnodes = b['nnodes'].to(device)
            tscaled = b['tscaled'].to(device)
            y = b['y'].to(device)

            optimizer.zero_grad()
            pred = model(L, u, v, m, tbin, mask, nnodes, tscaled)
            loss = criterion(pred, y)
            loss.backward()

            # Gradient clipping: Max Norm = 1.0 (Protocol #4)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            bs = len(b['idxs'])
            total_loss += loss.item() * bs
            total_count += bs

        train_msle = total_loss / total_count
        val_msle, val_male, val_mape, _, _ = evaluate(model, val_batches, len(val_data), device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_msle)
        new_lr = optimizer.param_groups[0]['lr']

        improved = False
        if val_msle < best_val_msle - 1e-6:
            best_val_msle = val_msle
            best_epoch = epoch
            epochs_no_improve = 0
            improved = True
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1

        dt = time.time() - t_ep
        mark = "*" if improved else " "
        lr_msg = f" -> {new_lr:.6g}" if new_lr != current_lr else ""
        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} [{dt:5.1f}s] | "
            f"Train MSLE: {train_msle:.4f} | Val MSLE: {val_msle:.4f} (MALE: {val_male:.4f}) | "
            f"Best Val: {best_val_msle:.4f} (Ep {best_epoch}) | "
            f"LR: {current_lr:.6g}{lr_msg} | Wait: {epochs_no_improve}/{patience_es} {mark}",
            flush=True
        )

        history.append({
            'epoch': epoch,
            'train_msle': train_msle,
            'val_msle': val_msle,
            'val_male': val_male,
            'lr': current_lr,
            'time_sec': dt,
        })

        if epochs_no_improve >= patience_es:
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience_es} epochs).")
            break

    # Restore weights from the lowest validation MSLE checkpoint (Protocol #4)
    print(f"Restoring best checkpoint from Epoch {best_epoch} (Val MSLE = {best_val_msle:.4f})...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    val_msle, val_male, val_mape, val_preds, val_targets = evaluate(model, val_batches, len(val_data), device)
    test_msle, test_male, test_mape, test_preds, test_targets = evaluate(model, test_batches, len(test_data), device)

    val_raw = np.array([d['y_inc'] for d in val_data], dtype=np.float32)
    test_raw = np.array([d['y_inc'] for d in test_data], dtype=np.float32)

    val_m = compute_full_metrics(val_preds, val_targets, val_raw)
    test_m = compute_full_metrics(test_preds, test_targets, test_raw)
    elapsed_seed = time.time() - t_seed_start

    print(f"[Seed {seed} Final Results @ Best Val Checkpoint (Epoch {best_epoch})]", flush=True)
    print(f"  Validation -> MSLE (log2): {val_m['MSLE_log2']:.4f} | R2: {val_m['R2_log2']:.4f} | MAE: {val_m['MAE_count']:.2f}", flush=True)
    print(f"  Test       -> MSLE (log2): {test_m['MSLE_log2']:.4f} | R2: {test_m['R2_log2']:.4f} | MAE: {test_m['MAE_count']:.2f}", flush=True)
    print(f"[Seed {seed} DONE] Time: {elapsed_seed:.1f}s ({elapsed_seed/60:.2f} min) | Test MSLE (log2): {test_m['MSLE_log2']:.4f}", flush=True)

    np.savez_compressed(
        os.path.join(seed_dir, 'test_predictions.npz'),
        cascade_id=np.array([d['cascade_id'] for d in test_data]),
        pred_log2=test_preds.astype(np.float32),
        true_log2=test_targets.astype(np.float32),
        true_raw=test_raw.astype(np.float32),
    )

    pcounts = sum(p.numel() for p in model.parameters())
    report = {
        "model": "CasCN (Recurrent Cascades Convolution, ICDE 2019)",
        "seed": seed,
        "window_seconds": observation_window,
        "best_epoch": int(best_epoch),
        "best_val_msle_log2": float(val_m["MSLE_log2"]),
        "train_time_sec": elapsed_seed,
        "train_time_min": elapsed_seed / 60.0,
        "architecture": {
            "max_nodes": 100,
            "max_steps": 100,
            "num_hidden": 32,
            "chebyshev_K": 2,
            "n_time_interval": 6,
            "mlp_hidden": [32, 16, 1],
            "params": pcounts,
        },
        "val_metrics": val_m,
        "test_metrics": test_m,
    }
    with open(os.path.join(seed_dir, 'cascn_evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(seed_dir, 'training_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    return {
        'seed': seed,
        'best_epoch': best_epoch,
        'val_msle': val_m['MSLE_log2'],
        'test_msle': test_m['MSLE_log2'],
        'train_time_sec': elapsed_seed,
        'train_time_min': elapsed_seed / 60.0,
        'val_metrics': val_m,
        'test_metrics': test_m,
        'history': history,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='weibo_data/cascn_7200')
    parser.add_argument('--out_dir', type=str, default='cascn/runs_7200/cascn')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--observation_window', type=int, default=7200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=60)
    parser.add_argument('--lr_patience', type=int, default=3)
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading preprocessed Weibo cascades...")
    with open(os.path.join(args.data_dir, 'cascn_train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(args.data_dir, 'cascn_val.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    with open(os.path.join(args.data_dir, 'cascn_test.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    with open(os.path.join(args.data_dir, 'train_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    print(f"Dataset split verification: Train={len(train_data):,}, Val={len(val_data):,}, Test={len(test_data):,} (Total={len(train_data)+len(val_data)+len(test_data):,})")

    all_results = []
    for seed in args.seeds:
        res = train_single_seed(
            seed=seed,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            scaler=scaler,
            out_dir=args.out_dir,
            observation_window=args.observation_window,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            patience_es=args.patience,
            max_epochs=args.max_epochs,
            lr_patience=args.lr_patience,
            lr_factor=args.lr_factor,
            device=args.device,
        )
        all_results.append(res)

    existing_summary_path = os.path.join(args.out_dir, 'cascn_metrics_summary.json')
    merged_results = {}
    if os.path.exists(existing_summary_path):
        try:
            with open(existing_summary_path, 'r', encoding='utf-8') as f:
                old_summary = json.load(f)
                for r in old_summary.get('seeds_results', []):
                    merged_results[r['seed']] = r
        except Exception:
            pass

    for r in all_results:
        merged_results[r['seed']] = r

    combined_results = list(merged_results.values())
    combined_seeds = [r['seed'] for r in combined_results]
    val_msles = [r['val_msle'] for r in combined_results]
    test_msles = [r['test_msle'] for r in combined_results]

    summary = {
        'protocol': {
            'split_ratio': f'70% Train ({len(train_data)}) / 15% Val ({len(val_data)}) / 15% Test ({len(test_data)})',
            'split_strategy': 'Strictly Chronological by t0',
            'observation_window_sec': args.observation_window,
            'prediction_horizon_sec': 86400,
            'prediction_target': f'Delta P = R_24h - R_{args.observation_window}s',
            'loss_function': 'MSLE in log2 base',
            'batch_size': args.batch_size,
            'optimizer': f'AdamW (lr={args.lr}, weight_decay={args.weight_decay})',
            'lr_schedule': f'ReduceLROnPlateau (factor={args.lr_factor}, patience={args.lr_patience})',
            'grad_clip_max_norm': args.max_grad_norm,
            'early_stopping': f'Patience={args.patience} epochs (max {args.max_epochs} epochs)',
            'checkpoint': 'Best Validation MSLE Epoch',
            'seeds': combined_seeds,
            'feature_scaler': f'Fit on Train split ONLY ({len(train_data)} cascades)',
        },
        'val_msle_mean': float(np.mean(val_msles)),
        'val_msle_std': float(np.std(val_msles, ddof=1) if len(val_msles) > 1 else 0.0),
        'test_msle_mean': float(np.mean(test_msles)),
        'test_msle_std': float(np.std(test_msles, ddof=1) if len(test_msles) > 1 else 0.0),
        'seeds_results': combined_results,
    }

    with open(os.path.join(args.out_dir, 'cascn_metrics_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print("\n=======================================================", flush=True)
    print(f"FINAL MULTI-SEED SUMMARY (Seeds: {combined_seeds})", flush=True)
    print("=======================================================", flush=True)
    for r in combined_results:
        time_str = f" | Time: {r.get('train_time_min', 0):.2f} min" if 'train_time_min' in r else ""
        print(f"  Seed {r['seed']}: Best Ep = {r['best_epoch']:02d} | Val MSLE = {r['val_msle']:.4f} | Test MSLE = {r['test_msle']:.4f}{time_str}", flush=True)
    print("-------------------------------------------------------", flush=True)
    print(f"  Validation MSLE (log2): {summary['val_msle_mean']:.4f} ± {summary['val_msle_std']:.4f}", flush=True)
    print(f"  Test MSLE       (log2): {summary['test_msle_mean']:.4f} ± {summary['test_msle_std']:.4f}", flush=True)
    print("=======================================================", flush=True)
