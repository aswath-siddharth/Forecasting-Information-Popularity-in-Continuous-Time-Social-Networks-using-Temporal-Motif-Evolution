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

# ==============================================================================
# CasCN Architecture (Chen et al., ICDE 2019: "Information Diffusion Prediction
# via Recurrent Cascades Convolution")
# Faithfully implemented in PyTorch with exact algebraic factorization for
# one-hot incremental cascade adjacency additions + strict benchmark protocol.
# ==============================================================================

class CasCNModel(nn.Module):
    def __init__(
        self,
        max_nodes=100,
        max_steps=100,
        num_hidden=32,
        num_kernel=2,
        n_time_interval=6,
        n_hidden_dense1=32,
        n_hidden_dense2=16,
        stddev=0.01,
        use_scaled_time=True,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.max_steps = max_steps
        self.num_hidden = num_hidden
        self.num_kernel = num_kernel
        self.n_time_interval = n_time_interval
        self.use_scaled_time = use_scaled_time

        # In CasCN (model_sparse_graph_signal.py):
        # Wzxt, Wixt, Wfxt, Woxt have shape [K * feat_in, feat_out] where feat_in = max_nodes (100), K = 2, feat_out = 32
        # Initialized with RandomUniform(-0.1, 0.1).
        # We store them combined as W_x of shape [K, max_nodes, 4 * num_hidden] so that
        # W_x[0] is the K=0 Chebyshev weight and W_x[1] is the K=1 Chebyshev weight.
        self.W_x = nn.Parameter(torch.empty(num_kernel, max_nodes, 4 * num_hidden))
        nn.init.uniform_(self.W_x, -0.1, 0.1)

        # Wzht, Wiht, Wfht, Woht have shape [K * feat_out, feat_out]
        self.W_h = nn.Parameter(torch.empty(num_kernel, num_hidden, 4 * num_hidden))
        nn.init.uniform_(self.W_h, -0.1, 0.1)

        # Gate biases bzt, bit, bft, bot of shape [feat_out] each
        self.b_gates = nn.Parameter(torch.zeros(4 * num_hidden))

        # Time decay weights of shape [n_time_interval]
        self.time_weight = nn.Parameter(torch.empty(n_time_interval))
        nn.init.normal_(self.time_weight, mean=0.0, std=stddev)

        # Optional continuous scaled time modulation (using StandardScaler fit ONLY on Train split)
        if use_scaled_time:
            self.time_scale_proj = nn.Linear(2, num_hidden)
            nn.init.normal_(self.time_scale_proj.weight, mean=0.0, std=stddev)
            nn.init.zeros_(self.time_scale_proj.bias)

        # Dense readout layers (dense1: 32 -> 32, dense2: 32 -> 16, out: 16 -> 1) with ReLU activation
        self.dense1 = nn.Linear(num_hidden, n_hidden_dense1)
        self.dense2 = nn.Linear(n_hidden_dense1, n_hidden_dense2)
        self.out_layer = nn.Linear(n_hidden_dense2, 1)

        for layer in [self.dense1, self.dense2]:
            nn.init.normal_(layer.weight, mean=0.0, std=stddev)
            nn.init.constant_(layer.bias, 0.1)
        nn.init.normal_(self.out_layer.weight, mean=0.0, std=stddev)
        nn.init.constant_(self.out_layer.bias, 0.1)

        self.act = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, L_batch, u_batch, v_batch, m_batch, time_bin_batch, rnn_mask_batch, n_nodes_batch, time_scaled_batch=None):
        """
        Args:
            L_batch: (B, N_cur, N_cur) scaled directed Laplacian matrices
            u_batch: (B, T_cur) source node indices in [0, N_cur - 1]
            v_batch: (B, T_cur) destination node indices in [0, N_cur - 1]
            m_batch: (B, T_cur) 1.0 if new edge at step t, 0.0 if duplicate/padding
            time_bin_batch: (B, T_cur) integer bin in [0, n_time_interval - 1]
            rnn_mask_batch: (B, T_cur) 1.0 if valid step < sz[id], 0.0 if padding
            n_nodes_batch: (B,) number of active nodes in each cascade
            time_scaled_batch: (B, T_cur, 2) standardized time features (fit on train split only)
        """
        B, T_cur = u_batch.shape
        N_cur = L_batch.shape[1]
        H = self.num_hidden
        device = L_batch.device

        W_x0 = self.W_x[0]  # (max_nodes, 4*H)
        W_x1 = self.W_x[1]  # (max_nodes, 4*H)
        W_h0 = self.W_h[0]  # (H, 4*H)
        W_h1 = self.W_h[1]  # (H, 4*H)

        h = torch.zeros(B, N_cur, H, device=device, dtype=L_batch.dtype)
        c = torch.zeros(B, N_cur, H, device=device, dtype=L_batch.dtype)

        # Analytic tracking of zero-padded nodes (N_cur .. max_nodes-1)
        h_pad = torch.zeros(1, H, device=device, dtype=L_batch.dtype)
        c_pad = torch.zeros(1, H, device=device, dtype=L_batch.dtype)
        num_pad_nodes = (self.max_nodes - n_nodes_batch).unsqueeze(1).to(L_batch.dtype)  # (B, 1)

        AW0 = torch.zeros(B, N_cur, 4 * H, device=device, dtype=L_batch.dtype)
        AW1 = torch.zeros(B, N_cur, 4 * H, device=device, dtype=L_batch.dtype)

        # Pre-gather W_x0[v] and W_x1[v] across all T_cur steps
        w0_all = W_x0[v_batch] * m_batch.unsqueeze(-1)  # (B, T_cur, 4*H)
        w1_all = W_x1[v_batch] * m_batch.unsqueeze(-1)  # (B, T_cur, 4*H)
        u_idx_all = u_batch.unsqueeze(1).unsqueeze(3).expand(-1, 1, -1, 4 * H)  # (B, 1, T_cur, 4*H)

        hidden_seq = []
        for t in range(T_cur):
            w0_vt = w0_all[:, t, :].unsqueeze(1)  # (B, 1, 4*H)
            w1_vt = w1_all[:, t, :].unsqueeze(1)  # (B, 1, 4*H)
            idx_t = u_idx_all[:, :, t, :]         # (B, 1, 4*H)

            AW0 = AW0.scatter_add(1, idx_t, w0_vt)
            AW1 = AW1.scatter_add(1, idx_t, w1_vt)

            LAW1 = torch.bmm(L_batch, AW1)
            h1 = torch.bmm(L_batch, h)

            gates = AW0 + LAW1 + torch.matmul(h, W_h0) + torch.matmul(h1, W_h1) + self.b_gates
            zt, it, ft, ot = gates.chunk(4, dim=-1)
            c = torch.sigmoid(ft) * c + torch.sigmoid(it) * torch.tanh(zt)
            h = torch.sigmoid(ot) * torch.tanh(c)

            # Padded nodes update
            gates_pad = torch.matmul(h_pad, W_h0) + self.b_gates
            zp, ip, fp, op = gates_pad.chunk(4, dim=-1)
            c_pad = torch.sigmoid(fp) * c_pad + torch.sigmoid(ip) * torch.tanh(zp)
            h_pad = torch.sigmoid(op) * torch.tanh(c_pad)

            # Sum over active nodes (first n_nodes_batch[b] nodes) + padded nodes
            # Note: for nodes in [n_nodes_batch[b], N_cur), their L rows/cols are 0 and AW0/AW1 are 0,
            # so they ALREADY evolve identically to h_pad!
            # Thus h.sum(dim=1) + (max_nodes - N_cur) * h_pad equals the exact sum over all max_nodes!
            step_sum = h.sum(dim=1) + (self.max_nodes - N_cur) * h_pad  # (B, H)
            hidden_seq.append(step_sum)

        # Stack across time steps: (B, T_cur, H)
        hidden_states = torch.stack(hidden_seq, dim=1)

        # Incorporate standardized time features if enabled
        if self.use_scaled_time and time_scaled_batch is not None:
            hidden_states = hidden_states + self.time_scale_proj(time_scaled_batch)

        # Mask invalid steps with rnn_index
        hidden_states = hidden_states * rnn_mask_batch.unsqueeze(-1)

        # Multiply by learned time_decay weight according to time_interval_index
        tw = self.time_weight[time_bin_batch].unsqueeze(-1)  # (B, T_cur, 1)
        hidden_states = hidden_states * tw

        # Sum over sequence steps to form graph representation: (B, H)
        hidden_graph = hidden_states.sum(dim=1)

        # MLP Readout with ReLU on hidden layers and linear output (prevents dead output neuron)
        d1 = torch.relu(self.dense1(hidden_graph))
        d2 = torch.relu(self.dense2(d1))
        pred = self.out_layer(d2).squeeze(-1)  # (B,)
        return pred
