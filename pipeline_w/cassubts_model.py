"""CasSubTS, reimplemented from Wang et al., Scientific Reports 15:6801 (2025).

  "Fusing temporal and structural information via subgraph sampling and
   multi-head attention for information cascade prediction"
  https://doi.org/10.1038/s41598-025-91752-z

The authors released no code ("available from the corresponding author on
reasonable request"), so this is built from the paper's equations and Table 2
hyperparameters. Where the paper is specific we follow it exactly; where it is
silent the choice is marked DEVIATION below. A reimplementation is evidence about
the architecture, not about the authors' own trained system.

Pipeline (paper Fig. 3): subgraph sampling -> MH-GAT structural branch +
node-attention/Bi-GRU temporal branch -> SENet fusion -> MLP increment prediction.

DEVIATIONS, all forced by underspecification:
 1. Subgraph sampling. Eqs 7-8 define each subgraph by edges whose timestamp lies
    in [t_k, t_k+T], i.e. disjoint slices, but Fig. 3 draws progressively fuller
    graphs. We use cumulative snapshots (edges with t <= t_k), matching the figure
    and the CasCN lineage the paper situates itself in.
 2. Base node features. Eq 13 concatenates mapped in/out-degree onto "original
    node features" x_i, which the paper never defines; the released Weibo data has
    no node attributes, and a learned per-user embedding would be transductive over
    ~100k users. We use [depth, normalised first-activation time] — structural,
    inductive, and drawn from the same information our own model sees, so neither
    model gets an information advantage.
 3. In/out-degree are computed per subgraph rather than once on the whole cascade
    graph, which is the reading consistent with "capture dynamic node changes".
 4. SENet fusion. Eq 26 gives a CBAM-style avg+max channel gate but not what the
    channels are. We gate over the concatenated structural and temporal feature
    channels, which is the only reading consistent with Fig. 3's N x T x D -> T x D.
"""

from __future__ import annotations

import torch
from torch import nn


def scatter_softmax(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax over all entries sharing an index (eq 18), numerically stabilised."""
    m = torch.full((n,), float("-inf"), device=src.device, dtype=src.dtype)
    m = m.scatter_reduce(0, index, src, reduce="amax", include_self=True)
    m = torch.nan_to_num(m, neginf=0.0)
    e = (src - m[index]).exp()
    s = torch.zeros(n, device=src.device, dtype=src.dtype).index_add_(0, index, e)
    return e / (s[index] + 1e-16)


class MultiHeadGATLayer(nn.Module):
    """Eqs 14-20: per-head linear projection, LeakyReLU attention scores,
    neighbourhood softmax, weighted aggregation, then average across heads."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 2, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.heads, self.out_dim = heads, out_dim
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.leaky = nn.LeakyReLU(negative_slope)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_src: torch.Tensor, edge_dst: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        h = self.lin(x).view(n, self.heads, self.out_dim)

        # Self-loops keep isolated nodes (common in early snapshots) from vanishing.
        loop = torch.arange(n, device=x.device)
        src = torch.cat([edge_src, loop])
        dst = torch.cat([edge_dst, loop])

        score = self.leaky((h[src] * self.att_src).sum(-1) + (h[dst] * self.att_dst).sum(-1))

        out = torch.zeros(n, self.heads, self.out_dim, device=x.device, dtype=x.dtype)
        for k in range(self.heads):
            alpha = scatter_softmax(score[:, k], dst, n)
            out[:, k].index_add_(0, dst, h[src, k] * alpha.unsqueeze(-1))

        return out.mean(dim=1) + self.bias     # eq 20: average the k heads


class NodeAttentionPool(nn.Module):
    """Eq 21: a fully connected layer scores each node, scores are normalised,
    and the weighted node vectors are averaged into one subgraph vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
        w = scatter_softmax(self.score(x).squeeze(-1), batch, n_graphs)
        out = torch.zeros(n_graphs, x.shape[1], device=x.device, dtype=x.dtype)
        out.index_add_(0, batch, x * w.unsqueeze(-1))
        return out


class SENetFusion(nn.Module):
    """Eq 26: sigma(MLP(AvgPool(F)) + MLP(MaxPool(F))) gating over feature channels."""

    def __init__(self, dim: int, reduction: int = 2) -> None:
        super().__init__()
        hidden = max(1, dim // reduction)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.mlp(f) + self.mlp(f.abs()))
        return f * gate


class CasSubTS(nn.Module):
    def __init__(
        self,
        base_node_dim: int = 2,
        degree_dim: int = 8,
        gat_dim: int = 32,
        gat_heads: int = 2,
        gat_layers: int = 2,
        gru_hidden: int = 64,
        gru_layers: int = 2,
        n_subgraphs: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_subgraphs = n_subgraphs

        # Eq 12: in/out-degree mapped into a low-dimensional space.
        self.deg_in = nn.Sequential(nn.Linear(1, degree_dim), nn.ReLU())
        self.deg_out = nn.Sequential(nn.Linear(1, degree_dim), nn.ReLU())

        in_dim = base_node_dim + 2 * degree_dim        # eq 13
        self.gats = nn.ModuleList()
        for i in range(gat_layers):
            self.gats.append(MultiHeadGATLayer(in_dim if i == 0 else gat_dim, gat_dim, gat_heads))

        self.node_pool = NodeAttentionPool(gat_dim)
        self.gru = nn.GRU(gat_dim, gru_hidden, num_layers=gru_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if gru_layers > 1 else 0.0)

        fused_dim = gat_dim + 2 * gru_hidden
        self.fusion = SENetFusion(fused_dim)
        self.mlp = nn.Sequential(                       # Table 2: 64 -> 32 -> 1
            nn.Linear(fused_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def init_target_bias(self, mean_log_target: float) -> None:
        last = self.mlp[-1]
        if isinstance(last, nn.Linear) and last.bias is not None:
            nn.init.constant_(last.bias, float(mean_log_target))

    def forward(self, batch: dict) -> torch.Tensor:
        x_base = batch["x_base"]            # [N_nodes, base_node_dim]
        node_batch = batch["node_batch"]    # [N_nodes] -> cascade index
        n_graphs = int(batch["n_graphs"])
        device = x_base.device

        seq = []
        for k in range(self.n_subgraphs):
            src, dst = batch["edge_src"][k], batch["edge_dst"][k]

            indeg = torch.zeros(x_base.shape[0], device=device)
            outdeg = torch.zeros(x_base.shape[0], device=device)
            if src.numel() > 0:
                indeg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
                outdeg.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))

            x = torch.cat([
                x_base,
                self.deg_in(torch.log1p(indeg).unsqueeze(-1)),
                self.deg_out(torch.log1p(outdeg).unsqueeze(-1)),
            ], dim=-1)

            for gat in self.gats:
                x = torch.relu(gat(x, src, dst))

            seq.append(self.node_pool(x, node_batch, n_graphs))

        stacked = torch.stack(seq, dim=1)               # [B, N_subgraphs, gat_dim]
        h_struct = stacked.mean(dim=1)                  # structural branch
        gru_out, _ = self.gru(stacked)                  # eqs 22-24
        h_temporal = gru_out[:, -1, :]                  # temporal branch

        fused = self.fusion(torch.cat([h_struct, h_temporal], dim=-1))
        return self.mlp(fused)
