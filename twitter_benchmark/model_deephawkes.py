"""PyTorch reimplementation of DeepHawkes (Cao et al., ACM CIKM 2017) for Twitter cascades.

Reproduces the official architecture:
  1. User Embeddings: 50-D, initialized Uniform(0, 1), padding_idx=0 (<PAD>), 1 (<UNK>),
     with 0.2 embedding dropout.
  2. GRU Path Encoder: single-layer GRU (hidden_size=64) over diffusion paths.
  3. Non-Parametric Time Decay: 6 learnable interval weights covering [0, T_obs].
  4. Additive Sum Pooling across paths for each cascade.
  5. MLP Regression Head: 64 -> 32 -> 16 -> 1 with ReLU output.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class DeepHawkes(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 50,
        hidden_size: int = 64,
        n_time_intervals: int = 6,
        mlp_hidden_1: int = 32,
        mlp_hidden_2: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_size = int(hidden_size)
        self.n_time_intervals = int(n_time_intervals)

        self.user_emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.uniform_(self.user_emb.weight, a=0.0, b=1.0)
        with torch.no_grad():
            self.user_emb.weight[0].zero_()
        self.emb_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.time_weight = nn.Parameter(torch.empty(n_time_intervals, dtype=torch.float32))
        nn.init.normal_(self.time_weight, mean=0.0, std=0.01)

        self.fc1 = nn.Linear(hidden_size, mlp_hidden_1)
        self.fc2 = nn.Linear(mlp_hidden_1, mlp_hidden_2)
        self.out = nn.Linear(mlp_hidden_2, 1)
        for layer in (self.fc1, self.fc2, self.out):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.normal_(layer.bias, mean=0.0, std=0.01)
        self.act = nn.ReLU()

    def init_target_bias(self, mean_log_target: float) -> None:
        if self.out.bias is not None:
            nn.init.constant_(self.out.bias, float(mean_log_target))

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        emb = sum(p.numel() for p in self.user_emb.parameters())
        return {
            "total": int(total),
            "embedding": int(emb),
            "non_embedding": int(total - emb),
        }

    def forward(self, batch: dict[str, torch.Tensor | int]) -> torch.Tensor:
        path_users: torch.Tensor = batch["path_users"]      # [P, L_max]
        path_lens: torch.Tensor = batch["path_lens"]        # [P]
        time_bins: torch.Tensor = batch["time_bins"]        # [P]
        cascade_idx: torch.Tensor = batch["cascade_idx"]    # [P]
        n_cascades: int = int(batch["n_cascades"])

        if path_users.shape[0] == 0:
            dev = self.out.weight.device
            zeros = torch.zeros((n_cascades, self.hidden_size), device=dev)
            return self.act(self.out(self.act(self.fc2(self.act(self.fc1(zeros))))))

        emb = self.emb_dropout(self.user_emb(path_users))   # [P, L_max, D]
        packed = pack_padded_sequence(
            emb, path_lens.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)                           # [1, P, H]
        path_repr = h_n.squeeze(0)                          # [P, H]

        w = self.time_weight[time_bins].unsqueeze(1)        # [P, 1]
        decayed = path_repr * w                             # [P, H]

        # Additive sum pooling across paths for each cascade
        cascade_repr = torch.zeros((n_cascades, self.hidden_size), device=path_repr.device)
        cascade_repr.index_add_(0, cascade_idx, decayed)

        x = self.act(self.fc1(cascade_repr))
        x = self.act(self.fc2(x))
        out = self.out(x)
        return self.act(out)
