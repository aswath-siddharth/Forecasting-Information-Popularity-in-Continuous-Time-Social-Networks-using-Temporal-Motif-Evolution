"""PyTorch reimplementation of DeepHawkes (Cao et al., ACM CIKM 2017).

Faithfully reproduces the official TensorFlow implementation
``CaoQi92/DeepHawkes/deep_learning/model_sparse.py`` (class ``SDPP``):

    1. User Embedding Layer: ``embedding_dim = 50``, initialized ``Uniform(0, 1)``
       (``0`` reserved for ``<PAD>``, ``1`` for ``<UNK>``), followed by embedding
       dropout (``p = 0.2``, matching ``dropout_prob = 0.8`` keep probability).
    2. Self-Exciting Path Encoding (GRU): single-layer GRU with hidden size
       ``2 * n_hidden_gru = 64`` (``n_hidden_gru = 32`` in ``run_sparse.py``).
       Extracts the final valid hidden state ``h_j in R^{64}`` of each diffusion path.
    3. Non-Parametric Time Decay Effect: ``n_time_intervals = 6`` equal intervals
       covering ``[0, T_obs)``, with learnable scalar weights ``time_weight``
       initialized ``Normal(0, 0.01)``.
    4. Additive Sum Pooling: ``V_c = sum_{j=1}^{n_c} time_weight[b_j] * h_j``.
    5. MLP Prediction Head: ``64 -> 32 (ReLU) -> 16 (ReLU) -> 1 (ReLU)``,
       weights initialized ``Normal(0, 0.01)``.
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

        # 1. User embeddings initialized Uniform(0, 1) as in model_sparse.py (initializer2)
        self.user_emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.uniform_(self.user_emb.weight, a=0.0, b=1.0)
        with torch.no_grad():
            self.user_emb.weight[0].zero_()
        self.emb_dropout = nn.Dropout(dropout)

        # 2. GRU path encoder (2 * n_hidden_gru = 64 in official model_sparse.py)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # 3. Non-parametric time decay weights initialized Normal(0, 0.01)
        self.time_weight = nn.Parameter(torch.empty(n_time_intervals, dtype=torch.float32))
        nn.init.normal_(self.time_weight, mean=0.0, std=0.01)

        # 4. MLP regression head: 64 -> 32 -> 16 -> 1 with ReLU output (non-negative log2 target)
        self.fc1 = nn.Linear(hidden_size, mlp_hidden_1)
        self.fc2 = nn.Linear(mlp_hidden_1, mlp_hidden_2)
        self.out = nn.Linear(mlp_hidden_2, 1)
        for layer in (self.fc1, self.fc2, self.out):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.normal_(layer.bias, mean=0.0, std=0.01)
        self.act = nn.ReLU()

    def init_target_bias(self, mean_log_target: float) -> None:
        """Initialize the output bias to the training-set mean of log2(y + 1)."""
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
        """Forward pass over a flattened batch of diffusion paths."""
        path_users: torch.Tensor = batch["path_users"]      # [P, L_max]
        path_lens: torch.Tensor = batch["path_lens"]        # [P]
        time_bins: torch.Tensor = batch["time_bins"]        # [P]
        cascade_idx: torch.Tensor = batch["cascade_idx"]    # [P]
        n_cascades: int = int(batch["n_cascades"])

        emb = self.emb_dropout(self.user_emb(path_users))   # [P, L_max, D_emb]
        packed = pack_padded_sequence(
            emb, path_lens.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)                           # [1, P, H]
        path_repr = h_n.squeeze(0)                          # [P, H]

        # Non-parametric time decay scaling per path
        decay = self.time_weight[time_bins].unsqueeze(-1)   # [P, 1]
        weighted = path_repr * decay                        # [P, H]

        # Additive sum pooling over all paths in each cascade
        cascade_repr = torch.zeros(
            (n_cascades, self.hidden_size),
            device=weighted.device,
            dtype=weighted.dtype,
        )
        cascade_repr.index_add_(0, cascade_idx, weighted)   # [B, H]

        h = self.act(self.fc1(cascade_repr))
        h = self.act(self.fc2(h))
        return self.act(self.out(h))                        # [B, 1]
