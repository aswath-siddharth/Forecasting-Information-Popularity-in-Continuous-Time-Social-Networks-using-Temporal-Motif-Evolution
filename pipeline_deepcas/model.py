"""DeepCas: An End-to-end Predictor of Information Cascades (WWW 2017).

Faithful PyTorch implementation based on Cheng Li et al., WWW 2017 and the official
github repository (https://github.com/chengli-um/DeepCas).

Architecture:
- Node Embedding layer (learnable embeddings for diffusion network users)
- BiGRU Walk Encoder (encodes sampled structural random walk paths)
- Step-level Attention (soft attention pooling over the sequence steps of each walk)
- Cascade-level Attention (soft attention pooling over the random walks of the cascade)
- Multi-Layer Perceptron (MLP regression head predicting log2(increment + 1))
"""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class DeepCasModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 50,
        hidden_gru: int = 32,
        walk_length: int = 10,
        attention_dim: int = 64,
        dense1_dim: int = 64,
        dense2_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_gru = hidden_gru
        self.walk_length = walk_length

        # 1. Node Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(p=dropout)

        # 2. Bidirectional GRU Walk Encoder
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_gru,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.gru_out_dim = hidden_gru * 2  # Bidirectional

        # 3. Step Attention across the walk steps
        self.step_attention = nn.Parameter(torch.zeros(walk_length))
        nn.init.normal_(self.step_attention, mean=0.0, std=0.01)

        # 4. Cascade Walk Attention across walks
        self.walk_attention_scorer = nn.Sequential(
            nn.Linear(self.gru_out_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

        # 5. Prediction MLP Head (takes cascade context + observed cascade size)
        head_in_dim = self.gru_out_dim + 1  # 2*hidden_gru + log2(N_obs + 1)
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, dense1_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dense1_dim, dense2_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dense2_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_target_bias(self, mean_log_target: float) -> None:
        """Start the prediction head at the training set mean."""
        last_linear = self.head[-1]
        if isinstance(last_linear, nn.Linear) and last_linear.bias is not None:
            nn.init.constant_(last_linear.bias, float(mean_log_target))

    def forward(self, walks: torch.Tensor, obs_size: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            walks: (batch_size, n_walks, walk_length) int64 tensor of node IDs
            obs_size: (batch_size,) float32 tensor of observed cascade sizes N(T_obs)
        Returns:
            pred: (batch_size,) predicted log2(increment + 1)
        """
        B, N_w, L = walks.shape

        # Flatten walks for batch processing: (B * N_w, L)
        flat_walks = walks.view(B * N_w, L)

        # Node Embedding: (B * N_w, L, embedding_dim)
        embeds = self.embedding(flat_walks)
        embeds = self.embed_dropout(embeds)

        # BiGRU encoding: (B * N_w, L, 2 * hidden_gru)
        gru_out, _ = self.gru(embeds)

        # Step Attention: (L,) -> softmax -> (1, L, 1)
        step_weights = F.softmax(self.step_attention, dim=0).view(1, L, 1)
        # Pooled walk representation: sum_t(weight_t * h_t) -> (B * N_w, 2 * hidden_gru)
        walk_reprs = (gru_out * step_weights).sum(dim=1)

        # Reshape to (B, N_w, 2 * hidden_gru)
        walk_reprs = walk_reprs.view(B, N_w, self.gru_out_dim)

        # Walk Attention: (B, N_w, 1) -> (B, N_w)
        walk_scores = self.walk_attention_scorer(walk_reprs).squeeze(-1)
        walk_weights = F.softmax(walk_scores, dim=1).unsqueeze(-1)  # (B, N_w, 1)

        # Cascade Graph representation: (B, 2 * hidden_gru)
        cascade_repr = (walk_reprs * walk_weights).sum(dim=1)

        # Integrate log2(N_obs + 1)
        log_obs = torch.log2(obs_size.clamp_min(1.0)).unsqueeze(-1)  # (B, 1)
        combined = torch.cat([cascade_repr, log_obs], dim=-1)        # (B, 2 * hidden_gru + 1)

        # Prediction: (B, 1) -> (B,)
        pred = self.head(combined).squeeze(-1)
        return pred
