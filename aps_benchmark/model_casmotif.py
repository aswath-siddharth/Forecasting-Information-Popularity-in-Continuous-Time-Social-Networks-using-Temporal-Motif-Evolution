"""Attention-GRU regressor for CasMotif on APS citation cascade benchmark."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalAdditiveAttention(nn.Module):
    """Additive Bahdanau-style attention over dynamic cascade sequences."""

    def __init__(self, hidden_dim: int, attention_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attention_dim, bias=True)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: [B, T, H], mask: [B, T] (bool, True for valid tokens)
        scores = self.v(torch.tanh(self.proj(h))).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(~mask, -1e9)
        alpha = torch.softmax(scores, dim=-1)                   # [B, T]
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)   # [B, H]
        return context, alpha


class CascadeAttentionGRURegressor(nn.Module):
    """CasMotif architecture: 15-D Causal Motif vectors -> GRU -> Attention -> MLP."""

    def __init__(
        self,
        input_dim: int = 15,
        hidden_size: int = 64,
        attention_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False,
        )
        self.attention = TemporalAdditiveAttention(hidden_size, attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], mask: [B, T]
        h, _ = self.gru(x)
        context, _ = self.attention(h, mask)
        out = self.head(self.dropout(context))
        return out

    def init_target_bias(self, target_mean: float) -> None:
        final_linear = self.head[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.constant_(final_linear.bias, float(target_mean))
