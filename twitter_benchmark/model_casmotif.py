"""CasMotif Model Architecture for Twitter Cascade Popularity Forecasting.
BiGRU + LearnedAttentionPooling + BottleneckHead (49,824 parameters).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from pipeline_c.model import BottleneckHead, LearnedAttentionPooling


class CascadeAttentionGRURegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 15,
        hidden_size: int = 64,
        num_layers: int = 1,
        attention_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        encoded_dim = hidden_size * 2
        self.pooling = LearnedAttentionPooling(encoded_dim, attention_dim)
        self.head = BottleneckHead(encoded_dim, dropout)

    def init_target_bias(self, mean_log_target: float) -> None:
        last = self.head.layers[-1]
        if isinstance(last, nn.Linear) and last.bias is not None:
            nn.init.constant_(last.bias, float(mean_log_target))

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        lengths = (~padding_mask).sum(dim=1).clamp_min(1).detach().cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        hidden, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
        context, _ = self.pooling(hidden, padding_mask)
        return self.head(context)
