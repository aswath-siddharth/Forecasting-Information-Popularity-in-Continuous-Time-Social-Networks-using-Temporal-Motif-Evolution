"""Standard CasMotif Model Architecture for APS Cascade Popularity Forecasting.
BiGRU + LearnedAttentionPooling + BottleneckHead (49,824 parameters).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from pipeline_c.model import BottleneckHead, LearnedAttentionPooling


class CascadeAttentionGRURegressor(nn.Module):
    """Standard CasMotif: LayerNorm(15) -> BiGRU(64) -> LearnedAttentionPooling -> BottleneckHead."""

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

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # valid_mask: [B, T] bool (True for valid tokens from data_casmotif.py)
        padding_mask = ~valid_mask
        x = self.input_norm(x)
        lengths = valid_mask.sum(dim=1).clamp_min(1).detach().cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        hidden, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
        context, _ = self.pooling(hidden, padding_mask)
        return self.head(context)

    def init_target_bias(self, target_mean: float) -> None:
        final_linear = self.head.layers[-1]
        if isinstance(final_linear, nn.Linear) and final_linear.bias is not None:
            nn.init.constant_(final_linear.bias, float(target_mean))
