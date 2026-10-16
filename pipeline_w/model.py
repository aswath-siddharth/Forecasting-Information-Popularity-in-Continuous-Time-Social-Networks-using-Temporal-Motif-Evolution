"""Regression head over the same BiGRU + attention encoder used for the Reddit task.

The encoder is deliberately identical to ``pipeline_c.model.CascadeAttentionGRU`` --
the point of the benchmark is that *our* architecture transfers to the published
dataset, so only the output interpretation and loss change (scalar log-popularity
with MSE, rather than a logit with focal loss).
"""

from __future__ import annotations

import math

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
        side_dim: int = 0,
    ) -> None:
        super().__init__()
        self.side_dim = side_dim
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
        # Cascade-level side features (publish hour, momentum) join the attended
        # context just before the head; side_dim=0 is the original architecture.
        self.head = BottleneckHead(encoded_dim + side_dim, dropout)

    def init_target_bias(self, mean_log_target: float) -> None:
        """Start the head predicting the training mean, so early epochs spend their
        gradient on the signal rather than on finding the target's scale."""
        last = self.head.layers[-1]
        if isinstance(last, nn.Linear) and last.bias is not None:
            nn.init.constant_(last.bias, float(mean_log_target))

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor,
                side: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_norm(x)
        lengths = (~padding_mask).sum(dim=1).clamp_min(1).detach().cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        hidden, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
        context, _ = self.pooling(hidden, padding_mask)
        if self.side_dim:
            context = torch.cat([context, side], dim=-1)
        return self.head(context)

    @torch.no_grad()
    def attention_weights(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Expose alpha_j for the 'which early events mattered' analysis."""
        x = self.input_norm(x)
        lengths = (~padding_mask).sum(dim=1).clamp_min(1).detach().cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        hidden, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
        _, weights = self.pooling(hidden, padding_mask)
        return weights


class PoolingRegressor(nn.Module):
    """Ablation baseline: identical head, but mean/max pooling instead of the
    BiGRU + attention encoder. Isolates how much the sequence model actually buys."""

    def __init__(self, mode: str, input_dim: int = 15, dropout: float = 0.3) -> None:
        super().__init__()
        if mode not in {"mean", "max"}:
            raise ValueError("mode must be 'mean' or 'max'")
        self.mode = mode
        self.input_norm = nn.LayerNorm(input_dim)
        self.head = BottleneckHead(input_dim, dropout)

    def init_target_bias(self, mean_log_target: float) -> None:
        last = self.head.layers[-1]
        if isinstance(last, nn.Linear) and last.bias is not None:
            nn.init.constant_(last.bias, float(mean_log_target))

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        valid = (~padding_mask).unsqueeze(-1)
        if self.mode == "mean":
            pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        else:
            pooled = x.masked_fill(~valid, -torch.inf).amax(dim=1)
            pooled = torch.nan_to_num(pooled, neginf=0.0)
        return self.head(pooled)


class MeanPredictor:
    """Non-learned reference point: always predict the training mean of log2(y+1).
    Any model that cannot beat this has learned nothing from the cascade."""

    def __init__(self, mean_log_target: float) -> None:
        self.mean_log_target = float(mean_log_target)

    def predict(self, n: int) -> torch.Tensor:
        return torch.full((n, 1), self.mean_log_target, dtype=torch.float32)


def build_model(name: str, input_dim: int, hidden_size: int, num_layers: int,
                attention_dim: int, dropout: float, side_dim: int = 0) -> nn.Module:
    if name == "attention_gru":
        return CascadeAttentionGRURegressor(input_dim, hidden_size, num_layers, attention_dim,
                                            dropout, side_dim)
    if name == "mean_pool":
        return PoolingRegressor("mean", input_dim, dropout)
    if name == "max_pool":
        return PoolingRegressor("max", input_dim, dropout)
    raise ValueError(f"Unknown model: {name}")
