from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LearnedAttentionPooling(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int = 64) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.scorer(hidden_states).squeeze(-1)
        scores = scores.masked_fill(padding_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, weights


class BottleneckHead(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CascadeAttentionGRU(nn.Module):
    def __init__(
        self,
        input_dim: int = 20,
        hidden_size: int = 64,
        num_layers: int = 1,
        attention_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout,
        )
        encoded_dim = hidden_size * 2
        self.pooling = LearnedAttentionPooling(encoded_dim, attention_dim)
        self.head = BottleneckHead(encoded_dim, dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        lengths = (~padding_mask).sum(dim=1).detach().cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed)
        hidden_states, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=x.shape[1])
        context, _ = self.pooling(hidden_states, padding_mask)
        return self.head(context)


class PoolingBaseline(nn.Module):
    def __init__(
        self,
        mode: str,
        input_dim: int = 20,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if mode not in {"mean", "max"}:
            raise ValueError("mode must be 'mean' or 'max'")
        self.mode = mode
        self.head = BottleneckHead(input_dim, dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid_mask = (~padding_mask).unsqueeze(-1)
        if self.mode == "mean":
            summed = (x * valid_mask).sum(dim=1)
            counts = valid_mask.sum(dim=1).clamp_min(1)
            pooled = summed / counts
        else:
            pooled = x.masked_fill(~valid_mask, -torch.inf).amax(dim=1)
            pooled = torch.nan_to_num(pooled, neginf=0.0)
        return self.head(pooled)

