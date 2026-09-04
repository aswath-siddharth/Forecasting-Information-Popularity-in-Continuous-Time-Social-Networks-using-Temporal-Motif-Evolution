from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BinaryFocalLossWithLogits(nn.Module):
    """Binary Focal Loss with Logits for extreme class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy well-classified background examples to prevent them from dominating gradients.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_weight = alpha_t * torch.pow((1.0 - p_t).clamp_min(0.0), self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


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
        scores = scores.masked_fill(padding_mask, -1e4)
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

    def init_prior_bias(self, prior_prob: float = 0.05) -> None:
        """Initialize final linear layer bias to log-odds of prior positive rate: ln(p / (1 - p))."""
        prior_prob = min(max(prior_prob, 1e-4), 1.0 - 1e-4)
        bias_val = math.log(prior_prob / (1.0 - prior_prob))
        last_linear = self.layers[-1]
        if isinstance(last_linear, nn.Linear) and last_linear.bias is not None:
            nn.init.constant_(last_linear.bias, bias_val)

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
        self.input_norm = nn.LayerNorm(input_dim)
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

    def init_prior_bias(self, prior_prob: float = 0.05) -> None:
        self.head.init_prior_bias(prior_prob)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        lengths = (~padding_mask).sum(dim=1).clamp_min(1).detach().cpu()
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
        self.input_norm = nn.LayerNorm(input_dim)
        self.head = BottleneckHead(input_dim, dropout)

    def init_prior_bias(self, prior_prob: float = 0.05) -> None:
        self.head.init_prior_bias(prior_prob)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        valid_mask = (~padding_mask).unsqueeze(-1)
        if self.mode == "mean":
            summed = (x * valid_mask).sum(dim=1)
            counts = valid_mask.sum(dim=1).clamp_min(1)
            pooled = summed / counts
        else:
            pooled = x.masked_fill(~valid_mask, -torch.inf).amax(dim=1)
            pooled = torch.nan_to_num(pooled, neginf=0.0)
        return self.head(pooled)

