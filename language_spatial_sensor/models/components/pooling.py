"""Pooling strategies: sequence of token features → single scene-text vector.

All poolers have the same signature:
    forward(x, padding_mask) -> (B, D)

    x:            (B, S, D)  token sequence (text CLS prepended + object tokens)
    padding_mask: (B, S)     True = padding, should be ignored

Registered in POOLING_REGISTRY so the active strategy is selected by
cfg.pooling_type without changing LSSModel code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LSSConfig
from ..registry import POOLING_REGISTRY


@POOLING_REGISTRY.register("mean")
class MeanPooling(nn.Module):
    """Average over non-padded token positions."""

    def __init__(self, cfg: LSSConfig) -> None:  # noqa: ARG002
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,                           # (B, S, D)
        padding_mask: torch.Tensor | None = None,  # (B, S) True = pad
    ) -> torch.Tensor:                             # (B, D)
        if padding_mask is None:
            return x.mean(dim=1)
        # mask out padding before averaging
        mask = ~padding_mask                         # (B, S)  True = valid
        x_masked = x * mask.unsqueeze(-1).float()
        return x_masked.sum(dim=1) / mask.float().sum(dim=1, keepdim=True).clamp(min=1)


@POOLING_REGISTRY.register("max")
class MaxPooling(nn.Module):
    """Element-wise max over non-padded positions."""

    def __init__(self, cfg: LSSConfig) -> None:  # noqa: ARG002
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,                           # (B, S, D)
        padding_mask: torch.Tensor | None = None,  # (B, S)
    ) -> torch.Tensor:                             # (B, D)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
        return x.max(dim=1).values


@POOLING_REGISTRY.register("attention")
class AttentionPooling(nn.Module):
    """Learned query vector attends over the full sequence.

    A single trainable query q ∈ R^D computes a weighted sum of the token
    sequence.  This lets the model emphasise task-relevant tokens (e.g. anchor
    objects, preposition-bearing text) rather than averaging everything equally.
    """

    def __init__(self, cfg: LSSConfig) -> None:
        super().__init__()
        self.query   = nn.Parameter(torch.empty(cfg.hidden_dim))
        self.key_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        nn.init.normal_(self.query, std=cfg.hidden_dim ** -0.5)

    def forward(
        self,
        x: torch.Tensor,                           # (B, S, D)
        padding_mask: torch.Tensor | None = None,  # (B, S)
    ) -> torch.Tensor:                             # (B, D)
        # scores: (B, S)
        scores = torch.matmul(self.key_proj(x), self.query)   # (B, S)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)                   # (B, S)
        return torch.einsum("bs,bsd->bd", weights, x)
