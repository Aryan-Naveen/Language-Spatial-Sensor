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


@POOLING_REGISTRY.register("query_token")
class QueryTokenPooling(nn.Module):
    """DETR-style target query token.

    Owns a learnable query vector Q_target ∈ R^D.  LSSModel detects this pooler
    (via ``hasattr(pool, "prepend")``) and inserts the query immediately after
    the text CLS before the fusion transformer, so the token cross-attends to
    both text and objects during fusion.  Its post-fusion state is extracted as
    the pooled context vector — replacing the "average the scene" inductive
    bias with a "what does the unobserved target attend to" signal.

    Sequence layout when active:  [text_CLS, Q_target, obj_1, …, obj_N]
    """

    def __init__(self, cfg: LSSConfig) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(cfg.hidden_dim))
        nn.init.normal_(self.query, std=cfg.hidden_dim ** -0.5)

    def prepend(
        self,
        seq: torch.Tensor,           # (B, 1+N, D)  [text_CLS, obj_1, …, obj_N]
        padding_mask: torch.Tensor,  # (B, 1+N)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Insert Q_target at position 1; return extended (seq, mask)."""
        B = seq.size(0)
        q = self.query.to(seq.dtype).view(1, 1, -1).expand(B, 1, -1)  # (B, 1, D)
        new_seq = torch.cat([seq[:, :1], q, seq[:, 1:]], dim=1)        # (B, 2+N, D)
        q_mask = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
        new_mask = torch.cat(
            [padding_mask[:, :1], q_mask, padding_mask[:, 1:]], dim=1
        )                                                              # (B, 2+N)
        return new_seq, new_mask

    def forward(
        self,
        x: torch.Tensor,                           # (B, 2+N, D)  post-fusion
        padding_mask: torch.Tensor | None = None,  # unused
    ) -> torch.Tensor:                             # (B, D)
        # Position 1 is the Q_target slot (position 0 is text CLS).
        return x[:, 1]


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
