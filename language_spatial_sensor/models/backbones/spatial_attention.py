"""3D-VisTA spatial attention backbone.

MultiHeadAttentionSpatial adds a per-head learned bias from pairwise
geometric features to the standard QK attention logits.  This lets the model
attend differently to nearby vs. far objects and understand directional
relationships without positional encodings.

Registered under BACKBONE_REGISTRY as "vista".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LSSConfig
from ..registry import BACKBONE_REGISTRY


class MultiHeadAttentionSpatial(nn.Module):
    """Standard MHA + additive spatial bias from pairwise geometric features.

    Args:
        d_model:     token feature dimension (hidden_dim)
        num_heads:   number of attention heads
        spatial_dim: dimension of pairwise spatial features (``cfg.spatial_relation_dim``)
        dropout:     attention dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        spatial_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.dropout_p = dropout

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Project spatial features to one scalar bias per head.
        # bias=False: spatial features are zero-mean; no constant offset needed.
        self.spatial_proj = nn.Linear(spatial_dim, num_heads, bias=False)

        # Attention-capture hook (off by default; enabled by viz tooling).
        self._store_attn: bool = False
        self._last_attn_weights: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,                    # (B, N, D)
        spatial_relations: torch.Tensor,    # (B, N, N, spatial_dim)
        key_padding_mask: torch.Tensor | None = None,  # (B, N) True = padded
    ) -> torch.Tensor:                      # (B, N, D)
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim
        scale = d ** -0.5

        def _split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, N, H, d).transpose(1, 2)   # (B, H, N, d)

        q = _split_heads(self.q_proj(x))
        k = _split_heads(self.k_proj(x))
        v = _split_heads(self.v_proj(x))

        # (B, H, N, N) attention logits
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale

        # spatial bias: (B, N, N, H) → (B, H, N, N)
        spatial_bias = self.spatial_proj(spatial_relations).permute(0, 3, 1, 2)
        attn_logits  = attn_logits + spatial_bias

        if key_padding_mask is not None:
            # mask out padded key positions — shape broadcast: (B, 1, 1, N)
            attn_logits = attn_logits.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        attn_weights = F.softmax(attn_logits, dim=-1)
        # If every key is padded for a query row, logits are all -inf → softmax NaN.
        # Zero those weights so the head contributes nothing instead of poisoning the graph.
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        if self._store_attn:
            self._last_attn_weights = attn_weights.detach().to(torch.float32).cpu()
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)

        out = torch.matmul(attn_weights, v)              # (B, H, N, d)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class SpatialEncoderLayer(nn.Module):
    """Single Spatial block: spatial MHA → LayerNorm → FFN → LayerNorm."""

    def __init__(self, cfg: LSSConfig) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttentionSpatial(
            d_model    = cfg.hidden_dim,
            num_heads  = cfg.num_heads,
            spatial_dim = cfg.spatial_relation_dim,
            dropout    = cfg.dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.hidden_dim),
        )
        self.norm1   = nn.LayerNorm(cfg.hidden_dim)
        self.norm2   = nn.LayerNorm(cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,                    # (B, N, D)
        spatial_relations: torch.Tensor,    # (B, N, N, spatial_dim)
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:                      # (B, N, D)
        # pre-norm residual style
        x = x + self.dropout(
            self.self_attn(self.norm1(x), spatial_relations, key_padding_mask)
        )
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


@BACKBONE_REGISTRY.register("identity")
class IdentityBackbone(nn.Module):
    """Pass-through backbone for the 'no spatial backbone' ablation.

    Returns object features unchanged, ignoring pairwise spatial relations and
    the padding mask. Tests whether the explicit ViSTA-style spatial bias is
    pulling its weight, or whether the global fusion transformer alone can
    learn geometry from raw bbox tokens.
    """

    def __init__(self, cfg: LSSConfig) -> None:  # noqa: ARG002
        super().__init__()

    def forward(
        self,
        obj_features: torch.Tensor,                       # (B, N, D)
        spatial_relations: torch.Tensor,                  # (B, N, N, R) — unused
        key_padding_mask: torch.Tensor | None = None,     # (B, N)        — unused
    ) -> torch.Tensor:                                    # (B, N, D)
        return obj_features


@BACKBONE_REGISTRY.register("scene_spatial")
class SceneSpatialEncoder(nn.Module):
    """Stack of ViSTAEncoderLayers.

    Refines all N object features using pairwise geometric context.  Because
    anchors share the same role embedding (E_anchor), the encoder learns a
    geometry-aware strategy for 'query object relates to anchor' without any
    sensitivity to anchor ordering.
    """

    def __init__(self, cfg: LSSConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [SpatialEncoderLayer(cfg) for _ in range(cfg.num_spatial_layers)]
        )

    def forward(
        self,
        obj_features: torch.Tensor,         # (B, N, D)
        spatial_relations: torch.Tensor,    # (B, N, N, spatial_dim)
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:                      # (B, N, D)
        x = obj_features
        for layer in self.layers:
            x = layer(x, spatial_relations, key_padding_mask)
        return x
