"""LSSModel: thin wiring layer that assembles the full forward pass.

This file contains no novel logic — it delegates to the registered
backbone/pooling/head components.  To swap in a new component, register it
in the appropriate registry and set the matching cfg field.

Forward pass shape trace (B=batch, N=max_objects, L=text_len, D=hidden_dim):

    text_input_ids        (B, L)        → TextEncoder
    text_attention_mask   (B, L)        ↗

    obj_clip_features     (B, N, D_clip) → VisionEncoder
    obj_is_anchor         (B, N)         ↗

    obj_bboxes            (B, N, 6)  → SpatialRelationMLP → (B, N, N, 12)
      (bboxes are in region-normalized frame; coord_scale/coord_shift carry the transform)

    [obj features (B,N,D)] + [spatial (B,N,N,12)] → BACKBONE → (B,N,D)

    prepend text CLS      → sequence (B, 1+N, D)
    add modality tags

    GlobalFusionTransformer               → (B, 1+N, D)

    Pooling                               → (B, D)

    FiLMLayer(coord_scale, coord_shift)   → (B, D)   [if cfg.use_film]

    Head(coord_scale, coord_shift)        → GaussianPrediction(mu, L) in world frame
"""

import torch
import torch.nn as nn

from .config import LSSConfig
from .components.encoders import SpatialRelationMLP, TextEncoder, VisionEncoder
from .components.heads import FiLMLayer
from .registry import BACKBONE_REGISTRY, HEAD_REGISTRY, POOLING_REGISTRY

# Side-effect imports: register all built-in components into their registries.
import language_spatial_sensor.models.backbones.spatial_attention  # noqa: F401
import language_spatial_sensor.models.components.pooling            # noqa: F401
import language_spatial_sensor.models.components.heads              # noqa: F401


class GlobalFusionTransformer(nn.Module):
    """Standard transformer encoder (no spatial bias) over the full token sequence.

    Processes the concatenated sequence [text_CLS | obj_1 | … | obj_N], allowing
    text to attend to anchor-tagged objects when resolving spatial prepositions.
    """

    def __init__(self, cfg: LSSConfig) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = cfg.hidden_dim,
            nhead           = cfg.num_heads,
            dim_feedforward = cfg.ffn_dim,
            dropout         = cfg.dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,   # pre-norm (more stable)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers      = cfg.num_fusion_layers,
            enable_nested_tensor = False,
        )

    def forward(
        self,
        x: torch.Tensor,                           # (B, 1+N, D)
        src_key_padding_mask: torch.Tensor | None = None,  # (B, 1+N)
    ) -> torch.Tensor:                             # (B, 1+N, D)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class LSSModel(nn.Module):
    """Language-Spatial Sensor model.

    Construct with an LSSConfig; all component choices (backbone, pooling,
    head) are resolved through their registries.

    Example::

        cfg   = LSSConfig(pooling_type="attention", num_fusion_layers=4)
        model = LSSModel(cfg)
        pred  = model(text_input_ids, text_attention_mask,
                      obj_clip_features, obj_bboxes,
                      obj_is_anchor, obj_padding_mask)
        # pred.mu:  (B, 3)
        # pred.L:   (B, 3, 3)
    """

    # Modality tag indices — must match VisionEncoder._MODALITY_* constants.
    _MODALITY_TEXT = 0
    _MODALITY_OBJ  = 1

    def __init__(self, cfg: LSSConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ── Encoders ──────────────────────────────────────────────────────────
        self.text_enc    = TextEncoder(cfg)
        self.vision_enc  = VisionEncoder(cfg)
        self.spatial_enc = SpatialRelationMLP(cfg)

        # ── Backbone: object feature spatial refinement ────────────────────────
        self.backbone = BACKBONE_REGISTRY.build(cfg.backbone_type, cfg)

        # ── Global fusion ──────────────────────────────────────────────────────
        self.fusion = GlobalFusionTransformer(cfg)

        # Modality tag shared with VisionEncoder; [TEXT] tag applied here
        self.modality_embed = nn.Embedding(2, cfg.hidden_dim)

        # ── Pooling ────────────────────────────────────────────────────────────
        self.pool = POOLING_REGISTRY.build(cfg.pooling_type, cfg)

        # ── FiLM conditioning (optional) ───────────────────────────────────────
        self.film: FiLMLayer | None = FiLMLayer(cfg) if cfg.use_film else None

        # ── Output head ────────────────────────────────────────────────────────
        self.head = HEAD_REGISTRY.build(cfg.head_type, cfg)

    def forward(
        self,
        text_input_ids: torch.Tensor,       # (B, L)
        text_attention_mask: torch.Tensor,  # (B, L)
        obj_clip_features: torch.Tensor,    # (B, N, D_clip)
        obj_bboxes: torch.Tensor,           # (B, N, 6)   [cx,cy,cz, w,h,l] region frame
        obj_is_anchor: torch.Tensor,        # (B, N)      bool / {0,1}
        obj_padding_mask: torch.Tensor,     # (B, N)      True = padded slot
        coord_scale: torch.Tensor,          # (B, 3)      per-axis region size
        coord_shift: torch.Tensor,          # (B, 3)      region centre in world frame
    ):
        B, N, _ = obj_clip_features.shape

        # ── 1. Encode text (CLS token) ─────────────────────────────────────────
        text_feat = self.text_enc(text_input_ids, text_attention_mask)  # (B, D)

        # ── 2. Encode vision + add role / modality tags ────────────────────────
        obj_feat = self.vision_enc(obj_clip_features, obj_is_anchor)    # (B, N, D)

        # ── 3. Compute pairwise spatial relations ──────────────────────────────
        spatial_rel = self.spatial_enc(obj_bboxes)                      # (B, N, N, 12)

        # ── 4. Spatial refinement backbone (e.g. ViSTA) ───────────────────────
        obj_feat = self.backbone(obj_feat, spatial_rel, obj_padding_mask)  # (B, N, D)

        # ── 5. Prepend text CLS and add modality tag ───────────────────────────
        text_feat = text_feat.unsqueeze(1)                              # (B, 1, D)
        text_feat = text_feat + self.modality_embed.weight[self._MODALITY_TEXT]

        seq = torch.cat([text_feat, obj_feat], dim=1)                  # (B, 1+N, D)

        # Extend padding mask: text token is never padded
        text_valid   = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
        fusion_mask  = torch.cat([text_valid, obj_padding_mask], dim=1)  # (B, 1+N)

        # ── 6. Global fusion ───────────────────────────────────────────────────
        seq = self.fusion(seq, fusion_mask)                             # (B, 1+N, D)

        # ── 7. Pool → scene-text context vector ───────────────────────────────
        ctx = self.pool(seq, fusion_mask)                               # (B, D)

        # ── 7b. FiLM conditioning (region scale/shift → γ*ctx + β) ────────────
        if self.film is not None:
            ctx = self.film(ctx, coord_scale, coord_shift)              # (B, D)

        # ── 8. Predict 3-D Gaussian (region frame → world frame) ──────────────
        return self.head(ctx, coord_scale, coord_shift)
