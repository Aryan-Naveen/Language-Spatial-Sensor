"""Input encoders for text, vision, and spatial branches.

All modules assume batched tensor inputs — the tensorizer in
language_spatial_sensor/pipeline/tensorizer.py is responsible for
converting SpatialQuery objects into these tensors before they arrive here.

Tensor conventions (B=batch, N=max_objects, L=text_seq_len):
    text_input_ids      (B, L)       long
    text_attention_mask (B, L)       long  {0,1}
    obj_clip_features   (B, N, D_clip)  float32   pre-extracted, frozen CLIP
    obj_bboxes          (B, N, 6)    float32   [cx, cy, cz, w, h, l]
    obj_is_anchor       (B, N)       bool / {0,1}
    obj_padding_mask    (B, N)       bool  True = padded slot (no real object)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from ..config import LSSConfig


# ── Text encoder ──────────────────────────────────────────────────────────────

class TextEncoder(nn.Module):
    """BERT → CLS token → linear projection to hidden_dim."""

    def __init__(self, cfg: LSSConfig) -> None:
        super().__init__()
        # attn_implementation="sdpa" uses PyTorch SDPA, which dispatches to
        # flash attention automatically when running in bf16/fp16 on CUDA.
        self.bert = AutoModel.from_pretrained(
            cfg.text_model, attn_implementation="sdpa"
        )
        if cfg.freeze_text:
            for p in self.bert.parameters():
                p.requires_grad_(False)
        self.proj = nn.Linear(cfg.bert_dim, cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:                 # (B, D)
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]   # CLS token
        return self.dropout(self.proj(cls))


# ── Vision encoder ─────────────────────────────────────────────────────────────

class VisionEncoder(nn.Module):
    """Project frozen CLIP features and add role + modality embeddings.

    Role embedding encodes anchor vs. non-anchor status using a single shared
    E_anchor vector so permutation invariance between multiple anchors is
    guaranteed — no positional index is used.

    Modality tags ([OBJ] = 1) are added here; the [TEXT] tag (= 0) is added
    by LSSModel when prepending the text CLS token.
    """

    _ROLE_NON_ANCHOR = 0
    _ROLE_ANCHOR     = 1
    _MODALITY_TEXT   = 0
    _MODALITY_OBJ    = 1

    def __init__(self, cfg: LSSConfig) -> None:
        super().__init__()
        self.clip_proj    = nn.Linear(cfg.clip_dim, cfg.hidden_dim)
        # 2-entry embedding: index 0 = non-anchor, index 1 = anchor
        self.role_embed     = nn.Embedding(2, cfg.hidden_dim)
        # 2-entry embedding: index 0 = TEXT, index 1 = OBJ
        # self.modality_embed = nn.Embedding(2, cfg.hidden_dim)
        self.norm    = nn.LayerNorm(cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        clip_features: torch.Tensor,  # (B, N, D_clip)
        is_anchor: torch.Tensor,       # (B, N)  bool / {0,1}
    ) -> torch.Tensor:                 # (B, N, D)
        x = self.clip_proj(clip_features)                          # (B, N, D)
        x = x + self.role_embed(is_anchor.long())                  # broadcast anchor role
        # x = x + self.modality_embed.weight[self._MODALITY_OBJ]     # same OBJ tag for all
        return self.dropout(self.norm(x))


# ── Spatial relation features ──────────────────────────────────────────────────

def calc_pairwise_locs(
    bboxes: torch.Tensor,  # (B, N, 6)  [cx, cy, cz, w, h, l]
    eps: float = 1e-10,
    pairwise_rel_type: str = "mlp",
    spatial_dist_norm: bool = True,
    spatial_dim: int = 12,
) -> torch.Tensor:
    """Pairwise geometry for spatial attention bias.

    - ``mlp``: concat ``[cx,cy,cz,w,h,l]`` for object *i* and object *j* → **12** dims.
    - ``center`` / ``vertical_bottom``: normalized distances and direction ratios →
      ``spatial_dim`` in ``{1, 4, 5}`` (``spatial_dist_norm`` / ``spatial_dim`` ignored for ``mlp``).
    """
    obj_centers = bboxes[..., :3]
    obj_whls = bboxes[..., 3:6]

    if pairwise_rel_type == "mlp":
        obj_locs = torch.cat([obj_centers, obj_whls], dim=-1)  # (B, N, 6)
        n = obj_locs.size(1)
        left = obj_locs.unsqueeze(2).expand(-1, -1, n, -1)
        right = obj_locs.unsqueeze(1).expand(-1, n, -1, -1)
        return torch.cat([left, right], dim=-1)  # (B, N, N, 12)

    pairwise_locs = obj_centers.unsqueeze(2) - obj_centers.unsqueeze(1)  # (B, N, N, 3)
    pairwise_dists = torch.sqrt(torch.sum(pairwise_locs**2, dim=-1) + eps)  # (B, N, N)

    if spatial_dist_norm:
        max_dists = pairwise_dists.reshape(pairwise_dists.size(0), -1).max(dim=1).values.clamp(
            min=eps
        )
        norm_pairwise_dists = pairwise_dists / max_dists.view(-1, 1, 1)
    else:
        norm_pairwise_dists = pairwise_dists

    if spatial_dim == 1:
        return norm_pairwise_dists.unsqueeze(-1)

    pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2] ** 2, dim=-1) + eps)

    if pairwise_rel_type == "center":
        pairwise_feats = torch.stack(
            [
                norm_pairwise_dists,
                pairwise_locs[..., 2] / pairwise_dists,
                pairwise_dists_2d / pairwise_dists,
                pairwise_locs[..., 1] / pairwise_dists_2d,
                pairwise_locs[..., 0] / pairwise_dists_2d,
            ],
            dim=-1,
        )
    elif pairwise_rel_type == "vertical_bottom":
        bottom_centers = obj_centers.clone()
        bottom_centers[:, :, 2] = bottom_centers[:, :, 2] - obj_whls[:, :, 2]
        bottom_pairwise_locs = bottom_centers.unsqueeze(2) - bottom_centers.unsqueeze(1)
        bottom_pairwise_dists = torch.sqrt(torch.sum(bottom_pairwise_locs**2, dim=-1) + eps)
        bottom_pairwise_dists_2d = torch.sqrt(
            torch.sum(bottom_pairwise_locs[..., :2] ** 2, dim=-1) + eps
        )
        pairwise_feats = torch.stack(
            [
                norm_pairwise_dists,
                bottom_pairwise_locs[..., 2] / bottom_pairwise_dists,
                bottom_pairwise_dists_2d / bottom_pairwise_dists,
                pairwise_locs[..., 1] / pairwise_dists_2d,
                pairwise_locs[..., 0] / pairwise_dists_2d,
            ],
            dim=-1,
        )
    else:
        raise ValueError(
            f"pairwise_rel_type must be 'center', 'vertical_bottom', or 'mlp', got {pairwise_rel_type!r}"
        )

    if spatial_dim == 4:
        pairwise_feats = pairwise_feats[..., 1:]
    elif spatial_dim != 5:
        raise ValueError(f"spatial_dim must be 1, 4, or 5 for center/vertical_bottom, got {spatial_dim}")

    return pairwise_feats


class SpatialRelationMLP(nn.Module):
    """Wrap calc_pairwise_locs with an optional learned projection.

    Raw pairwise features (dim ``cfg.spatial_relation_dim``) are projected through
    a small MLP before use as spatial bias in ``MultiHeadAttentionSpatial``.

    When ``cfg.condition_spatial_on_text`` is True, the hidden layer of the MLP is
    FiLM-conditioned on the text CLS embedding.  The FiLM generator is initialised
    so that γ ≈ 1 and β ≈ 0, making the conditioning a near-identity at start and
    letting gradients learn how the query should reshape the pairwise bias (e.g.
    up-weight vertical pairs when the text contains "above").
    """

    def __init__(self, cfg: LSSConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.spatial_relation_dim, cfg.spatial_mlp_hidden)
        self.fc2 = nn.Linear(cfg.spatial_mlp_hidden, cfg.spatial_relation_dim)

        self.text_film: nn.Linear | None = None
        if cfg.condition_spatial_on_text:
            self.text_film = nn.Linear(cfg.hidden_dim, 2 * cfg.spatial_mlp_hidden)
            with torch.no_grad():
                nn.init.zeros_(self.text_film.weight)
                bias = torch.zeros(2 * cfg.spatial_mlp_hidden)
                bias[: cfg.spatial_mlp_hidden] = 1.0
                self.text_film.bias.copy_(bias)

    def forward(
        self,
        bboxes: torch.Tensor,                # (B, N, 6)
        text_cls: torch.Tensor | None = None,  # (B, D)  required when condition_spatial_on_text
    ) -> torch.Tensor:                        # (B, N, N, R)
        raw = calc_pairwise_locs(
            bboxes,
            eps=1e-10,
            pairwise_rel_type=self.cfg.pairwise_rel_type,
            spatial_dist_norm=self.cfg.spatial_pairwise_dist_norm,
            spatial_dim=self.cfg.spatial_relation_dim,
        )
        h = F.relu(self.fc1(raw))             # (B, N, N, H)

        if self.text_film is not None:
            if text_cls is None:
                raise ValueError(
                    "SpatialRelationMLP: condition_spatial_on_text=True but text_cls is None"
                )
            film = self.text_film(text_cls)   # (B, 2H)
            H = self.cfg.spatial_mlp_hidden
            gamma = film[:, :H].unsqueeze(1).unsqueeze(1)  # (B, 1, 1, H)
            beta  = film[:, H:].unsqueeze(1).unsqueeze(1)  # (B, 1, 1, H)
            h = gamma * h + beta

        return self.fc2(h)


# ── Anchor-centric object embedding ──────────────────────────────────────────

class AnchorCentricEmbedding(nn.Module):
    """Add a per-object (centre − anchor-centroid) embedding to obj features.

    Forces the encoder to reason about relative offsets to the referenced
    anchor(s) rather than absolute region-frame coordinates, improving
    generalisation across room layouts.

    Anchor centroid = mean of centres over anchor-tagged (non-padded) objects.
    If no anchor is present, the centroid falls back to the origin.
    """

    def __init__(self, cfg: LSSConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

    def forward(
        self,
        obj_feat: torch.Tensor,       # (B, N, D)
        bboxes: torch.Tensor,         # (B, N, 6)
        is_anchor: torch.Tensor,      # (B, N)
        padding_mask: torch.Tensor,   # (B, N)
    ) -> torch.Tensor:                # (B, N, D)
        centers = bboxes[..., :3]                                  # (B, N, 3)
        anchor_mask = is_anchor.bool() & ~padding_mask.bool()      # (B, N)
        w = anchor_mask.to(centers.dtype).unsqueeze(-1)            # (B, N, 1)
        denom = w.sum(dim=1).clamp(min=1.0)                        # (B, 1)
        centroid = (centers * w).sum(dim=1) / denom                # (B, 3)
        delta = centers - centroid.unsqueeze(1)                    # (B, N, 3)
        return obj_feat + self.mlp(delta)
