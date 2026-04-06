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
        self.bert = AutoModel.from_pretrained(cfg.text_model)
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
    bboxes: torch.Tensor,   # (B, N, 6)  [cx, cy, cz, w, h, l]
    eps: float = 1e-6,
) -> torch.Tensor:          # (B, N, N, 12)
    """Compute 12-D geometric pairwise features between all object pairs.

    Feature layout (12 total):
        [0:3]   delta_center  — i→j offset in world coords
        [3]     log_dist      — log L2 distance between centres
        [4:7]   unit_dir      — unit direction vector i→j
        [7:10]  log_size_ratio— log(w_i/w_j, h_i/h_j, l_i/l_j)
        [10]    log_vol_ratio — log(vol_i / vol_j)
        [11]    norm_dist     — dist / (cbrt(vol_i) + cbrt(vol_j))
    """
    centers = bboxes[..., :3]   # (B, N, 3)
    sizes   = bboxes[..., 3:]   # (B, N, 3)  W H L

    # (B, N, N, 3): offset from j to i  (i = row, j = col)
    delta = centers.unsqueeze(2) - centers.unsqueeze(1)             # (B, N, N, 3)

    dist     = delta.norm(dim=-1, keepdim=True).clamp(min=eps)      # (B, N, N, 1)
    log_dist = dist.log()                                            # (B, N, N, 1)
    unit_dir = delta / dist                                          # (B, N, N, 3)

    si = sizes.unsqueeze(2).expand_as(delta)                        # (B, N, N, 3)
    sj = sizes.unsqueeze(1).expand_as(delta)                        # (B, N, N, 3)

    log_size_ratio = (si / sj.clamp(min=eps)).log()                 # (B, N, N, 3)

    vol_i     = si.prod(dim=-1, keepdim=True)                       # (B, N, N, 1)
    vol_j     = sj.prod(dim=-1, keepdim=True)
    log_vol_ratio = (vol_i / vol_j.clamp(min=eps)).log()            # (B, N, N, 1)

    scale     = (vol_i.clamp(min=eps) ** (1/3) +
                 vol_j.clamp(min=eps) ** (1/3)).clamp(min=eps)
    norm_dist = dist / scale                                         # (B, N, N, 1)

    return torch.cat(
        [delta, log_dist, unit_dir, log_size_ratio, log_vol_ratio, norm_dist],
        dim=-1,
    )   # (B, N, N, 3+1+3+3+1+1) == (B, N, N, 12)


class SpatialRelationMLP(nn.Module):
    """Wrap calc_pairwise_locs with an optional learned projection.

    The raw 12-D geometric features are projected through a small MLP before
    being used as spatial bias in MultiHeadAttentionSpatial.  Keeping this as
    a module (rather than a bare function) lets the projection be learned
    end-to-end and makes it easy to swap the raw-feature formula later.
    """

    def __init__(self, cfg: LSSConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.spatial_relation_dim, cfg.spatial_mlp_hidden),
            nn.ReLU(),
            nn.Linear(cfg.spatial_mlp_hidden, cfg.spatial_relation_dim),
        )

    def forward(self, bboxes: torch.Tensor) -> torch.Tensor:
        # (B, N, 6) → (B, N, N, 12)
        raw = calc_pairwise_locs(bboxes)
        return self.mlp(raw)
