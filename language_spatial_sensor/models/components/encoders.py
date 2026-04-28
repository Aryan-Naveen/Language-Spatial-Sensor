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
    """HuggingFace text backbone → pooled vector → linear projection to hidden_dim.

    The backbone hidden size is read from the loaded model's config, so every
    HF checkpoint (bert-base/large, distilbert, roberta, sentence-transformers,
    …) is accepted without updating ``cfg.bert_dim``. ``cfg.bert_dim`` is
    ignored and kept only for config back-compat.

    Pooling:
      * ``sentence-transformers/*`` — trained with attention-mask-weighted
        mean pooling followed by L2 normalisation; we replicate both so the
        checkpoint is used as intended.
      * everything else — CLS token (position 0), which is how BERT/RoBERTa/
        DistilBERT/MPNet expose a sequence summary.
    """

    def __init__(self, cfg: LSSConfig) -> None:
        super().__init__()
        # SDPA dispatches to flash attention automatically in bf16/fp16 on
        # CUDA. Not every HF model class supports it — fall back to eager
        # rather than failing a sweep trial on older architectures.
        try:
            self.bert = AutoModel.from_pretrained(
                cfg.text_model, attn_implementation="sdpa"
            )
        except (ValueError, TypeError):
            self.bert = AutoModel.from_pretrained(cfg.text_model)

        if cfg.freeze_text:
            for p in self.bert.parameters():
                p.requires_grad_(False)

        text_hidden = int(self.bert.config.hidden_size)
        self.proj = nn.Linear(text_hidden, cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

        self.pool_mode: str = (
            "mean" if cfg.text_model.startswith("sentence-transformers/") else "cls"
        )

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:                 # (B, D)
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, L, D_text)

        if self.pool_mode == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)   # (B, L, 1)
            summed = (hidden * mask).sum(dim=1)                     # (B, D_text)
            denom = mask.sum(dim=1).clamp(min=1.0)                  # (B, 1)
            pooled = F.normalize(summed / denom, p=2, dim=-1)
        else:
            pooled = hidden[:, 0]       # CLS / <s>

        return self.dropout(self.proj(pooled))


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

def _topological_pairwise_feats(
    obj_centers: torch.Tensor,  # (B, N, 3)
    obj_whls: torch.Tensor,     # (B, N, 3)
    eps: float = 1e-10,
) -> torch.Tensor:
    """Topology features: IoU, vertical support, axis gaps → (B, N, N, 7).

    Distinguishes containment / contact / support, which center-to-center
    distance alone conflates.  Channels:
      0: 3D IoU of AABBs
      1: vertical support  z_bottom_i - z_top_j   (≈0 ⇒ i sits on top of j)
      2: vertical support  z_bottom_j - z_top_i   (≈0 ⇒ j sits on top of i)
      3: signed axis gap along x  (negative ⇒ overlap)
      4: signed axis gap along y
      5: signed axis gap along z
      6: surface-to-surface L2 distance  sqrt(max(0,gap_x)^2 + ... )
    """
    half = obj_whls * 0.5
    lo = obj_centers - half            # (B, N, 3)  min corner
    hi = obj_centers + half            # (B, N, 3)  max corner

    lo_i = lo.unsqueeze(2); lo_j = lo.unsqueeze(1)     # (B, N, 1, 3), (B, 1, N, 3)
    hi_i = hi.unsqueeze(2); hi_j = hi.unsqueeze(1)

    # Intersection / union for 3D IoU
    inter_lo = torch.maximum(lo_i, lo_j)
    inter_hi = torch.minimum(hi_i, hi_j)
    inter_wlh = (inter_hi - inter_lo).clamp(min=0.0)
    inter_vol = inter_wlh.prod(dim=-1)                 # (B, N, N)
    vol_i = obj_whls.prod(dim=-1).unsqueeze(2)
    vol_j = obj_whls.prod(dim=-1).unsqueeze(1)
    union_vol = (vol_i + vol_j - inter_vol).clamp(min=eps)
    iou = inter_vol / union_vol                        # (B, N, N)

    # Vertical support: z_bottom_i == z_top_j ⇒ i rests on top of j
    z_bot_i = lo[..., 2].unsqueeze(2); z_top_i = hi[..., 2].unsqueeze(2)
    z_bot_j = lo[..., 2].unsqueeze(1); z_top_j = hi[..., 2].unsqueeze(1)
    v_support_ij = z_bot_i - z_top_j                   # (B, N, N)
    v_support_ji = z_bot_j - z_top_i

    # Signed axis gaps: max(lo_i, lo_j) - min(hi_i, hi_j) per axis.
    # Negative ⇒ axes overlap; positive ⇒ gap length.
    gap_xyz = inter_lo - inter_hi                      # (B, N, N, 3)  == -(inter_hi - inter_lo)
    gap_pos = gap_xyz.clamp(min=0.0)
    surface_dist = torch.sqrt((gap_pos ** 2).sum(dim=-1) + eps)  # (B, N, N)

    return torch.stack(
        [
            iou,
            v_support_ij,
            v_support_ji,
            gap_xyz[..., 0],
            gap_xyz[..., 1],
            gap_xyz[..., 2],
            surface_dist,
        ],
        dim=-1,
    )  # (B, N, N, 7)


def _geometric_algebra_pairwise_feats(
    obj_centers: torch.Tensor,  # (B, N, 3)
    obj_whls: torch.Tensor,     # (B, N, 3)
) -> torch.Tensor:
    """Grade decomposition of the pair (c_i, c_j) in 3D → (B, N, N, 7).

      vector (3):   c_i - c_j                       relative displacement
      bivector (3): c_i ∧ c_j                       oriented area of the pair (plane of interaction)
      trivector (1): det([c_i, c_j, s_i + s_j])     signed volume — 0 iff c_i, c_j, and
                                                    the combined size vector are coplanar, so
                                                    it carries size-aware chirality info that is
                                                    absent from the pure 2-vector wedge.
    """
    c_i = obj_centers.unsqueeze(2)        # (B, N, 1, 3)
    c_j = obj_centers.unsqueeze(1)        # (B, 1, N, 3)

    vec = c_i - c_j                       # (B, N, N, 3)

    # Bivector (wedge): in 3D with basis e_x∧e_y, e_x∧e_z, e_y∧e_z
    bi_xy = c_i[..., 0] * c_j[..., 1] - c_i[..., 1] * c_j[..., 0]
    bi_xz = c_i[..., 0] * c_j[..., 2] - c_i[..., 2] * c_j[..., 0]
    bi_yz = c_i[..., 1] * c_j[..., 2] - c_i[..., 2] * c_j[..., 1]

    # Trivector: signed 3-volume spanned by (c_i, c_j, s_i + s_j).
    s_sum = (obj_whls.unsqueeze(2) + obj_whls.unsqueeze(1))  # (B, N, N, 3)
    cross_ij = torch.cross(
        c_i.expand_as(s_sum), c_j.expand_as(s_sum), dim=-1
    )                                                         # (B, N, N, 3)
    tri = (cross_ij * s_sum).sum(dim=-1)                      # (B, N, N)

    return torch.stack(
        [
            vec[..., 0], vec[..., 1], vec[..., 2],
            bi_xy, bi_xz, bi_yz,
            tri,
        ],
        dim=-1,
    )  # (B, N, N, 7)


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
    - ``topological``: IoU / vertical-support / surface-gap signature → **7** dims.
      Targets prepositions that center distance conflates: "on", "inside", "touching".
    - ``geometric_algebra``: vector + bivector + trivector grade decomposition → **7** dims.
      Captures the "plane of interaction" (bivector) and size-aware chirality (trivector).
    """
    obj_centers = bboxes[..., :3]
    obj_whls = bboxes[..., 3:6]

    if pairwise_rel_type == "mlp":
        obj_locs = torch.cat([obj_centers, obj_whls], dim=-1)  # (B, N, 6)
        n = obj_locs.size(1)
        left = obj_locs.unsqueeze(2).expand(-1, -1, n, -1)
        right = obj_locs.unsqueeze(1).expand(-1, n, -1, -1)
        return torch.cat([left, right], dim=-1)  # (B, N, N, 12)

    if pairwise_rel_type == "topological":
        if spatial_dim != 7:
            raise ValueError(f"spatial_dim must be 7 for 'topological', got {spatial_dim}")
        return _topological_pairwise_feats(obj_centers, obj_whls, eps=eps)

    if pairwise_rel_type == "geometric_algebra":
        if spatial_dim != 7:
            raise ValueError(f"spatial_dim must be 7 for 'geometric_algebra', got {spatial_dim}")
        return _geometric_algebra_pairwise_feats(obj_centers, obj_whls)

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
            "pairwise_rel_type must be one of "
            "'mlp', 'center', 'vertical_bottom', 'topological', 'geometric_algebra'; "
            f"got {pairwise_rel_type!r}"
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
    conditioned on the text CLS embedding.  ``cfg.spatial_conditioning_type``
    selects the conditioning mechanism:
      * ``"film"``      — affine modulation h ← γ·h + β (γ≈1, β≈0 at init).
      * ``"gate"``      — multiplicative sigmoid gate h ← σ(Wt)·h (gate≈1 at init).
      * ``"film_gate"`` — gate first, then FiLM; lets the text both suppress
                          channels (gate ∈ [0,1]) and shift/rescale them (FiLM).
    All generators are initialised so that conditioning is a near-identity at
    start, letting gradients learn how the query should reshape the pairwise
    bias (e.g. up-weight vertical pairs when the text contains "above").
    """

    _COND_TYPES = ("film", "gate", "film_gate")

    def __init__(self, cfg: LSSConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.spatial_relation_dim, cfg.spatial_mlp_hidden)
        self.fc2 = nn.Linear(cfg.spatial_mlp_hidden, cfg.spatial_relation_dim)

        self.text_film: nn.Linear | None = None
        self.text_gate: nn.Linear | None = None

        if cfg.condition_spatial_on_text:
            if cfg.spatial_conditioning_type not in self._COND_TYPES:
                raise ValueError(
                    f"spatial_conditioning_type must be one of {self._COND_TYPES}, "
                    f"got {cfg.spatial_conditioning_type!r}"
                )

            if cfg.spatial_conditioning_type in ("film", "film_gate"):
                self.text_film = nn.Linear(cfg.hidden_dim, 2 * cfg.spatial_mlp_hidden)
                with torch.no_grad():
                    nn.init.zeros_(self.text_film.weight)
                    bias = torch.zeros(2 * cfg.spatial_mlp_hidden)
                    bias[: cfg.spatial_mlp_hidden] = 1.0   # γ init = 1, β init = 0
                    self.text_film.bias.copy_(bias)

            if cfg.spatial_conditioning_type in ("gate", "film_gate"):
                # Init so sigmoid(bias)=1 → gate is a near-identity at start.
                # Large positive bias (+4 → σ≈0.982) is standard for "warm"
                # gates that the optimiser can softly close as needed.
                self.text_gate = nn.Linear(cfg.hidden_dim, cfg.spatial_mlp_hidden)
                with torch.no_grad():
                    nn.init.zeros_(self.text_gate.weight)
                    nn.init.constant_(self.text_gate.bias, 4.0)

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

        if self.text_film is not None or self.text_gate is not None:
            if text_cls is None:
                raise ValueError(
                    "SpatialRelationMLP: condition_spatial_on_text=True but text_cls is None"
                )

            if self.text_gate is not None:
                gate = torch.sigmoid(self.text_gate(text_cls))     # (B, H)
                h = gate.unsqueeze(1).unsqueeze(1) * h              # (B, 1, 1, H) * (B, N, N, H)

            if self.text_film is not None:
                film = self.text_film(text_cls)                     # (B, 2H)
                H = self.cfg.spatial_mlp_hidden
                gamma = film[:, :H].unsqueeze(1).unsqueeze(1)
                beta  = film[:, H:].unsqueeze(1).unsqueeze(1)
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
