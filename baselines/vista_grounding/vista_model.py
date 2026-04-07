"""3D-ViSTA Grounding Model with GMM output head.

Wraps 3D-ViSTA's pretrained scene encoder and text encoder, adds a
GMMCholeskyHead to predict a K-component Gaussian mixture over 3D space.

3D-ViSTA architecture (used here):
    1. PointTokenizeEncoder   — per-object PointNet++ + spatial transformer
                                input:  obj_pcds (B,O,P,6), obj_locs (B,O,6),
                                        obj_masks (B,O), obj_sem_masks (B,O)
                                output: obj_embeds (B,O,768)
    2. BertModel              — text encoding
                                input:  txt_ids (B,L), txt_masks (B,L)
                                output: last_hidden_state (B,L,768)
    3. UnifiedSpatialCrossEncoderV2 — cross-modal fusion
                                input:  lang (B,L,768), obj (B,O,768), locs (B,O,6)
                                output: lang_fused (B,L,768), obj_fused (B,O,768)
    4. Mean pool over valid objects → scene_feature (B, 768)
    5. GMMCholeskyHead        → GMMPrediction (logits, mu, L)

The original ViSTA grounding head (og3d_logits) is replaced by our GMM head.
Pretrained weights for modules 1-3 can be loaded from a ViSTA checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

# ── sys.path setup ─────────────────────────────────────────────────────────────
_LSS_ROOT    = Path(__file__).resolve().parents[2]
_VISTA_ROOT  = _LSS_ROOT / "baselines" / "3dvista"
_COMMON_ROOT = _LSS_ROOT / "baselines"


def _prepend_sys_path(path: Path) -> None:
    """Put *path* at sys.path[0], removing prior copies.

    3D-VisTA uses top-level imports like ``from dataset.path_config``. If
    ``PYTHONPATH`` already contained ``_VISTA_ROOT`` later in the list, the old
    ``if p not in sys.path: insert`` logic skipped reordering, leaving
    ``.../vista_grounding`` (script dir) before ``3dvista`` so ``dataset.py``
    here shadowed the real ``dataset`` package.
    """
    p = str(path.resolve())
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)


# Final order: baselines → 3dvista → lss (each call prepends).
_prepend_sys_path(_LSS_ROOT)
_prepend_sys_path(_VISTA_ROOT)
_prepend_sys_path(_COMMON_ROOT)

# 3D-ViSTA imports (lazy — only needed when this module is imported)
from model.vision.point_encoder import PointTokenizeEncoder
from model.vision.unified_encoder import UnifiedSpatialCrossEncoderV2
from model.language.lang_encoder import get_bert_lang_encoder

from common.gmm import GMMCholeskyHead, GMMPrediction


class ViSTAGroundingModel(nn.Module):
    """3D-ViSTA scene encoder + GMM grounding head.

    Args:
        hidden_size:        Feature dimension of ViSTA modules (default 768).
        num_components:     Number of GMM mixture components K (default 5).
        num_text_layers:    Number of BERT hidden layers (default 4).
        num_spatial_layers: Number of spatial transformer layers in point encoder (default 4).
        spatial_dim:        Pairwise spatial feature dimension (default 5).
        dim_loc:            Object location feature dimension (default 6).
        gmm_hidden_dim:     Hidden dim of the GMM head MLP.
        min_sigma:          Minimum diagonal Cholesky entry (region frame).
        dropout:            Dropout in GMM head MLP.
        vista_ckpt_path:    Optional path to a 3D-ViSTA pretrained checkpoint (.pth).
                            The checkpoint should be a dict with keys:
                            'lang_encoder', 'point_encoder', 'unified_encoder'.
                            Loaded with strict=False so missing/extra keys are ignored.
        freeze_layers:      Number of ViSTA backbone layers to freeze (0 = full finetune,
                            -1 = freeze all ViSTA modules, N > 0 = freeze first N
                            spatial transformer layers in the point encoder).
    """

    def __init__(
        self,
        hidden_size:        int   = 768,
        num_components:     int   = 5,
        num_text_layers:    int   = 4,
        num_spatial_layers: int   = 4,
        spatial_dim:        int   = 5,
        dim_loc:            int   = 6,
        gmm_hidden_dim:     int   = 256,
        min_sigma:          float = 0.001,
        dropout:            float = 0.1,
        vista_ckpt_path:    str   = "",
        freeze_layers:      int   = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # ── ViSTA modules ──────────────────────────────────────────────────────
        self.lang_encoder = get_bert_lang_encoder(num_hidden_layer=num_text_layers)

        self.point_encoder = PointTokenizeEncoder(
            backbone="pointnet++",
            hidden_size=hidden_size,
            freeze_feature=False,
            num_attention_heads=12,
            spatial_dim=spatial_dim,
            num_layers=num_spatial_layers,
            dim_loc=dim_loc,
        )

        self.unified_encoder = UnifiedSpatialCrossEncoderV2(
            hidden_size=hidden_size,
        )

        # ── GMM head ───────────────────────────────────────────────────────────
        self.gmm_head = GMMCholeskyHead(
            in_dim=hidden_size,
            num_components=num_components,
            hidden_dim=gmm_hidden_dim,
            min_sigma=min_sigma,
            dropout=dropout,
        )

        # ── Load pretrained weights ────────────────────────────────────────────
        if vista_ckpt_path:
            self._load_pretrained(vista_ckpt_path)

        # ── Freeze strategy ────────────────────────────────────────────────────
        self._apply_freeze(freeze_layers)

    # ── Weight loading ─────────────────────────────────────────────────────────

    def _load_pretrained(self, ckpt_path: str) -> None:
        """Load ViSTA pretrained weights with strict=False.

        The GMM head is always randomly initialized (its keys won't be in the
        checkpoint).  ViSTA's original grounding/QA/caption heads are ignored.
        """
        print(f"[ViSTAGroundingModel] Loading pretrained weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # ViSTA checkpoints store sub-module state dicts under named keys
        for module_name, module in [
            ("lang_encoder",    self.lang_encoder),
            ("point_encoder",   self.point_encoder),
            ("unified_encoder", self.unified_encoder),
        ]:
            if module_name in ckpt:
                missing, unexpected = module.load_state_dict(
                    ckpt[module_name], strict=False
                )
                print(
                    f"  [{module_name}] loaded — "
                    f"missing: {len(missing)}, unexpected: {len(unexpected)}"
                )
            else:
                print(f"  [{module_name}] key not found in checkpoint, skipping")

    # ── Freeze strategy ────────────────────────────────────────────────────────

    def _apply_freeze(self, n_layers: int) -> None:
        """Freeze ViSTA backbone parameters.

        n_layers = -1:  freeze lang_encoder + point_encoder + unified_encoder
        n_layers =  0:  no freezing (full finetune)
        n_layers >  0:  freeze first n_layers spatial transformer layers in
                        point_encoder; everything else remains trainable
        """
        if n_layers == 0:
            return

        if n_layers == -1:
            for m in [self.lang_encoder, self.point_encoder, self.unified_encoder]:
                for p in m.parameters():
                    p.requires_grad_(False)
            print("[ViSTAGroundingModel] All ViSTA backbone parameters frozen")
            return

        # Freeze first n_layers spatial transformer layers of point encoder
        frozen = 0
        if hasattr(self.point_encoder, "spatial_encoder"):
            layers = list(self.point_encoder.spatial_encoder.layers)
            for layer in layers[:n_layers]:
                for p in layer.parameters():
                    p.requires_grad_(False)
                frozen += 1
        print(
            f"[ViSTAGroundingModel] Froze {frozen} spatial transformer layers "
            f"in point_encoder (requested {n_layers})"
        )

    # ── Optimizer parameter groups ─────────────────────────────────────────────

    def parameter_groups(
        self,
        lr: float,
        lang_lr_scale: float = 0.1,
        backbone_lr_scale: float = 0.1,
    ) -> list[dict]:
        """Return AdamW parameter groups with per-module learning rates.

        Groups:
            1. lang_encoder (BERT) — lr * lang_lr_scale
            2. point_encoder + unified_encoder (ViSTA backbone) — lr * backbone_lr_scale
            3. gmm_head — lr (full)
        """
        lang_ids     = {id(p) for p in self.lang_encoder.parameters()}
        backbone_ids = {
            id(p) for p in list(self.point_encoder.parameters()) +
                            list(self.unified_encoder.parameters())
        }

        head_params     = [p for p in self.gmm_head.parameters() if p.requires_grad]
        lang_params     = [p for p in self.lang_encoder.parameters() if p.requires_grad]
        backbone_params = [
            p for p in list(self.point_encoder.parameters()) +
                       list(self.unified_encoder.parameters())
            if p.requires_grad
        ]

        return [
            {"params": lang_params,     "lr": lr * lang_lr_scale,     "name": "lang"},
            {"params": backbone_params, "lr": lr * backbone_lr_scale, "name": "backbone"},
            {"params": head_params,     "lr": lr,                     "name": "gmm_head"},
        ]

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        obj_pcds:          Tensor,  # (B, O, P, 6)
        obj_locs:          Tensor,  # (B, O, 6)
        obj_masks:         Tensor,  # (B, O) bool, True = valid object
        txt_ids:           Tensor,  # (B, L)
        txt_masks:         Tensor,  # (B, L)
        scene_id:          Any = None,
        language:          Any = None,
        target_xyz_world:  Any = None,
        target_bbox_world: Any = None,
    ) -> GMMPrediction:
        """Forward pass.

        Returns:
            GMMPrediction with shapes:
                logits: (B, K)
                mu:     (B, K, 3)  — world frame
                L:      (B, K, 3, 3)
        """
        # ── 1. Text encoding ───────────────────────────────────────────────────
        # BERT: last_hidden_state (B, L, 768)
        lang_out     = self.lang_encoder(txt_ids, attention_mask=txt_masks)
        lang_feats   = lang_out.last_hidden_state                    # (B, L, 768)

        # ── 2. Point cloud encoding ────────────────────────────────────────────
        # PointTokenizeEncoder returns (obj_embeds, obj_embeds_pre, obj_sem_cls)
        # obj_sem_masks: same as obj_masks (no separate semantic validity mask needed)
        obj_embeds, _, _ = self.point_encoder(
            obj_pcds,               # (B, O, P, 6)
            obj_locs,               # (B, O, 6)
            obj_masks,              # (B, O) bool
            obj_masks,              # obj_sem_masks — reuse obj_masks
        )                           # → (B, O, 768)

        # ── 3. Cross-modal fusion ──────────────────────────────────────────────
        lang_fused, obj_fused = self.unified_encoder(
            lang_feats,             # (B, L, 768)
            txt_masks,              # (B, L)
            obj_embeds,             # (B, O, 768)
            obj_locs,               # (B, O, 6)
            obj_masks,              # (B, O)
        )                           # → (B, L, 768), (B, O, 768)

        # ── 4. Pool over valid objects → scene feature ─────────────────────────
        # Mask out padded objects before averaging
        mask_f = obj_masks.float().unsqueeze(-1)             # (B, O, 1)
        n_valid = obj_masks.sum(dim=-1, keepdim=True).clamp(min=1).float()  # (B, 1)
        scene_feat = (obj_fused * mask_f).sum(dim=1) / n_valid  # (B, 768)

        # ── 5. GMM head ────────────────────────────────────────────────────────
        return self.gmm_head(scene_feat)
