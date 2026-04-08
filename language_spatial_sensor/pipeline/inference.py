"""LSSInference: load a trained checkpoint and run single-sample inference.

Wraps LSSModel, Tensorizer, and CollateFn so the caller only needs to provide
a Proposal and a base SpatialQuery.

Typical usage::

    from language_spatial_sensor.pipeline.inference import LSSInference

    model = LSSInference("checkpoints/best.pt", device="cuda")
    pred  = model.predict(proposal, base_query)
    # pred.mu: (3,)   predicted 3-D centre in world frame
    # pred.L:  (3, 3) Cholesky factor (Σ = L @ Lᵀ)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import replace
from pathlib import Path

import torch

from language_spatial_sensor.core.schema import SpatialQuery
from language_spatial_sensor.core.ontology import VALID_NYU40_LABELS
from language_spatial_sensor.models import GaussianPrediction, LSSConfig, LSSModel
from language_spatial_sensor.pipeline.tensorizer import Tensorizer, build_clip_label_map
from language_spatial_sensor.training.dataset import CollateFn
from language_spatial_sensor.proposer.schema import Proposal

logger = logging.getLogger(__name__)


class LSSInference:
    """Stateful inference wrapper around a trained LSSModel checkpoint.

    Args:
        ckpt_path:       Path to a .pt checkpoint saved during training.
                         Expected keys: 'model' (state_dict), optionally 'cfg'.
        clip_model_name: CLIP variant used at training time (default ViT-B/32).
        device:          Torch device string, e.g. "cuda" or "cpu".
        max_objects:     Sequence length; should match training config.
        max_text_len:    BERT token length; should match training config.
        extra_labels:    Extra label strings to pre-embed with CLIP beyond the
                         standard NYU40 set.  Pass all unique labels seen in
                         your dataset for best coverage.
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        clip_model_name: str = "ViT-B/32",
        device: str = "cuda",
        max_objects: int = 100,
        max_text_len: int = 64,
        extra_labels: list[str] | None = None,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.max_objects = max_objects
        self.max_text_len = max_text_len

        # ── Load checkpoint ───────────────────────────────────────────────────
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        cfg = self._extract_cfg(ckpt, max_objects, max_text_len)

        # ── Reconstruct model ─────────────────────────────────────────────────
        model = LSSModel(cfg)
        state_dict = ckpt.get("model", ckpt)
        # Strip torch.compile prefix if present
        state_dict = {
            k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(self.device)
        self.model = model
        self.cfg = cfg

        # ── Build CLIP label map ──────────────────────────────────────────────
        all_labels = list(VALID_NYU40_LABELS) + (extra_labels or [])
        logger.info("Building CLIP label map for %d labels …", len(all_labels))
        clip_map = build_clip_label_map(
            list(set(all_labels)),
            clip_model_name=clip_model_name,
            device="cpu",
        )

        # ── Tensorizer + collate ──────────────────────────────────────────────
        self.tensorizer = Tensorizer(
            max_objects=max_objects,
            clip_embedding_map=clip_map,
            clip_dim=cfg.clip_dim,
        )
        self.collate_fn = CollateFn(
            tokenizer_name=cfg.text_model,
            max_text_len=max_text_len,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def predict(
        self,
        proposal: Proposal,
        base_query: SpatialQuery,
    ) -> GaussianPrediction:
        """Predict a 3-D Gaussian for this proposal given the masked scene graph.

        The base_query's scene_graph, pc, and object_split are reused as-is.
        Only the language, anchor identifiers, and region are overridden by
        the proposal.

        Returns:
            GaussianPrediction with mu (3,) and L (3,3) squeezed (batch dim removed).
        """
        modified = self._apply_proposal(base_query, proposal)
        sample   = self.tensorizer.tensorize(modified)
        batch    = self.collate_fn([sample])
        batch    = self._to_device(batch)

        pred: GaussianPrediction = self.model(
            text_input_ids      = batch.text_input_ids,
            text_attention_mask = batch.text_attention_mask,
            obj_clip_features   = batch.obj_clip_features,
            obj_bboxes          = batch.obj_bboxes,
            obj_is_anchor       = batch.obj_is_anchor,
            obj_padding_mask    = batch.obj_padding_mask,
            coord_scale         = batch.coord_scale,
            coord_shift         = batch.coord_shift,
        )

        # Remove batch dimension (B=1)
        return GaussianPrediction(
            mu = pred.mu.squeeze(0).cpu(),   # (3,)
            L  = pred.L.squeeze(0).cpu(),    # (3, 3)
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _apply_proposal(base_query: SpatialQuery, proposal: Proposal) -> SpatialQuery:
        """Return a shallow copy of base_query with proposal overrides applied."""
        return replace(
            base_query,
            language              = proposal.utterance,
            gt_anchor_object_ids  = proposal.anchor_ids,
            gt_anchor_room_id     = proposal.region_id,
        )

    def _to_device(self, batch):
        """Move all tensor fields of a TensorizerOutput to self.device."""
        from dataclasses import fields as dc_fields
        kwargs = {}
        for f in dc_fields(batch):
            val = getattr(batch, f.name)
            if isinstance(val, torch.Tensor):
                kwargs[f.name] = val.to(self.device)
            else:
                kwargs[f.name] = val
        return type(batch)(**kwargs)

    @staticmethod
    def _extract_cfg(ckpt: dict, max_objects: int, max_text_len: int) -> LSSConfig:
        """Reconstruct LSSConfig from checkpoint, with fallback to defaults."""
        raw_cfg = ckpt.get("cfg", {})
        if isinstance(raw_cfg, dict):
            model_cfg = raw_cfg.get("model", raw_cfg)
            # Build LSSConfig from whatever keys are present; unknown keys are ignored
            known = {
                f.name: model_cfg[f.name]
                for f in LSSConfig.__dataclass_fields__.values()  # type: ignore[attr-defined]
                if f.name in model_cfg
            }
            # Override sequence limits if explicitly provided
            known.setdefault("max_objects", max_objects)
            known.setdefault("max_text_len", max_text_len)
            return LSSConfig(**known)
        # cfg stored as LSSConfig directly
        if isinstance(raw_cfg, LSSConfig):
            return replace(raw_cfg, max_objects=max_objects, max_text_len=max_text_len)
        logger.warning("No model cfg found in checkpoint — using LSSConfig defaults")
        return LSSConfig(max_objects=max_objects, max_text_len=max_text_len)
