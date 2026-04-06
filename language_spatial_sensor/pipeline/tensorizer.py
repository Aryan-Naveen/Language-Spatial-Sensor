"""Tensorizer: converts SpatialQuery → TensorizerOutput for LSSModel.forward().

Typical usage::

    from language_spatial_sensor.core.ontology import VALID_NYU40_LABELS
    from language_spatial_sensor.pipeline.tensorizer import Tensorizer, build_clip_label_map

    # Once at startup — encodes all object labels with frozen CLIP text encoder
    label_map = build_clip_label_map(all_obj_labels)
    tensorizer = Tensorizer(clip_embedding_map=label_map)

    output = tensorizer.tensorize(query)   # TensorizerOutput (no batch dim)
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoTokenizer

from language_spatial_sensor.core.schema import (
    RegionInfo,
    SpatialQuery,
    TensorizerOutput,
)


# ── Public helper ─────────────────────────────────────────────────────────────

def build_clip_label_map(
    labels: list[str],
    clip_model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Encode a list of label strings with a frozen CLIP text encoder.

    Returns a dict mapping each label to a (512,) float32 numpy array.
    Call once before training; pass the result to Tensorizer.
    """
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    model = CLIPTextModel.from_pretrained(clip_model_name).to(device).eval()

    result: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for label in labels:
            inputs = tokenizer(label, return_tensors="pt", padding=True).to(device)
            feat = model(**inputs).pooler_output.squeeze(0).cpu().numpy()
            result[label] = feat.astype(np.float32)

    return result


# ── Tensorizer ────────────────────────────────────────────────────────────────

class Tensorizer:
    """Converts a SpatialQuery into model-ready TensorizerOutput tensors.

    Args:
        tokenizer_name:     HuggingFace tokenizer for text (should match TextEncoder).
        max_text_len:       Padding/truncation length for BERT tokens.
        max_objects:        Max objects per scene; shorter scenes are zero-padded.
        clip_embedding_map: Dict mapping obj.label → (clip_dim,) float32 array.
                            Built once at startup via build_clip_label_map().
                            Missing labels fall back to zero vectors.
        clip_dim:           Dimensionality of CLIP features (512 for ViT-B/32).
    """

    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        max_text_len: int = 64,
        max_objects: int = 100,
        clip_embedding_map: dict[str, np.ndarray] | None = None,
        clip_dim: int = 512,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_text_len = max_text_len
        self.max_objects = max_objects
        self.clip_embedding_map: dict[str, np.ndarray] = clip_embedding_map or {}
        self.clip_dim = clip_dim

    # ── Public entry point ────────────────────────────────────────────────────

    def tensorize(self, query: SpatialQuery) -> TensorizerOutput:
        """Convert a single SpatialQuery into model-ready tensors."""
        # A. Resolve region bounding box
        region_id, region_bbox = self._resolve_region(query)

        # B. Coordinate frame: shift = region center, scale = per-axis size
        coord_shift, coord_scale = self._compute_coord_frame(region_bbox)

        # C. Object crop + normalization
        clip_feats, bboxes, is_anchor, padding_mask = self._tensorize_objects(
            query, region_id, coord_shift, coord_scale
        )

        # D. Text tokenization
        input_ids, attention_mask = self._tokenize(query.language)

        # E. Target supervision — kept in world frame for loss
        target_xyz_world = torch.from_numpy(query.target_xyz.astype(np.float32))
        target_bbox_world = self._resolve_target_bbox(query)

        return TensorizerOutput(
            text_input_ids=input_ids,
            text_attention_mask=attention_mask,
            obj_clip_features=torch.from_numpy(clip_feats),
            obj_bboxes=torch.from_numpy(bboxes),
            obj_is_anchor=torch.from_numpy(is_anchor),
            obj_padding_mask=torch.from_numpy(padding_mask),
            coord_shift=torch.from_numpy(coord_shift),
            coord_scale=torch.from_numpy(coord_scale),
            target_xyz_world=target_xyz_world,
            target_bbox_world=target_bbox_world,
        )

    # ── Internal steps ────────────────────────────────────────────────────────

    def _resolve_region(
        self, query: SpatialQuery
    ) -> tuple[int | None, np.ndarray]:
        """Return (region_id, bbox_6) where bbox_6 = [x_min,y_min,z_min,x_max,y_max,z_max]."""
        region_id = query.gt_anchor_room_id

        if region_id is not None:
            for region in query.scene_graph.regions:
                if region.id == region_id:
                    bbox = self._region_bbox(region)
                    if bbox is not None:
                        return region_id, bbox

        # Fallback: derive bbox from point cloud extent (5th/95th percentile)
        lo = np.percentile(query.pc, 5, axis=0).astype(np.float32)
        hi = np.percentile(query.pc, 95, axis=0).astype(np.float32)
        return None, np.concatenate([lo, hi])

    @staticmethod
    def _region_bbox(region: RegionInfo) -> np.ndarray | None:
        """Extract [x_min,y_min,z_min,x_max,y_max,z_max] from region metadata."""
        m = region.metadata
        keys = ["bbox_x_min", "bbox_y_min", "bbox_z_min", "bbox_x_max", "bbox_y_max", "bbox_z_max"]
        if not all(k in m for k in keys):
            return None
        return np.array([m[k] for k in keys], dtype=np.float32)

    @staticmethod
    def _compute_coord_frame(
        region_bbox: np.ndarray,  # (6,) [x_min,y_min,z_min,x_max,y_max,z_max]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (coord_shift, coord_scale) for normalizing world→region frame.

        Normalization: p_region = (p_world - coord_shift) / coord_scale
        """
        bbox_min = region_bbox[:3]
        bbox_max = region_bbox[3:]
        coord_shift = ((bbox_min + bbox_max) / 2.0).astype(np.float32)
        coord_scale = np.maximum(bbox_max - bbox_min, 1e-3).astype(np.float32)
        return coord_shift, coord_scale

    def _tensorize_objects(
        self,
        query: SpatialQuery,
        region_id: int | None,
        coord_shift: np.ndarray,   # (3,)
        coord_scale: np.ndarray,   # (3,)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Crop objects to region, normalize to region frame, pad to max_objects.

        Returns:
            clip_feats    (max_objects, clip_dim)  float32
            bboxes        (max_objects, 6)         float32  [cx,cy,cz,w,h,l] region frame
            is_anchor     (max_objects,)           bool
            padding_mask  (max_objects,)           bool  True = padded slot
        """
        anchor_ids: set[int] = set(query.gt_anchor_object_ids or [])

        objs = [
            obj for obj in query.scene_graph.objects
            if region_id is None or obj.metadata.get("region_id") == region_id
        ]

        # Truncate if more objects than max_objects
        objs = objs[: self.max_objects]
        n = len(objs)

        clip_feats   = np.zeros((self.max_objects, self.clip_dim), dtype=np.float32)
        bboxes       = np.zeros((self.max_objects, 6),             dtype=np.float32)
        is_anchor    = np.zeros((self.max_objects,),               dtype=bool)
        padding_mask = np.ones((self.max_objects,),                dtype=bool)   # True=padded

        for i, obj in enumerate(objs):
            # CLIP features via label lookup
            clip_feats[i] = self.clip_embedding_map.get(
                obj.label, np.zeros(self.clip_dim, dtype=np.float32)
            )

            # Bbox: 8-corner format (24 floats) → [cx,cy,cz,w,h,l] in region frame
            if obj.bbox is not None:
                corners = np.array(obj.bbox, dtype=np.float32).reshape(8, 3)
                center_world = corners.mean(axis=0)
                size_world = corners.max(axis=0) - corners.min(axis=0)
            else:
                center_world = np.array(obj.position, dtype=np.float32)
                size_world = np.zeros(3, dtype=np.float32)

            center_region = (center_world - coord_shift) / coord_scale
            size_region   = size_world / coord_scale
            bboxes[i]     = np.concatenate([center_region, size_region])

            is_anchor[i]    = obj.id in anchor_ids
            padding_mask[i] = False   # real object slot

        return clip_feats, bboxes, is_anchor, padding_mask

    def _tokenize(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text to (input_ids, attention_mask), both shape (max_text_len,)."""
        enc = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    @staticmethod
    def _resolve_target_bbox(query: SpatialQuery) -> torch.Tensor:
        """Return target bbox (6,) in world frame; fall back to small cube if None."""
        if query.target_bbox is not None:
            return torch.from_numpy(query.target_bbox.astype(np.float32))
        # Fallback: ±0.1 m cube centred at target_xyz
        xyz = query.target_xyz.astype(np.float32)
        eps = np.full(3, 0.1, dtype=np.float32)
        return torch.from_numpy(np.concatenate([xyz - eps, xyz + eps]))
