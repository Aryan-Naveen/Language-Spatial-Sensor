"""Tensorizer: converts SpatialQuery → TensorizerOutput for LSSModel.forward().

Typical usage::

    from language_spatial_sensor.core.ontology import VALID_NYU40_LABELS
    from language_spatial_sensor.pipeline.tensorizer import Tensorizer, build_clip_label_map

    # Once at startup — encodes all object labels with frozen CLIP (openai/clip)
    label_map = build_clip_label_map(all_obj_labels)  # uses clip.load("ViT-B/32")
    tensorizer = Tensorizer(clip_embedding_map=label_map)

    output = tensorizer.tensorize(query)   # CachedSample (no batch dim)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from language_spatial_sensor.core.schema import (
    CachedSample,
    GroundedQuery,
    RegionInfo,
    SceneGraph,
    SpatialQuery,
)


# ── Public helper ─────────────────────────────────────────────────────────────

def build_clip_label_map(
    labels: list[str],
    clip_model_name: str = "ViT-B/32",
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Encode a list of label strings with a frozen CLIP text encoder.

    Uses the openai/clip package (import clip).
    Returns a dict mapping each label to a (512,) float32 numpy array.
    Call once before training; pass the result to Tensorizer.
    """
    import clip

    model, _ = clip.load(clip_model_name, device=device)
    model.eval()

    result: dict[str, np.ndarray] = {}
    with torch.no_grad():
        # clip.tokenize handles truncation to 77 tokens
        tokens = clip.tokenize(labels).to(device)           # (N, 77)
        feats  = model.encode_text(tokens)                  # (N, 512)
        feats  = feats / feats.norm(dim=-1, keepdim=True)   # L2-normalise

    for label, feat in zip(labels, feats):
        result[label] = feat.cpu().float().numpy()

    return result


# ── Tensorizer ────────────────────────────────────────────────────────────────

class Tensorizer:
    """Converts a SpatialQuery into a CachedSample for on-disk storage.

    Text tokenization is intentionally excluded here; it runs at batch time
    inside CollateFn so the tokenizer lives only in the training process.

    Args:
        max_objects:        Max objects per scene; shorter scenes are zero-padded.
        clip_embedding_map: Dict mapping obj.label → (clip_dim,) float32 array.
                            Built once at startup via build_clip_label_map().
                            Missing labels fall back to zero vectors.
        clip_dim:           Dimensionality of CLIP features (512 for ViT-B/32).
    """

    def __init__(
        self,
        max_objects: int = 100,
        clip_embedding_map: dict[str, np.ndarray] | None = None,
        clip_dim: int = 512,
    ) -> None:
        self.max_objects = max_objects
        self.clip_embedding_map: dict[str, np.ndarray] = clip_embedding_map or {}
        self.clip_dim = clip_dim

    # ── Public entry points ───────────────────────────────────────────────────

    def tensorize(self, query: SpatialQuery) -> CachedSample:
        """Convert a single SpatialQuery into a CachedSample for on-disk storage."""
        return self.tensorize_batch([query])[0]

    def tensorize_batch(self, queries: Sequence[SpatialQuery]) -> list[CachedSample]:
        """Convert many SpatialQueries with batched NumPy work and few torch conversions."""
        grounded = [GroundedQuery.from_spatial_query(q) for q in queries]
        return self.tensorize_grounded_batch(grounded)

    def tensorize_grounded(self, query: GroundedQuery) -> CachedSample:
        """Convert a single GroundedQuery into a CachedSample.

        Use this at inference time when a Proposer supplies the grounding hypothesis.
        """
        return self.tensorize_grounded_batch([query])[0]

    def tensorize_grounded_batch(self, queries: Sequence[GroundedQuery]) -> list[CachedSample]:
        """Like tensorize_batch but accepts GroundedQuery objects.

        The grounding (anchor_room_id, anchor_object_ids, language) may come from
        ground-truth annotations *or* from a Proposer — the tensorizer does not care.
        """
        if not queries:
            return []

        B = len(queries)
        region_bboxes = np.zeros((B, 6), dtype=np.float32)
        region_ids: list[int] = []
        for i, q in enumerate(queries):
            rid, bbox = self._resolve_region(q.scene_id, q.scene_graph, q.grounding.anchor_room_id)
            region_ids.append(rid)
            region_bboxes[i] = bbox

        coord_shift_b, coord_scale_b = self._compute_coord_frame_batched(region_bboxes)

        clip_feats = np.zeros((B, self.max_objects, self.clip_dim), dtype=np.float32)
        bboxes = np.zeros((B, self.max_objects, 6), dtype=np.float32)
        is_anchor = np.zeros((B, self.max_objects), dtype=bool)
        padding_mask = np.ones((B, self.max_objects), dtype=bool)

        for i, q in enumerate(queries):
            self._tensorize_objects_into(
                q.scene_graph,
                q.grounding.anchor_object_ids,
                region_ids[i],
                coord_shift_b[i],
                coord_scale_b[i],
                clip_feats[i],
                bboxes[i],
                is_anchor[i],
                padding_mask[i],
            )

        target_xyz = np.stack([
            q.target_xyz.astype(np.float32) if q.target_xyz is not None
            else np.zeros(3, dtype=np.float32)
            for q in queries
        ], axis=0)
        target_bbox_world = self._stack_target_bbox_world_grounded(queries)

        t_clip = torch.from_numpy(clip_feats)
        t_obj_bbox = torch.from_numpy(bboxes)
        t_is_anchor = torch.from_numpy(is_anchor)
        t_pad = torch.from_numpy(padding_mask)
        t_shift = torch.from_numpy(coord_shift_b)
        t_scale = torch.from_numpy(coord_scale_b)
        t_tgt_xyz = torch.from_numpy(target_xyz)
        t_tgt_bbox = torch.from_numpy(target_bbox_world)

        return [
            CachedSample(
                language=queries[i].grounding.language,
                obj_clip_features=t_clip[i],
                obj_bboxes=t_obj_bbox[i],
                obj_is_anchor=t_is_anchor[i],
                obj_padding_mask=t_pad[i],
                coord_shift=t_shift[i],
                coord_scale=t_scale[i],
                target_xyz_world=t_tgt_xyz[i],
                target_bbox_world=t_tgt_bbox[i],
            )
            for i in range(B)
        ]

    # ── Internal steps ────────────────────────────────────────────────────────

    def _resolve_region(
        self,
        scene_id: str,
        scene_graph: SceneGraph,
        anchor_room_id: int,
    ) -> tuple[int, np.ndarray]:
        """Return (region_id, bbox_6) where bbox_6 = [x_min,y_min,z_min,x_max,y_max,z_max]."""
        for region in scene_graph.regions:
            if region.id == anchor_room_id:
                bbox = self._region_bbox(region)
                if bbox is not None:
                    return anchor_room_id, bbox

        raise ValueError(f"Region ID {anchor_room_id} not found in scene graph for scene '{scene_id}'")

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

    @staticmethod
    def _compute_coord_frame_batched(
        region_bboxes: np.ndarray,  # (B, 6)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized coord frame; same math as `_compute_coord_frame` per row."""
        bbox_min = region_bboxes[:, :3]
        bbox_max = region_bboxes[:, 3:]
        coord_shift = ((bbox_min + bbox_max) / 2.0).astype(np.float32)
        coord_scale = np.maximum(bbox_max - bbox_min, 1e-3).astype(np.float32)
        return coord_shift, coord_scale

    def _tensorize_objects_into(
        self,
        scene_graph: SceneGraph,
        anchor_object_ids: list[int],
        region_id: int | None,
        coord_shift: np.ndarray,      # (3,)
        coord_scale: np.ndarray,      # (3,)
        out_clip: np.ndarray,         # (max_objects, clip_dim)
        out_bboxes: np.ndarray,       # (max_objects, 6)
        out_is_anchor: np.ndarray,    # (max_objects,)
        out_padding_mask: np.ndarray, # (max_objects,)
    ) -> None:
        """Crop objects to region, normalize to region frame, pad to max_objects (in-place)."""
        anchor_ids: set[int] = set(anchor_object_ids)

        objs = [
            obj for obj in scene_graph.objects
            if region_id is None or obj.metadata.get("region_id") == region_id
        ]

        objs = objs[: self.max_objects]

        out_padding_mask[:] = True

        for i, obj in enumerate(objs):
            out_clip[i] = self.clip_embedding_map.get(
                obj.label, np.zeros(self.clip_dim, dtype=np.float32)
            )

            if obj.bbox is not None:
                corners = np.array(obj.bbox, dtype=np.float32).reshape(8, 3)
                center_world = corners.mean(axis=0)
                size_world = corners.max(axis=0) - corners.min(axis=0)
            else:
                center_world = np.array(obj.position, dtype=np.float32)
                size_world = np.zeros(3, dtype=np.float32)

            center_region = (center_world - coord_shift) / coord_scale
            size_region   = size_world / coord_scale
            out_bboxes[i] = np.concatenate([center_region, size_region])

            out_is_anchor[i] = obj.id in anchor_ids
            out_padding_mask[i] = False

    @staticmethod
    def _target_bbox_world_numpy(query: SpatialQuery) -> np.ndarray:
        """Target bbox (6,) AABB [x_min,y_min,z_min,x_max,y_max,z_max] in world frame.

        Scene-graph ``target_bbox`` may be either a 6-vector (already AABB) or 8×3
        corners (24 floats), matching ``ObjectInfo.bbox`` elsewhere in the pipeline.
        """
        tb = query.target_bbox
        if tb is not None:
            arr = np.asarray(tb, dtype=np.float32).reshape(-1)
            if arr.size == 6:
                return arr.astype(np.float32)
            if arr.size == 24:
                corners = arr.reshape(8, 3)
                cmin = corners.min(axis=0)
                cmax = corners.max(axis=0)
                return np.concatenate([cmin, cmax]).astype(np.float32)
            raise ValueError(
                f"target_bbox must have 6 (AABB) or 24 (8 corners) values, got size {arr.size}"
            )
        xyz = query.target_xyz.astype(np.float32)
        eps = np.full(3, 0.1, dtype=np.float32)
        return np.concatenate([xyz - eps, xyz + eps]).astype(np.float32)

    def _stack_target_bbox_world_numpy(self, queries: Sequence[SpatialQuery]) -> np.ndarray:
        """Stack per-query target bbox (6,) → (B, 6)."""
        return np.stack(
            [self._target_bbox_world_numpy(q) for q in queries],
            axis=0,
        ).astype(np.float32)

    def _stack_target_bbox_world_grounded(self, queries: Sequence[GroundedQuery]) -> np.ndarray:
        """Stack per-GroundedQuery target bbox (6,) → (B, 6). Returns zeros when target is unknown."""
        rows = []
        for q in queries:
            if q.target_bbox is not None or q.target_xyz is not None:
                # Reuse SpatialQuery helper by constructing a minimal adapter
                tb = q.target_bbox
                xyz = q.target_xyz
                if tb is not None:
                    arr = np.asarray(tb, dtype=np.float32).reshape(-1)
                    if arr.size == 6:
                        rows.append(arr)
                        continue
                    if arr.size == 24:
                        corners = arr.reshape(8, 3)
                        rows.append(np.concatenate([corners.min(0), corners.max(0)]).astype(np.float32))
                        continue
                if xyz is not None:
                    eps = np.full(3, 0.1, dtype=np.float32)
                    xyz32 = xyz.astype(np.float32)
                    rows.append(np.concatenate([xyz32 - eps, xyz32 + eps]))
                    continue
            rows.append(np.zeros(6, dtype=np.float32))
        return np.stack(rows, axis=0).astype(np.float32)
