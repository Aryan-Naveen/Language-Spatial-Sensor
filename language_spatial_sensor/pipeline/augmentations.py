"""Data augmentations applied to TensorizerOutput at dataset load time.

Augmentations operate on already-tensorized (region-normalized) data so they
compose cleanly with the precomputed cache.  All transforms are in-place-free
and return new TensorizerOutput instances.

Available:
    RandomMaskObjects  — drop random non-anchor objects to simulate partial views
    RandomSceneRotation — random Z-axis rotation of the full scene in world frame
    RandomObjectJitter  — Gaussian noise on object center positions

Usage::

    augs = Compose([
        RandomMaskObjects(mask_prob=0.3),
        RandomSceneRotation(max_angle_deg=180.0),
        RandomObjectJitter(std=0.02),
    ])
    output = augs(item)   # item: TensorizerOutput
"""

from __future__ import annotations

import math
import random
from dataclasses import replace

import torch

from language_spatial_sensor.core.schema import TensorizerOutput


# ── Compose ───────────────────────────────────────────────────────────────────

class Compose:
    """Apply a sequence of augmentations."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, item: TensorizerOutput) -> TensorizerOutput:
        for t in self.transforms:
            item = t(item)
        return item


# ── (a) Random mask non-anchor objects ───────────────────────────────────────

class RandomMaskObjects:
    """Randomly zero-out non-anchor objects and mark them as padded.

    Simulates partial scene observability — the model must ground the language
    query without seeing every context object.

    Args:
        mask_prob: Per-object probability of masking a non-anchor object.
    """

    def __init__(self, mask_prob: float = 0.3) -> None:
        self.mask_prob = mask_prob

    def __call__(self, item: TensorizerOutput) -> TensorizerOutput:
        N = item.obj_padding_mask.shape[0]

        # Boolean mask: True if eligible for masking (non-anchor, non-padded)
        eligible = ~item.obj_is_anchor & ~item.obj_padding_mask  # (N,)
        # Sample which eligible objects to drop
        rand = torch.rand(N)
        drop = eligible & (rand < self.mask_prob)                # (N,)

        if not drop.any():
            return item

        new_clip   = item.obj_clip_features.clone()
        new_bboxes = item.obj_bboxes.clone()
        new_anchor = item.obj_is_anchor.clone()
        new_pad    = item.obj_padding_mask.clone()

        new_clip[drop]   = 0.0
        new_bboxes[drop] = 0.0
        new_pad[drop]    = True    # treat as padded so attention ignores them

        # Never leave zero real object slots: eligible is only *non-anchor* objects.
        # Statements with no anchors make every object eligible; independent drops
        # can mask 100% of them (e.g. one object → 30% at mask_prob=0.3), which
        # does not happen in the cache but does at train time.
        was_valid = ~item.obj_padding_mask
        if was_valid.any() and bool(new_pad[was_valid].all().item()):
            undo = torch.nonzero(drop & was_valid, as_tuple=False).view(-1)
            j = int(undo[torch.randint(len(undo), (1,)).item()].item())
            new_pad[j] = False
            new_clip[j] = item.obj_clip_features[j]
            new_bboxes[j] = item.obj_bboxes[j]

        return replace(
            item,
            obj_clip_features=new_clip,
            obj_bboxes=new_bboxes,
            obj_is_anchor=new_anchor,
            obj_padding_mask=new_pad,
        )


# ── (b) Random scene rotation (Z-axis) ───────────────────────────────────────

class RandomSceneRotation:
    """Uniform random rotation around the Z (vertical) axis.

    Operates in world frame:
        1. De-normalise object centres: c_world = c_region * scale + shift
        2. Rotate c_world and shift by the same R_z matrix
        3. Re-normalise with the rotated shift: c_region' = (R c_world - R shift) / scale
        4. Rotate target_xyz_world and the target bbox centre

    coord_scale is kept fixed (axis-aligned region dims are approximate after rotation,
    but this is acceptable for regularisation purposes).

    Args:
        max_angle_deg: Rotation sampled uniformly in [-max_angle_deg, +max_angle_deg].
    """

    def __init__(self, max_angle_deg: float = 180.0) -> None:
        self.max_angle_rad = math.radians(max_angle_deg)

    def __call__(self, item: TensorizerOutput) -> TensorizerOutput:
        theta = random.uniform(-self.max_angle_rad, self.max_angle_rad)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        device = item.coord_shift.device
        dtype  = item.coord_shift.dtype

        # 3×3 rotation matrix around Z
        R = torch.tensor(
            [[cos_t, -sin_t, 0.0],
             [sin_t,  cos_t, 0.0],
             [0.0,    0.0,   1.0]],
            dtype=dtype, device=device,
        )

        scale = item.coord_scale   # (3,)
        shift = item.coord_shift   # (3,)

        # ── Rotate object centres ────────────────────────────────────────────
        centers = item.obj_bboxes[:, :3].clone()   # (N, 3) region frame
        sizes   = item.obj_bboxes[:, 3:].clone()   # (N, 3) unchanged

        # de-normalise: c_world = c_region * scale + shift  [broadcast over N]
        centers_world = centers * scale + shift    # (N, 3)

        # rotate in world frame (R @ row_vector = row_vector @ R^T)
        centers_world_rot = centers_world @ R.T    # (N, 3)

        # new region centre after rotation
        shift_rot = R @ shift                      # (3,)

        # re-normalise
        centers_rot = (centers_world_rot - shift_rot) / scale  # (N, 3)

        # zero out padded slots
        centers_rot[item.obj_padding_mask] = 0.0

        new_bboxes = torch.cat([centers_rot, sizes], dim=-1)   # (N, 6)

        # ── Rotate target ────────────────────────────────────────────────────
        target_rot = R @ item.target_xyz_world     # (3,)

        # Rotate bbox: rotate centre, keep half-sizes
        bbox_min = item.target_bbox_world[:3]
        bbox_max = item.target_bbox_world[3:]
        bbox_center = (bbox_min + bbox_max) / 2.0
        bbox_half   = (bbox_max - bbox_min) / 2.0
        bbox_center_rot = R @ bbox_center
        bbox_rot = torch.cat([bbox_center_rot - bbox_half, bbox_center_rot + bbox_half])

        return replace(
            item,
            obj_bboxes=new_bboxes,
            coord_shift=shift_rot,
            target_xyz_world=target_rot,
            target_bbox_world=bbox_rot,
        )


# ── (c) Random object jitter ─────────────────────────────────────────────────

class RandomObjectJitter:
    """Add Gaussian noise to object centre positions (region-normalised frame).

    Simulates localisation noise in the sensor input.  Only non-padded objects
    are jittered.

    Args:
        std: Standard deviation of the noise in region-normalised units.
             (e.g. std=0.02 ≈ 2% of the region side length)
    """

    def __init__(self, std: float = 0.02) -> None:
        self.std = std

    def __call__(self, item: TensorizerOutput) -> TensorizerOutput:
        if self.std <= 0.0:
            return item

        new_bboxes = item.obj_bboxes.clone()
        valid = ~item.obj_padding_mask                       # (N,)

        noise = torch.randn(valid.sum().item(), 3) * self.std
        new_bboxes[valid, :3] = new_bboxes[valid, :3] + noise.to(
            device=new_bboxes.device, dtype=new_bboxes.dtype
        )

        return replace(item, obj_bboxes=new_bboxes)
