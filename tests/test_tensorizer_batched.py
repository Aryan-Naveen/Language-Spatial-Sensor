"""Tests for batched tensorizer coord frame and tensorize_batch parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from language_spatial_sensor.core.schema import ObjectInfo, RegionInfo, SceneGraph, SpatialQuery
from language_spatial_sensor.pipeline.tensorizer import Tensorizer


def _unit_cube_bbox_24() -> list[float]:
    pts = []
    for x in (0.0, 2.0):
        for y in (0.0, 3.0):
            for z in (0.0, 4.0):
                pts.extend([x, y, z])
    return pts


def _region_meta(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> dict:
    return {
        "bbox_x_min": x0,
        "bbox_y_min": y0,
        "bbox_z_min": z0,
        "bbox_x_max": x1,
        "bbox_y_max": y1,
        "bbox_z_max": z1,
    }


@pytest.mark.parametrize("batch_size", [1, 3, 16])
def test_compute_coord_frame_batched_matches_scalar(batch_size: int) -> None:
    rng = np.random.default_rng(42)
    # valid boxes: min < max per axis
    lows = rng.uniform(-5.0, 5.0, size=(batch_size, 3)).astype(np.float32)
    spans = rng.uniform(0.05, 10.0, size=(batch_size, 3)).astype(np.float32)
    highs = lows + spans
    region_bboxes = np.concatenate([lows, highs], axis=1).astype(np.float32)

    shift_b, scale_b = Tensorizer._compute_coord_frame_batched(region_bboxes)

    assert shift_b.shape == (batch_size, 3)
    assert scale_b.shape == (batch_size, 3)
    assert shift_b.dtype == np.float32
    assert scale_b.dtype == np.float32

    for i in range(batch_size):
        s, sc = Tensorizer._compute_coord_frame(region_bboxes[i])
        np.testing.assert_allclose(shift_b[i], s, rtol=0, atol=1e-6)
        np.testing.assert_allclose(scale_b[i], sc, rtol=0, atol=1e-6)


def test_compute_coord_frame_batched_degenerate_axis_clamps_scale() -> None:
    # Thin along X: span 1e-4 → scale should be 1e-3
    region = np.array(
        [0.0, 0.0, 0.0, 1e-4, 5.0, 5.0],
        dtype=np.float32,
    )
    shift, scale = Tensorizer._compute_coord_frame_batched(region.reshape(1, 6))
    np.testing.assert_allclose(scale[0, 0], 1e-3, rtol=0, atol=1e-9)
    np.testing.assert_allclose(scale[0, 1], 5.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(scale[0, 2], 5.0, rtol=0, atol=1e-6)


def test_compute_coord_frame_batched_hand_room_box() -> None:
    region = np.array([0.0, 0.0, 0.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(1, 6)
    shift, scale = Tensorizer._compute_coord_frame_batched(region)
    np.testing.assert_allclose(shift[0], [1.0, 1.5, 2.0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(scale[0], [2.0, 3.0, 4.0], rtol=0, atol=1e-6)


def _make_tensorizer() -> Tensorizer:
    dim = 512
    return Tensorizer(
        max_objects=10,
        clip_embedding_map={"chair": np.ones(dim, dtype=np.float32) * 0.25, "table": np.ones(dim, dtype=np.float32) * 0.5},
        clip_dim=dim,
    )


def test_target_bbox_world_numpy_corners_to_aabb() -> None:
    """8×3 corner list (24 floats) must become a 6-vector AABB."""
    corners = np.array(_unit_cube_bbox_24(), dtype=np.float32)
    q = SpatialQuery(
        scene_id="corners",
        scene_graph=SceneGraph(objects=[], regions=[]),
        pc=None,
        object_split=None,
        language="",
        target_xyz=np.zeros(3, dtype=np.float32),
        target_bbox=corners,
        gt_anchor_object_ids=None,
        gt_anchor_room_id=None,
    )
    out = Tensorizer._target_bbox_world_numpy(q)
    np.testing.assert_allclose(
        out, [0.0, 0.0, 0.0, 2.0, 3.0, 4.0], rtol=0, atol=1e-5
    )


def _spatial_query_target1() -> SpatialQuery:
    """Target id=1; remaining object id=2 in region 1."""
    sg = SceneGraph(
        objects=[
            ObjectInfo(
                id=2,
                label="table",
                position=[5.0, 5.0, 1.0],
                bbox=_unit_cube_bbox_24(),
                metadata={"region_id": 1},
            ),
        ],
        regions=[
            RegionInfo(
                id=1,
                label="room",
                position=[0.0, 0.0, 0.0],
                metadata=_region_meta(0.0, 0.0, 0.0, 10.0, 10.0, 3.0),
            ),
        ],
    )
    pc = np.random.randn(50, 3).astype(np.float32) * 0.5 + np.array([5.0, 5.0, 1.0], dtype=np.float32)
    split = np.ones(50, dtype=np.int64) * 2
    return SpatialQuery(
        scene_id="synth_a",
        scene_graph=sg,
        pc=pc,
        object_split=split,
        language="pick the table",
        target_xyz=np.array([9.0, 9.0, 2.0], dtype=np.float32),
        target_bbox=None,
        gt_anchor_object_ids=[2],
        gt_anchor_room_id=1,
    )


def _spatial_query_target2() -> SpatialQuery:
    """Target id=2; remaining object id=1 in region 1."""
    sg = SceneGraph(
        objects=[
            ObjectInfo(
                id=1,
                label="chair",
                position=[1.0, 2.0, 0.5],
                bbox=None,
                metadata={"region_id": 1},
            ),
        ],
        regions=[
            RegionInfo(
                id=1,
                label="room",
                position=[0.0, 0.0, 0.0],
                metadata=_region_meta(0.0, 0.0, 0.0, 10.0, 10.0, 3.0),
            ),
        ],
    )
    pc = np.random.randn(40, 3).astype(np.float32) * 0.2
    split = np.ones(40, dtype=np.int64)
    return SpatialQuery(
        scene_id="synth_b",
        scene_graph=sg,
        pc=pc,
        object_split=split,
        language="pick the chair",
        target_xyz=np.array([1.0, 2.0, 0.5], dtype=np.float32),
        target_bbox=np.array([0.0, 0.0, 0.0, 2.0, 3.0, 1.0], dtype=np.float32),
        gt_anchor_object_ids=None,
        gt_anchor_room_id=1,
    )


def _assert_cached_close(a, b) -> None:
    assert a.language == b.language
    torch.testing.assert_close(a.obj_clip_features, b.obj_clip_features)
    torch.testing.assert_close(a.obj_bboxes, b.obj_bboxes)
    torch.testing.assert_close(a.obj_is_anchor, b.obj_is_anchor)
    torch.testing.assert_close(a.obj_padding_mask, b.obj_padding_mask)
    torch.testing.assert_close(a.coord_shift, b.coord_shift)
    torch.testing.assert_close(a.coord_scale, b.coord_scale)
    torch.testing.assert_close(a.target_xyz_world, b.target_xyz_world)
    torch.testing.assert_close(a.target_bbox_world, b.target_bbox_world)


def test_tensorize_batch_matches_per_item() -> None:
    np.random.seed(0)
    t = _make_tensorizer()
    q1 = _spatial_query_target1()
    q2 = _spatial_query_target2()

    batch = t.tensorize_batch([q1, q2])
    one = [t.tensorize(q1), t.tensorize(q2)]

    assert len(batch) == 2
    _assert_cached_close(batch[0], one[0])
    _assert_cached_close(batch[1], one[1])


def test_tensorize_batch_single_matches_tensorize() -> None:
    np.random.seed(1)
    t = _make_tensorizer()
    q = _spatial_query_target1()
    _assert_cached_close(t.tensorize_batch([q])[0], t.tensorize(q))


def test_tensorize_batch_empty() -> None:
    t = _make_tensorizer()
    assert t.tensorize_batch([]) == []


def test_tensorize_empty_pointcloud_percentile_fallback() -> None:
    """No anchor region bbox + empty PC (all points removed with target) must not crash."""
    t = _make_tensorizer()
    sg = SceneGraph(
        objects=[
            ObjectInfo(
                id=1,
                label="chair",
                position=[2.0, 3.0, 1.0],
                bbox=None,
                metadata={},
            ),
        ],
        regions=[],
    )
    q = SpatialQuery(
        scene_id="empty_pc",
        scene_graph=sg,
        pc=np.zeros((0, 3), dtype=np.float32),
        object_split=np.zeros(0, dtype=np.int64),
        language="test",
        target_xyz=np.array([2.0, 3.0, 1.0], dtype=np.float32),
        target_bbox=None,
        gt_anchor_object_ids=None,
        gt_anchor_room_id=None,
    )
    out = t.tensorize(q)
    assert out.coord_shift.shape == (3,)
    assert out.coord_scale.shape == (3,)
    torch.testing.assert_close(
        out.coord_shift,
        torch.tensor([2.0, 3.0, 1.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        out.coord_scale,
        torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32),
    )
