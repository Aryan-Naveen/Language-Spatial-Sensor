from __future__ import annotations

import numpy as np

from language_spatial_sensor.core.schema import (
    ReferentialStatement,
    SceneGraph,
    SpatialQuery,
)


def build_spatial_query(
    scene_id: str,
    scene_graph: SceneGraph,
    statement: ReferentialStatement,
    points: np.ndarray | None = None,        # (N, 3) float — from load_pointcloud
    object_split: np.ndarray | None = None,  # (N,)   int   — from load_object_split
) -> SpatialQuery:
    target_id = statement.target_object_id

    # 1. Resolve target position from the unmodified scene graph
    target_obj = next(
        (o for o in scene_graph.objects if o.id == target_id), None
    )
    if target_obj is None:
        raise ValueError(
            f"target_object_id {target_id} not found in scene graph for scene '{scene_id}'"
        )
    target_xyz = np.array(target_obj.position, dtype=np.float32)
    target_bbox = np.array(target_obj.bbox, dtype=np.float32) if target_obj.bbox is not None else None

    # 2. Remove target object from scene graph (non-mutating)
    filtered_graph = SceneGraph(
        objects=[o for o in scene_graph.objects if o.id != target_id],
        regions=scene_graph.regions,
        relations=scene_graph.relations,
    )

    # 3. Remove target object's points from point cloud
    if points is not None and object_split is not None:
        keep = object_split != target_id
        masked_pc    = points[keep]
        masked_split = object_split[keep]
    else:
        masked_pc = None
        masked_split = None

    return SpatialQuery(
        scene_id=scene_id,
        scene_graph=filtered_graph,
        pc=masked_pc,
        object_split=masked_split,
        language=statement.text,
        target_xyz=target_xyz,
        target_bbox=target_bbox,
        gt_anchor_object_ids=[int(aid) for aid in statement.anchor_object_id] if statement.anchor_object_id else None,
        gt_anchor_room_id=int(statement.region[0]) if statement.region else None,
        )
