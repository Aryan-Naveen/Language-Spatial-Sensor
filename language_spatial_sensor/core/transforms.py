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
    points: np.ndarray,        # (N, 3) float — from load_pointcloud
    object_split: np.ndarray,  # (N,)   int   — from load_object_split
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

    # 2. Remove target object from scene graph (non-mutating)
    filtered_graph = SceneGraph(
        objects=[o for o in scene_graph.objects if o.id != target_id],
        regions=scene_graph.regions,
        relations=scene_graph.relations,
    )

    # 3. Remove target object's points from point cloud
    masked_pc = points[object_split != target_id]

    return SpatialQuery(
        scene_id=scene_id,
        scene_graph=filtered_graph,
        pc=masked_pc,
        language=statement.text,
        target_xyz=target_xyz,
        anchor_object_ids=statement.anchor_object_id,
    )
