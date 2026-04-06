from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class SpatialQuery:
    scene_id: str
    
    scene_graph: "SceneGraph"   # from schema
    pointcloud: Any | None
    
    language: str
    
    target_xyz: np.ndarray      # supervision
    
    # optional
    anchor_object_id: int | None = None
    anchor_room_id: int | None = None
    metadata: dict = field(default_factory=dict)