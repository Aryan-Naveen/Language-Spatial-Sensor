from dataclasses import dataclass, field
from typing import Any
import numpy as np

from pydantic import BaseModel



class ObjectInfo(BaseModel):
    id: int
    label: str
    position: list[float]  # (x, y, z)
    bbox: list[float] | None = None
    metadata: dict = {}

class RegionInfo(BaseModel):
    id: int
    label: str
    position: list[float]
    metadata: dict = {}

class SceneGraph(BaseModel):
    objects: list[ObjectInfo]
    regions: list[RegionInfo] = []
    
    # optional but useful
    relations: list[dict] = []  # edges

class ReferentialStatement(BaseModel):
    text: str
    target_object_id: int
    ambiguity: int
    
    # optional richer structure
    anchor_object_id: list[int] | None = None
    relation: str | None = None

class SceneData(BaseModel):
    scene_id: str
    
    scene_graph: SceneGraph
    statements: list[ReferentialStatement]
    
    # paths (optional)
    pointcloud_path: str | None = None

@dataclass
class SpatialQuery:
    scene_id: str
    
    scene_graph: SceneGraph
    pc : np.ndarray
    
    language: str
    
    target_xyz: np.ndarray      # supervision
    
    # optional
    anchor_object_ids: list[int] | None = None
    anchor_room_id: int | None = None
    metadata: dict = field(default_factory=dict)