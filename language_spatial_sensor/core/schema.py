from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch

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
    region: tuple | None = None  # (region_id, region_label)

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
    pc: np.ndarray           # (N, 3) — target object removed
    object_split: np.ndarray # (N,)   — per-point object IDs, aligned with pc

    language: str

    target_xyz: np.ndarray   # supervision
    target_bbox: np.ndarray | None = None  # (6,) — [x_min, y_min, z_min, x_max, y_max, z_max]

    # optional
    gt_anchor_object_ids: list[int] | None = None
    gt_anchor_room_id: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Grounding:
    """A single grounding hypothesis: the proposer's binding of scene elements to a language utterance.

    Contains which room to ground in, which objects serve as anchors, and the utterance.
    May come from ground-truth annotations (training) or a Proposer (inference).
    """
    anchor_room_id: int
    anchor_object_ids: list[int]
    language: str


@dataclass
class GroundedQuery:
    """Scene context paired with a single Grounding hypothesis — the tensorizer's input at inference time.

    At training time: build via ``GroundedQuery.from_spatial_query(sq)`` to lift gt fields.
    At inference time: build from a raw scene + a ``Grounding`` emitted by the Proposer.

    ``target_xyz`` / ``target_bbox`` are ``None`` when GT supervision is unavailable (pure inference).
    They can be set for evaluation so that the same type is used across the pipeline.
    """
    scene_id: str
    scene_graph: SceneGraph
    pc: np.ndarray
    object_split: np.ndarray
    grounding: Grounding
    target_xyz: np.ndarray | None = None
    target_bbox: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_spatial_query(
        cls,
        query: "SpatialQuery",
        grounding: "Grounding | None" = None,
    ) -> "GroundedQuery":
        """Construct from a SpatialQuery, optionally overriding the grounding.

        If *grounding* is ``None``, the gt anchor fields on *query* are used.
        Raises ``ValueError`` if those fields are ``None`` and no override is given.
        """
        if grounding is None:
            if query.gt_anchor_room_id is None:
                raise ValueError(
                    "SpatialQuery.gt_anchor_room_id is None; "
                    "provide an explicit Grounding or set the gt field."
                )
            grounding = Grounding(
                anchor_room_id=query.gt_anchor_room_id,
                anchor_object_ids=list(query.gt_anchor_object_ids or []),
                language=query.language,
            )
        return cls(
            scene_id=query.scene_id,
            scene_graph=query.scene_graph,
            pc=query.pc,
            object_split=query.object_split,
            grounding=grounding,
            target_xyz=query.target_xyz,
            target_bbox=query.target_bbox,
            metadata=query.metadata,
        )


@dataclass
class CachedSample:
    """What gets written to disk by scripts/preprocess.py.

    Holds all model-ready tensors except text tokens, plus the raw language
    string so the tokenizer can run at batch time in CollateFn.
    """
    language:          str             # raw text — tokenized by CollateFn
    obj_clip_features: torch.Tensor   # (N, 512)  float32
    obj_bboxes:        torch.Tensor   # (N, 6)    float32  [cx,cy,cz,w,h,l] in region frame
    obj_is_anchor:     torch.Tensor   # (N,)      bool
    obj_padding_mask:  torch.Tensor   # (N,)      bool  True=padded slot
    coord_shift:       torch.Tensor   # (3,)      float32  world→region translation
    coord_scale:       torch.Tensor   # (3,)      float32  per-axis scale (region bbox size)
    target_xyz_world:  torch.Tensor   # (3,)      float32  target center in world frame
    target_bbox_world: torch.Tensor   # (6,)      float32  [x_min,y_min,z_min,x_max,y_max,z_max]


@dataclass
class TensorizerOutput:
    """Batched tensors for LSSModel.forward(). Fields have a leading batch dim B.

    Built by CollateFn from a list of CachedSample objects.

    Coordinate convention:
        obj_bboxes are in region-normalized frame: (p_world - coord_shift) / coord_scale
        target_xyz_world and target_bbox_world are in world frame (used for loss computation)
    """
    text_input_ids:      torch.Tensor   # (B, L)    long
    text_attention_mask: torch.Tensor   # (B, L)    long {0,1}
    obj_clip_features:   torch.Tensor   # (B, N, 512)  float32
    obj_bboxes:          torch.Tensor   # (B, N, 6)    float32
    obj_is_anchor:       torch.Tensor   # (B, N,)      bool
    obj_padding_mask:    torch.Tensor   # (B, N,)      bool
    coord_shift:         torch.Tensor   # (B, 3,)      float32
    coord_scale:         torch.Tensor   # (B, 3,)      float32
    target_xyz_world:    torch.Tensor   # (B, 3,)      float32
    target_bbox_world:   torch.Tensor   # (B, 6,)      float32