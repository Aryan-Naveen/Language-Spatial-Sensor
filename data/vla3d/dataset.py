import json
from pathlib import Path

import numpy as np
import open3d as o3d

from language_spatial_sensor.core.schema import (
    ObjectInfo,
    RegionInfo,
    ReferentialStatement,
    SceneGraph,
)


class VLA3DScene:
    DATASETS = {
        "3RScan", "ARKitScenes", "HM3D", "Matterport", "Scannet", "Unity"
    }

    def __init__(self, path: Path):
        self.path = path
        self.scene_id = path.name
        # derive dataset name from grandparent directory
        self.dataset = path.parent.name

    def _prefix(self) -> str:
        return self.scene_id

    def _file(self, suffix: str) -> Path:
        return self.path / f"{self._prefix()}_{suffix}"

    # ------------------------------------------------------------------
    # SceneGraph
    # ------------------------------------------------------------------

    def load_scene_graph(self) -> SceneGraph:
        with open(self._file("scene_graph.json")) as f:
            data = json.load(f)

        objects: list[ObjectInfo] = []
        regions: list[RegionInfo] = []

        for region_raw in data["regions"].values():
            region_id = int(region_raw["region_id"])
            # compute region centroid from 8-corner bbox
            region_bbox = region_raw.get("region_bbox", [])
            if region_bbox:
                corners = np.array(region_bbox)
                region_pos = corners.mean(axis=0).tolist()
            else:
                region_pos = [0.0, 0.0, 0.0]

            regions.append(RegionInfo(
                id=region_id,
                label=region_raw.get("region_name", ""),
                position=region_pos,
            ))

            for obj_raw in region_raw.get("objects", []):
                obj_id = int(obj_raw["object_id"])
                label = obj_raw.get("nyu40_label") or obj_raw.get("raw_label", "")
                center = obj_raw.get("center", [0.0, 0.0, 0.0])

                # bbox is 8 corners [[x,y,z], ...]; flatten to 24 floats
                bbox_corners = obj_raw.get("bbox")
                bbox = [v for corner in bbox_corners for v in corner] if bbox_corners else None

                metadata = {
                    "region_id": region_id,
                    "nyu_label": obj_raw.get("nyu_label"),
                    "nyu40_label": obj_raw.get("nyu40_label"),
                    "raw_label": obj_raw.get("raw_label"),
                    "volume": obj_raw.get("volume"),
                    "size": obj_raw.get("size"),
                }

                objects.append(ObjectInfo(
                    id=obj_id,
                    label=label,
                    position=center,
                    bbox=bbox,
                    metadata=metadata,
                ))

        return SceneGraph(objects=objects, regions=regions)

    # ------------------------------------------------------------------
    # ReferentialStatements
    # ------------------------------------------------------------------

    def load_statements(self) -> list[ReferentialStatement]:
        with open(self._file("referential_statements.json")) as f:
            data = json.load(f)

        statements: list[ReferentialStatement] = []

        for region_stmts in data["regions"].values():
            for text, entries in region_stmts.items():
                for entry in entries:
                    target_id = int(entry["target_index"])
                    anchors = entry.get("anchors", {})
                    anchor_ids = []
                    if anchors:
                        anchor_ids = [int(anchor["index"]) for anchor in anchors.values()]

                    # TODO: update the ambiguity definition
                    ambiguity = 0

                    statements.append(ReferentialStatement(
                        text=text,
                        target_object_id=target_id,
                        ambiguity=ambiguity,
                        anchor_object_id=anchor_ids,
                        relation=entry.get("relation"),
                    ))

        return statements

    # ------------------------------------------------------------------
    # Point cloud
    # ------------------------------------------------------------------

    def load_pointcloud(self) -> o3d.geometry.PointCloud:
        return o3d.io.read_point_cloud(str(self._file("pc_result.ply")))

    def load_object_split(self) -> np.ndarray:
        """Per-point object ID assignments."""
        return np.load(str(self._file("object_split.npy")))

    def load_region_split(self) -> np.ndarray:
        """Per-point region ID assignments."""
        return np.load(str(self._file("region_split.npy")))


class VLA3DDataset:
    def __init__(self, root: Path):
        self.root = root

    def scenes(self) -> list[VLA3DScene]:
        return [VLA3DScene(p) for p in sorted(self.root.iterdir()) if p.is_dir()]

    def get_scene(self, scene_id: str) -> VLA3DScene:
        return VLA3DScene(self.root / scene_id)

    def __len__(self) -> int:
        return sum(1 for p in self.root.iterdir() if p.is_dir())

    def __iter__(self):
        return iter(self.scenes())


class VLA3D:
    _DATASET_DIRS = ["3RScan", "ARKitScenes", "HM3D", "Matterport", "Scannet", "Unity"]

    def __init__(self, root: Path):
        self.root = root

    def datasets(self) -> dict[str, VLA3DDataset]:
        return {
            d.lower(): VLA3DDataset(self.root / d)
            for d in self._DATASET_DIRS
            if (self.root / d).exists()
        }

    def get_dataset(self, name: str) -> VLA3DDataset:
        """name is case-insensitive (e.g. '3rscan', 'hm3d')."""
        for d in self._DATASET_DIRS:
            if d.lower() == name.lower():
                path = self.root / d
                if not path.exists():
                    raise FileNotFoundError(f"{path} does not exist")
                return VLA3DDataset(path)
        raise ValueError(f"Unknown dataset '{name}'. Options: {self._DATASET_DIRS}")
