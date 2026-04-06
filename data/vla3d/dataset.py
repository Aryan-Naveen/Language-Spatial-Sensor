import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d

from language_spatial_sensor.core.ontology import VALID_NYU40_LABELS, VALID_REGION_LABELS
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
            region_meta: dict = {}
            if region_bbox:
                corners = np.array(region_bbox)
                region_pos = corners.mean(axis=0).tolist()
                region_meta["bbox_x_min"] = float(corners[:, 0].min())
                region_meta["bbox_x_max"] = float(corners[:, 0].max())
                region_meta["bbox_y_min"] = float(corners[:, 1].min())
                region_meta["bbox_y_max"] = float(corners[:, 1].max())
                region_meta["bbox_z_min"] = float(corners[:, 2].min())
                region_meta["bbox_z_max"] = float(corners[:, 2].max())
            else:
                region_pos = [0.0, 0.0, 0.0]

            regions.append(RegionInfo(
                id=region_id,
                label=region_raw.get("region_name", ""),
                position=region_pos,
                metadata=region_meta,
            ))

            for obj_raw in region_raw.get("objects", []):
                obj_id = int(obj_raw["object_id"])
                label = obj_raw.get("raw_label", "")
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

    _NYU40_OTHER = {"otherprop", "otherfurniture", "otherstructure"}

    def _semantic_label(self, metadata: dict) -> str:
        """Return the semantic bin for an object.

        Uses nyu40_label for most objects; falls back to raw_label for the
        catch-all nyu40 categories so they aren't collapsed into one bucket.
        """
        nyu40 = metadata.get("nyu40_label") or ""
        if nyu40 in self._NYU40_OTHER:
            return metadata.get("raw_label") or nyu40
        return nyu40 or metadata.get("raw_label") or ""

    def load_statements(self, scene_graph: SceneGraph | None = None) -> list[ReferentialStatement]:
        with open(self._file("referential_statements.json")) as f:
            data = json.load(f)

        # Build per-object semantic label and a scene-wide frequency count.
        if scene_graph is None:
            scene_graph = self.load_scene_graph()
        obj_sem: dict[int, str] = {
            obj.id: self._semantic_label(obj.metadata)
            for obj in scene_graph.objects
        }
        sem_counts: Counter[str] = Counter(obj_sem.values())

        statements: list[ReferentialStatement] = []

        for rid, region_stmts in data["regions"].items():
            region_semantics = region_stmts.get("region", "")
            for text, entries in region_stmts.items():
                if text == 'region':
                    continue
                for entry in entries:
                    target_id = int(entry["target_index"])
                    anchors = entry.get("anchors", {})
                    anchor_ids = []
                    if anchors:
                        anchor_ids = [int(anchor["index"]) for anchor in anchors.values()]

                    # Ambiguity = product over anchors of (# scene objects with
                    # the same semantic label, excluding the anchor itself).
                    if not anchor_ids:
                        ambiguity = 0
                    else:
                        ambiguity = 1
                        for aid in anchor_ids:
                            sem = obj_sem.get(aid, "")
                            others = max(sem_counts.get(sem, 1), 0)
                            ambiguity *= others
                    text = "There is a " + text[4:]
                    statements.append(ReferentialStatement(
                        text=text,
                        target_object_id=target_id,
                        ambiguity=ambiguity,
                        anchor_object_id=anchor_ids,
                        relation=entry.get("relation"),
                        region=(rid, region_semantics)
                    ))

        return statements

    # ------------------------------------------------------------------
    # Synthetic region-grounding statements
    # ------------------------------------------------------------------

    def generate_region_statements(
        self,
        scene_graph: SceneGraph | None = None,
    ) -> list[ReferentialStatement]:
        """Generate synthetic "there is a <object> in the <region>" statements.

        Only produces statements for scenes with more than one region, where
        the region label is in VALID_REGION_LABELS and the object's nyu40_label
        is in VALID_NYU40_LABELS.  Ambiguity is the count of objects sharing
        the same nyu40 label within the same region.
        """
        if scene_graph is None:
            scene_graph = self.load_scene_graph()

        if len(scene_graph.regions) <= 1:
            return []

        valid_regions: dict[int, str] = {
            r.id: r.label.lower().strip()
            for r in scene_graph.regions
            if r.label.lower().strip() in VALID_REGION_LABELS
        }
        if not valid_regions:
            return []

        region_objects: dict[int, list] = defaultdict(list)
        for obj in scene_graph.objects:
            rid = obj.metadata.get("region_id")
            if rid is None:
                continue
            rid = int(rid)
            if rid not in valid_regions:
                continue
            label = str(obj.label).lower().strip()
            if label in VALID_NYU40_LABELS:
                region_objects[rid].append((obj, label))

        stmts: list[ReferentialStatement] = []
        for rid, obj_pairs in region_objects.items():
            region_label = valid_regions[rid]
            label_counts = Counter(label for _, label in obj_pairs)
            for obj, label in obj_pairs:
                stmts.append(ReferentialStatement(
                    text=f"there is a {label} in the {region_label}",
                    target_object_id=obj.id,
                    ambiguity=-1,
                    anchor_object_id=None,
                    relation="in region",
                    region=(str(rid), region_label),
                ))
        return stmts

    # ------------------------------------------------------------------
    # Point cloud
    # ------------------------------------------------------------------

    def load_pointcloud(self) -> o3d.geometry.PointCloud:
        return o3d.io.read_point_cloud(str(self._file("pc_result.ply")))

    def load_object_split(self) -> np.ndarray:
        """Per-point object ID assignments (shape: N,).

        The raw file stores a compact (K, 2) array where each row is
        [object_id, cumulative_end_index], meaning points are laid out
        contiguously per object.  This method expands that into a flat
        per-point label array so callers can use it as a boolean mask.
        """
        raw = np.load(str(self._file("object_split.npy")))
        if raw.ndim == 2 and raw.shape[1] == 2:
            n_points = int(raw[-1, 1])
            per_point = np.empty(n_points, dtype=np.int64)
            start = 0
            for obj_id, end in raw:
                per_point[start:int(end)] = int(obj_id)
                start = int(end)
            return per_point
        return raw

    def load_region_split(self) -> np.ndarray:
        """Per-point region ID assignments (shape: N,).

        Expands the same compact (K, 2) format as load_object_split.
        """
        raw = np.load(str(self._file("region_split.npy")))
        if raw.ndim == 2 and raw.shape[1] == 2:
            n_points = int(raw[-1, 1])
            per_point = np.empty(n_points, dtype=np.int64)
            start = 0
            for obj_id, end in raw:
                per_point[start:int(end)] = int(obj_id)
                start = int(end)
            return per_point
        return raw


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
