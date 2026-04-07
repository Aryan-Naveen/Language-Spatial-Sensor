"""Preprocess VLA-3D data for 3D-ViSTA grounding baseline.

Produces per-sample .pt files containing per-object point clouds (same format
as 3D-ViSTA's PointTokenizeEncoder) plus grounding targets.

Scene splits and statement filtering respect the same rules as the LSS
pipeline (via ``build_splits``):
  - Dataset selection: ``cfg.data.datasets``
  - Relation filter:   ``VALID_RELATIONS``
  - Object filter:     ``VALID_NYU40_LABELS`` (applied to target and anchors)

In addition, when building per-object point clouds we further restrict the
object list to objects whose ``nyu40_label`` is in ``VALID_NYU40_LABELS``,
mirroring the structural-label filtering done in ``viz/bev.py``.

Usage::

    # From the lss/ root directory:
    python baselines/vista_grounding/preprocess.py

    # Override cache dir or data root:
    python baselines/vista_grounding/preprocess.py cache.dir=/fast/ssd/cache/vista

    # Force-rebuild:
    python baselines/vista_grounding/preprocess.py cache.force_rebuild=true
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import BertTokenizer

# ── sys.path: make lss/ root importable ───────────────────────────────────────
_LSS_ROOT = Path(__file__).resolve().parents[2]
if str(_LSS_ROOT) not in sys.path:
    sys.path.insert(0, str(_LSS_ROOT))

from data.vla3d.splits import DataSplits, SplitRecord, build_splits
from language_spatial_sensor.core.ontology import VALID_NYU40_LABELS
from language_spatial_sensor.core.transforms import build_spatial_query


# ── Constants matching 3D-ViSTA defaults ──────────────────────────────────────

_VISTA_PCD_POINTS = 1024  # points sampled per object sub-cloud
_VISTA_PCD_DIM    = 6     # [dx, dy, dz, 0, 0, 0]  (ViSTA uses XYZ + 3 extra)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _atomic_torch_save(obj, path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".pt.tmp", dir=str(path.parent), text=False)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(obj, f)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _nyu40_label(obj) -> str:
    return str(obj.metadata.get("nyu40_label", "")).lower().strip()


def _is_valid_object(obj) -> bool:
    """True iff the object's nyu40_label is in the allowlist."""
    return _nyu40_label(obj) in VALID_NYU40_LABELS


def _sample_object_pcd(
    points: np.ndarray,        # (N, 3) full scene XYZ
    object_split: np.ndarray,  # (N,) per-point object IDs
    obj_id: int,
    n_points: int = _VISTA_PCD_POINTS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:               # (n_points, 6)
    """Extract points belonging to ``obj_id`` and sample to ``n_points``.

    The sub-cloud is centred on the object mean so that PointNet++ can learn
    local geometry regardless of scene position.  The last 3 of the 6 output
    channels are zeroed (placeholder for color/normal features).

    Returns all-zeros if the object has no associated points.
    """
    if rng is None:
        rng = np.random.default_rng()

    mask = object_split == obj_id
    pts  = points[mask]                           # (M, 3)

    if len(pts) == 0:
        return np.zeros((n_points, 6), dtype=np.float32)

    center = pts.mean(axis=0)
    pts    = pts - center                          # center the sub-cloud

    idx     = rng.choice(len(pts), size=n_points, replace=(len(pts) < n_points))
    sampled = pts[idx]                             # (n_points, 3)

    result = np.zeros((n_points, 6), dtype=np.float32)
    result[:, :3] = sampled
    return result


def _obj_loc(obj) -> np.ndarray:
    """6-d location vector (cx,cy,cz, sx,sy,sz) from an ObjectInfo.

    sx,sy,sz are the full-width extents of the bounding box.
    Falls back to zeros if bbox data is missing.
    """
    center = np.asarray(obj.position, dtype=np.float32)[:3]
    if obj.bbox is not None:
        corners = np.asarray(obj.bbox, dtype=np.float32).reshape(-1, 3)
        size    = corners.max(axis=0) - corners.min(axis=0)
    else:
        size = np.zeros(3, dtype=np.float32)
    return np.concatenate([center, size])          # (6,)


def _process_split(
    split_name: str,
    records: list[SplitRecord],
    cache_split_dir: Path,
    tokenizer: BertTokenizer,
    max_text_len: int,
    max_objects: int,
    force_rebuild: bool,
) -> None:
    """Tensorize all records in one split and save .pt files + manifest."""
    manifest_path = cache_split_dir / "manifest.json"
    if manifest_path.exists() and not force_rebuild:
        print(f"  [{split_name}] cache exists — skipping (use force_rebuild=true to override)")
        return

    cache_split_dir.mkdir(parents=True, exist_ok=True)

    by_scene: dict[str, list[tuple[int, SplitRecord]]] = defaultdict(list)
    for global_idx, record in enumerate(records):
        by_scene[record.scene.scene_id].append((global_idx, record))

    manifest: list[str] = []
    failed = 0
    rng = np.random.default_rng(42)

    for scene_id, indexed_records in tqdm(by_scene.items(), desc=f"  [{split_name}]"):
        scene = indexed_records[0][1].scene
        try:
            sg = scene.load_scene_graph()
        except Exception as e:
            print(f"  [warn] Skipping scene {scene_id}: {e}")
            failed += len(indexed_records)
            continue

        # Load full scene point cloud once per scene
        try:
            import open3d as o3d
            pcd          = scene.load_pointcloud()
            points       = np.asarray(pcd.points, dtype=np.float32)
            object_split = scene.load_object_split()
        except Exception as e:
            print(f"  [warn] Failed to load point cloud for {scene_id}: {e}")
            failed += len(indexed_records)
            continue

        # Filter to valid objects only (same allowlist as LSS / BEV viz)
        valid_objects = [obj for obj in sg.objects if _is_valid_object(obj)]
        valid_objects  = valid_objects[:max_objects]
        n_valid = len(valid_objects)
        O = max_objects

        # Build per-object tensors (shared across all statements in this scene)
        obj_pcds  = np.zeros((O, _VISTA_PCD_POINTS, _VISTA_PCD_DIM), dtype=np.float32)
        obj_locs  = np.zeros((O, 6), dtype=np.float32)
        obj_masks = np.zeros(O, dtype=bool)

        for i, obj in enumerate(valid_objects):
            obj_pcds[i]  = _sample_object_pcd(points, object_split, obj.id, rng=rng)
            obj_locs[i]  = _obj_loc(obj)
            obj_masks[i] = True

        for global_idx, record in indexed_records:
            try:
                query = build_spatial_query(scene_id, sg, record.statement)
            except Exception as e:
                print(f"  [warn] Failed build_spatial_query {scene_id}: {e}")
                failed += 1
                continue

            # Tokenize
            enc      = tokenizer(
                query.language,
                max_length=max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            txt_ids  = enc["input_ids"].squeeze(0)      # (L,)
            txt_mask = enc["attention_mask"].squeeze(0)  # (L,)

            sample = {
                # Identity for BEV viz
                "scene_id": scene_id,
                "language": query.language,

                # 3D-ViSTA inputs
                "obj_pcds":  torch.from_numpy(obj_pcds),   # (O, P, 6)
                "obj_locs":  torch.from_numpy(obj_locs),   # (O, 6)
                "obj_masks": torch.from_numpy(obj_masks),  # (O,) bool
                "txt_ids":   txt_ids,                       # (L,)
                "txt_masks": txt_mask,                      # (L,)

                # Grounding targets (world frame, same as LSS)
                "target_xyz_world":  torch.tensor(
                    np.asarray(query.target_xyz,  dtype=np.float32)),  # (3,)
                "target_bbox_world": torch.tensor(
                    np.asarray(query.target_bbox, dtype=np.float32)),  # (6,)
            }

            fname = f"{scene_id}_{global_idx:07d}.pt"
            _atomic_torch_save(sample, cache_split_dir / fname)
            manifest.append(fname)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"  [{split_name}] saved {len(manifest):,} samples "
        f"({failed} failed) → {cache_split_dir}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(
    config_path=".",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    print("Building data splits...")
    # build_splits uses cfg.data.datasets for dataset selection and internally
    # applies _filter_statements (VALID_RELATIONS + VALID_NYU40_LABELS filter).
    splits: DataSplits = build_splits(cfg.data)
    print(splits.summary())

    print(f"Loading tokenizer: {cfg.model.text_model}")
    tokenizer = BertTokenizer.from_pretrained(cfg.model.text_model)

    cache_dir = Path(cfg.cache.dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {cache_dir}")

    split_map = {
        "train":      splits.train,
        "val_seen":   splits.val_seen,
        "val_unseen": splits.val_unseen,
    }

    for split_name, records in split_map.items():
        if not records:
            print(f"  [{split_name}] empty — skipping")
            continue
        _process_split(
            split_name,
            records,
            cache_dir / split_name,
            tokenizer=tokenizer,
            max_text_len=cfg.model.max_text_len,
            max_objects=cfg.model.max_objects,
            force_rebuild=cfg.cache.force_rebuild,
        )

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
