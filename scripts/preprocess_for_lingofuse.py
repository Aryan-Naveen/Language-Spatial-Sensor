"""
Preprocess Matterport (VLA-3D) scenes into LingoFuse episode bundles.

Output layout:
    output_dir/
      manifest.json                    # [{episode_id, scene_id}, ...]
      <scene_id>/
        <episode_id>/
          metadata.json                # utterances list + episode-level target
          scene_graph.json             # flat SceneGraph (target object removed)
          pc_xyz.npy                   # (N, 3) float32  [only with --pointcloud]
          object_split.npy             # (N,)   int64    [only with --pointcloud]

Usage:
    python scripts/preprocess_for_lingofuse.py \\
        --data_root /path/to/lingofuse/data \\
        --scene_ids 17DRP5sb8fy \\
        --output_dir /path/to/lingofuse/data/17DRP5sb8fy/episodes \\
        --ambiguity_min 0 --ambiguity_max 5 \\
        --target_labels chair sofa lamp \\
        --n_samples 50 \\
        --seed 42

Note: --output_dir should point to the scene's episodes/ subdirectory.
The script writes <scene_id>/<episode_id>/ inside that directory,
which matches the layout load_episodes() expects.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

# Ensure lss/ root is on the path when invoked from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.vla3d.dataset import VLA3DScene
from language_spatial_sensor.core.ontology import VALID_RELATIONS
from language_spatial_sensor.core.transforms import build_spatial_query


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _scene_graph_to_dict(sg) -> dict:
    """Convert an LSS SceneGraph (pydantic) to a plain dict for JSON output."""
    objects = []
    for o in sg.objects:
        obj_dict: dict = {
            "id":    o.id,
            "label": o.label,
            "position": list(o.position),
        }
        if o.bbox is not None:
            # bbox in LSS is a flat list (24 floats = 8 corners × 3).
            # Reduce to a 6-element AABB that lingofuse expects.
            arr = np.array(o.bbox, dtype=np.float32).reshape(-1, 3)
            obj_dict["bbox"] = [
                float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 2].min()),
                float(arr[:, 0].max()), float(arr[:, 1].max()), float(arr[:, 2].max()),
            ]
        rid = o.metadata.get("region_id")
        if rid is not None:
            obj_dict["region_id"] = int(rid)
        nyu40 = o.metadata.get("nyu40_label")
        if nyu40:
            obj_dict["nyu40_label"] = str(nyu40)
        objects.append(obj_dict)

    regions = []
    for r in sg.regions:
        regions.append({
            "id":       r.id,
            "label":    r.label,
            "position": list(r.position),
        })

    return {"objects": objects, "regions": regions}


# ── Per-scene processing ──────────────────────────────────────────────────────

def _process_scene(
    scene: VLA3DScene,
    output_dir: Path,
    ambiguity_min: int,
    ambiguity_max: int,
    target_labels: list[str] | None,
    relation_types: list[str] | None,
    n_samples: int,
    include_pointcloud: bool,
    rng: random.Random,
) -> list[dict]:
    """Process one scene and write episode bundles. Returns manifest entries."""
    try:
        scene_graph = scene.load_scene_graph()
        statements  = scene.load_statements(scene_graph)
    except Exception as e:
        print(f"  [skip] {scene.scene_id}: failed to load — {e}")
        return []

    # ── Filter statements ─────────────────────────────────────────────────────
    filtered = []
    for stmt in statements:
        if not (ambiguity_min <= stmt.ambiguity <= ambiguity_max):
            continue
        if stmt.relation not in VALID_RELATIONS:
            continue
        if target_labels:
            target_obj = next((o for o in scene_graph.objects if o.id == stmt.target_object_id), None)
            if target_obj is None:
                continue
            nyu40 = (target_obj.metadata.get("nyu40_label") or "").lower()
            if nyu40 not in target_labels:
                continue
        if relation_types and stmt.relation not in relation_types:
            continue
        filtered.append(stmt)

    if not filtered:
        print(f"  [skip] {scene.scene_id}: no statements match filters")
        return []

    sampled = rng.sample(filtered, min(n_samples, len(filtered)))

    # ── Optionally load point cloud once per scene ────────────────────────────
    pc_xyz: np.ndarray | None = None
    object_split: np.ndarray | None = None
    if include_pointcloud:
        try:
            pcd = scene.load_pointcloud()
            pc_xyz = np.asarray(pcd.points, dtype=np.float32)
            object_split = scene.load_object_split()
        except Exception as e:
            print(f"  [warn] {scene.scene_id}: point cloud unavailable — {e}")

    scene_out = output_dir / scene.scene_id
    scene_out.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict] = []

    # Build a fast object_id → target_label lookup for the whole scene.
    # Mirrors LSS's _semantic_label logic: for catch-all nyu40 bins, prefer raw_label.
    _NYU40_CATCHALL = {"otherprop", "otherfurniture", "otherstructure"}
    def _target_label(obj_metadata: dict) -> str:
        nyu40 = (obj_metadata.get("nyu40_label") or "").lower().strip()
        if nyu40 in _NYU40_CATCHALL:
            return (obj_metadata.get("raw_label") or nyu40).lower().strip()
        return nyu40 or (obj_metadata.get("raw_label") or "").lower().strip()

    obj_label_map = {o.id: _target_label(o.metadata) for o in scene_graph.objects}

    for idx, stmt in enumerate(sampled):
        episode_id = f"{scene.scene_id}_{idx:07d}"
        ep_dir = scene_out / episode_id
        ep_dir.mkdir(exist_ok=True)

        try:
            query = build_spatial_query(
                scene.scene_id, scene_graph, stmt,
                points=pc_xyz,
                object_split=object_split,
            )
        except ValueError as e:
            print(f"  [warn] skipping episode {episode_id}: {e}")
            continue

        # ── metadata.json ─────────────────────────────────────────────────────
        region_id = int(stmt.region[0]) if stmt.region else None
        metadata = {
            "episode_id":       episode_id,
            "scene_id":         scene.scene_id,
            "target_object_id": stmt.target_object_id,
            "target_label":     obj_label_map.get(stmt.target_object_id, ""),
            "target_xyz":       query.target_xyz.tolist(),
            "target_bbox":      query.target_bbox.tolist() if query.target_bbox is not None else None,
            "utterances": [
                {
                    "timestep":          0,
                    "text":              stmt.text,
                    "anchor_object_ids": stmt.anchor_object_id or [],
                    "ambiguity":         stmt.ambiguity,
                    "relation":          stmt.relation,
                    "region_id":         region_id,
                }
            ],
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # ── scene_graph.json ──────────────────────────────────────────────────
        with open(ep_dir / "scene_graph.json", "w") as f:
            json.dump(_scene_graph_to_dict(query.scene_graph), f, indent=2)

        # ── point cloud arrays (optional) ─────────────────────────────────────
        if include_pointcloud and query.pc is not None:
            np.save(ep_dir / "pc_xyz.npy",       query.pc.astype(np.float32))
            np.save(ep_dir / "object_split.npy",  query.object_split.astype(np.int64))

        manifest_entries.append({"episode_id": episode_id, "scene_id": scene.scene_id})

    print(f"  {scene.scene_id}: wrote {len(manifest_entries)} episodes")
    return manifest_entries


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root",    required=True,  help="Path to VLA-3D Matterport dataset root")
    parser.add_argument("--output_dir",   required=True,  help="Where to write episode bundles")
    parser.add_argument("--scene_ids",    nargs="*",      help="Specific scene IDs to process (default: all)")
    parser.add_argument("--ambiguity_min", type=int, default=0,  help="Minimum ambiguity level (inclusive)")
    parser.add_argument("--ambiguity_max", type=int, default=100, help="Maximum ambiguity level (inclusive)")
    parser.add_argument("--target_labels", nargs="*",     help="NYU40 labels to include, e.g. chair sofa lamp")
    parser.add_argument("--relation_types", nargs="*",    help="Relation types to include, e.g. near on")
    parser.add_argument("--n_samples",   type=int, default=50,   help="Max episodes per scene")
    parser.add_argument("--seed",        type=int, default=42,   help="Random seed for sampling")
    parser.add_argument("--pointcloud",  action="store_true",    help="Include pc_xyz.npy and object_split.npy")
    args = parser.parse_args()

    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_labels  = [l.lower() for l in args.target_labels]  if args.target_labels  else None
    relation_types = [r.lower() for r in args.relation_types] if args.relation_types else None
    rng = random.Random(args.seed)

    # Discover scenes
    if args.scene_ids:
        scene_dirs = [data_root / sid for sid in args.scene_ids]
    else:
        scene_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())

    print(f"Processing {len(scene_dirs)} scene(s) → {output_dir}")

    all_manifest: list[dict] = []
    for scene_dir in scene_dirs:
        if not scene_dir.is_dir():
            print(f"  [skip] {scene_dir}: not a directory")
            continue
        scene = VLA3DScene(scene_dir)
        entries = _process_scene(
            scene, output_dir,
            ambiguity_min=args.ambiguity_min,
            ambiguity_max=args.ambiguity_max,
            target_labels=target_labels,
            relation_types=relation_types,
            n_samples=args.n_samples,
            include_pointcloud=args.pointcloud,
            rng=rng,
        )
        all_manifest.extend(entries)

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(all_manifest, f, indent=2)

    print(f"\nDone. {len(all_manifest)} total episodes written to {output_dir}/manifest.json")


if __name__ == "__main__":
    main()
