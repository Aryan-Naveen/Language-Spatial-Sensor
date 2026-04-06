"""Test script: randomly sample a SpatialQuery from a VLA3D dataset and render BEV.

Usage examples:
    python scripts/test_viz.py --data_root /data/vla3d --dataset scannet
    python scripts/test_viz.py --data_root /data/vla3d --dataset hm3d --seed 7
    python scripts/test_viz.py --data_root /data/vla3d --dataset scannet \\
        --scene_id scene0000_00 --save /tmp/bev.png
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.vla3d.dataset import VLA3D
from language_spatial_sensor.core.transforms import build_spatial_query
from viz.bev import render_bev


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise a random SpatialQuery as BEV.")
    p.add_argument("--data_root", required=True, type=Path,
                   help="Root directory of the VLA3D dataset (contains 3RScan/, Scannet/, …).")
    p.add_argument("--dataset", default="scannet",
                   help="Dataset name (3rscan | arkitscenes | hm3d | matterport | scannet | unity).")
    p.add_argument("--scene_id", default=None,
                   help="Specific scene ID to load; random if omitted.")
    p.add_argument("--save", default=None, type=Path,
                   help="If given, save figure to this path instead of displaying.")
    p.add_argument("--resolution", default=0.05, type=float,
                   help="BEV grid cell size in metres (default: 0.05).")
    p.add_argument("--seed", default=None, type=int,
                   help="Random seed for reproducible sampling.")
    p.add_argument("--anchor_highlight", action="store_true",
                   help="Grayscale all objects except those sharing a class with the anchor.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # --- Load dataset -------------------------------------------------------
    vla3d = VLA3D(args.data_root)
    dataset = vla3d.get_dataset(args.dataset)

    if args.scene_id:
        scene = dataset.get_scene(args.scene_id)
    else:
        scenes = dataset.scenes()
        if not scenes:
            print(f"[error] No scenes found in dataset '{args.dataset}' at {args.data_root}")
            sys.exit(1)
        scene = random.choice(scenes)

    print(f"Dataset  : {args.dataset}")
    print(f"Scene ID : {scene.scene_id}")

    # --- Load scene data ----------------------------------------------------
    scene_graph  = scene.load_scene_graph()
    statements   = scene.load_statements()

    if not statements:
        print("[error] Scene has no referential statements.")
        sys.exit(1)

    statement = random.choice(statements)

    pcd          = scene.load_pointcloud()
    points       = np.asarray(pcd.points, dtype=np.float32)
    object_split = scene.load_object_split()

    print(f"Points   : {points.shape[0]:,}  z∈[{points[:,2].min():.2f}, {points[:,2].max():.2f}] m")
    print(f"Query    : \"{statement.text}\"")
    print(f"Target ID: {statement.target_object_id}")

    print("points shape:", points.shape)
    print("object_split shape:", object_split.shape, "dtype:", object_split.dtype)
    print("object_split sample:", object_split[:5])

    # --- Build SpatialQuery -------------------------------------------------
    query = build_spatial_query(
        scene_id=scene.scene_id,
        scene_graph=scene_graph,
        statement=statement,
        points=points,
        object_split=object_split,
    )

    print(f"PC after target removal: {query.pc.shape[0]:,} points")

    # --- BEV render ---------------------------------------------------------
    fig = render_bev(query, resolution=args.resolution, anchor_highlight=args.anchor_highlight)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
