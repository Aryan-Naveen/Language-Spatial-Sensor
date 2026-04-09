"""BEV + camera trajectory alignment check for VLA3D scenes.

Loads a random referential statement, renders the semantic BEV, finds the longest
time-contiguous run of poses whose (x,y) lies in the BEV footprint and (by default)
z lies in the same vertical slab as ``render_bev``, then overlays that polyline.

Poses from ``poses.csv`` are transformed once: **90° CCW about +z through the origin**
``(tx, ty) → (-ty, tx)`` (matches Matterport / scan alignment for scenes like
``1LXtFkjw3qL``).

Poses file is ``poses.csv`` in the scene directory (header:
``timestamp_ns,tx,ty,tz,qw,qx,qy,qz``). If you only have ``poses.txt``, pass
``--poses`` explicitly.

Example:
    python scripts/bev_pose_alignment.py \\
        --data_root /path/to/VLA-3D_dataset --dataset matterport \\
        --scene_id 1LXtFkjw3qL --save /tmp/bev.png

Four PNGs (same 90° rotation, pivot = each BEV corner):
    python scripts/bev_pose_alignment.py ... --save-four-corners /tmp/out/bev
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.vla3d.dataset import VLA3D
from language_spatial_sensor.core.transforms import build_spatial_query
from viz.bev import (
    add_bev_trajectory_overlay,
    fit_floor_ceiling_semantic,
    render_bev,
)

# Single fixed transform for Matterport-style poses → BEV / point-cloud frame.
POSE_XY_ROTATION_DEG_CCW = 90.0
POSE_XY_ROTATION_PIVOT_XY = (0.0, 0.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BEV + longest in-slab pose segment overlay.")
    p.add_argument("--data_root", required=True, type=Path,
                   help="VLA3D root (e.g. .../Matterport) or top-level dataset root.")
    p.add_argument("--dataset", default="matterport",
                   help="Dataset name when data_root is the VLA3D parent (default: matterport).")
    p.add_argument("--scene_id", required=True,
                   help="Scene folder name (e.g. 17DRP5sb8fy).")
    p.add_argument("--poses", default=None, type=Path,
                   help="Path to poses CSV (default: <scene>/poses.csv).")
    p.add_argument("--save", default=None, type=Path,
                   help="Save figure to this path instead of displaying.")
    p.add_argument("--resolution", default=0.25, type=float,
                   help="BEV grid cell size in metres (default: 0.25).")
    p.add_argument("--seed", default=None, type=int,
                   help="Random seed for statement sampling.")
    p.add_argument(
        "--xy-only",
        action="store_true",
        help="Gate poses on BEV XY extent only (ignore z slab).",
    )
    p.add_argument(
        "--save-four-corners",
        type=Path,
        metavar="PREFIX",
        default=None,
        help="Save four PNGs PREFIX_ll.png, … PREFIX_ur.png (same 90° z-rotation, "
             "pivot = each BEV corner). Parent directories are created.",
    )
    return p.parse_args()


def longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    """Inclusive (start, end) indices of the longest contiguous True run."""
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not np.any(mask):
        return None
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    lengths = ends - starts + 1
    j = int(np.argmax(lengths))
    return int(starts[j]), int(ends[j])


def _axis_xy_bounds(ax) -> tuple[float, float, float, float]:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)


def pose_mask(
    tx: np.ndarray,
    ty: np.ndarray,
    tz: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float | None,
    z_max: float | None,
    xy_only: bool,
) -> np.ndarray:
    inside_xy = (tx >= x_min) & (tx <= x_max) & (ty >= y_min) & (ty <= y_max)
    if xy_only or z_min is None or z_max is None:
        return inside_xy
    return inside_xy & (tz > z_min) & (tz < z_max)


def load_poses_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (timestamp_ns (T,), xyz (T, 3))."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"Expected >= 4 columns in {path}, got {data.shape[1]}")
    ts = data[:, 0].astype(np.int64)
    xyz = data[:, 1:4].astype(np.float64)
    return ts, xyz


def rotate_xy_about_z(
    xyz: np.ndarray,
    degrees_ccw: float,
    pivot_xy: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Rotate (x, y) about a vertical axis through ``pivot_xy``; z unchanged.

    Positive ``degrees_ccw`` is counter-clockwise when looking down the +z axis.
    With pivot (0, 0), 90° maps (x, y) → (-y, x). With a general pivot, the same
    rigid motion is applied in the plane.
    """
    if degrees_ccw == 0.0:
        return xyz
    px, py = float(pivot_xy[0]), float(pivot_xy[1])
    rad = np.deg2rad(degrees_ccw)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    out = np.array(xyz, dtype=np.float64, copy=True)
    x = out[:, 0] - px
    y = out[:, 1] - py
    out[:, 0] = c * x - s * y + px
    out[:, 1] = s * x + c * y + py
    return out


def _bev_corner_pivots(
    xm0: float, xm1: float, ym0: float, ym1: float,
) -> list[tuple[str, tuple[float, float]]]:
    """Axis-aligned corners of the BEV rectangle (xmin/xmax × ymin/ymax)."""
    return [
        ("ll", (xm0, ym0)),
        ("lr", (xm1, ym0)),
        ("ul", (xm0, ym1)),
        ("ur", (xm1, ym1)),
    ]


def _mask_and_longest_run(
    tx: np.ndarray,
    ty: np.ndarray,
    tz: np.ndarray,
    xm0: float,
    xm1: float,
    ym0: float,
    ym1: float,
    z_min: float,
    z_max: float,
    xy_only: bool,
) -> tuple[np.ndarray, bool, tuple[int, int] | None]:
    mask = pose_mask(tx, ty, tz, xm0, xm1, ym0, ym1, z_min, z_max, xy_only=xy_only)
    run = longest_true_run(mask)
    used_xy = xy_only
    if run is None and not xy_only:
        mask_xy = pose_mask(tx, ty, tz, xm0, xm1, ym0, ym1, z_min, z_max, xy_only=True)
        run_xy = longest_true_run(mask_xy)
        if run_xy is not None:
            mask = mask_xy
            used_xy = True
            run = run_xy
    return mask, used_xy, run


def _save_corner_figures(
    *,
    prefix: Path,
    query,
    xyz_raw: np.ndarray,
    z_min: float,
    z_max: float,
    rotation_deg: float,
    resolution: float,
    xy_only_arg: bool,
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stem = prefix.name

    fig0 = render_bev(query, resolution=resolution)
    xm0, xm1, ym0, ym1 = _axis_xy_bounds(fig0.axes[0])
    plt.close(fig0)

    print(f"BEV corners (pivots): ll=({xm0:.3f},{ym0:.3f}) lr=({xm1:.3f},{ym0:.3f}) "
          f"ul=({xm0:.3f},{ym1:.3f}) ur=({xm1:.3f},{ym1:.3f})")
    print(f"Rotation {rotation_deg:g}° CCW @ each corner → four figures under {prefix.parent}/")

    for suffix, pivot in _bev_corner_pivots(xm0, xm1, ym0, ym1):
        xyz = rotate_xy_about_z(
            np.array(xyz_raw, copy=True), rotation_deg, pivot_xy=pivot,
        )
        tx, ty, tz = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        mask, used_xy, run = _mask_and_longest_run(
            tx, ty, tz, xm0, xm1, ym0, ym1, z_min, z_max, xy_only_arg,
        )
        n = len(tx)

        fig = render_bev(query, resolution=resolution)
        ax = fig.axes[0]
        title = fig.axes[0].get_title()
        fig.axes[0].set_title(
            f"{title}\npivot {suffix} ({pivot[0]:.2f}, {pivot[1]:.2f}) · "
            f"{rotation_deg:g}° CCW",
            fontsize=8,
        )

        if n > 1:
            add_bev_trajectory_overlay(
                ax, xyz[:, :2],
                color="0.75", linewidth=0.8, alpha=0.35, zorder=8,
                linestyle="--",
            )
        if run is not None:
            i0, i1 = run
            add_bev_trajectory_overlay(
                ax, xyz[i0 : i1 + 1, :2],
                color="cyan", linewidth=2.5, alpha=0.95, zorder=10,
            )
        else:
            print(f"[warn] corner {suffix}: no in-bounds contiguous segment")

        out_path = prefix.parent / f"{stem}_{suffix}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        n_in = int(mask.sum())
        print(
            f"  {suffix}: pivot {pivot}  inside {n_in}/{n}  "
            f"gate={'XY' if used_xy else 'XY+Z'}  → {out_path}",
        )


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    dataset = VLA3D(args.data_root, args.dataset)
    scene = dataset.get_scene(args.scene_id)

    scene_graph = scene.load_scene_graph()
    statements = scene.load_statements(scene_graph)
    if not statements:
        print("[error] Scene has no referential statements.")
        sys.exit(1)

    statement = random.choice(statements)
    pcd = scene.load_pointcloud()
    points = np.asarray(pcd.points, dtype=np.float32)
    object_split = scene.load_object_split()

    poses_path = args.poses if args.poses is not None else scene.path / "poses.csv"
    if not poses_path.is_file():
        print(f"[error] Poses file not found: {poses_path}")
        sys.exit(1)

    ts, xyz = load_poses_csv(poses_path)
    xyz_raw = np.array(xyz, copy=True)

    query = build_spatial_query(
        scene_id=scene.scene_id,
        scene_graph=scene_graph,
        statement=statement,
        points=points,
        object_split=object_split,
    )

    z_min, z_max = fit_floor_ceiling_semantic(
        query.scene_graph, query.target_xyz, query.pc,
    )

    if args.save_four_corners is not None:
        print(f"Dataset   : {args.dataset}")
        print(f"Scene ID  : {scene.scene_id}")
        print(f"Query     : \"{statement.text}\"")
        print(f"Poses     : {poses_path} ({len(xyz_raw)} rows)")
        _save_corner_figures(
            prefix=args.save_four_corners,
            query=query,
            xyz_raw=xyz_raw,
            z_min=z_min,
            z_max=z_max,
            rotation_deg=POSE_XY_ROTATION_DEG_CCW,
            resolution=args.resolution,
            xy_only_arg=args.xy_only,
        )
        return

    xyz = rotate_xy_about_z(
        np.array(xyz_raw, copy=True),
        POSE_XY_ROTATION_DEG_CCW,
        pivot_xy=POSE_XY_ROTATION_PIVOT_XY,
    )
    tx, ty, tz = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    fig = render_bev(query, resolution=args.resolution)
    ax = fig.axes[0]
    xm0, xm1, ym0, ym1 = _axis_xy_bounds(ax)

    xy_only = args.xy_only
    mask = pose_mask(tx, ty, tz, xm0, xm1, ym0, ym1, z_min, z_max, xy_only=xy_only)
    run = longest_true_run(mask)
    used_xy_only = xy_only

    if run is None and not xy_only:
        mask_xy = pose_mask(tx, ty, tz, xm0, xm1, ym0, ym1, z_min, z_max, xy_only=True)
        run = longest_true_run(mask_xy)
        if run is not None:
            print(
                "[warn] No poses in XY+Z slab; falling back to XY-only longest segment.",
            )
            mask = mask_xy
            used_xy_only = True

    n = len(tx)
    n_in = int(mask.sum())
    frac = n_in / n if n else 0.0

    print(f"Dataset   : {args.dataset}")
    print(f"Scene ID  : {scene.scene_id}")
    print(f"Query     : \"{statement.text}\"")
    print(f"Poses     : {poses_path} ({n} rows)")
    px, py = POSE_XY_ROTATION_PIVOT_XY
    print(
        f"Pose XY   : rotated {POSE_XY_ROTATION_DEG_CCW:g}° CCW about z @ "
        f"origin ({px:.3f}, {py:.3f})",
    )
    print(f"Z slab    : ({z_min:.3f}, {z_max:.3f}) m  (from fit_floor_ceiling_semantic)")
    print(f"BEV XY    : x∈[{xm0:.3f}, {xm1:.3f}]  y∈[{ym0:.3f}, {ym1:.3f}]")
    print(f"Gate      : {'XY only' if used_xy_only else 'XY + Z slab'}")
    print(f"Inside    : {n_in} / {n} ({100.0 * frac:.1f}%)")

    if run is None:
        print("[error] No poses fall inside the BEV XY extent.")
        sys.exit(1)

    i0, i1 = run
    seg_len = i1 - i0 + 1
    print(
        f"Longest   : indices [{i0}, {i1}] (len={seg_len}); "
        f"timestamp_ns [{ts[i0]}, {ts[i1]}]",
    )
    print("Overlay   : gray dashed = all poses; cyan = longest in-gate segment")

    if n > 1:
        add_bev_trajectory_overlay(
            ax, xyz[:, :2],
            color="0.75", linewidth=0.8, alpha=0.35, zorder=8,
            linestyle="--",
        )
    add_bev_trajectory_overlay(
        ax, xyz[i0 : i1 + 1, :2],
        color="cyan", linewidth=2.5, alpha=0.95, zorder=10,
    )
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
