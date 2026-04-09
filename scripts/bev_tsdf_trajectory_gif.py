"""Habitat stage-2: fuse RGB-D along the longest in-BEV trajectory segment into a TSDF,
project to a BEV occupancy slice (same voxel size as ``render_bev``), crop to 2 m around
the agent, and write a GIF on top of the semantic BEV.

Expects per-scene layout (Matterport / VLA3D):
  ``color/rgb_%07d.png``, ``depth/depth_%07d.tiff``, ``poses.csv``, ``camera_info.yaml``.

Writes:
  * ``<out_stem>.gif`` — BEV + TSDF occupancy + trajectory (no on-image title).
  * ``<out_stem>_camera_rgb.gif`` — same timeline, raw pinhole RGB frames.
  * ``<out_stem>_dual.mp4`` — **RGB | BEV/TSDF** side-by-side (H.264; needs ``pip install imageio-ffmpeg``).
  * ``<out_stem>_metadata.txt`` — includes ``language_utterance`` and run parameters.

Requires: open3d, numpy, matplotlib, pillow, imageio, imageio-ffmpeg (for MP4), tifffile.

Example:
    python scripts/bev_tsdf_trajectory_gif.py \\
        --data_root /path/to/VLA-3D_dataset --dataset matterport \\
        --scene_id 1LXtFkjw3qL --out /tmp/tsdf.gif --seed 0
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.vla3d.dataset import VLA3D
from language_spatial_sensor.core.transforms import build_spatial_query
from viz.bev import fit_floor_ceiling_semantic, render_bev_underlay_rgb

# Load pose helpers + fixed Matterport rotation from sibling script
_bpa_path = Path(__file__).resolve().parent / "bev_pose_alignment.py"
_spec = importlib.util.spec_from_file_location("bev_pose_alignment", _bpa_path)
_bpa = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_bpa)

POSE_ROT_DEG = _bpa.POSE_XY_ROTATION_DEG_CCW
POSE_PIVOT = _bpa.POSE_XY_ROTATION_PIVOT_XY
load_poses_csv = _bpa.load_poses_csv
rotate_xy_about_z = _bpa.rotate_xy_about_z
longest_true_run = _bpa.longest_true_run
pose_mask = _bpa.pose_mask

# Camera-frame axis for optical direction on BEV. VLA3D Matterport poses here match +Z forward;
# use neg_z if your export uses OpenGL-style −Z forward.
_LOOK_AXIS_VECTORS: dict[str, np.ndarray] = {
    "neg_z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
    "pos_z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "neg_x": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "pos_x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "neg_y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "pos_y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TSDF + BEV GIFs and side-by-side RGB|BEV MP4 along trajectory segment.",
    )
    p.add_argument("--data_root", required=True, type=Path)
    p.add_argument("--dataset", default="matterport")
    p.add_argument("--scene_id", required=True)
    p.add_argument("--poses", default=None, type=Path)
    p.add_argument("--out", required=True, type=Path, help="Output .gif path.")
    p.add_argument("--resolution", type=float, default=0.25,
                   help="BEV voxel size / TSDF voxel_length (m).")
    p.add_argument("--active-radius", type=float, default=2.0,
                   help="Show fused occupancy only within this XY radius of the agent (m).")
    p.add_argument("--depth-trunc", type=float, default=5.0, help="RGB-D depth truncation (m).")
    p.add_argument("--sdf-trunc-mult", type=float, default=4.0,
                   help="sdf_trunc = mult * voxel_length.")
    p.add_argument("--max-frames", type=int, default=120,
                   help="Max frames along segment (uniform subsample; default 120).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--xy-only", action="store_true")
    p.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Matplotlib raster DPI for BEV underlay and composite frames (default: 220).",
    )
    p.add_argument(
        "--bev-max-inches",
        type=float,
        default=14.0,
        help="Larger matplotlib figure inch size for BEV PNG underlay (default: 14).",
    )
    p.add_argument(
        "--composite-max-inches",
        type=float,
        default=14.0,
        help="Figure inch size for TSDF composite frames (default: 14).",
    )
    p.add_argument(
        "--gif-duration",
        type=float,
        default=1.0 / 3.0,
        help="Seconds per GIF frame (~3 fps default; use 0.5 for slower).",
    )
    p.add_argument(
        "--out-mp4",
        type=Path,
        default=None,
        help="Combined RGB|BEV MP4 path (default: <out_stem>_dual.mp4 next to --out).",
    )
    p.add_argument(
        "--no-mp4",
        action="store_true",
        help="Skip writing the side-by-side MP4.",
    )
    p.add_argument(
        "--video-fps",
        type=float,
        default=3.0,
        help="Frames per second for the dual MP4 (default: 3; lower feels less rushed).",
    )
    p.add_argument(
        "--video-crf",
        type=int,
        default=18,
        help="x264 CRF for MP4 (lower = higher quality / larger; default 18).",
    )
    p.add_argument(
        "--look-axis",
        choices=tuple(_LOOK_AXIS_VECTORS.keys()),
        default="pos_z",
        help="Camera-frame axis for viewing direction on BEV (default pos_z for VLA3D Matterport; try neg_z if reversed).",
    )
    p.add_argument(
        "--heading-arrow-m",
        type=float,
        default=0.5,
        help="Arrow length in metres for look direction on BEV (default: 0.5).",
    )
    p.add_argument(
        "--no-heading-arrow",
        action="store_true",
        help="Do not draw look-direction arrow on BEV frames.",
    )
    return p.parse_args()


def load_camera_info(path: Path) -> dict:
    d: dict = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        try:
            d[k] = float(v) if "." in v else int(v)
        except ValueError:
            d[k] = v
    return d


def quat_wxyz_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Rotation matrix (camera→world or world frame orientation of camera axes)."""
    n = qw * qw + qx * qx + qy * qy + qz * qz
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s * qw * qx, s * qw * qy, s * qw * qz
    xx, xy, xz = s * qx * qx, s * qx * qy, s * qx * qz
    yy, yz, zz = s * qy * qy, s * qy * qz, s * qz * qz
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose_row_to_T_wc(row: np.ndarray) -> np.ndarray:
    """4×4 world-from-camera: X_world = R_wc @ X_cam + t (Open3D-style)."""
    tx, ty, tz = float(row[1]), float(row[2]), float(row[3])
    qw, qx, qy, qz = float(row[4]), float(row[5]), float(row[6]), float(row[7])
    R = quat_wxyz_to_R(qw, qx, qy, qz)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def world_z_rotation_4x4(degrees_ccw: float, pivot_xy: tuple[float, float]) -> np.ndarray:
    """Rigid transform: rotate XY about vertical axis through ``pivot_xy`` (matches ``rotate_xy_about_z``)."""
    px, py = float(pivot_xy[0]), float(pivot_xy[1])
    rad = np.deg2rad(degrees_ccw)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 3] = -c * px + s * py + px
    T[1, 3] = -s * px - c * py + py
    return T


def align_camera_pose_to_bev_frame(T_wc_raw: np.ndarray) -> np.ndarray:
    """Apply the same world transform as pose positions (Matterport → BEV / PLY frame)."""
    S = world_z_rotation_4x4(POSE_ROT_DEG, POSE_PIVOT)
    return S @ T_wc_raw


def look_heading_xy_unit(T_wc: np.ndarray, forward_cam: np.ndarray) -> tuple[float, float] | None:
    """Unit direction in world XY of the camera look axis (horizontal projection)."""
    R = T_wc[:3, :3]
    v = R @ np.asarray(forward_cam, dtype=np.float64).reshape(3)
    n = float(np.hypot(v[0], v[1]))
    if n < 1e-8:
        return None
    return v[0] / n, v[1] / n


def read_depth_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile

        d = tifffile.imread(str(path))
    except Exception:
        d = np.array(Image.open(path))
    return np.asarray(d, dtype=np.float32)


def try_import_open3d():
    try:
        import open3d as o3d

        return o3d
    except ImportError as e:
        print("[error] open3d is required. Install with: pip install open3d")
        raise SystemExit(1) from e


def _resize_rgb_to_height(rgb: np.ndarray, target_h: int) -> np.ndarray:
    """Resize RGB uint8 (H,W,3) so height is ``target_h`` (LANCZOS)."""
    h, w = rgb.shape[:2]
    if h == target_h:
        return rgb
    new_w = max(1, int(round(w * target_h / h)))
    return np.asarray(
        Image.fromarray(rgb).resize((new_w, target_h), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def _dual_panel_rgb_bev(cam_rgb: np.ndarray, bev_frame: np.ndarray, sep_px: int = 6) -> np.ndarray:
    """Horizontal concat: camera (scaled to BEV height) | separator | BEV/TSDF composite."""
    hb = bev_frame.shape[0]
    cam_s = _resize_rgb_to_height(cam_rgb, hb)
    sep = np.full((hb, sep_px, 3), 28, dtype=np.uint8)
    return np.hstack([cam_s, sep, bev_frame])


def _pad_to_h264_macroblock(img: np.ndarray, block: int = 16) -> np.ndarray:
    """Pad so H×W are multiples of ``block`` (even + encoder-friendly)."""
    h, w = img.shape[:2]
    hp = ((h + block - 1) // block) * block
    wp = ((w + block - 1) // block) * block
    if hp == h and wp == w:
        return img
    out = np.zeros((hp, wp, 3), dtype=np.uint8)
    out[:h, :w] = img
    return out


def _write_dual_mp4(
    path: Path,
    frames_camera: list[np.ndarray],
    frames_bev: list[np.ndarray],
    *,
    fps: float,
    crf: int,
) -> None:
    """Side-by-side MP4 via imageio FFmpeg (requires imageio-ffmpeg)."""
    dual = [_pad_to_h264_macroblock(_dual_panel_rgb_bev(c, b)) for c, b in zip(frames_camera, frames_bev)]
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        str(path),
        dual,
        fps=float(fps),
        codec="libx264",
        ffmpeg_params=[
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
        ],
    )


def _composite_figsize(
    xspan: float, yspan: float, max_inches: float = 10.0,
) -> tuple[float, float]:
    if xspan < 1e-9:
        xspan = 1.0
    if yspan < 1e-9:
        yspan = 1.0
    if xspan >= yspan:
        return float(max_inches), max_inches * yspan / xspan
    return max_inches * xspan / yspan, float(max_inches)


def occupancy_slice_from_pcd(
    points_xyz: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    resolution: float,
    agent_xy: tuple[float, float],
    active_radius: float,
    max_points: int = 80_000,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """2D occupancy heatmap (H,W) float in [0,1] and same extent as BEV."""
    if len(points_xyz) == 0:
        nx = max(1, int(np.ceil((x_max - x_min) / resolution)))
        ny = max(1, int(np.ceil((y_max - y_min) / resolution)))
        return np.zeros((ny, nx), dtype=np.float32), (x_min, x_max, y_min, y_max)

    p = points_xyz
    if len(p) > max_points:
        rng = np.random.default_rng(0)
        p = p[rng.choice(len(p), size=max_points, replace=False)]

    m = (p[:, 2] >= z_min) & (p[:, 2] <= z_max)
    p = p[m]
    if len(p) == 0:
        nx = max(1, int(np.ceil((x_max - x_min) / resolution)))
        ny = max(1, int(np.ceil((y_max - y_min) / resolution)))
        return np.zeros((ny, nx), dtype=np.float32), (x_min, x_max, y_min, y_max)

    nx = int(np.ceil((x_max - x_min) / resolution))
    ny = int(np.ceil((y_max - y_min) / resolution))
    nx, ny = max(1, nx), max(1, ny)

    h, _, _ = np.histogram2d(
        p[:, 0],
        p[:, 1],
        bins=[nx, ny],
        range=[[x_min, x_max], [y_min, y_max]],
    )
    occ = (h.T > 0).astype(np.float32)
    hmax = float(h.T.max()) if h.T.size else 1.0
    if hmax > 0:
        occ = np.clip(h.T / hmax, 0.0, 1.0)

    xc = x_min + (np.arange(nx) + 0.5) * (x_max - x_min) / nx
    yc = y_min + (np.arange(ny) + 0.5) * (y_max - y_min) / ny
    X, Y = np.meshgrid(xc, yc)
    ax, ay = agent_xy
    dist = np.sqrt((X - ax) ** 2 + (Y - ay) ** 2)
    occ = np.where(dist <= active_radius, occ, 0.0)
    return occ.astype(np.float32), (x_min, x_max, y_min, y_max)


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    o3d = try_import_open3d()

    dataset = VLA3D(args.data_root, args.dataset)
    scene = dataset.get_scene(args.scene_id)
    scene_dir = scene.path
    cam_path = scene_dir / "camera_info.yaml"
    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    if not cam_path.is_file():
        print(f"[error] Missing {cam_path}")
        sys.exit(1)
    if not color_dir.is_dir() or not depth_dir.is_dir():
        print(f"[error] Need color/ and depth/ under {scene_dir}")
        sys.exit(1)

    cam = load_camera_info(cam_path)
    W, H = int(cam["width"]), int(cam["height"])
    fx, fy, cx, cy = float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    scene_graph = scene.load_scene_graph()
    statements = scene.load_statements(scene_graph)
    if not statements:
        print("[error] No referential statements.")
        sys.exit(1)
    statement = random.choice(statements)

    pcd = scene.load_pointcloud()
    points = np.asarray(pcd.points, dtype=np.float32)
    object_split = scene.load_object_split()

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

    poses_path = args.poses if args.poses is not None else scene_dir / "poses.csv"
    data = np.loadtxt(poses_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    xyz_raw = data[:, 1:4].astype(np.float64)

    xyz = rotate_xy_about_z(
        np.hstack([xyz_raw, np.zeros((len(xyz_raw), 1))])[:, :3].copy(),
        POSE_ROT_DEG,
        pivot_xy=POSE_PIVOT,
    )
    tx, ty, tz = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    bev_rgb, (xm0, xm1, ym0, ym1) = render_bev_underlay_rgb(
        query, args.resolution, dpi=args.dpi, max_inches=args.bev_max_inches,
    )

    xy_only = args.xy_only
    mask = pose_mask(tx, ty, tz, xm0, xm1, ym0, ym1, z_min, z_max, xy_only=xy_only)
    run = longest_true_run(mask)
    if run is None and not xy_only:
        mask = pose_mask(tx, ty, tz, xm0, xm1, ym0, ym1, z_min, z_max, xy_only=True)
        run = longest_true_run(mask)
    if run is None:
        print("[error] No trajectory segment inside BEV.")
        sys.exit(1)

    i0, i1 = run
    indices = list(range(i0, i1 + 1))
    if len(indices) > args.max_frames:
        step = max(1, len(indices) // args.max_frames)
        indices = indices[::step][: args.max_frames]

    print(f"Segment [{i0}, {i1}] → {len(indices)} GIF frames (step subsampled).")
    print(f"BEV extent x∈[{xm0:.2f},{xm1:.2f}] y∈[{ym0:.2f},{ym1:.2f}]  voxel {args.resolution} m")

    L, R, B, T = xm0, xm1, ym0, ym1
    fw, fh = _composite_figsize(R - L, T - B, max_inches=args.composite_max_inches)

    voxel_length = float(args.resolution)
    sdf_trunc = voxel_length * float(args.sdf_trunc_mult)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )

    frames_bev: list[np.ndarray] = []
    frames_camera: list[np.ndarray] = []

    for step_idx, k in enumerate(indices):
        row = data[k]
        T_wc = align_camera_pose_to_bev_frame(pose_row_to_T_wc(row))

        rgb_path = color_dir / f"rgb_{k:07d}.png"
        dep_path = depth_dir / f"depth_{k:07d}.tiff"
        if not rgb_path.is_file() or not dep_path.is_file():
            print(f"[warn] skip frame {k}: missing rgb or depth")
            continue

        cam_rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)

        color = o3d.io.read_image(str(rgb_path))
        dep_m = read_depth_tiff(dep_path)
        dep_mm = np.clip(dep_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
        depth_o3d = o3d.geometry.Image(dep_mm)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth_o3d,
            depth_scale=1000.0,
            depth_trunc=float(args.depth_trunc),
            convert_rgb_to_intensity=False,
        )
        extrinsic_w2c = np.linalg.inv(T_wc)
        volume.integrate(rgbd, intrinsic, extrinsic_w2c)

        fused = volume.extract_point_cloud()
        pts = np.asarray(fused.points)
        occ, _ = occupancy_slice_from_pcd(
            pts,
            x_min=L,
            x_max=R,
            y_min=B,
            y_max=T,
            z_min=z_min,
            z_max=z_max,
            resolution=voxel_length,
            agent_xy=(float(tx[k]), float(ty[k])),
            active_radius=float(args.active_radius),
        )

        fig, ax = plt.subplots(figsize=(fw, fh), dpi=args.dpi)
        fig.subplots_adjust(0.0, 0.0, 1.0, 1.0)
        ax.axis("off")
        ax.imshow(bev_rgb, extent=[L, R, B, T], origin="lower", interpolation="nearest")
        occ_rgba = np.zeros((*occ.shape, 4), dtype=np.float32)
        occ_rgba[..., 0] = 1.0
        occ_rgba[..., 1] = 0.35
        occ_rgba[..., 2] = 0.1
        occ_rgba[..., 3] = np.clip(occ * 0.65, 0.0, 1.0)
        ax.imshow(occ_rgba, extent=[L, R, B, T], origin="lower", interpolation="nearest")
        path_idx = indices[: step_idx + 1]
        seg_x = tx[path_idx]
        seg_y = ty[path_idx]
        ax.plot(seg_x, seg_y, color="white", linewidth=2.0, alpha=0.85, zorder=8)
        ax.plot(seg_x, seg_y, color="black", linewidth=0.8, alpha=0.9, zorder=7)
        ax.scatter([tx[k]], [ty[k]], c="lime", s=120, edgecolors="black", linewidths=1.2, zorder=10)
        if not args.no_heading_arrow:
            fcam = _LOOK_AXIS_VECTORS[args.look_axis]
            u = look_heading_xy_unit(T_wc, fcam)
            if u is not None:
                ux, uy = u
                alen = float(args.heading_arrow_m)
                px, py = float(tx[k]), float(ty[k])
                ax.annotate(
                    "",
                    xytext=(px, py),
                    xy=(px + ux * alen, py + uy * alen),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "facecolor": "#FF00FF",
                        "edgecolor": "#4B0082",
                        "linewidth": 2.4,
                        "mutation_scale": 22,
                    },
                    zorder=12,
                )
        circ = patches.Circle(
            (float(tx[k]), float(ty[k])),
            float(args.active_radius),
            fill=False,
            edgecolor="cyan",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            zorder=9,
        )
        ax.add_patch(circ)
        ax.set_xlim(L, R)
        ax.set_ylim(B, T)
        ax.set_aspect("equal")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        frames_bev.append(frame)
        frames_camera.append(cam_rgb)

    if not frames_bev:
        print("[error] No frames rendered.")
        sys.exit(1)

    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    mp4_path: Path | None = None
    if not args.no_mp4:
        mp4_path = (
            args.out_mp4
            if args.out_mp4 is not None
            else args.out.parent / f"{args.out.stem}_dual.mp4"
        ).resolve()

    meta_path = args.out.parent / f"{args.out.stem}_metadata.txt"
    meta_lines = [
        f"scene_id: {scene.scene_id}",
        f"dataset: {args.dataset}",
        f"language_utterance: {statement.text}",
        f"seed: {args.seed if args.seed is not None else '(none)'}",
        f"trajectory_segment_indices: [{i0}, {i1}] (inclusive)",
        f"gif_pose_indices: {indices}",
        f"frames_written: {len(frames_bev)}",
        f"bev_resolution_m: {args.resolution}",
        f"active_radius_m: {args.active_radius}",
        f"composite_dpi: {args.dpi}",
        f"gif_duration_s: {args.gif_duration}",
        f"bev_gif: {args.out.name}",
        f"rgb_gif: {args.out.stem}_camera_rgb.gif",
        f"dual_mp4: {mp4_path.name if mp4_path is not None else '(skipped)'}",
        f"video_fps: {args.video_fps}",
        f"video_crf: {args.video_crf}",
        f"look_axis: {args.look_axis}",
        f"heading_arrow_m: {args.heading_arrow_m}",
        f"heading_arrow: {not args.no_heading_arrow}",
    ]
    meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    print(f"Wrote {meta_path}")

    # High–pixel-count GIFs: Pillow quantizes to 256 colours; subrectangles helps file size.
    gif_kw: dict = {
        "duration": float(args.gif_duration),
        "loop": 0,
        "subrectangles": True,
    }
    try:
        imageio.mimsave(str(args.out), frames_bev, **gif_kw)
    except TypeError:
        gif_kw.pop("subrectangles", None)
        imageio.mimsave(str(args.out), frames_bev, **gif_kw)

    rgb_out = args.out.parent / f"{args.out.stem}_camera_rgb.gif"
    try:
        imageio.mimsave(str(rgb_out), frames_camera, **gif_kw)
    except TypeError:
        kw2 = {k: v for k, v in gif_kw.items() if k != "subrectangles"}
        imageio.mimsave(str(rgb_out), frames_camera, **kw2)

    print(f"Wrote {len(frames_bev)} BEV/TSDF frames → {args.out}")
    print(f"Wrote {len(frames_camera)} RGB frames → {rgb_out}")

    if mp4_path is not None:
        try:
            _write_dual_mp4(
                mp4_path,
                frames_camera,
                frames_bev,
                fps=args.video_fps,
                crf=args.video_crf,
            )
            print(
                f"Wrote dual MP4 (RGB | BEV/TSDF) @ {args.video_fps} fps → {mp4_path}",
            )
        except Exception as e:
            print(f"[error] MP4 export failed: {e}")
            print("        Install: pip install imageio-ffmpeg")
            raise SystemExit(1) from e


if __name__ == "__main__":
    main()
