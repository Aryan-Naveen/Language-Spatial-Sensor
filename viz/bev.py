"""Bird's Eye View (BEV) occupancy grid visualization.

Pipeline:
    1. fit_floor_ceiling  — RANSAC-fit floor and ceiling z-levels
    2. make_bev_grid      — filter points, project to XY, bin into occupancy grid
    3. render_bev         — full pipeline returning a matplotlib Figure
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from language_spatial_sensor.core.schema import SpatialQuery


# ---------------------------------------------------------------------------
# Floor / ceiling fitting
# ---------------------------------------------------------------------------

def _ransac_plane(points: np.ndarray, distance_threshold: float, num_iterations: int) -> list[float]:
    """Run Open3D RANSAC plane segmentation. Returns [a, b, c, d] coefficients."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    plane_model, _ = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=num_iterations,
    )
    return plane_model  # [a, b, c, d]  → ax + by + cz + d = 0


def fit_floor_ceiling(
    pc: np.ndarray,
    percentile: float = 0.15,
    distance_threshold: float = 0.05,
    num_iterations: int = 1000,
    horizontal_threshold: float = 0.8,
) -> tuple[float, float]:
    """RANSAC-fit floor and ceiling z-levels from a point cloud.

    Samples the bottom/top `percentile` fraction of points by z, fits a plane
    to each via RANSAC, then extracts the z-intercept.  Falls back to raw
    percentiles when the fitted plane is not sufficiently horizontal (e.g.
    outdoor scenes or noisy captures).

    Args:
        pc: (N, 3) float array of XYZ points.
        percentile: Fraction of points (by z) used for floor/ceiling fitting.
        distance_threshold: RANSAC inlier distance in metres.
        num_iterations: RANSAC iterations.
        horizontal_threshold: Minimum |c| for a plane to be considered horizontal.

    Returns:
        (z_floor, z_ceiling) — z-levels to filter against.
    """
    z = pc[:, 2]
    n = len(z)
    k = max(3, int(n * percentile))  # at least 3 points for RANSAC

    idx_sorted = np.argsort(z)
    floor_pts = pc[idx_sorted[:k]]
    ceil_pts  = pc[idx_sorted[-k:]]

    def _extract_z(pts: np.ndarray, fallback_z: float) -> float:
        try:
            model = _ransac_plane(pts, distance_threshold, num_iterations)
            a, b, c, d = model
            if abs(c) >= horizontal_threshold:
                return -d / c
        except Exception:
            pass
        return fallback_z

    z_floor   = _extract_z(floor_pts,  float(np.percentile(z,  5)))
    z_ceiling = _extract_z(ceil_pts,   float(np.percentile(z, 95)))

    # Sanity guard: ensure floor < ceiling
    if z_floor >= z_ceiling:
        z_floor   = float(np.percentile(z,  5))
        z_ceiling = float(np.percentile(z, 95))

    return z_floor, z_ceiling


# ---------------------------------------------------------------------------
# BEV occupancy grid
# ---------------------------------------------------------------------------

def make_bev_grid(
    pc: np.ndarray,
    z_min: float,
    z_max: float,
    resolution: float = 0.05,
    margin: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Project a height-filtered slice of the point cloud onto a 2-D occupancy grid.

    Args:
        pc: (N, 3) float array.
        z_min: Discard points at or below this z (floor level).
        z_max: Discard points at or above this z (ceiling level).
        resolution: Grid cell size in metres.
        margin: Extra border around the XY extent in metres.

    Returns:
        grid: (H, W) bool array — True where at least one point projects.
        meta: dict with keys ``x_min``, ``y_min``, ``resolution``, ``width``, ``height``.
    """
    mask = (pc[:, 2] > z_min) & (pc[:, 2] < z_max)
    pts = pc[mask]

    if len(pts) == 0:
        # Degenerate case: return a tiny empty grid
        return np.zeros((1, 1), dtype=bool), {
            "x_min": 0.0, "y_min": 0.0, "resolution": resolution,
            "width": 1, "height": 1,
        }

    x_min = pts[:, 0].min() - margin
    x_max = pts[:, 0].max() + margin
    y_min = pts[:, 1].min() - margin
    y_max = pts[:, 1].max() + margin

    width  = int(np.ceil((x_max - x_min) / resolution))
    height = int(np.ceil((y_max - y_min) / resolution))

    col = np.clip(np.floor((pts[:, 0] - x_min) / resolution).astype(int), 0, width  - 1)
    row = np.clip(np.floor((pts[:, 1] - y_min) / resolution).astype(int), 0, height - 1)

    grid = np.zeros((height, width), dtype=bool)
    grid[row, col] = True

    meta = {
        "x_min": x_min,
        "y_min": y_min,
        "resolution": resolution,
        "width": width,
        "height": height,
    }
    return grid, meta


# ---------------------------------------------------------------------------
# Full render pipeline
# ---------------------------------------------------------------------------

def render_bev(
    query: SpatialQuery,
    resolution: float = 0.05,
    percentile: float = 0.15,
    figsize: tuple[int, int] = (8, 8),
) -> plt.Figure:
    """Run the full floor/ceiling → BEV pipeline and return a matplotlib Figure.

    The figure shows:
    - Binary BEV occupancy grid (white = occupied, black = empty).
    - Red × marker at the ground-truth target position.
    - Scene ID and (truncated) language query as the title.
    - Axis ticks in metres.

    Args:
        query: SpatialQuery whose `.pc` field is used.
        resolution: BEV grid cell size in metres.
        percentile: Fraction of points used for RANSAC floor/ceiling fitting.
        figsize: Matplotlib figure size in inches.

    Returns:
        A ``matplotlib.figure.Figure`` — call ``.show()`` or ``.savefig(path)`` on it.
    """
    z_floor, z_ceiling = fit_floor_ceiling(query.pc, percentile=percentile)
    grid, meta = make_bev_grid(query.pc, z_min=z_floor, z_max=z_ceiling, resolution=resolution)

    fig, ax = plt.subplots(figsize=figsize)

    # Occupancy grid — flip so +y is up
    ax.imshow(
        grid,
        origin="lower",
        cmap="gray_r",
        interpolation="nearest",
        extent=[
            meta["x_min"],
            meta["x_min"] + meta["width"]  * resolution,
            meta["y_min"],
            meta["y_min"] + meta["height"] * resolution,
        ],
    )

    # Target marker
    tx, ty = float(query.target_xyz[0]), float(query.target_xyz[1])
    ax.plot(tx, ty, marker="x", color="red", markersize=12, markeredgewidth=2.5, zorder=5)

    # Labels
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    lang_display = query.language if len(query.language) <= 80 else query.language[:77] + "..."
    ax.set_title(f"{query.scene_id}\n\"{lang_display}\"", fontsize=9)

    legend = [
        mpatches.Patch(color="white", label="occupied"),
        plt.Line2D([0], [0], marker="x", color="red", linestyle="None",
                   markersize=8, markeredgewidth=2, label="target"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.7)

    # Annotation: floor / ceiling z
    ax.annotate(
        f"floor z={z_floor:.2f}m  ceiling z={z_ceiling:.2f}m",
        xy=(0.01, 0.01), xycoords="axes fraction",
        fontsize=7, color="gray",
    )

    fig.tight_layout()
    return fig
