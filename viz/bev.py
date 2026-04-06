"""Bird's Eye View (BEV) occupancy grid visualization.

Pipeline (semantic mode, default when query.object_split is available):
    1. filter_object_points  — keep only non-structural object points
    2. make_semantic_bev     — project to XY, bin into an RGB occupancy image
    3. render_bev            — produce a matplotlib Figure
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from language_spatial_sensor.core.schema import SpatialQuery, SceneGraph
from language_spatial_sensor.core.ontology import VALID_NYU40_LABELS, VALID_REGION_LABELS


# ---------------------------------------------------------------------------
# Structural label filter
# ---------------------------------------------------------------------------

# Points belonging to these object labels are excluded from the BEV so that
# floors, ceilings, and walls don't dominate the occupancy image.
STRUCTURAL_LABELS: frozenset[str] = frozenset({
    "floor", "flooring", "floor mat",
    "ceiling",
    "wall panel", "wall plug",
    "window", "windowsill",
    "door", "doorframe",
    "curtain", "blinds", "shower curtain",
    "unknown",
})


# ---------------------------------------------------------------------------
# Semantic color mapping
# ---------------------------------------------------------------------------

# Keyed by exact NYU40 label strings. Only objects whose nyu40_label appears
# here are included in the BEV; everything else is silently dropped.
_PALETTE: dict[str, tuple[float, float, float]] = {
    "cabinet":       (0.55, 0.34, 0.29),
    "bed":           (0.84, 0.15, 0.16),
    "chair":         (0.12, 0.47, 0.71),
    "sofa":          (0.17, 0.63, 0.17),
    "table":         (1.00, 0.50, 0.05),
    "bookshelf":     (0.89, 0.47, 0.76),
    "picture":       (0.80, 0.40, 0.40),
    "counter":       (0.90, 0.60, 0.60),
    "blinds":        (0.70, 0.85, 0.70),
    "desk":          (0.58, 0.40, 0.74),
    "shelves":       (0.75, 0.50, 0.80),
    "curtain":       (0.95, 0.70, 0.80),
    "dresser":       (0.65, 0.40, 0.20),
    "pillow":        (0.74, 0.74, 0.13),
    "mirror":        (0.60, 0.85, 0.90),
    "floormat":      (0.50, 0.75, 0.50),
    "clothes":       (0.95, 0.60, 0.75),
    "books":         (0.40, 0.55, 0.30),
    "refrigerator":  (0.60, 0.80, 0.20),
    "television":    (0.25, 0.25, 0.25),
    "paper":         (0.90, 0.90, 0.80),
    "towel":         (0.80, 0.70, 0.90),
    "showercurtain": (0.95, 0.75, 0.85),
    "box":           (0.80, 0.70, 0.40),
    "whiteboard":    (0.95, 0.95, 0.95),
    "person":        (1.00, 0.35, 0.35),
    "nightstand":    (0.70, 0.45, 0.25),
    "toilet":        (0.94, 0.67, 0.37),
    "sink":          (0.68, 0.78, 0.91),
    "lamp":          (0.09, 0.75, 0.81),
    "bathtub":       (0.26, 0.78, 0.87),
    "bag":           (0.60, 0.40, 0.80),
    "otherprop":     (0.50, 0.50, 0.50),
    "otherfurniture": (0.70, 0.70, 0.70),
    "otherstructure": (0.30, 0.30, 0.30),
}


def nyu40_to_color(nyu40_label: str) -> tuple[float, float, float]:
    """Return an RGB color for an NYU40 label. Raises KeyError if not in palette."""
    return _PALETTE[nyu40_label.lower().strip()]


# ---------------------------------------------------------------------------
# Floor / ceiling fitting (semantic)
# ---------------------------------------------------------------------------

_FLOOR_NYU40   = {"floor"}
_CEILING_NYU40 = {"ceiling"}
_FLOOR_RAW     = {"floor", "flooring", "floor mat"}
_CEILING_RAW   = {"ceiling"}


def fit_floor_ceiling_semantic(
    scene_graph: SceneGraph,
    target_xyz: np.ndarray,
    pc: np.ndarray,
) -> tuple[float, float]:
    """Determine z-bounds from scene-graph semantics.

    Floor z  : top surface of the highest floor object whose top is still
               below the target object.
    Ceiling z: bottom surface of the lowest ceiling object whose bottom is
               still above the target object.
    Falls back to point-cloud percentiles when no matching objects are found.
    """
    target_z = float(target_xyz[2])

    floor_z:   float | None = None
    ceiling_z: float | None = None

    for obj in scene_graph.objects:
        nyu = str(obj.metadata.get("nyu40_label", "")).lower().strip()
        raw = obj.label.lower().strip()

        is_floor   = nyu in _FLOOR_NYU40   or raw in _FLOOR_RAW
        is_ceiling = nyu in _CEILING_NYU40 or raw in _CEILING_RAW

        if not (is_floor or is_ceiling):
            continue

        if obj.bbox is not None:
            z_vals = np.array(obj.bbox[2::3])   # every 3rd value starting at index 2
            obj_z_top = float(z_vals.max())
            obj_z_bot = float(z_vals.min())
        else:
            obj_z_top = obj_z_bot = float(obj.position[2])

        if is_floor and obj_z_top < target_z:
            if floor_z is None or obj_z_top > floor_z:
                floor_z = obj_z_top

        if is_ceiling and obj_z_bot > target_z:
            if ceiling_z is None or obj_z_bot < ceiling_z:
                ceiling_z = obj_z_bot

    # Fallback to point-cloud percentiles for whichever bound is missing
    z = pc[:, 2]
    if floor_z is None:
        floor_z = float(np.percentile(z, 5))
    if ceiling_z is None:
        ceiling_z = float(np.percentile(z, 95))

    if floor_z >= ceiling_z:
        floor_z, ceiling_z = float(np.percentile(z, 5)), float(np.percentile(z, 95))

    return floor_z, ceiling_z


# ---------------------------------------------------------------------------
# Semantic point filtering
# ---------------------------------------------------------------------------

def filter_object_points(
    pc: np.ndarray,
    object_split: np.ndarray,
    scene_graph: SceneGraph,
    z_min: float | None = None,
    z_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Positive-allowlist filter: keep only points that are in a known palette
    label AND within the given z-range.

    Args:
        pc: (N, 3) point cloud.
        object_split: (N,) per-point object IDs.
        scene_graph: SceneGraph whose objects define label→id mapping.
        z_min: Discard points at or below this z (floor level).
        z_max: Discard points at or above this z (ceiling level).

    Returns:
        (filtered_pc, filtered_split)
    """
    # Positive allowlist: only objects whose nyu40_label is in the shared ontology
    valid_ids = np.array([
        obj.id for obj in scene_graph.objects
        if str(obj.metadata.get("nyu40_label", "")).lower().strip() in VALID_NYU40_LABELS
    ], dtype=np.int64)

    keep = np.isin(object_split, valid_ids)

    if z_min is not None:
        keep &= pc[:, 2] > z_min
    if z_max is not None:
        keep &= pc[:, 2] < z_max

    return pc[keep], object_split[keep]


# ---------------------------------------------------------------------------
# BEV grid builders
# ---------------------------------------------------------------------------

def make_bev_grid(
    pc: np.ndarray,
    z_min: float,
    z_max: float,
    resolution: float = 0.05,
    margin: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Binary BEV occupancy grid (fallback, no semantic info).

    Returns:
        grid: (H, W) bool array.
        meta: dict with ``x_min``, ``y_min``, ``resolution``, ``width``, ``height``.
    """
    pts = pc[(pc[:, 2] > z_min) & (pc[:, 2] < z_max)]
    if len(pts) == 0:
        return np.zeros((1, 1), dtype=bool), {
            "x_min": 0.0, "y_min": 0.0, "resolution": resolution,
            "width": 1, "height": 1,
        }

    x_min = pts[:, 0].min() - margin
    x_max = pts[:, 0].max() + margin
    y_min = pts[:, 1].min() - margin
    y_max = pts[:, 1].max() + margin

    W = int(np.ceil((x_max - x_min) / resolution))
    H = int(np.ceil((y_max - y_min) / resolution))

    col = np.clip(np.floor((pts[:, 0] - x_min) / resolution).astype(int), 0, W - 1)
    row = np.clip(np.floor((pts[:, 1] - y_min) / resolution).astype(int), 0, H - 1)

    grid = np.zeros((H, W), dtype=bool)
    grid[row, col] = True

    return grid, {"x_min": x_min, "y_min": y_min, "resolution": resolution,
                  "width": W, "height": H}


_GRAYSCALE = np.array([0.55, 0.55, 0.55], dtype=np.float32)
_HIGHLIGHT  = np.array([1.00, 0.30, 0.20], dtype=np.float32)  # vivid coral-red


def make_semantic_bev(
    pc: np.ndarray,
    object_split: np.ndarray,
    scene_graph: SceneGraph,
    resolution: float = 0.05,
    margin: float = 0.5,
    highlight_labels: set[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Semantic RGB BEV image.

    Each occupied cell is coloured by the object's NYU40 label.  Background
    cells are black (0, 0, 0).

    When ``highlight_labels`` is provided the image switches to a two-tone
    mode: cells whose label is in the set are drawn in coral-red, all other
    occupied cells are rendered in grayscale.

    Returns:
        image: (H, W, 3) float32 RGB image.
        meta: same keys as make_bev_grid.
    """
    if len(pc) == 0:
        return np.ones((1, 1, 3), dtype=np.float32), {
            "x_min": 0.0, "y_min": 0.0, "resolution": resolution,
            "width": 1, "height": 1,
        }

    # Build object_id → RGB color only for IDs present in this filtered split.
    present_ids = set(int(i) for i in np.unique(object_split))
    id_to_obj = {obj.id: obj for obj in scene_graph.objects}

    def _label_gray(lbl: str) -> np.ndarray:
        """Deterministic per-class gray in [0.30, 0.72] so classes stay distinct."""
        v = 0.30 + (abs(hash(lbl)) % 256) / 256 * 0.42
        return np.array([v, v, v], dtype=np.float32)

    def _color_for(oid: int) -> np.ndarray:
        obj = id_to_obj.get(oid)
        if obj is None:
            return _GRAYSCALE
        lbl = str(obj.metadata.get("nyu40_label", "")).lower().strip()
        if highlight_labels is not None:
            return _HIGHLIGHT if lbl in highlight_labels else _label_gray(lbl)
        return np.array(nyu40_to_color(lbl), dtype=np.float32)

    id_to_color: dict[int, np.ndarray] = {
        oid: _color_for(oid) for oid in present_ids if oid in id_to_obj
    }
    # Per-point colors (any point with an unrecognised id gets skipped upstream,
    # but keep a fallback for -1 / background points just in case)
    default_color = np.array([0.4, 0.4, 0.4], dtype=np.float32)
    point_colors = np.stack([
        id_to_color.get(int(oid), default_color) for oid in object_split
    ])  # (N, 3)

    x_min = pc[:, 0].min() - margin
    x_max = pc[:, 0].max() + margin
    y_min = pc[:, 1].min() - margin
    y_max = pc[:, 1].max() + margin

    W = int(np.ceil((x_max - x_min) / resolution))
    H = int(np.ceil((y_max - y_min) / resolution))

    col = np.clip(np.floor((pc[:, 0] - x_min) / resolution).astype(int), 0, W - 1)
    row = np.clip(np.floor((pc[:, 1] - y_min) / resolution).astype(int), 0, H - 1)

    image = np.ones((H, W, 3), dtype=np.float32)  # white background

    # Sort points by z ascending so higher-z objects paint over lower-z ones
    z_order = np.argsort(pc[:, 2])

    if highlight_labels is not None:
        # Determine which points belong to a highlighted object
        is_highlight = np.array([
            str(id_to_obj[int(oid)].metadata.get("nyu40_label", "")).lower().strip()
            in highlight_labels
            if int(oid) in id_to_obj else False
            for oid in object_split
        ])
        # Within each group respect z-order; highlighted group goes on top of background
        hi_in_z = is_highlight[z_order]
        for idx in (z_order[~hi_in_z], z_order[hi_in_z]):
            image[row[idx], col[idx]] = point_colors[idx]
    else:
        image[row[z_order], col[z_order]] = point_colors[z_order]

    meta = {"x_min": x_min, "y_min": y_min, "resolution": resolution,
            "width": W, "height": H}
    return image, meta



# ---------------------------------------------------------------------------
# Full render pipeline
# ---------------------------------------------------------------------------

def render_bev(
    query: SpatialQuery,
    resolution: float = 0.05,
    figsize: tuple[int, int] = (8, 8),
    include_legend: bool = True,
    anchor_highlight: bool = False,
) -> plt.Figure:
    """Render a BEV figure from a SpatialQuery.

    Uses semantic filtering (palette allowlist + floor/ceiling z-bounds) and
    per-label colouring.

    Args:
        query: SpatialQuery to visualise.
        resolution: Grid cell size in metres.
        figsize: Matplotlib figure size in inches.
        include_legend: Whether to draw the per-label colour legend.
        anchor_highlight: When True, renders all occupied cells in grayscale
            except objects sharing a semantic class with any anchor object,
            which are drawn in coral-red.

    Returns:
        ``matplotlib.figure.Figure`` — call ``.show()`` or ``.savefig(path)``.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    if query.object_split is None:
        print(f"Cannot render semantic BEV for scene '{query.scene_id}' (missing object_split).")
        return fig

    # Fit floor/ceiling semantically using scene-graph floor/ceiling objects
    z_min, z_max = fit_floor_ceiling_semantic(query.scene_graph, query.target_xyz, query.pc)

    # Positive allowlist: palette labels only, within z-range
    pc_obj, split_obj = filter_object_points(
        query.pc, query.object_split, query.scene_graph,
        z_min=z_min, z_max=z_max,
    )

    # Resolve anchor NYU40 labels for highlight mode
    highlight_labels: set[str] | None = None
    if anchor_highlight and query.anchor_object_ids:
        id_to_obj_full = {obj.id: obj for obj in query.scene_graph.objects}
        highlight_labels = {
            str(id_to_obj_full[aid].metadata.get("nyu40_label", "")).lower().strip()
            for aid in query.anchor_object_ids
            if aid in id_to_obj_full
        } - {""}

    image, meta = make_semantic_bev(
        pc_obj, split_obj, query.scene_graph,
        resolution=resolution,
        highlight_labels=highlight_labels,
    )
    extent = [
        meta["x_min"],
        meta["x_min"] + meta["width"]  * resolution,
        meta["y_min"],
        meta["y_min"] + meta["height"] * resolution,
    ]
    ax.imshow(image, origin="lower", interpolation="nearest", extent=extent)

    # Legend
    if include_legend:
        if highlight_labels:
            present_ids = set(int(i) for i in np.unique(split_obj))
            id_to_obj   = {obj.id: obj for obj in query.scene_graph.objects}
            gray_seen: dict[str, np.ndarray] = {}
            for oid in present_ids:
                obj = id_to_obj.get(oid)
                if obj is None:
                    continue
                lbl = str(obj.metadata.get("nyu40_label", "")).lower().strip()
                if lbl and lbl not in highlight_labels and lbl not in gray_seen:
                    gv = 0.30 + (abs(hash(lbl)) % 256) / 256 * 0.42
                    gray_seen[lbl] = np.array([gv, gv, gv])
            legend_handles = [
                mpatches.Patch(color=_HIGHLIGHT, label=", ".join(sorted(highlight_labels))),
                *[mpatches.Patch(color=c, label=lbl) for lbl, c in sorted(gray_seen.items())],
            ]
        else:
            present_ids = set(int(i) for i in np.unique(split_obj))
            id_to_obj   = {obj.id: obj for obj in query.scene_graph.objects}
            seen: dict[str, tuple[float, float, float]] = {}
            for oid in present_ids:
                if oid in id_to_obj:
                    lbl = str(id_to_obj[oid].metadata.get("nyu40_label", "")).lower().strip()
                    if lbl and lbl not in seen:
                        seen[lbl] = nyu40_to_color(lbl)
            legend_handles = [
                mpatches.Patch(color=c, label=lbl)
                for lbl, c in sorted(seen.items())
            ]
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper right", fontsize=6,
                      framealpha=0.8, ncol=2)

    note = f"{len(pc_obj):,} object pts"

    # Region bounding boxes — only when there are multiple regions
    _REGION_MIN_POINTS = 50  # minimum BEV points from a region to draw its box

    # Build object_id -> region_id lookup, then count BEV points per region
    obj_to_region: dict[int, int] = {
        obj.id: int(obj.metadata["region_id"])
        for obj in query.scene_graph.objects
        if "region_id" in obj.metadata
    }
    region_point_counts: dict[int, int] = {}
    for oid in split_obj:
        rid = obj_to_region.get(int(oid))
        if rid is not None:
            region_point_counts[rid] = region_point_counts.get(rid, 0) + 1

    regions = query.scene_graph.regions
    if len(regions) > 1:
        region_colors = plt.cm.Set2.colors  # 8 distinct pastel colours
        color_idx = 0
        for region in regions:
            m = region.metadata
            bbox_keys = ("bbox_x_min", "bbox_x_max", "bbox_y_min", "bbox_y_max")
            if not all(k in m for k in bbox_keys):
                continue

            # Label filter
            if region.label.lower().strip() not in VALID_REGION_LABELS:
                continue

            # Point threshold: skip regions with too few plotted BEV points
            if region_point_counts.get(region.id, 0) < _REGION_MIN_POINTS:
                continue

            rx0, rx1 = m["bbox_x_min"], m["bbox_x_max"]
            ry0, ry1 = m["bbox_y_min"], m["bbox_y_max"]
            i = color_idx
            color_idx += 1
            color = region_colors[i % len(region_colors)]
            rect = mpatches.FancyBboxPatch(
                (rx0, ry0), rx1 - rx0, ry1 - ry0,
                boxstyle="square,pad=0",
                linewidth=1.8, edgecolor=color, facecolor="none",
                linestyle="--", zorder=4,
            )
            ax.add_patch(rect)
            ax.text(
                rx0 + (rx1 - rx0) * 0.02, ry1,
                region.label or f"region {region.id}",
                fontsize=11, color=color, fontweight="bold",
                va="bottom", ha="left", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color,
                          alpha=0.75, linewidth=0.8),
            )

    # Target marker — gold star
    tx, ty = float(query.target_xyz[0]), float(query.target_xyz[1])
    ax.plot(tx, ty, marker="*", color="#FFD700", markersize=16,
            markeredgecolor="black", markeredgewidth=0.8, zorder=5)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    lang = query.language if len(query.language) <= 80 else query.language[:77] + "..."
    ax.set_title(f"{query.scene_id}\n\"{lang}\"", fontsize=9)
    ax.annotate(note, xy=(0.01, 0.01), xycoords="axes fraction",
                fontsize=7, color="lightgray")

    fig.tight_layout()
    return fig
