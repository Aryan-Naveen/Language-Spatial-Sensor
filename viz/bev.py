"""Bird's Eye View (BEV) occupancy grid visualization.

Pipeline (semantic mode, default when query.object_split is available):
    1. filter_object_points  — keep only non-structural object points
    2. make_semantic_bev     — project to XY, bin into an RGB occupancy image
    3. render_bev            — produce a matplotlib Figure

For training-loop visualization:
    render_bev_with_sample_overlay  — renders anchor-highlight BEV, then hexbins samples on top
"""

from __future__ import annotations

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


def _nyu40_label(obj) -> str:
    """Normalized NYU40 label for a scene-graph object."""
    return str(obj.metadata.get("nyu40_label", "")).lower().strip()


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
        nyu = _nyu40_label(obj)
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
    valid_ids = np.array([
        obj.id for obj in scene_graph.objects
        if _nyu40_label(obj) in VALID_NYU40_LABELS
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
# Vivid magenta — chosen to stay well clear of every entry in ``_PALETTE``
# so the anchor highlight never collides with a semantic colour when
# ``semantic_background=True``. Nearest palette neighbour is ``person``
# (RGB distance ≈ 0.39).
_HIGHLIGHT  = np.array([0.93, 0.11, 0.65], dtype=np.float32)


def make_semantic_bev(
    pc: np.ndarray,
    object_split: np.ndarray,
    scene_graph: SceneGraph,
    resolution: float = 0.05,
    margin: float = 0.5,
    highlight_labels: set[str] | None = None,
    highlight_object_ids: set[int] | None = None,
    semantic_background: bool = False,
    anchor_semantic_fill: bool = False,
) -> tuple[np.ndarray, dict]:
    """Semantic RGB BEV image.

    Each occupied cell is coloured by the object's NYU40 label.  Background
    cells are black (0, 0, 0).

    When ``highlight_labels`` is provided the image switches to a two-tone
    mode: cells whose label is in the set are drawn in magenta, all other
    occupied cells are rendered in grayscale.

    When ``highlight_object_ids`` is provided, highlights those specific
    object IDs in magenta (takes precedence over ``highlight_labels``).

    When ``semantic_background`` is True **and** a highlight set is active,
    non-highlighted cells keep their NYU40 palette colour instead of being
    flattened to grayscale — useful when you want an object-aware backdrop
    under a density overlay.

    When ``anchor_semantic_fill`` is True **and** ``highlight_object_ids`` is
    given, the mode inverts: highlighted (anchor) cells are painted with
    their NYU40 palette colour, and every other cell is drawn in per-label
    grayscale. The magenta ``_HIGHLIGHT`` fill is not used in this mode —
    callers that still want to mark anchors can overlay outlines separately.

    Returns:
        image: (H, W, 3) float32 RGB image.
        meta: same keys as make_bev_grid.
    """
    if len(pc) == 0:
        return np.ones((1, 1, 3), dtype=np.float32), {
            "x_min": 0.0, "y_min": 0.0, "resolution": resolution,
            "width": 1, "height": 1,
        }

    present_ids = set(int(i) for i in np.unique(object_split))
    id_to_obj = {obj.id: obj for obj in scene_graph.objects}

    def _label_gray(lbl: str) -> np.ndarray:
        """Deterministic per-class gray in [0.30, 0.72] so classes stay distinct."""
        v = 0.30 + (abs(hash(lbl)) % 256) / 256 * 0.42
        return np.array([v, v, v], dtype=np.float32)

    def _semantic_or_gray(lbl: str, oid: int) -> np.ndarray:
        """NYU40 palette colour with grayscale fallback if the label is missing."""
        try:
            return np.array(nyu40_to_color(lbl), dtype=np.float32)
        except KeyError:
            return _label_gray(lbl or str(oid))

    def _background_color(lbl: str, oid: int) -> np.ndarray:
        """Colour used for non-highlighted objects in highlight mode."""
        if semantic_background:
            return _semantic_or_gray(lbl, oid)
        return _label_gray(lbl or str(oid))

    def _color_for(oid: int) -> np.ndarray:
        obj = id_to_obj.get(oid)
        if obj is None:
            return _GRAYSCALE
        lbl = _nyu40_label(obj)
        if highlight_object_ids is not None:
            if anchor_semantic_fill:
                return (
                    _semantic_or_gray(lbl, oid) if oid in highlight_object_ids
                    else _label_gray(lbl or str(oid))
                )
            return _HIGHLIGHT if oid in highlight_object_ids else _background_color(lbl, oid)
        if highlight_labels is not None:
            if anchor_semantic_fill:
                return (
                    _semantic_or_gray(lbl, oid) if lbl in highlight_labels
                    else _label_gray(lbl or str(oid))
                )
            return _HIGHLIGHT if lbl in highlight_labels else _background_color(lbl, oid)
        return np.array(nyu40_to_color(lbl), dtype=np.float32)

    id_to_color: dict[int, np.ndarray] = {
        oid: _color_for(oid) for oid in present_ids if oid in id_to_obj
    }
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

    in_highlight_mode = highlight_object_ids is not None or highlight_labels is not None
    if in_highlight_mode:
        if highlight_object_ids is not None:
            is_highlight = np.array([int(oid) in highlight_object_ids for oid in object_split])
        else:
            is_highlight = np.array([
                _nyu40_label(id_to_obj[int(oid)]) in highlight_labels
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
# Legend helper
# ---------------------------------------------------------------------------

def _build_legend_handles(
    split_obj: np.ndarray,
    highlight_labels: set[str] | None,
    id_to_obj: dict,
    highlight_object_ids: set[int] | None = None,
) -> list[mpatches.Patch]:
    """Build matplotlib legend patches for the BEV image."""
    present_ids = set(int(i) for i in np.unique(split_obj))

    if highlight_object_ids is not None:
        gray_seen: dict[str, np.ndarray] = {}
        hi_labels: list[str] = []
        for oid in present_ids:
            obj = id_to_obj.get(oid)
            if obj is None:
                continue
            lbl = _nyu40_label(obj)
            if oid in highlight_object_ids:
                hi_labels.append(lbl or str(oid))
            elif lbl and lbl not in gray_seen:
                gv = 0.30 + (abs(hash(lbl)) % 256) / 256 * 0.42
                gray_seen[lbl] = np.array([gv, gv, gv])
        label_str = ", ".join(sorted(set(hi_labels))) if hi_labels else "anchors"
        return [
            mpatches.Patch(color=_HIGHLIGHT, label=label_str),
            *[mpatches.Patch(color=c, label=lbl) for lbl, c in sorted(gray_seen.items())],
        ]
    elif highlight_labels:
        gray_seen2: dict[str, np.ndarray] = {}
        for oid in present_ids:
            obj = id_to_obj.get(oid)
            if obj is None:
                continue
            lbl = _nyu40_label(obj)
            if lbl and lbl not in highlight_labels and lbl not in gray_seen2:
                gv = 0.30 + (abs(hash(lbl)) % 256) / 256 * 0.42
                gray_seen2[lbl] = np.array([gv, gv, gv])
        return [
            mpatches.Patch(color=_HIGHLIGHT, label=", ".join(sorted(highlight_labels))),
            *[mpatches.Patch(color=c, label=lbl) for lbl, c in sorted(gray_seen2.items())],
        ]
    else:
        seen: dict[str, tuple[float, float, float]] = {}
        for oid in present_ids:
            if oid in id_to_obj:
                lbl = _nyu40_label(id_to_obj[oid])
                if lbl and lbl not in seen:
                    seen[lbl] = nyu40_to_color(lbl)
        return [
            mpatches.Patch(color=c, label=lbl)
            for lbl, c in sorted(seen.items())
        ]


# ---------------------------------------------------------------------------
# Full render pipeline
# ---------------------------------------------------------------------------

def render_bev(
    query: SpatialQuery,
    resolution: float = 0.05,
    figsize: tuple[int, int] = (8, 8),
    include_legend: bool = True,
    anchor_highlight: bool = False,
    highlight_object_ids: set[int] | None = None,
    show_target: bool = True,
    semantic_background: bool = False,
    anchor_semantic_fill: bool = False,
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
            which are drawn in magenta.
        highlight_object_ids: When provided, highlights these specific object
            IDs in magenta (takes precedence over anchor_highlight).
        show_target: When False, suppresses the gold-star GT target marker.

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

    # Build once; reused for highlight resolution and legend
    id_to_obj = {obj.id: obj for obj in query.scene_graph.objects}

    # Resolve highlight mode: explicit object IDs take precedence over label-based anchor_highlight
    highlight_labels: set[str] | None = None
    if highlight_object_ids is None and anchor_highlight and query.gt_anchor_object_ids:
        highlight_labels = {
            _nyu40_label(id_to_obj[aid])
            for aid in query.gt_anchor_object_ids
            if aid in id_to_obj
        } - {""}

    image, meta = make_semantic_bev(
        pc_obj, split_obj, query.scene_graph,
        resolution=resolution,
        highlight_labels=highlight_labels,
        highlight_object_ids=highlight_object_ids,
        semantic_background=semantic_background,
        anchor_semantic_fill=anchor_semantic_fill,
    )
    extent = [
        meta["x_min"],
        meta["x_min"] + meta["width"]  * resolution,
        meta["y_min"],
        meta["y_min"] + meta["height"] * resolution,
    ]
    ax.imshow(image, origin="lower", interpolation="nearest", extent=extent)

    if include_legend:
        legend_handles = _build_legend_handles(split_obj, highlight_labels, id_to_obj, highlight_object_ids)
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper right", fontsize=6,
                      framealpha=0.8, ncol=2)

    note = f"{len(pc_obj):,} object pts"

    # Region bounding boxes — only when there are multiple regions
    _REGION_MIN_POINTS = 50  # minimum BEV points from a region to draw its box

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

            if region.label.lower().strip() not in VALID_REGION_LABELS:
                continue

            if region_point_counts.get(region.id, 0) < _REGION_MIN_POINTS:
                continue

            rx0, rx1 = m["bbox_x_min"], m["bbox_x_max"]
            ry0, ry1 = m["bbox_y_min"], m["bbox_y_max"]
            color = region_colors[color_idx % len(region_colors)]
            color_idx += 1
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

    # Target marker — gold star (only when GT target is available)
    if show_target:
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


# ---------------------------------------------------------------------------
# Training-loop BEV: SpatialQuery + position samples → heatmap overlay
# ---------------------------------------------------------------------------

def render_bev_with_sample_overlay(
    query: SpatialQuery,
    samples_xyz: np.ndarray,  # (K, 3) position samples in world frame
    resolution: float = 1, # voxel size in metres for the density overlay
    heatmap_alpha: float = 0.35,
    highlight_object_ids: set[int] | None = None,
    show_target: bool = True,
    semantic_background: bool = False,
    include_legend: bool = False,
    anchor_highlight: bool | None = None,
    background_alpha: float = 1.0,
    anchor_semantic_fill: bool = False,
) -> plt.Figure:
    """Render the anchor-highlight BEV, then overlay a per-voxel sample-proportion grid.

    The underlying occupancy grid is produced by ``render_bev(anchor_highlight=True)``
    — identical to ``scripts/test_viz --anchor_highlight``.

    Each BEV voxel is coloured by the proportion of ``samples_xyz`` that fall in it
    (count / total_samples).  The colormap is always anchored to [0, 1] so the same
    proportion value maps to the same colour across every epoch and every scene.
    Voxels with zero samples are fully transparent, leaving the occupancy grid visible.

    The interface accepts raw position **samples** rather than a distribution object
    so it generalises to diffusion-transformer output without code changes.

    Args:
        query:          SpatialQuery with point cloud and anchor metadata loaded.
        samples_xyz:    (K, 3) array of position samples in world frame.
        resolution:     Voxel side length in metres — match to BEV resolution.
        heatmap_alpha:  Opacity of occupied voxels (0 = invisible, 1 = opaque).

    Returns:
        ``matplotlib.figure.Figure``
    """
    if anchor_highlight is None:
        anchor_highlight = highlight_object_ids is None

    fig = render_bev(
        query,
        anchor_highlight=anchor_highlight,
        highlight_object_ids=highlight_object_ids,
        include_legend=False,
        show_target=show_target,
        semantic_background=semantic_background,
        anchor_semantic_fill=anchor_semantic_fill,
    )
    ax = fig.axes[0]

    # Mute the semantic backdrop so the density overlay reads as primary.
    if background_alpha < 1.0 and ax.images:
        ax.images[0].set_alpha(background_alpha)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    W = max(int(np.ceil((x_max - x_min) / resolution)), 1)
    H = max(int(np.ceil((y_max - y_min) / resolution)), 1)

    counts, _, _ = np.histogram2d(
        samples_xyz[:, 0], samples_xyz[:, 1],
        bins=[W, H],
        range=[[x_min, x_max], [y_min, y_max]],
    )
    proportion = (counts / len(samples_xyz)).T  # (H, W), values in [0, 1]

    # Log-scale for more signal at low densities.
    # log(proportion + eps) ∈ [log(eps), 0]; remap that fixed interval to [0, 1]
    # so 0 proportion always → cool end and 1 proportion always → warm end.
    _EPS = 1e-8
    _LOG_MIN = np.log(_EPS)   # fixed lower bound ≈ -18.4
    _LOG_MAX = np.log(0.2)    # fixed upper bound ≈ -1.6; proportions ≥ 0.2 saturate to warm end
    log_prop = np.log(proportion + _EPS)
    normalized = np.clip((log_prop - _LOG_MIN) / (_LOG_MAX - _LOG_MIN), 0.0, 1.0)

    rgba = plt.cm.plasma(normalized)             # (H, W, 4)
    # Transparent where proportion is below threshold; opaque elsewhere
    rgba[..., 3] = np.where(proportion > 0, heatmap_alpha, 0.0)

    # ── Mask voxels outside all region bounding boxes ─────────────────────────
    # Use every region bbox in the scene graph (including regions excluded from
    # the BEV render due to invalid semantics) as the authoritative scene extent.
    # Voxels whose centres fall outside every bbox are set to alpha=0 (white).
    region_boxes = []
    for region in query.scene_graph.regions:
        m = region.metadata
        if all(k in m for k in ("bbox_x_min", "bbox_x_max", "bbox_y_min", "bbox_y_max")):
            region_boxes.append((
                float(m["bbox_x_min"]), float(m["bbox_x_max"]),
                float(m["bbox_y_min"]), float(m["bbox_y_max"]),
            ))

    vox_cx = x_min + (np.arange(W) + 0.5) * resolution  # (W,)
    vox_cy = y_min + (np.arange(H) + 0.5) * resolution  # (H,)
    cx_grid, cy_grid = np.meshgrid(vox_cx, vox_cy)       # (H, W) each

    inside = np.zeros((H, W), dtype=bool)
    for rx0, rx1, ry0, ry1 in region_boxes:
        inside |= (cx_grid >= rx0) & (cx_grid <= rx1) & (cy_grid >= ry0) & (cy_grid <= ry1)

    rgba[~inside, 3] = 0.0

    ax.imshow(
        rgba,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        interpolation="nearest",
        zorder=4,  # above BEV image, below gold star (zorder=5)
    )

    if include_legend:
        _attach_density_and_markers_legend(
            fig, ax,
            log_min=_LOG_MIN,
            log_max=_LOG_MAX,
            heatmap_alpha=heatmap_alpha,
            show_anchor=highlight_object_ids is not None and len(highlight_object_ids) > 0,
            show_target=show_target,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Structural anchor / GT-target / mode markers (GMM overlay)
# ---------------------------------------------------------------------------

def _object_xy_aabb(obj) -> tuple[float, float, float, float] | None:
    """Axis-aligned XY bbox (x0, y0, x1, y1) from a scene-graph object.

    Uses the 8-corner world-frame bbox if present, otherwise falls back to
    a small square around ``obj.position``.
    """
    if obj.bbox is not None:
        corners = np.asarray(obj.bbox, dtype=np.float32).reshape(-1, 3)
        x0, x1 = float(corners[:, 0].min()), float(corners[:, 0].max())
        y0, y1 = float(corners[:, 1].min()), float(corners[:, 1].max())
        return x0, y0, x1, y1
    pos = getattr(obj, "position", None)
    if pos is None:
        return None
    px, py = float(pos[0]), float(pos[1])
    return px - 0.15, py - 0.15, px + 0.15, py + 0.15


def _draw_anchor_outlines(
    ax: plt.Axes,
    query: SpatialQuery,
    anchor_object_ids: set[int],
    color: np.ndarray | str = None,
) -> None:
    """Draw dashed outlines + small ``A{i}: {label}`` tags for each anchor.

    Structural (outline-only) styling — avoids competing with the density
    heatmap for attention.
    """
    if not anchor_object_ids:
        return
    color = _HIGHLIGHT if color is None else color
    id_to_obj = {obj.id: obj for obj in query.scene_graph.objects}
    for i, oid in enumerate(sorted(anchor_object_ids), start=1):
        obj = id_to_obj.get(int(oid))
        if obj is None:
            continue
        aabb = _object_xy_aabb(obj)
        if aabb is None:
            continue
        x0, y0, x1, y1 = aabb
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=1.6, edgecolor=color, facecolor="none",
            linestyle="--", zorder=5,
        ))
        ax.text(
            x0, y1,
            f"A{i}: {obj.label or f'#{oid}'}",
            fontsize=6.5, color=color, ha="left", va="bottom", zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.15", fc="white",
                ec=color, alpha=0.75, linewidth=0.5,
            ),
        )


def _draw_gt_bullseye(
    ax: plt.Axes,
    x: float,
    y: float,
    outer_radius: float = 0.22,
    inner_radius: float = 0.07,
    crosshair_len: float = 0.32,
) -> None:
    """Bullseye marker for GT target: crosshair + outer ring + inner dot.

    Chosen over a star because it reads as *measurement-like* and integrates
    with the probabilistic density rather than competing with it.
    """
    ax.plot(
        [x - crosshair_len, x + crosshair_len], [y, y],
        color="black", linewidth=0.7, zorder=6, solid_capstyle="round",
    )
    ax.plot(
        [x, x], [y - crosshair_len, y + crosshair_len],
        color="black", linewidth=0.7, zorder=6, solid_capstyle="round",
    )
    ax.add_patch(mpatches.Circle(
        (x, y), radius=outer_radius,
        linewidth=1.3, edgecolor="black", facecolor="none", zorder=7,
    ))
    ax.add_patch(mpatches.Circle(
        (x, y), radius=inner_radius,
        linewidth=0.6, edgecolor="black", facecolor="white", zorder=8,
    ))


# ---------------------------------------------------------------------------
# Density colorbar + anchor/target legend (used by the sample/GMM overlays)
# ---------------------------------------------------------------------------

class _BullseyeProxy:
    """Sentinel handle used only to key into ``BullseyeHandler`` in a legend."""
    pass


class _BullseyeHandler:
    """Custom legend handler that draws the GT bullseye (crosshair + ring + dot).

    Matches ``_draw_gt_bullseye`` — otherwise the legend would misrepresent the
    actual marker.
    """

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle

        x0, y0 = handlebox.xdescent, handlebox.ydescent
        w, h = handlebox.width, handlebox.height
        cx = x0 + w / 2.0
        cy = y0 + h / 2.0
        r_out = min(w, h) * 0.35
        r_in = min(w, h) * 0.12
        r_cross = min(w, h) * 0.55

        artists = [
            Line2D([cx - r_cross, cx + r_cross], [cy, cy],
                   color="black", linewidth=0.7, solid_capstyle="round"),
            Line2D([cx, cx], [cy - r_cross, cy + r_cross],
                   color="black", linewidth=0.7, solid_capstyle="round"),
            Circle((cx, cy), radius=r_out,
                   edgecolor="black", facecolor="none", linewidth=1.1),
            Circle((cx, cy), radius=r_in,
                   edgecolor="black", facecolor="white", linewidth=0.5),
        ]
        for a in artists:
            handlebox.add_artist(a)
        return artists[-1]


def _attach_density_and_markers_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    log_min: float,
    log_max: float,
    heatmap_alpha: float,
    show_anchor: bool,
    show_target: bool,
    show_modes: bool = False,
    anchor_style: str = "outline",   # "outline" (GMM overlay) | "fill" (legacy)
    target_style: str = "bullseye",  # "bullseye" (GMM overlay) | "star" (legacy)
) -> None:
    """Add a density colorbar and a minimal marker legend to ``ax``.

    The colorbar maps plasma values back to the underlying sample proportion
    (the overlay normalises on log-scale, so the colorbar ticks are at fixed
    proportions 1e-6 … 0.2).
    """
    from matplotlib import colors as mcolors
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D

    # Density colorbar — tick at a handful of proportions that span log_min..log_max.
    tick_props = [1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.2]
    tick_norms = [
        float(np.clip((np.log(p) - log_min) / (log_max - log_min), 0.0, 1.0))
        for p in tick_props
    ]
    sm = ScalarMappable(norm=mcolors.Normalize(vmin=0.0, vmax=1.0), cmap=plt.cm.plasma)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=ax, fraction=0.035, pad=0.02, shrink=0.75,
        ticks=tick_norms, alpha=heatmap_alpha,
    )
    cbar.ax.set_yticklabels([f"{p:g}" for p in tick_props], fontsize=7)
    cbar.set_label("density (sample proportion)", fontsize=8)
    cbar.outline.set_linewidth(0.4)

    handles: list = []
    labels: list[str] = []
    handler_map: dict = {}

    if show_anchor:
        if anchor_style == "outline":
            handles.append(Line2D(
                [0, 1], [0, 0], color=_HIGHLIGHT, linestyle="--", linewidth=1.6,
            ))
            labels.append("anchor bbox")
        else:
            handles.append(mpatches.Patch(color=_HIGHLIGHT))
            labels.append("anchor object(s)")
    if show_target:
        if target_style == "bullseye":
            proxy = _BullseyeProxy()
            handles.append(proxy)
            labels.append("GT location")
            handler_map[_BullseyeProxy] = _BullseyeHandler()
        else:
            handles.append(Line2D(
                [0], [0], marker="*", color="none", markerfacecolor="#FFD700",
                markeredgecolor="black", markeredgewidth=0.8, markersize=12,
            ))
            labels.append("GT location")
    if show_modes:
        handles.append(Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.0, markersize=7,
        ))
        labels.append("GMM mode (size ∝ weight)")

    if handles:
        ax.legend(
            handles=handles, labels=labels, loc="upper right", fontsize=7,
            framealpha=0.85, ncol=1, handler_map=handler_map or None,
        )


# ---------------------------------------------------------------------------
# GMM overlay (for LanguageSpatialSensor pipeline output)
# ---------------------------------------------------------------------------

def render_bev_with_gmm_overlay(
    query: SpatialQuery,
    gmm_result,  # GMMResult — avoid circular import
    n_samples: int = 5000,
    resolution: float = 0.25,
    heatmap_alpha: float = 0.35,
    show_component_means: bool = True,
    show_target: bool = False,
    seed: int = 42,
    include_legend: bool = False,
    background_alpha: float = 0.55,
    full_color_backdrop: bool = False,
    semantic_background: bool = True,  # kept for backward compat; ignored
) -> plt.Figure:
    """Render BEV with a GMM density overlay, structural anchor outlines,
    and a bullseye GT marker.

    Visual hierarchy (strongest → weakest):

        density heatmap ▸ GT bullseye + GMM modes ▸ anchor outlines ▸ backdrop.

    Backdrop policy (default): anchor objects keep their NYU40 palette
    colour to signal what drove the prediction; every other object is drawn
    in per-label grayscale so the density stays dominant.  Pass
    ``full_color_backdrop=True`` to fall back to the classic behaviour where
    every object is painted by its NYU40 palette colour.  Anchors also
    receive a dashed magenta outline and an ``A{i}: {label}`` tag in both
    modes.  The GT location uses a crosshair + ring bullseye so it reads as
    measurement-like rather than categorical.

    Args:
        query:               SpatialQuery with point cloud data for BEV.
        gmm_result:         :class:`GMMResult` from LanguageSpatialSensor.predict().
        n_samples:          Samples drawn from the GMM for the density estimate.
        resolution:         Voxel size in metres for the density overlay.
        heatmap_alpha:      Opacity of the density overlay.
        show_component_means: Mark each Gaussian component with a weight-sized circle.
        show_target:        Draw the GT location bullseye marker.
        seed:               Random seed for deterministic sampling.
        include_legend:     Attach the density colorbar + anchor/GT/mode legend.
        background_alpha:   Opacity of the semantic backdrop (<1 mutes it so the
                            density overlay dominates the visual hierarchy).
        full_color_backdrop: If True, paint every object by its NYU40 palette
                            colour instead of the default grayscale-except-anchors
                            backdrop.
        semantic_background: Deprecated no-op kept for backward compatibility.

    Returns:
        ``matplotlib.figure.Figure``
    """
    del semantic_background  # always uses the anchor_semantic_fill backdrop now
    anchor_ids: set[int] = set()
    for g in gmm_result.groundings:
        anchor_ids.update(g.anchor_object_ids)

    samples_xyz = gmm_result.sample(n_samples, seed=seed)
    # Backdrop: default = anchor cells keep NYU40 palette, rest go grayscale;
    # full_color_backdrop = classic semantic palette everywhere.  We always
    # draw our own structural anchor outlines + bullseye target afterwards,
    # so the magenta fill-highlight is never enabled here.
    fig = render_bev_with_sample_overlay(
        query, samples_xyz,
        resolution=resolution,
        heatmap_alpha=heatmap_alpha,
        highlight_object_ids=(
            None if full_color_backdrop else (anchor_ids if anchor_ids else None)
        ),
        show_target=False,
        semantic_background=False,
        include_legend=False,
        anchor_highlight=False,
        background_alpha=background_alpha,
        anchor_semantic_fill=not full_color_backdrop,
    )
    ax = fig.axes[0]

    _draw_anchor_outlines(ax, query, anchor_ids)

    if show_target:
        tx, ty = float(query.target_xyz[0]), float(query.target_xyz[1])
        _draw_gt_bullseye(ax, tx, ty)

    if show_component_means:
        mus = gmm_result.mus.numpy()           # (K, 3)
        weights = gmm_result.weights.numpy()   # (K,)
        for k in range(len(weights)):
            mx, my = float(mus[k, 0]), float(mus[k, 1])
            w = float(weights[k])
            ax.plot(
                mx, my, "o",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.0,
                markersize=5 + 8 * w,  # modest range so high-w doesn't dominate
                zorder=7,
            )
            ax.annotate(
                f"{w:.2f}",
                (mx, my),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=6.5, color="black",
                zorder=8,
                bbox=dict(
                    boxstyle="round,pad=0.12", fc="white", ec="black",
                    alpha=0.8, linewidth=0.4,
                ),
            )

    if include_legend:
        _EPS = 1e-8
        _attach_density_and_markers_legend(
            fig, ax,
            log_min=float(np.log(_EPS)),
            log_max=float(np.log(0.2)),
            heatmap_alpha=heatmap_alpha,
            show_anchor=bool(anchor_ids),
            show_target=show_target,
            show_modes=show_component_means,
            anchor_style="outline",
            target_style="bullseye",
        )

    fig.tight_layout()
    return fig
