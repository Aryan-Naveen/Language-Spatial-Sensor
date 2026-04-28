"""Per-layer attention visualization for the SceneSpatialEncoder backbone.

Given a trained ``LanguageSpatialSensor`` plus a scene graph and utterance,
render a BEV figure per spatial-attention layer that shows:

    * the anchor object shaded red
    * every other object colored by how much attention the anchor's query
      row assigns to it in that layer (averaged across heads by default)

The entry point is :func:`render_attention_bev`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from language_spatial_sensor.core.schema import (
    Grounding,
    GroundedQuery,
    SceneGraph,
    SpatialQuery,
)
from language_spatial_sensor.models.backbones.spatial_attention import (
    MultiHeadAttentionSpatial,
    SceneSpatialEncoder,
)
from language_spatial_sensor.pipeline.distribution_predictor import _to_device
from viz.bev import (
    _nyu40_label,
    fit_floor_ceiling_semantic,
    filter_object_points,
    make_semantic_bev,
)


# ---------------------------------------------------------------------------
# Attention capture
# ---------------------------------------------------------------------------

@contextmanager
def _capture_attention(
    backbone: SceneSpatialEncoder,
) -> Iterator[list[MultiHeadAttentionSpatial]]:
    """Temporarily enable attention-weight caching on every backbone layer."""
    attn_modules = [layer.self_attn for layer in backbone.layers]
    for m in attn_modules:
        m._store_attn = True
        m._last_attn_weights = None
    try:
        yield attn_modules
    finally:
        for m in attn_modules:
            m._store_attn = False


# ---------------------------------------------------------------------------
# Object-order reconstruction
# ---------------------------------------------------------------------------

def _objects_in_tensor(
    scene_graph: SceneGraph,
    anchor_room_id: int,
    max_objects: int,
) -> list:
    """Replay the tensorizer's per-region object ordering."""
    objs = [
        obj for obj in scene_graph.objects
        if obj.metadata.get("region_id") == anchor_room_id
    ]
    return objs[:max_objects]


def _object_center_xy(obj) -> tuple[float, float]:
    """World-frame (x, y) centre for an ObjectInfo."""
    if obj.bbox is not None:
        corners = np.asarray(obj.bbox, dtype=np.float32).reshape(-1, 3)
        c = corners.mean(axis=0)
    else:
        c = np.asarray(obj.position, dtype=np.float32)
    return float(c[0]), float(c[1])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_attention_bev(
    sensor,                           # LanguageSpatialSensor
    scene_graph: SceneGraph,
    pc: np.ndarray,
    object_split: np.ndarray,
    utterance: str,
    scene_id: str = "attention_demo",
    grounding: Grounding | None = None,
    resolution: float = 0.15,
    head_reduction: str = "mean",     # "mean" or "max"
    cmap_name: str = "viridis",
    marker_size: float = 260.0,
    ncols: int = 2,
) -> plt.Figure:
    """Render per-layer anchor attention as a grid of BEV subplots.

    Args:
        sensor:         Initialised ``LanguageSpatialSensor``.
        scene_graph:    Scene graph for the scene being queried.
        pc:             (N, 3) point cloud (any frame matching ``object_split``).
        object_split:   (N,) per-point object IDs aligned with ``pc``.
        utterance:      Natural-language query string.
        scene_id:       Scene identifier (used for caching inside the proposer).
        grounding:      Optional explicit grounding; when ``None`` the proposer
                        is run and the highest-confidence grounding is used.
        resolution:     BEV grid cell size in metres.
        head_reduction: How to collapse the per-head attention vectors
                        ("mean" or "max").
        cmap_name:      Name of a matplotlib colormap for non-anchor objects.
        marker_size:    Size of attention-coloured scatter markers.
        ncols:          Number of columns in the subplot grid.
    """
    # --- 1. Pick a grounding -------------------------------------------------
    if grounding is None:
        proposer_query = SpatialQuery(
            scene_id=scene_id,
            scene_graph=scene_graph,
            pc=pc,
            object_split=object_split,
            language=utterance,
            target_xyz=np.zeros(3, dtype=np.float32),
        )
        groundings = sensor.proposer.propose(proposer_query)
        if not groundings:
            raise RuntimeError("Proposer returned no groundings for attention viz.")
        grounding = max(groundings, key=lambda g: g.confidence)

    # --- 2. Tensorize a single-sample batch ---------------------------------
    grounded = GroundedQuery(
        scene_id=scene_id,
        scene_graph=scene_graph,
        pc=pc,
        object_split=object_split,
        grounding=grounding,
    )
    sample = sensor.tensorizer.tensorize_grounded(grounded)
    batch = sensor.collate_fn([sample])
    device = sensor.device
    batch = _to_device(batch, device)

    objs_in_tensor = _objects_in_tensor(
        scene_graph,
        grounding.anchor_room_id,
        sensor.tensorizer.max_objects,
    )
    anchor_ids_set = set(grounding.anchor_object_ids)
    anchor_slots = [
        i for i, obj in enumerate(objs_in_tensor) if obj.id in anchor_ids_set
    ]
    if not anchor_slots:
        raise RuntimeError(
            "No anchor objects found in tensor slots for the chosen grounding."
        )

    # --- 3. Run forward with attention capture ------------------------------
    backbone = sensor.model.backbone
    with _capture_attention(backbone) as attn_modules:
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        ):
            sensor.model(
                batch.text_input_ids,
                batch.text_attention_mask,
                batch.obj_clip_features,
                batch.obj_bboxes,
                batch.obj_is_anchor,
                batch.obj_padding_mask,
                batch.coord_scale,
                batch.coord_shift,
            )
        layer_attn = [m._last_attn_weights for m in attn_modules]

    # --- 4. Reduce to per-object attention vectors --------------------------
    n_obj = len(objs_in_tensor)
    anchor_idx_t = torch.tensor(anchor_slots, dtype=torch.long)
    per_layer_vec: list[np.ndarray] = []
    for weights in layer_attn:
        # weights: (1, H, N_padded, N_padded)
        row = weights[0, :, anchor_idx_t, :]  # (H, |anchors|, N_padded)
        row = row.mean(dim=1)                  # average across anchor rows → (H, N_padded)
        if head_reduction == "mean":
            vec = row.mean(dim=0)
        elif head_reduction == "max":
            vec = row.max(dim=0).values
        else:
            raise ValueError(f"Unknown head_reduction: {head_reduction}")
        per_layer_vec.append(vec[:n_obj].numpy())

    # --- 5. Build BEV base (shared across subplots) -------------------------
    target_proxy = np.asarray(
        _object_center_xy(objs_in_tensor[anchor_slots[0]]) + (0.0,),
        dtype=np.float32,
    )
    z_min, z_max = fit_floor_ceiling_semantic(scene_graph, target_proxy, pc)
    pc_obj, split_obj = filter_object_points(
        pc, object_split, scene_graph, z_min=z_min, z_max=z_max,
    )
    image, meta = make_semantic_bev(
        pc_obj, split_obj, scene_graph,
        resolution=resolution,
        highlight_object_ids=set(grounding.anchor_object_ids),
    )
    extent = [
        meta["x_min"],
        meta["x_min"] + meta["width"] * resolution,
        meta["y_min"],
        meta["y_min"] + meta["height"] * resolution,
    ]

    # --- 6. Precompute per-object marker coordinates & labels --------------
    marker_xy = np.array(
        [_object_center_xy(obj) for obj in objs_in_tensor], dtype=np.float32,
    )
    obj_labels = [_nyu40_label(obj) or str(obj.id) for obj in objs_in_tensor]
    anchor_mask = np.zeros(n_obj, dtype=bool)
    anchor_mask[anchor_slots] = True

    # --- 7. Plot grid -------------------------------------------------------
    num_layers = len(per_layer_vec)
    nrows = (num_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.0 * ncols, 6.0 * nrows),
        squeeze=False,
    )
    cmap = plt.get_cmap(cmap_name)

    anchor_label = ", ".join(sorted({
        obj_labels[i] for i in anchor_slots
    }))
    fig.suptitle(
        f'{scene_id} · "{utterance}"\nanchor: {anchor_label} '
        f'(ids={grounding.anchor_object_ids})',
        fontsize=11,
    )

    for layer_idx in range(num_layers):
        ax = axes[layer_idx // ncols][layer_idx % ncols]
        ax.imshow(image, origin="lower", interpolation="nearest", extent=extent)

        vec = per_layer_vec[layer_idx].copy()
        # Mask anchor entries from vmax so other-object variation is visible.
        vec_for_scale = vec.copy()
        vec_for_scale[anchor_mask] = -np.inf
        vmax = float(vec_for_scale.max()) if np.isfinite(vec_for_scale).any() else 1.0
        vmin = float(vec[~anchor_mask].min()) if (~anchor_mask).any() else 0.0
        if vmax <= vmin:
            vmax = vmin + 1e-6

        non_anchor = ~anchor_mask
        if non_anchor.any():
            sc = ax.scatter(
                marker_xy[non_anchor, 0],
                marker_xy[non_anchor, 1],
                c=vec[non_anchor],
                s=marker_size,
                cmap=cmap,
                vmin=vmin, vmax=vmax,
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )
            fig.colorbar(sc, ax=ax, shrink=0.7, label="attention")

        ax.scatter(
            marker_xy[anchor_mask, 0],
            marker_xy[anchor_mask, 1],
            c="red",
            s=marker_size * 1.4,
            edgecolors="black",
            linewidths=1.2,
            marker="*",
            zorder=6,
            label="anchor",
        )

        for i in range(n_obj):
            ax.annotate(
                obj_labels[i],
                (marker_xy[i, 0], marker_xy[i, 1]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=6,
                color="black",
                zorder=7,
                bbox=dict(
                    boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6,
                ),
            )

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"layer {layer_idx + 1} · heads={head_reduction}", fontsize=10)
        ax.legend(
            handles=[mpatches.Patch(color="red", label="anchor")],
            loc="upper right", fontsize=7, framealpha=0.8,
        )

    # Hide unused axes
    for idx in range(num_layers, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
