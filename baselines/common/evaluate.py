"""Shared evaluation and visualization utilities for 3D grounding baselines.

``evaluate()`` returns the same metric dict structure as ``lss/train.py`` so
runs are directly comparable in wandb.

``generate_bev_plots()`` reloads raw VLA3D scenes (for the spatial query) and
overlays GMM samples using the existing ``render_bev_with_sample_overlay``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

# ── sys.path: make lss/ root importable ───────────────────────────────────────
_LSS_ROOT = Path(__file__).resolve().parents[2]
if str(_LSS_ROOT) not in sys.path:
    sys.path.insert(0, str(_LSS_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from viz.bev import render_bev_with_sample_overlay
from baselines.common.gmm import GMMPrediction, sample_from_gmm

if TYPE_CHECKING:
    import torch.nn as nn


# ── GMM marginal CDF (diagnostic metric, matches LSS marginal_cdf_at_gt) ──────

def gmm_marginal_cdf_at_gt(
    pred: GMMPrediction,
    target_bbox_world: Tensor,  # (B, 6) [x_min,y_min,z_min, x_max,y_max,z_max]
) -> tuple[Tensor, Tensor]:
    """Mixture-weighted per-axis Gaussian mass over the target AABB.

    For each component k, computes the same erf-based marginal mass as
    ``lss/heads.py:marginal_cdf_at_gt``, then takes a π-weighted average.

    Returns:
        axis_mass:  (B, 3)  mixture-averaged per-axis interval mass.
        mass_prod:  (B,)    product of axis_mass over axes.
    """
    B, K = pred.logits.shape
    weights = F.softmax(pred.logits, dim=-1)  # (B, K)

    bbox_min = target_bbox_world[:, :3]  # (B, 3)
    bbox_max = target_bbox_world[:, 3:]  # (B, 3)

    # Per-component marginal variances: diag(L_k @ L_k^T) = row-wise sum of squares of L_k
    # pred.L: (B, K, 3, 3)
    var = (pred.L ** 2).sum(dim=-1).clamp(min=1e-12)   # (B, K, 3)
    sigma = var.sqrt()                                  # (B, K, 3)

    # Per-component per-axis CDF mass
    # shape: (B, K, 3)
    mu  = pred.mu                                       # (B, K, 3)
    z_min = (bbox_min.unsqueeze(1) - mu) / (sigma * math.sqrt(2.0))
    z_max = (bbox_max.unsqueeze(1) - mu) / (sigma * math.sqrt(2.0))
    per_component_axis_mass = 0.5 * (torch.erf(z_max) - torch.erf(z_min))  # (B, K, 3)

    # Mixture-weighted average: (B, K) x (B, K, 3) → (B, 3)
    axis_mass = (weights.unsqueeze(-1) * per_component_axis_mass).sum(dim=1)
    mass_prod = axis_mass.prod(dim=-1)  # (B,)

    return axis_mass, mass_prod


# ── Evaluation loop ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: "nn.Module",
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    loss_fn,
    loss_kwargs: dict,
) -> dict[str, float]:
    """Run one validation pass and return metrics matching the LSS eval dict.

    Metric keys (identical to lss/train.py):
        loss, mean_dist, acc,
        gt_bbox_marginal_mass_mean, gt_bbox_marginal_prod_mean

    For GMM: accuracy and distance use the MAP component (highest π_k).

    Args:
        model:          The grounding model; forward(**batch_dict) → GMMPrediction.
        loader:         Validation DataLoader.
        device:         Compute device.
        autocast_dtype: torch.bfloat16 / torch.float16 / torch.float32.
        loss_fn:        Callable(pred, target_xyz, **loss_kwargs) → scalar Tensor.
        loss_kwargs:    Extra kwargs for loss_fn (lambda_dist, entropy_reg, etc.).
    """
    model.eval()
    total_loss = total_dist = total_acc = total_cdf_axis = total_cdf_prod = 0.0
    n = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}

        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            pred: GMMPrediction = model(**batch)
            loss = loss_fn(pred, batch["target_xyz_world"], **loss_kwargs)

        B = pred.logits.shape[0]
        total_loss += loss.item() * B

        # Cast to float32 for metric computations
        pred_f = GMMPrediction(
            logits=pred.logits.float(),
            mu=pred.mu.float(),
            L=pred.L.float(),
        )
        target_xyz  = batch["target_xyz_world"].float()
        target_bbox = batch["target_bbox_world"].float()

        # MAP component mean
        k_star = pred_f.logits.argmax(dim=-1)                       # (B,)
        mu_map = pred_f.mu[torch.arange(B, device=device), k_star]  # (B, 3)

        # Distance (MAP mean → target)
        dist = (mu_map - target_xyz).norm(dim=-1)
        total_dist += dist.sum().item()

        # Accuracy: MAP mean inside target bbox
        inside = (
            (mu_map >= target_bbox[:, :3]).all(dim=-1) &
            (mu_map <= target_bbox[:, 3:]).all(dim=-1)
        )
        total_acc += inside.sum().item()

        # Marginal CDF mass
        axis_mass, mass_prod = gmm_marginal_cdf_at_gt(pred_f, target_bbox)
        total_cdf_axis += axis_mass.sum().item()
        total_cdf_prod += mass_prod.sum().item()

        n += B

    model.train()
    denom = max(n, 1)
    return {
        "loss":                       total_loss / denom,
        "mean_dist":                  total_dist / denom,
        "acc":                        total_acc  / denom,
        "gt_bbox_marginal_mass_mean": total_cdf_axis / max(denom * 3, 1),
        "gt_bbox_marginal_prod_mean": total_cdf_prod / denom,
    }


# ── BEV plot generation ────────────────────────────────────────────────────────

def _load_spatial_query(
    scene_id: str,
    language: str,
    data_root: Path,
    datasets: list[str],
):
    """Reload a SpatialQuery from raw VLA3D files (same logic as lss/train.py)."""
    from data.vla3d.dataset import VLA3DScene

    _CANONICAL = {
        "3rscan": "3RScan", "arkitscenes": "ARKitScenes", "hm3d": "HM3D",
        "matterport": "Matterport", "scannet": "Scannet", "unity": "Unity",
    }

    scene = None
    for ds_name in datasets:
        scene_path = data_root / _CANONICAL.get(ds_name.lower(), ds_name) / scene_id
        if scene_path.exists():
            scene = VLA3DScene(scene_path)
            break
    if scene is None:
        raise FileNotFoundError(
            f"Scene '{scene_id}' not found under {data_root} in {datasets}"
        )

    from language_spatial_sensor.core.transforms import build_spatial_query

    sg         = scene.load_scene_graph()
    statements = scene.load_statements(sg)
    stmt       = next((s for s in statements if s.text == language), None)
    if stmt is None:
        raise ValueError(f"Statement not found in scene '{scene_id}': {language!r}")

    import open3d as o3d
    pcd          = scene.load_pointcloud()
    points       = np.asarray(pcd.points, dtype=np.float32)
    object_split = scene.load_object_split()

    return build_spatial_query(scene_id, sg, stmt, points=points, object_split=object_split)


@torch.no_grad()
def generate_bev_plots(
    model: "nn.Module",
    dataset,                    # ViSTAGroundingDataset (or similar)
    device: torch.device,
    autocast_dtype: torch.dtype,
    collate_fn,
    n_scenes: int,
    n_samples: int,
    viz_seed: int,
    data_root: Path,
    datasets: list[str],
) -> list[tuple[str, plt.Figure]]:
    """Draw BEV + GMM sample-density overlay for N random val scenes.

    ``dataset[idx]`` must have keys ``scene_id`` and ``language`` in addition
    to model inputs.  The raw scene is reloaded from VLA3D to get the spatial
    query needed by ``render_bev_with_sample_overlay``.

    Returns:
        List of (language_text, matplotlib Figure) pairs; scenes that fail to
        load are skipped silently.
    """
    rng     = np.random.RandomState(viz_seed)
    indices = rng.choice(len(dataset), size=min(n_scenes, len(dataset)), replace=False)

    model.eval()
    figs: list[tuple[str, plt.Figure]] = []

    for idx in indices:
        item     = dataset[int(idx)]
        scene_id = item["scene_id"]
        language = item["language"]

        try:
            query = _load_spatial_query(scene_id, language, data_root, datasets)
        except Exception as e:
            print(f"[viz] Skipping {scene_id}: {e}")
            continue

        batch = collate_fn([item])
        batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}

        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            pred: GMMPrediction = model(**batch)

        # Sample from the GMM (single batch item 0)
        pred0 = GMMPrediction(
            logits=pred.logits[0].float(),
            mu=pred.mu[0].float(),
            L=pred.L[0].float(),
        )
        samples_xyz = sample_from_gmm(
            pred0.logits, pred0.mu, pred0.L, n_samples
        ).cpu().numpy()  # (n_samples, 3)

        fig = render_bev_with_sample_overlay(query, samples_xyz=samples_xyz, resolution=0.25)
        figs.append((language, fig))

    model.train()
    return figs
