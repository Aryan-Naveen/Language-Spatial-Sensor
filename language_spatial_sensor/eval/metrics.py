"""Evaluation metrics for the Language Spatial Sensor pipeline.

Three metric groups:

(a) Proposer precision / recall
    Measures how well the LLM-generated proposals cover the ground-truth
    anchor objects and region.

(b) GMM CDF likelihood at the GT bounding box
    Uses per-axis Gaussian marginals (matching the bbox_cdf_loss used at
    training time) to measure how much Gaussian mass the model places inside
    the target AABB.  Wraps ``marginal_cdf_at_gt`` from heads.py.

(c) Expected Calibration Error (ECE)
    Checks whether predicted uncertainty corresponds to empirical coverage.
    For each confidence level α, it checks what fraction of GT positions fall
    inside the α-confidence ellipsoid of the (best) GMM component, and
    compares against α.

All "aggregate" functions operate on lists of per-sample result dicts produced
by EvalRunner.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from scipy.special import gammainc  # type: ignore[import-untyped]

from language_spatial_sensor.core.schema import SpatialQuery
from language_spatial_sensor.models.components.heads import (
    GaussianPrediction,
    marginal_cdf_at_gt,
)
from language_spatial_sensor.pipeline.language_sensor import GMMPrediction
from language_spatial_sensor.proposer.schema import Proposal


# ── (a) Proposer precision / recall ──────────────────────────────────────────

def proposer_precision(proposals: list[Proposal], query: SpatialQuery) -> float:
    """Fraction of generated proposals that are "correct".

    A proposal is correct if:
      * Its region_id matches query.gt_anchor_room_id, AND
      * All of its anchor_ids are contained in query.gt_anchor_object_ids.
    """
    if not proposals:
        return 0.0
    gt_anchors = set(query.gt_anchor_object_ids or [])
    gt_region  = query.gt_anchor_room_id
    n_correct  = sum(
        1 for p in proposals
        if p.region_id == gt_region and set(p.anchor_ids).issubset(gt_anchors)
    )
    return n_correct / len(proposals)


def proposer_recall(proposals: list[Proposal], query: SpatialQuery) -> float:
    """1.0 if at least one proposal is correct, else 0.0."""
    gt_anchors = set(query.gt_anchor_object_ids or [])
    gt_region  = query.gt_anchor_room_id
    for p in proposals:
        if p.region_id == gt_region and set(p.anchor_ids).issubset(gt_anchors):
            return 1.0
    return 0.0


def proposer_mrr(proposals: list[Proposal], query: SpatialQuery) -> float:
    """Reciprocal rank of the first correct proposal (0.0 if none correct)."""
    gt_anchors = set(query.gt_anchor_object_ids or [])
    gt_region  = query.gt_anchor_room_id
    for rank, p in enumerate(proposals, start=1):
        if p.region_id == gt_region and set(p.anchor_ids).issubset(gt_anchors):
            return 1.0 / rank
    return 0.0


# ── (b) GMM CDF likelihood at GT bbox ────────────────────────────────────────

def _to_aabb6(bbox: np.ndarray) -> np.ndarray:
    """Normalize any bbox representation to a 6-element AABB.

    Accepts:
      - (6,)  already AABB [x_min, y_min, z_min, x_max, y_max, z_max]
      - (N*3,) or (N, 3) for any N — treats as N corner points, returns min/max
    """
    arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if arr.size == 6:
        return arr
    if arr.size % 3 == 0:
        corners = arr.reshape(-1, 3)
        return np.concatenate([corners.min(axis=0), corners.max(axis=0)])
    raise ValueError(f"Cannot convert bbox with {arr.size} values to AABB")


def cdf_likelihood(
    gmm: GMMPrediction,
    target_bbox_world: np.ndarray,  # (6,) or any corner format
) -> dict[str, float]:
    """Weighted mixture CDF mass at the GT bounding box.

    For each GMM component k:
        axis_mass_k, mass_prod_k = marginal_cdf_at_gt(pred_k, bbox)

    Final mass = sum_k  weights_k * mass_k

    Returns a dict with keys: axis_x, axis_y, axis_z, joint.
    """
    bbox_t = torch.from_numpy(_to_aabb6(target_bbox_world)).float().unsqueeze(0)  # (1, 6)

    weighted_axis = np.zeros(3, dtype=np.float64)
    weighted_joint = 0.0

    for k, (mu_k, L_k, w_k) in enumerate(
        zip(gmm.means, gmm.Ls, gmm.weights)
    ):
        pred_k = GaussianPrediction(
            mu = torch.from_numpy(mu_k).float().unsqueeze(0),  # (1, 3)
            L  = torch.from_numpy(L_k).float().unsqueeze(0),   # (1, 3, 3)
        )
        axis_mass, mass_prod = marginal_cdf_at_gt(pred_k, bbox_t)
        axis_np  = axis_mass.squeeze(0).numpy()   # (3,)
        joint_np = float(mass_prod.squeeze(0))    # scalar

        weighted_axis  += w_k * axis_np
        weighted_joint += w_k * joint_np

    return {
        "cdf_axis_x": float(weighted_axis[0]),
        "cdf_axis_y": float(weighted_axis[1]),
        "cdf_axis_z": float(weighted_axis[2]),
        "cdf_joint":  float(weighted_joint),
    }


# ── (c) Expected Calibration Error (ECE) ─────────────────────────────────────

def mahalanobis_dist(gmm: GMMPrediction, gt_xyz: np.ndarray) -> float:
    """Mahalanobis distance from gt_xyz to the GMM mean component (highest weight).

    Uses the component with the largest weight as a representative.
    d = sqrt((gt - mu)^T Σ^{-1} (gt - mu))  where  Σ = L @ L^T.
    """
    best_k = int(np.argmax(gmm.weights))
    mu = gmm.means[best_k]     # (3,)
    L  = gmm.Ls[best_k]        # (3, 3)

    diff = gt_xyz - mu         # (3,)
    # Solve L @ v = diff (forward substitution) then ||v||
    v = np.linalg.solve(L, diff)
    return float(np.sqrt(np.dot(v, v)))


def _chi2_threshold_3d(alpha: float) -> float:
    """Chi-squared quantile for 3 DOF at confidence level alpha.

    The Mahalanobis distance squared follows χ²(3) under the Gaussian model,
    so the alpha-confidence ellipsoid corresponds to d² ≤ quantile.
    """
    # scipy.special.gammaincinv(a, p) solves gammainc(a, x) = p
    from scipy.special import gammaincinv  # type: ignore[import-untyped]
    return float(2.0 * gammaincinv(1.5, alpha))   # χ²(3) = 2 * Gamma(3/2)


def ece(
    records: list[dict[str, Any]],
    levels: list[float] | None = None,
) -> dict[str, Any]:
    """Expected Calibration Error over a collection of result records.

    For each confidence level α, the α-confidence ellipsoid should contain α
    fraction of ground-truth positions.  ECE = mean |actual - α|.

    Args:
        records: List of per-sample dicts with key "mahalanobis_dist" (float).
        levels:  Confidence levels to evaluate (default [0.5, 0.68, 0.90, 0.95]).

    Returns dict with:
        ece:                float — mean absolute calibration error
        calibration_curve:  list of [expected, actual] pairs
    """
    if levels is None:
        levels = [0.5, 0.68, 0.90, 0.95]

    dists = np.array([r["mahalanobis_dist"] for r in records], dtype=np.float64)
    if len(dists) == 0:
        return {"ece": float("nan"), "calibration_curve": []}

    curve: list[list[float]] = []
    abs_errors: list[float] = []

    for alpha in levels:
        threshold_d = math.sqrt(_chi2_threshold_3d(alpha))
        actual_coverage = float((dists <= threshold_d).mean())
        curve.append([alpha, actual_coverage])
        abs_errors.append(abs(actual_coverage - alpha))

    return {
        "ece": float(np.mean(abs_errors)),
        "calibration_curve": curve,
    }


# ── Aggregation helpers ───────────────────────────────────────────────────────

def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute all three metric groups over a list of per-sample result dicts.

    Each record must contain (at minimum):
        proposer_precision, proposer_recall, proposer_mrr,
        n_proposals,
        cdf_axis_x, cdf_axis_y, cdf_axis_z, cdf_joint,
        mahalanobis_dist
    """
    if not records:
        return {"n_samples": 0}

    n = len(records)

    def _mean(key: str) -> float:
        vals = [r[key] for r in records if key in r]
        return float(np.mean(vals)) if vals else float("nan")

    def _median(key: str) -> float:
        vals = [r[key] for r in records if key in r]
        return float(np.median(vals)) if vals else float("nan")

    ece_result = ece(records)

    return {
        # (a) Proposer
        "proposer_precision":    _mean("proposer_precision"),
        "proposer_recall":       _mean("proposer_recall"),
        "proposer_mrr":          _mean("proposer_mrr"),
        "n_proposals_mean":      _mean("n_proposals"),
        # (b) CDF likelihood
        "cdf_axis_x_mean":       _mean("cdf_axis_x"),
        "cdf_axis_y_mean":       _mean("cdf_axis_y"),
        "cdf_axis_z_mean":       _mean("cdf_axis_z"),
        "cdf_joint_mean":        _mean("cdf_joint"),
        "cdf_joint_median":      _median("cdf_joint"),
        # (c) ECE
        "mahalanobis_mean":      _mean("mahalanobis_dist"),
        "mahalanobis_median":    _median("mahalanobis_dist"),
        "ece":                   ece_result["ece"],
        "calibration_curve":     ece_result["calibration_curve"],
        "n_samples":             n,
    }


def breakdown_by(
    records: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    """Group records by ``key`` and compute aggregate_metrics for each group.

    Args:
        records: Full list of per-sample result dicts.
        key:     Field to group by, e.g. "relation" or "ambiguity".

    Returns:
        Dict mapping each group value (as a string) to its aggregate metrics.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        val = r.get(key)
        if val is not None:
            groups[str(val)].append(r)

    return {group_key: aggregate_metrics(group_records)
            for group_key, group_records in sorted(groups.items())}
