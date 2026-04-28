"""Metrics for evaluating the Language Spatial Sensor pipeline.

Provides both proposer-level metrics (anchor precision / recall) and
end-to-end distribution metrics (CDF, RMSE, NLL, ANEES).  All functions
accept lists so they can be vectorised or grouped by relation / ambiguity.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from language_spatial_sensor.core.schema import Grounding, SpatialQuery
from language_spatial_sensor.pipeline.language_sensor import GMMResult


# ---------------------------------------------------------------------------
# Proposer-level metrics
# ---------------------------------------------------------------------------

def proposer_precision_recall(
    groundings_per_query: list[list[Grounding]],
    queries: list[SpatialQuery],
) -> dict[str, Any]:
    """Precision and recall of proposed anchor object IDs vs ground-truth.

    For each query, the *proposed set* is the union of anchor_object_ids
    across all K hypotheses.  The *ground-truth set* is
    ``query.gt_anchor_object_ids``.

    Returns overall averages and per-ambiguity-level breakdowns.
    """
    from evaluation.benchmark import ambiguity_sort_key

    precisions, recalls = [], []
    ambiguities: list[Any] = []
    num_proposed: list[int] = []

    for gs, q in zip(groundings_per_query, queries):
        gt_ids = set(q.gt_anchor_object_ids or [])
        proposed_ids = {aid for g in gs for aid in g.anchor_object_ids}

        tp = len(gt_ids & proposed_ids)
        prec = tp / len(proposed_ids) if proposed_ids else 0.0
        rec = tp / len(gt_ids) if gt_ids else 0.0
        precisions.append(prec)
        recalls.append(rec)
        num_proposed.append(len(proposed_ids))

        # Preserve whatever bucket label the driver attached (int for raw
        # ambiguity integers, or bucket strings like "5+" from
        # run_benchmark_ambig).
        ambiguity = q.metadata.get("ambiguity", 0) if q.metadata else 0
        ambiguities.append(ambiguity)

    out: dict[str, Any] = {
        "proposer_precision": float(np.mean(precisions)),
        "proposer_recall": float(np.mean(recalls)),
        "proposer_num_proposed_mean": float(np.mean(num_proposed)),
        "n": len(queries),
    }

    # Per-ambiguity breakdown — key type is whatever the driver used.
    by_level: dict[Any, list[int]] = defaultdict(list)
    for i, level in enumerate(ambiguities):
        by_level[level].append(i)
    for level in sorted(by_level, key=ambiguity_sort_key):
        idx = by_level[level]
        out[f"proposer_precision_ambig_{level}"] = float(np.mean([precisions[i] for i in idx]))
        out[f"proposer_recall_ambig_{level}"] = float(np.mean([recalls[i] for i in idx]))
        out[f"n_ambig_{level}"] = len(idx)

    return out


# ---------------------------------------------------------------------------
# End-to-end distribution metrics
# ---------------------------------------------------------------------------

def gmm_cdf_at_gt(
    results: list[GMMResult],
    queries: list[SpatialQuery],
) -> list[float]:
    """CDF of the true object location under each predicted GMM.

    Uses the product-of-marginals (axis-aligned) approximation matching
    the training loss.  Returns one value per query in [0, 1].
    """
    cdfs = []
    for r, q in zip(results, queries):
        if q.target_bbox is not None:
            bbox = np.asarray(q.target_bbox, dtype=np.float32).reshape(-1)
            if bbox.size == 24:
                corners = bbox.reshape(8, 3)
                bbox = np.concatenate([corners.min(0), corners.max(0)])
        else:
            xyz = q.target_xyz.astype(np.float32)
            eps = np.full(3, 0.1, dtype=np.float32)
            bbox = np.concatenate([xyz - eps, xyz + eps])
        cdfs.append(r.cdf_bbox(bbox))
    return cdfs


def gmm_rmse(
    results: list[GMMResult],
    queries: list[SpatialQuery],
) -> list[float]:
    """Euclidean distance from the GMM weighted mean to the GT target centre."""
    dists = []
    for r, q in zip(results, queries):
        gmm_mean = r.mean  # (3,)
        target = q.target_xyz.astype(np.float32)
        dists.append(float(np.linalg.norm(gmm_mean - target)))
    return dists


def gmm_nll_at_gt(
    results: list[GMMResult],
    queries: list[SpatialQuery],
) -> list[float]:
    """Negative log-likelihood at the GT target centre under the GMM."""
    nlls = []
    for r, q in zip(results, queries):
        x = torch.as_tensor(q.target_xyz, dtype=torch.float32)
        nlls.append(-float(r.log_prob(x)))
    return nlls


def gmm_nees(
    results: list[GMMResult],
    queries: list[SpatialQuery],
) -> list[float]:
    """Per-query Normalised Estimation Error Squared under the moment-matched
    mixture Gaussian.

    The full mixture — Gaussian components **plus the conformal uniform tail
    if present** — is collapsed to a single effective Gaussian
    ``N(μ_mix, Σ_mix)`` via the law of total (co)variance::

        μ_mix = Σ_k w̃_k μ_k  + α μ_e
        Σ_mix = Σ_k w̃_k (Σ_k + (μ_k - μ_mix)(μ_k - μ_mix)ᵀ)
              + α  ((q²/5) Σ_e + (μ_e - μ_mix)(μ_e - μ_mix)ᵀ)

    where ``w̃_k = (1-α) w_k`` and ``α = conformal.weight``; ``Σ_k = L_k L_kᵀ``
    and ``Σ_e = L_e L_eᵀ``. Without a conformal tail (α = 0) this reduces to
    the plain GMM moment match.

    The uniform tail's (q²/5) Σ_e factor comes from the covariance of a
    uniform distribution over a 3D ball scaled into the Mahalanobis ellipsoid.

    NEES for one query is ``(x - μ_mix)ᵀ Σ_mix⁻¹ (x - μ_mix)``. The average
    across queries (ANEES) is the calibration diagnostic: for a well-calibrated
    3D predictor ANEES ≈ 3 (>3 = over-confident, <3 = under-confident).
    Adding a conformal tail inflates Σ_mix, so ANEES drops toward 3.
    """
    nees_list: list[float] = []
    for r, q in zip(results, queries):
        w = r.weights.to(torch.float64)                       # (K,)
        mus = r.mus.to(torch.float64)                         # (K, 3)
        Ls = r.Ls.to(torch.float64)                           # (K, 3, 3)
        sigmas = Ls @ Ls.transpose(-2, -1)                    # (K, 3, 3)

        alpha = float(r.conformal.weight) if r.conformal is not None else 0.0
        w_tilde = (1.0 - alpha) * w                           # (K,)

        mu_mix = (w_tilde.unsqueeze(-1) * mus).sum(0)         # (3,)
        if r.conformal is not None:
            mu_e = r.conformal.mu.to(torch.float64)
            mu_mix = mu_mix + alpha * mu_e

        diff = mus - mu_mix                                   # (K, 3)
        outer = diff.unsqueeze(-1) @ diff.unsqueeze(-2)       # (K, 3, 3)
        sigma_mix = (w_tilde.view(-1, 1, 1) * (sigmas + outer)).sum(0)  # (3, 3)

        if r.conformal is not None:
            L_e = r.conformal.L.to(torch.float64)
            q_val = float(r.conformal.q)
            sigma_e = L_e @ L_e.T
            diff_e = (mu_e - mu_mix).unsqueeze(-1)             # (3, 1)
            sigma_mix = sigma_mix + alpha * (
                (q_val ** 2 / 5.0) * sigma_e
                + diff_e @ diff_e.T
            )

        target = torch.as_tensor(q.target_xyz, dtype=torch.float64)
        delta = target - mu_mix

        try:
            sol = torch.linalg.solve(sigma_mix, delta)
        except torch._C._LinAlgError:
            sol = torch.linalg.pinv(sigma_mix) @ delta
        nees_list.append(float(delta @ sol))
    return nees_list


# ---------------------------------------------------------------------------
# Component-wise NEES (mode-aware — preferred for ambiguous / multi-modal)
# ---------------------------------------------------------------------------

def _component_nees(r: GMMResult, target_xyz: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-component NEES and normalised weights for a single query.

    Returns:
        ``(eps, w)`` where ``eps[k] = (x* − μ_k)ᵀ Σ_k⁻¹ (x* − μ_k)`` and
        ``w`` is the K-vector of renormalised component weights (sums to 1).

    The conformal uniform tail is intentionally ignored — it has no component
    centre, and ``run_benchmark_ambig`` does not use conformal approaches.
    """
    mus = r.mus.to(torch.float64)                  # (K, 3)
    Ls = r.Ls.to(torch.float64)                    # (K, 3, 3)
    w = r.weights.to(torch.float64)                # (K,)
    target = torch.as_tensor(target_xyz, dtype=torch.float64)
    diff = target.unsqueeze(0) - mus               # (K, 3)

    # Solve L_k L_kᵀ z = diff_k with two triangular solves per component.
    #   y_k = L_k⁻¹ diff_k          (forward substitution)
    #   z_k = L_kᵀ⁻¹ y_k            (back substitution)
    # Then ε_k = diffᵀ z = yᵀ y.
    y = torch.linalg.solve_triangular(
        Ls, diff.unsqueeze(-1), upper=False,
    ).squeeze(-1)                                  # (K, 3)
    eps = (y * y).sum(dim=-1)                      # (K,)

    w_sum = float(w.sum())
    w_norm = w / w_sum if w_sum > 0 else torch.full_like(w, 1.0 / len(w))
    return eps, w_norm


def gmm_nees_min(
    results: list[GMMResult],
    queries: list[SpatialQuery],
) -> list[float]:
    """Per-query **minimum** component NEES: ``ε = min_k ε_k``.

    "Best matching mode" — measures how close the GT is to whichever
    component explains it best.  Useful in the ambiguous setting where a
    good multi-modal prediction should have *at least one* well-calibrated
    component near the GT, even if the others are far away.  Loses the
    clean χ² reference of moment-matched ANEES, but preserves multimodality.
    """
    vals: list[float] = []
    for r, q in zip(results, queries):
        eps, _ = _component_nees(r, q.target_xyz)
        vals.append(float(eps.min()))
    return vals


def gmm_nees_weighted(
    results: list[GMMResult],
    queries: list[SpatialQuery],
) -> list[float]:
    """Per-query **weight-averaged** component NEES: ``ε = Σ_k w_k ε_k``.

    Penalises components that put mass far from the GT in proportion to
    their weight.  Sits between moment-matched ANEES (single-Gaussian
    collapse) and min-NEES (best-mode only) — it keeps the full mixture's
    weight structure without discarding bad modes.
    """
    vals: list[float] = []
    for r, q in zip(results, queries):
        eps, w = _component_nees(r, q.target_xyz)
        vals.append(float((w * eps).sum()))
    return vals


# ---------------------------------------------------------------------------
# Aggregate by group
# ---------------------------------------------------------------------------

def _group_metric(
    values: list[float],
    groups: list,
) -> dict[Any, dict[str, float]]:
    """Group values and compute mean/std/median/IQR/count per group."""
    by_group: dict[Any, list[float]] = defaultdict(list)
    for v, g in zip(values, groups):
        by_group[g].append(v)
    out: dict[Any, dict[str, float]] = {}
    for g, vs in sorted(by_group.items(), key=lambda x: str(x[0])):
        arr = np.asarray(vs, dtype=np.float64)
        q25, q75 = np.percentile(arr, [25, 75])
        out[g] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "q25": float(q25),
            "q75": float(q75),
            "iqr": float(q75 - q25),
            "count": len(vs),
        }
    return out


# ---------------------------------------------------------------------------
# All-in-one
# ---------------------------------------------------------------------------

def compute_all_metrics(
    results: list[GMMResult],
    queries: list[SpatialQuery],
) -> dict[str, Any]:
    """Compute all end-to-end metrics and group by relation / ambiguity.

    Returns a nested dict with keys:
        - ``overall``: aggregate metrics (includes ``anees`` — target ≈ 3 for
          a well-calibrated 3D predictor)
        - ``by_relation``: per-relation breakdowns
        - ``by_ambiguity``: per-ambiguity breakdowns
    """
    cdfs = gmm_cdf_at_gt(results, queries)
    rmses = gmm_rmse(results, queries)
    nlls = gmm_nll_at_gt(results, queries)
    nees = gmm_nees(results, queries)
    nees_min = gmm_nees_min(results, queries)
    nees_w = gmm_nees_weighted(results, queries)

    # Whether GMM mean falls inside GT bbox
    in_bbox = []
    for r, q in zip(results, queries):
        mu = r.mean
        if q.target_bbox is not None:
            tb = np.asarray(q.target_bbox, dtype=np.float32).reshape(-1)
            if tb.size == 24:
                corners = tb.reshape(8, 3)
                tb = np.concatenate([corners.min(0), corners.max(0)])
            inside = bool(np.all(mu >= tb[:3]) and np.all(mu <= tb[3:]))
        else:
            inside = float(np.linalg.norm(mu - q.target_xyz)) < 0.2
        in_bbox.append(inside)

    relations = [q.metadata.get("relation", "unknown") for q in queries]
    ambiguities = [q.metadata.get("ambiguity", 0) for q in queries]

    nll_q25, nll_q75 = np.percentile(nlls, [25, 75]) if nlls else (0.0, 0.0)
    # ANEES is by definition mean(NEES); std/median are over the per-query
    # NEES distribution, not "std of ANEES".
    overall = {
        "cdf_mean": float(np.mean(cdfs)),
        "cdf_median": float(np.median(cdfs)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)) if rmses else 0.0,
        "rmse_median": float(np.median(rmses)),
        "nll_mean": float(np.mean(nlls)),
        "nll_std": float(np.std(nlls)) if nlls else 0.0,
        "nll_median": float(np.median(nlls)) if nlls else 0.0,
        "nll_q25": float(nll_q25),
        "nll_q75": float(nll_q75),
        "nll_iqr": float(nll_q75 - nll_q25),
        "accuracy": float(np.mean(in_bbox)),
        "anees": float(np.mean(nees)) if nees else 0.0,
        "nees_std": float(np.std(nees)) if nees else 0.0,
        "nees_median": float(np.median(nees)) if nees else 0.0,
        # Component-wise (mode-aware) variants — used by the ambiguity table.
        # ANEES_min is reported as median + IQR: its distribution is heavily
        # right-skewed (a single wrong mode can blow up the mean), so mean is
        # not representative. Median/IQR reflect the typical-query calibration.
        "anees_min": float(np.mean(nees_min)) if nees_min else 0.0,
        "nees_min_std": float(np.std(nees_min)) if nees_min else 0.0,
        "nees_min_median": float(np.median(nees_min)) if nees_min else 0.0,
        "nees_min_q25": float(np.percentile(nees_min, 25)) if nees_min else 0.0,
        "nees_min_q75": float(np.percentile(nees_min, 75)) if nees_min else 0.0,
        "nees_min_iqr": (
            float(np.percentile(nees_min, 75) - np.percentile(nees_min, 25))
            if nees_min else 0.0
        ),
        "anees_w": float(np.mean(nees_w)) if nees_w else 0.0,
        "nees_w_std": float(np.std(nees_w)) if nees_w else 0.0,
        "n": len(queries),
    }

    per_query_items = [
        ("cdf", cdfs), ("rmse", rmses), ("nll", nlls),
        ("nees", nees), ("nees_min", nees_min), ("nees_w", nees_w),
    ]
    return {
        "overall": overall,
        "per_query": {name: list(vals) for name, vals in per_query_items},
        "by_relation": {
            name: _group_metric(vals, relations) for name, vals in per_query_items
        },
        "by_ambiguity": {
            name: _group_metric(vals, ambiguities) for name, vals in per_query_items
        },
    }
