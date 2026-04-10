"""Metrics for evaluating the Language Spatial Sensor pipeline.

Provides both proposer-level metrics (anchor precision / recall) and
end-to-end distribution metrics (CDF, RMSE, NLL, ECE).  All functions
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
    precisions, recalls = [], []
    ambiguities: list[int] = []
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

        ambiguity = q.metadata.get("ambiguity", 0) if q.metadata else 0
        ambiguities.append(int(ambiguity))

    out: dict[str, Any] = {
        "proposer_precision": float(np.mean(precisions)),
        "proposer_recall": float(np.mean(recalls)),
        "proposer_num_proposed_mean": float(np.mean(num_proposed)),
        "n": len(queries),
    }

    # Per-ambiguity breakdown
    by_level: dict[int, list[int]] = defaultdict(list)
    for i, level in enumerate(ambiguities):
        by_level[level].append(i)
    for level in sorted(by_level):
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


def gmm_ece(
    cdfs: list[float],
    in_bbox: list[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error.

    Bins samples by their predicted CDF value and compares the average
    predicted mass to the actual fraction of samples where the GT falls
    inside the predicted high-density region.
    """
    cdfs_arr = np.array(cdfs)
    acc_arr = np.array(in_bbox, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (cdfs_arr >= lo) & (cdfs_arr < hi)
        if not mask.any():
            continue
        avg_conf = cdfs_arr[mask].mean()
        avg_acc = acc_arr[mask].mean()
        ece += mask.sum() / len(cdfs_arr) * abs(avg_acc - avg_conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Aggregate by group
# ---------------------------------------------------------------------------

def _group_metric(
    values: list[float],
    groups: list,
) -> dict[Any, dict[str, float]]:
    """Group values and compute mean/std/count per group."""
    by_group: dict[Any, list[float]] = defaultdict(list)
    for v, g in zip(values, groups):
        by_group[g].append(v)
    return {
        g: {
            "mean": float(np.mean(vs)),
            "std": float(np.std(vs)),
            "count": len(vs),
        }
        for g, vs in sorted(by_group.items(), key=lambda x: str(x[0]))
    }


# ---------------------------------------------------------------------------
# All-in-one
# ---------------------------------------------------------------------------

def compute_all_metrics(
    results: list[GMMResult],
    queries: list[SpatialQuery],
) -> dict[str, Any]:
    """Compute all end-to-end metrics and group by relation / ambiguity.

    Returns a nested dict with keys:
        - ``overall``: aggregate metrics
        - ``by_relation``: per-relation breakdowns
        - ``by_ambiguity``: per-ambiguity breakdowns
    """
    cdfs = gmm_cdf_at_gt(results, queries)
    rmses = gmm_rmse(results, queries)
    nlls = gmm_nll_at_gt(results, queries)

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

    ece = gmm_ece(cdfs, in_bbox)

    relations = [q.metadata.get("relation", "unknown") for q in queries]
    ambiguities = [q.metadata.get("ambiguity", 0) for q in queries]

    overall = {
        "cdf_mean": float(np.mean(cdfs)),
        "cdf_median": float(np.median(cdfs)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_median": float(np.median(rmses)),
        "nll_mean": float(np.mean(nlls)),
        "accuracy": float(np.mean(in_bbox)),
        "ece": ece,
        "n": len(queries),
    }

    return {
        "overall": overall,
        "by_relation": {
            metric: _group_metric(vals, relations)
            for metric, vals in [("cdf", cdfs), ("rmse", rmses), ("nll", nlls)]
        },
        "by_ambiguity": {
            metric: _group_metric(vals, ambiguities)
            for metric, vals in [("cdf", cdfs), ("rmse", rmses), ("nll", nlls)]
        },
    }
