"""Benchmark runner: compare multiple approaches on the same queries.

Usage::

    from evaluation.benchmark import load_eval_queries, run_benchmark
    from evaluation.approach import GMMApproach
    from language_spatial_sensor.pipeline.distribution_predictor import LSSGaussianPredictor
    from language_spatial_sensor.pipeline.proposer import GroundTruthProposer, LLMProposer

    queries = load_eval_queries(
        data_root="/path/to/VLA-3D",
        datasets=["Unity", "3RScan"],
        split="val_seen",
        max_samples=100,       # mini-val
    )

    lss = LSSGaussianPredictor("best.pt")
    approaches = {
        "gt":    GMMApproach(GroundTruthProposer(), lss),
        "ollama": GMMApproach(LLMProposer(model="qwen2.5:32b"), lss),
    }

    results = run_benchmark(approaches, queries)
    print(results["gt"]["overall"])
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm
import torch

if TYPE_CHECKING:
    from data.vla3d.splits import SplitRecord
from language_spatial_sensor.core.schema import SpatialQuery
from language_spatial_sensor.core.transforms import build_spatial_query
from language_spatial_sensor.pipeline.language_sensor import GMMResult
from evaluation.metrics import compute_all_metrics, proposer_precision_recall


# ---------------------------------------------------------------------------
# Load evaluation queries from raw VLA-3D data
# ---------------------------------------------------------------------------

def load_eval_records(
    data_root: str | Path,
    datasets: list[str],
    split: str = "val_seen",
    seed: int = 42,
    val_seen_stmt_frac: float = 0.05,
    val_unseen_scene_frac: float = 0.05,
) -> list[SplitRecord]:
    """Return the split's ``SplitRecord``s (cheap metadata — no scene data loaded).

    Delegates to :func:`data.vla3d.splits.build_splits` so the returned records
    are byte-identical to training. Use this when you want to filter or
    subsample at the record level before paying the cost of loading
    pointclouds — see :func:`records_to_queries`.
    """
    from omegaconf import OmegaConf

    from data.vla3d.splits import build_splits

    cfg = OmegaConf.create({
        "data_root": str(data_root),
        "datasets":  list(datasets),
        "splits": {
            "seed":                  seed,
            "val_seen_stmt_frac":    val_seen_stmt_frac,
            "val_unseen_scene_frac": val_unseen_scene_frac,
        },
    })
    splits = build_splits(cfg)
    print(
        f"[split-sizes] val_seen={len(splits.val_seen):,}  "
        f"val_unseen={len(splits.val_unseen):,}  "
        f"train={len(splits.train):,}"
    )

    if split == "val_seen":
        return splits.val_seen
    if split == "val_unseen":
        return splits.val_unseen
    raise ValueError(f"Unknown split {split!r} (expected 'val_seen' or 'val_unseen')")


def records_to_queries(
    records: list[SplitRecord],
    desc: str = "Loading queries",
    load_pointclouds: bool = True,
) -> list[SpatialQuery]:
    """Materialize ``SpatialQuery`` objects from records.

    Scene data is loaded once per distinct scene. When ``load_pointclouds`` is
    False, ``pc`` and ``object_split`` on the resulting queries are ``None`` —
    use for LLM-only benchmarks where pointclouds would otherwise blow up RAM.
    """
    by_scene: dict[str, list] = defaultdict(list)
    for rec in records:
        by_scene[rec.scene.scene_id].append(rec)

    queries: list[SpatialQuery] = []
    for scene_id, recs in tqdm(by_scene.items(), desc=desc):
        scene = recs[0].scene
        try:
            sg = scene.load_scene_graph()
            if load_pointclouds:
                pcd       = scene.load_pointcloud()
                points    = np.asarray(pcd.points, dtype=np.float32)
                obj_split = scene.load_object_split()
            else:
                points    = None
                obj_split = None
        except Exception as e:
            print(f"[eval] Skipping {scene_id}: {e}")
            continue

        for rec in recs:
            stmt = rec.statement
            try:
                q = build_spatial_query(scene_id, sg, stmt, points, obj_split)
                q.metadata["relation"]         = stmt.relation
                q.metadata["ambiguity"]        = stmt.ambiguity
                q.metadata["target_object_id"] = stmt.target_object_id
                queries.append(q)
            except Exception as e:
                print(f"[eval] Skipping stmt in {scene_id}: {e}")

    return queries


def load_eval_queries(
    data_root: str | Path,
    datasets: list[str],
    split: str = "val_seen",
    max_samples: int | None = None,
    seed: int = 42,
    val_seen_stmt_frac: float = 0.05,
    val_unseen_scene_frac: float = 0.05,
    load_pointclouds: bool = True,
) -> list[SpatialQuery]:
    """Load SpatialQuery objects for evaluation.

    Thin wrapper over :func:`load_eval_records` + :func:`records_to_queries`.
    ``max_samples`` subsamples at the record level, so only the subsampled
    scenes' pointclouds get materialized.

    For benchmarks that need custom record-level filtering (e.g. by ambiguity)
    before subsampling, call :func:`load_eval_records` directly.
    """
    records = load_eval_records(
        data_root=data_root,
        datasets=datasets,
        split=split,
        seed=seed,
        val_seen_stmt_frac=val_seen_stmt_frac,
        val_unseen_scene_frac=val_unseen_scene_frac,
    )
    if max_samples is not None and len(records) > max_samples:
        rng_sub = random.Random(seed + 1)
        records = rng_sub.sample(records, max_samples)

    queries = records_to_queries(
        records, desc=f"Loading {split}", load_pointclouds=load_pointclouds,
    )
    print(f"Loaded {len(queries)} queries for {split}")
    return queries


def load_mini_val(
    data_root: str | Path,
    datasets: list[str],
    n: int = 100,
    seed: int = 42,
) -> list[SpatialQuery]:
    """Load a small deterministic subset of val_seen for fast iteration."""
    return load_eval_queries(
        data_root=data_root,
        datasets=datasets,
        split="val_seen",
        max_samples=n,
        seed=seed,
    )


def _resolve_training_data_cfg(
    config_dir: str | Path,
    data_root_override: str | Path | None,
    datasets_override: list[str] | None,
) -> tuple[str, list[str], Any]:
    """Load the training data config and apply optional overrides."""
    from omegaconf import OmegaConf

    cfg_path = Path(config_dir) / "data" / "vla3d.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found. Pass config_dir= pointing at the Hydra config root."
        )
    data_cfg = OmegaConf.load(cfg_path)

    data_root = data_root_override if data_root_override is not None else data_cfg.data_root
    datasets = datasets_override if datasets_override is not None else list(data_cfg.datasets)
    splits = data_cfg.splits

    print(
        f"[split-load] config={cfg_path} datasets={datasets} "
        f"seed={splits.seed} val_seen_stmt_frac={splits.val_seen_stmt_frac} "
        f"val_unseen_scene_frac={splits.val_unseen_scene_frac}"
    )
    return data_root, datasets, splits


def load_training_split_records(
    config_dir: str | Path = "experiments/cfgs",
    split: str = "val_seen",
    data_root_override: str | Path | None = None,
    datasets_override: list[str] | None = None,
) -> list[SplitRecord]:
    """Return training-aligned ``SplitRecord``s without loading any scene data.

    Same split construction as :func:`load_training_split_queries` (so the
    returned records are byte-identical to training), but skips the expensive
    per-scene pointcloud load. Use this when you want to filter/subsample
    records before materialising ``SpatialQuery`` objects via
    :func:`records_to_queries`.
    """
    data_root, datasets, splits = _resolve_training_data_cfg(
        config_dir, data_root_override, datasets_override,
    )
    return load_eval_records(
        data_root=data_root,
        datasets=datasets,
        split=split,
        seed=splits.seed,
        val_seen_stmt_frac=splits.val_seen_stmt_frac,
        val_unseen_scene_frac=splits.val_unseen_scene_frac,
    )


def load_training_split_queries(
    config_dir: str | Path = "experiments/cfgs",
    split: str = "val_seen",
    max_samples: int | None = None,
    data_root_override: str | Path | None = None,
    datasets_override: list[str] | None = None,
    load_pointclouds: bool = True,
) -> list[SpatialQuery]:
    """Load eval queries using the exact split params the model was trained on.

    Reads ``experiments/cfgs/data/vla3d.yaml`` (or whatever ``config_dir``
    points to) and forwards ``data_root``, ``datasets``, ``splits.seed``,
    ``splits.val_seen_stmt_frac``, ``splits.val_unseen_scene_frac`` to
    :func:`load_eval_queries`.  This guarantees the returned val_seen /
    val_unseen records are byte-identical to the ones the model saw during
    training.

    Args:
        config_dir:        Path to the Hydra config root (contains ``data/vla3d.yaml``).
        split:             ``"val_seen"`` or ``"val_unseen"``.
        max_samples:       Subsample the split deterministically (``rng.Random(seed+1)``).
                           ``None`` keeps the full training-equivalent split.
        data_root_override: Override the config's ``data_root`` (e.g. if the dataset
                           was moved).  Does NOT affect which records are kept.
        datasets_override:  Override the config's dataset list.  WARNING: changing
                           this from training *will* change which records appear
                           in the split — only use if you consciously want a subset.
        load_pointclouds:  If False, skip per-scene pointcloud loading
                           (``pc``/``object_split`` are ``None`` on every query).
                           Set to False for LLM-only benchmarks.
    """
    data_root, datasets, splits = _resolve_training_data_cfg(
        config_dir, data_root_override, datasets_override,
    )
    return load_eval_queries(
        data_root=data_root,
        datasets=datasets,
        split=split,
        max_samples=max_samples,
        seed=splits.seed,
        val_seen_stmt_frac=splits.val_seen_stmt_frac,
        val_unseen_scene_frac=splits.val_unseen_scene_frac,
        load_pointclouds=load_pointclouds,
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _run_one(approach, query: SpatialQuery) -> GMMResult:
    """Dispatch one query to either a :class:`GMMApproach` or the legacy
    :class:`LanguageSpatialSensor` (kwargs-based signature).
    """
    predict = approach.predict
    # GMMApproach-style: single SpatialQuery argument.
    try:
        return predict(query)
    except TypeError:
        pass
    # Legacy kwargs-based sensor.
    return predict(
        scene_graph=query.scene_graph,
        utterance=query.language,
        scene_id=query.scene_id,
        pc=query.pc,
        object_split=query.object_split,
        target_xyz=query.target_xyz,
        target_bbox=query.target_bbox,
        gt_anchor_object_ids=query.gt_anchor_object_ids,
        gt_anchor_room_id=query.gt_anchor_room_id,
    )


def run_benchmark(
    approaches: dict[str, "Any"],
    queries: list[SpatialQuery],
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run all approaches on the same queries and collect metrics.

    Args:
        approaches: Mapping from approach name to object exposing
                    ``.predict(query) -> GMMResult``.  In practice
                    :class:`evaluation.approach.GMMApproach`, though any
                    duck-compatible object works (including the legacy
                    :class:`LanguageSpatialSensor`).
        queries:    List of SpatialQuery objects with GT fields populated.
        verbose:    Print progress and summary.

    Returns:
        Dict mapping approach name -> metrics dict (with 'overall',
        'by_relation', 'by_ambiguity', 'proposer' keys).
    """
    all_results: dict[str, dict[str, Any]] = {}

    for name, approach in approaches.items():
        if verbose:
            print(f"\n=== {name} ===")

        results: list[GMMResult] = []
        groundings_per_query: list[list] = []
        failed = 0

        for q in tqdm(queries, desc=name, disable=not verbose):
            try:
                r = _run_one(approach, q)
                results.append(r)
                groundings_per_query.append(r.groundings)
            except Exception as e:
                if verbose:
                    print(f"  [warn] {q.scene_id}: {e}")
                failed += 1
                # Append dummy result
                results.append(GMMResult(
                    groundings=[],
                    mus=torch.zeros(1, 3),
                    Ls=torch.eye(3).unsqueeze(0),
                    weights=torch.ones(1),
                ))
                groundings_per_query.append([])

        # Filter out failed queries for metrics
        valid = [(r, q, g) for r, q, g in zip(results, queries, groundings_per_query) if g]
        if not valid:
            all_results[name] = {"error": "all queries failed"}
            continue

        valid_results = [v[0] for v in valid]
        valid_queries = [v[1] for v in valid]
        valid_groundings = [v[2] for v in valid]

        metrics = compute_all_metrics(valid_results, valid_queries)
        proposer_metrics = proposer_precision_recall(valid_groundings, valid_queries)
        metrics["proposer"] = proposer_metrics
        metrics["n_failed"] = failed

        if verbose:
            o = metrics["overall"]
            p = proposer_metrics
            print(
                f"  CDF={o['cdf_mean']:.4f}  "
                f"RMSE={o['rmse_mean']:.3f}±{o['rmse_std']:.3f}m  "
                f"NLL={o['nll_mean']:.2f}  Acc={o['accuracy']:.3f}  "
                f"ANEES={o['anees']:.3f} (σ={o['nees_std']:.3f})  "
                f"ANEES_min med={o['nees_min_median']:.3f} "
                f"IQR=[{o['nees_min_q25']:.3f}, {o['nees_min_q75']:.3f}]  "
                f"ANEES_w={o['anees_w']:.3f}"
            )
            print(
                f"  Proposer P={p['proposer_precision']:.3f}  "
                f"R={p['proposer_recall']:.3f}  "
                f"({p['n']} queries, {failed} failed)"
            )

        all_results[name] = metrics

    return all_results


def plot_cdf_histogram(
    results: dict[str, dict[str, Any]],
    metric: str = "cdf",
    bins: int = 20,
    ax=None,
):
    """Plot a density histogram of per-query scores across approaches.

    Args:
        results: Output of :func:`run_benchmark`.
        metric:  One of ``"cdf"``, ``"rmse"``, ``"nll"``.
        bins:    Histogram bin count (or a bin-edge array passed to matplotlib).
        ax:      Optional matplotlib Axes to draw onto.  A new figure is created if omitted.

    Returns:
        The matplotlib Axes containing the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    range_ = (0.0, 1.0) if metric == "cdf" else None

    for name, m in results.items():
        vals = m.get("per_query", {}).get(metric)
        if not vals:
            continue
        ax.hist(
            vals,
            bins=bins,
            range=range_,
            density=True,
            alpha=0.5,
            label=f"{name} (n={len(vals)})",
        )

    ax.set_xlabel({"cdf": "CDF at GT", "rmse": "RMSE (m)", "nll": "NLL"}.get(metric, metric))
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def print_comparison_table(
    results: dict[str, dict[str, Any]],
) -> None:
    """Print a formatted comparison table across approaches.

    ``ANEES`` is the Average Normalised Estimation Error Squared under the
    moment-matched mixture Gaussian — target ≈ 3 for a well-calibrated 3D
    predictor (>3 = overconfident, <3 = underconfident).
    """
    header = f"{'Approach':<15} {'CDF':>8} {'RMSE':>8} {'NLL':>8} {'NLL(IQR)':>9} {'Acc':>8} {'ANEES':>8} {'P(prop)':>8} {'R(prop)':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        if "error" in m:
            print(f"{name:<15} {'ERROR':>8}")
            continue
        o = m["overall"]
        p = m.get("proposer", {})
        print(
            f"{name:<15} "
            f"{o.get('cdf_mean', 0):.4f}  "
            f"{o.get('rmse_mean', 0):.3f}m "
            f"{o.get('nll_mean', 0):>7.2f}  "
            f"{o.get('nll_iqr', 0):>8.2f}  "
            f"{o.get('accuracy', 0):.3f}   "
            f"{o.get('anees', 0):>7.3f}  "
            f"{p.get('proposer_precision', 0):.3f}   "
            f"{p.get('proposer_recall', 0):.3f}"
        )
    print("ANEES: target ≈ 3 for calibrated 3D predictions (>3 overconfident, <3 underconfident).")


def print_no_ambiguity_table(results: dict[str, dict[str, Any]]) -> None:
    """Compact table for the no-ambiguity benchmark: RMSE, NLL, ANEES ± std.

    ANEES target ≈ 3 for calibrated 3D predictions.  Adding a conformal tail
    inflates the moment-matched Σ, so conformal variants should show lower
    ANEES than their non-conformal counterparts.
    """
    header = (
        f"{'Approach':<20} "
        f"{'RMSE(m) mean±std / med':>28} "
        f"{'NLL mean±std / med':>28} "
        f"{'ANEES mean±std / med':>28} "
        f"{'n':>6}"
    )
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        if "error" in m:
            print(f"{name:<20} {'ERROR':>28}")
            continue
        o = m["overall"]
        rmse = (
            f"{o.get('rmse_mean', 0):>7.3f} ± {o.get('rmse_std', 0):<7.3f} "
            f"/ {o.get('rmse_median', 0):>7.3f}"
        )
        nll = (
            f"{o.get('nll_mean', 0):>7.3f} ± {o.get('nll_std', 0):<7.3f} "
            f"/ {o.get('nll_median', 0):>7.3f}"
        )
        anees = (
            f"{o.get('anees', 0):>7.3f} ± {o.get('nees_std', 0):<7.3f} "
            f"/ {o.get('nees_median', 0):>7.3f}"
        )
        print(
            f"{name:<20} "
            f"{rmse:>28} "
            f"{nll:>28} "
            f"{anees:>28} "
            f"{o.get('n', 0):>6d}"
        )
    print(
        "ANEES: target ≈ 3 for calibrated 3D predictions "
        "(>3 overconfident, <3 underconfident). "
        "Conformal variants moment-match the Gaussian + uniform-ellipsoid tail."
    )


def print_by_relation_table(results: dict[str, dict[str, Any]]) -> None:
    """Per-relation breakdown: one block per relation, one row per approach.

    Pulls from ``m["by_relation"]`` (produced by :func:`compute_all_metrics`).
    ANEES is the per-relation mean/std of the per-query NEES values, so each
    block's ANEES is a genuine per-relation calibration diagnostic (not the
    approach-wide ANEES).
    """
    # Collect the union of relations actually present across all approaches.
    relations: set = set()
    for m in results.values():
        if "error" in m:
            continue
        relations |= set(m.get("by_relation", {}).get("rmse", {}).keys())
    if not relations:
        return

    header = (
        f"{'Approach':<20} "
        f"{'RMSE(m) mean±std / med':>28} "
        f"{'NLL mean±std / med':>28} "
        f"{'ANEES mean±std / med':>28} "
        f"{'n':>6}"
    )
    for rel in sorted(relations, key=str):
        print(f"\n-- relation: {rel} --")
        print(header)
        print("-" * len(header))
        for name, m in results.items():
            if "error" in m:
                print(f"{name:<20} {'ERROR':>28}")
                continue
            br = m.get("by_relation", {})
            r_stats = br.get("rmse", {}).get(rel, {})
            n_stats = br.get("nll", {}).get(rel, {})
            e_stats = br.get("nees", {}).get(rel, {})
            count = r_stats.get("count", n_stats.get("count", e_stats.get("count", 0)))
            if count == 0:
                continue
            rmse = (
                f"{r_stats.get('mean', 0):>7.3f} ± {r_stats.get('std', 0):<7.3f} "
                f"/ {r_stats.get('median', 0):>7.3f}"
            )
            nll = (
                f"{n_stats.get('mean', 0):>7.3f} ± {n_stats.get('std', 0):<7.3f} "
                f"/ {n_stats.get('median', 0):>7.3f}"
            )
            anees = (
                f"{e_stats.get('mean', 0):>7.3f} ± {e_stats.get('std', 0):<7.3f} "
                f"/ {e_stats.get('median', 0):>7.3f}"
            )
            print(
                f"{name:<20} "
                f"{rmse:>28} "
                f"{nll:>28} "
                f"{anees:>28} "
                f"{count:>6d}"
            )


def ambiguity_sort_key(lvl: Any) -> tuple:
    """Sort key for ambiguity labels.

    Integer levels sort numerically.  Any ``"N+"`` bucket label (e.g.
    ``"5+"``) is parsed into the leading integer and sorts *after* the
    corresponding integer bucket.  Anything else falls back to lexicographic
    at the end.
    """
    s = str(lvl)
    if s.endswith("+"):
        try:
            return (1, int(s[:-1]))
        except ValueError:
            pass
    try:
        return (0, int(s))
    except ValueError:
        return (2, s)  # type: ignore[return-value]


def print_by_ambiguity_table(results: dict[str, dict[str, Any]]) -> None:
    """Per-ambiguity-level table: RMSE, NLL, ANEES_min, ANEES_w per (approach, ambiguity).

    Reads ``m["by_ambiguity"]`` produced by :func:`compute_all_metrics` — one
    row per ``(approach, ambiguity_level)`` pair.  The moment-matched ANEES
    is replaced by two mode-aware variants that preserve multimodality:

        * ``ANEES_min`` = mean over queries of ``min_k (x* − μ_k)ᵀ Σ_k⁻¹ (x* − μ_k)``
          — best-matching-mode calibration.
        * ``ANEES_w``   = mean over queries of ``Σ_k w_k (x* − μ_k)ᵀ Σ_k⁻¹ (x* − μ_k)``
          — weight-averaged across components.

    Neither has a clean χ² reference (unlike the moment-matched variant) —
    use ``ANEES_min`` as "did *any* mode explain the GT?" and ``ANEES_w`` as
    "on average across components".  Levels may be integers or bucket
    labels (e.g. ``"5+"``); sorting uses :func:`ambiguity_sort_key`.
    """
    header = (
        f"{'Approach':<20} {'Ambig':>6} {'n':>6} "
        f"{'RMSE(m)':>10} {'NLL':>10} "
        f"{'ANEES med':>10} {'ANEES IQR':>18} "
        f"{'ANEES_min med':>14} {'ANEES_min IQR':>20} "
        f"{'ANEES_w':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        if "error" in m:
            print(f"{name:<20} {'ERROR':>6}")
            continue
        by_ambig = m.get("by_ambiguity", {})
        rmse_by = by_ambig.get("rmse", {})
        nll_by = by_ambig.get("nll", {})
        nees_by = by_ambig.get("nees", {})
        nees_min_by = by_ambig.get("nees_min", {})
        nees_w_by = by_ambig.get("nees_w", {})
        levels = sorted(
            set(rmse_by.keys()) | set(nll_by.keys())
            | set(nees_by.keys())
            | set(nees_min_by.keys()) | set(nees_w_by.keys()),
            key=ambiguity_sort_key,
        )
        for lvl in levels:
            r = rmse_by.get(lvl, {})
            n = nll_by.get(lvl, {})
            e = nees_by.get(lvl, {})
            emin = nees_min_by.get(lvl, {})
            ew = nees_w_by.get(lvl, {})
            count = r.get(
                "count", n.get("count", emin.get("count", ew.get("count", 0))),
            )
            anees_iqr_range = f"[{e.get('q25', 0):.2f}, {e.get('q75', 0):.2f}]"
            iqr_range = f"[{emin.get('q25', 0):.2f}, {emin.get('q75', 0):.2f}]"
            print(
                f"{name:<20} "
                f"{str(lvl):>6} {count:>6} "
                f"{r.get('mean', 0):>10.3f} "
                f"{n.get('mean', 0):>10.3f} "
                f"{e.get('median', 0):>10.3f} "
                f"{anees_iqr_range:>18} "
                f"{emin.get('median', 0):>14.3f} "
                f"{iqr_range:>20} "
                f"{ew.get('mean', 0):>10.3f}"
            )
    print(
        "ANEES_min: min over components (reported as median + IQR because the "
        "per-query distribution is right-skewed — a single bad mode can dominate "
        "the mean). ANEES_w: weight-averaged across components (per-query mean)."
    )
