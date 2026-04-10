"""Benchmark runner: compare multiple proposer approaches on the same queries.

Usage::

    from evaluation.benchmark import load_eval_queries, run_benchmark
    from language_spatial_sensor.pipeline.language_sensor import LanguageSpatialSensor
    from language_spatial_sensor.pipeline.proposer import GroundTruthProposer, LLMProposer

    queries = load_eval_queries(
        data_root="/path/to/VLA-3D",
        datasets=["Unity", "3RScan"],
        split="val_seen",
        max_samples=100,       # mini-val
    )

    approaches = {
        "gt":    LanguageSpatialSensor("best.pt", GroundTruthProposer()),
        "ollama": LanguageSpatialSensor("best.pt", LLMProposer(model="qwen2.5:32b")),
    }

    results = run_benchmark(approaches, queries)
    print(results["gt"]["overall"])
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from language_spatial_sensor.core.schema import SpatialQuery
from language_spatial_sensor.core.transforms import build_spatial_query
from language_spatial_sensor.pipeline.language_sensor import GMMResult, LanguageSpatialSensor
from evaluation.metrics import compute_all_metrics, proposer_precision_recall


# ---------------------------------------------------------------------------
# Load evaluation queries from raw VLA-3D data
# ---------------------------------------------------------------------------

_CANONICAL = {
    "3rscan": "3RScan", "arkitscenes": "ARKitScenes", "hm3d": "HM3D",
    "matterport": "Matterport", "scannet": "Scannet", "unity": "Unity",
}


def load_eval_queries(
    data_root: str | Path,
    datasets: list[str],
    split: str = "val_seen",
    max_samples: int | None = None,
    seed: int = 42,
    val_seen_stmt_frac: float = 0.05,
    val_unseen_scene_frac: float = 0.05,
) -> list[SpatialQuery]:
    """Load SpatialQuery objects for evaluation.

    Reproduces the same splits as ``data/vla3d/splits.py`` to select val_seen
    queries, then builds full SpatialQuery objects (including point clouds)
    grouped by scene for efficient loading.

    Args:
        data_root:    Path to VLA-3D dataset root.
        datasets:     List of dataset names (e.g. ``["Unity", "3RScan"]``).
        split:        ``"val_seen"`` or ``"val_unseen"``.
        max_samples:  If set, subsample to this many queries (mini-val).
        seed:         Random seed for reproducibility.
    """
    from data.vla3d.dataset import VLA3D, VLA3DScene
    from language_spatial_sensor.core.ontology import VALID_NYU40_LABELS, VALID_RELATIONS

    data_root = Path(data_root)
    rng = random.Random(seed)

    # Collect all scenes
    all_scenes: list[VLA3DScene] = []
    for name in datasets:
        canonical = _CANONICAL.get(name.lower(), name)
        ds_path = data_root / canonical
        if ds_path.exists():
            all_scenes.extend(VLA3D(ds_path).scenes())

    if not all_scenes:
        raise FileNotFoundError(f"No scenes found under {data_root}")

    # Reproduce scene-level split
    shuffled = all_scenes[:]
    rng.shuffle(shuffled)
    n_unseen = max(1, round(len(shuffled) * val_unseen_scene_frac))
    unseen_scenes = set(s.scene_id for s in shuffled[-n_unseen:])
    seen_scenes = [s for s in shuffled if s.scene_id not in unseen_scenes]
    unseen_scene_list = [s for s in shuffled if s.scene_id in unseen_scenes]

    # Filter statements helper
    def _filter(sg, stmts):
        obj_label = {
            obj.id: str(obj.metadata.get("nyu40_label", "")).lower().strip()
            for obj in sg.objects
        }
        return [
            s for s in stmts
            if s.relation in VALID_RELATIONS
            and obj_label.get(s.target_object_id, "") in VALID_NYU40_LABELS
            and all(obj_label.get(aid, "") in VALID_NYU40_LABELS for aid in (s.anchor_object_id or []))
        ]

    # Collect split records: (scene, statement)
    records: list[tuple[VLA3DScene, Any]] = []

    target_scenes = seen_scenes if split == "val_seen" else unseen_scene_list

    for scene in target_scenes:
        try:
            sg = scene.load_scene_graph()
            stmts = _filter(sg, scene.load_statements(sg))
        except Exception:
            continue

        if split == "val_seen":
            rng_stmt = random.Random(seed)
            rng_stmt.shuffle(stmts)
            n_val = max(1, round(len(stmts) * val_seen_stmt_frac)) if stmts else 0
            for s in stmts[:n_val]:
                records.append((scene, s))
        else:
            for s in stmts:
                records.append((scene, s))

    if max_samples is not None and len(records) > max_samples:
        rng_sub = random.Random(seed + 1)
        records = rng_sub.sample(records, max_samples)

    # Build SpatialQuery objects, grouped by scene for efficient PC loading
    by_scene: dict[str, list[tuple[VLA3DScene, Any]]] = defaultdict(list)
    for scene, stmt in records:
        by_scene[scene.scene_id].append((scene, stmt))

    queries: list[SpatialQuery] = []
    for scene_id, scene_stmts in tqdm(by_scene.items(), desc=f"Loading {split}"):
        scene = scene_stmts[0][0]
        try:
            sg = scene.load_scene_graph()
            pcd = scene.load_pointcloud()
            points = np.asarray(pcd.points, dtype=np.float32)
            obj_split = scene.load_object_split()
        except Exception as e:
            print(f"[eval] Skipping {scene_id}: {e}")
            continue

        for _, stmt in scene_stmts:
            try:
                q = build_spatial_query(scene_id, sg, stmt, points, obj_split)
                q.metadata["relation"] = stmt.relation
                q.metadata["ambiguity"] = stmt.ambiguity
                q.metadata["target_object_id"] = stmt.target_object_id
                queries.append(q)
            except Exception as e:
                print(f"[eval] Skipping stmt in {scene_id}: {e}")

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


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    approaches: dict[str, LanguageSpatialSensor],
    queries: list[SpatialQuery],
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run all approaches on the same queries and collect metrics.

    Args:
        approaches: Mapping from approach name to configured sensor.
        queries:    List of SpatialQuery objects with GT fields populated.
        verbose:    Print progress and summary.

    Returns:
        Dict mapping approach name -> metrics dict (with 'overall',
        'by_relation', 'by_ambiguity', 'proposer' keys).
    """
    all_results: dict[str, dict[str, Any]] = {}

    for name, sensor in approaches.items():
        if verbose:
            print(f"\n=== {name} ===")

        results: list[GMMResult] = []
        groundings_per_query: list[list] = []
        failed = 0

        for q in tqdm(queries, desc=name, disable=not verbose):
            try:
                r = sensor.predict(
                    scene_graph=q.scene_graph,
                    utterance=q.language,
                    scene_id=q.scene_id,
                    pc=q.pc,
                    object_split=q.object_split,
                    target_xyz=q.target_xyz,
                    target_bbox=q.target_bbox,
                )
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
                f"  CDF={o['cdf_mean']:.4f}  RMSE={o['rmse_mean']:.3f}m  "
                f"NLL={o['nll_mean']:.2f}  Acc={o['accuracy']:.3f}  ECE={o['ece']:.4f}"
            )
            print(
                f"  Proposer P={p['proposer_precision']:.3f}  "
                f"R={p['proposer_recall']:.3f}  "
                f"({p['n']} queries, {failed} failed)"
            )

        all_results[name] = metrics

    return all_results


def print_comparison_table(
    results: dict[str, dict[str, Any]],
) -> None:
    """Print a formatted comparison table across approaches."""
    header = f"{'Approach':<15} {'CDF':>8} {'RMSE':>8} {'NLL':>8} {'Acc':>8} {'ECE':>8} {'P(prop)':>8} {'R(prop)':>8}"
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
            f"{o.get('accuracy', 0):.3f}   "
            f"{o.get('ece', 0):.4f}  "
            f"{p.get('proposer_precision', 0):.3f}   "
            f"{p.get('proposer_recall', 0):.3f}"
        )
