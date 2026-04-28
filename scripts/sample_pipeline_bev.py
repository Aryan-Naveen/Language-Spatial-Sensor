#!/usr/bin/env python3
"""Render a BEV heatmap from the full Language Spatial Sensor pipeline.

Picks one record from a chosen dataset+split (by ``--index``, ``--random``,
or ``--scene-id``), runs the full pipeline
(:class:`LLMProposer` -> :class:`LSSGaussianPredictor` -> :class:`GMMResult`),
and overlays the GMM density on a semantic BEV using
:func:`viz.bev.render_bev_with_gmm_overlay`.

Usage::

    # Fixed record from 3RScan
    python scripts/sample_pipeline_bev.py --dataset 3RScan --index 0

    # Random 3RScan record, with a specific checkpoint
    python scripts/sample_pipeline_bev.py --dataset Scannet --random \\
        --checkpoint checkpoints/best.pt

    # Target a specific scene
    python scripts/sample_pipeline_bev.py --dataset Scannet \\
        --scene-id scene0010_00
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.approach import GMMApproach  # noqa: E402
from evaluation.benchmark import (  # noqa: E402
    load_training_split_records,
    records_to_queries,
)
from language_spatial_sensor.core.ontology import VALID_REGION_LABELS  # noqa: E402
from language_spatial_sensor.pipeline.distribution_predictor import (  # noqa: E402
    LSSGaussianPredictor,
)
from language_spatial_sensor.pipeline.proposer import LLMProposer  # noqa: E402
from viz.bev import render_bev_with_gmm_overlay  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        required=True,
        help="Single dataset name to filter records "
        "(e.g. 3RScan, ARKitScenes, HM3D, Matterport, Scannet, Unity).",
    )
    p.add_argument(
        "--split",
        default="val_seen",
        choices=["val_seen", "val_unseen"],
    )
    p.add_argument(
        "--index",
        type=int,
        default=0,
        help="Pick the Nth record after deterministic shuffle. Ignored if --random or --scene-id.",
    )
    p.add_argument(
        "--random",
        action="store_true",
        help="Pick a uniformly random record (non-deterministic).",
    )
    p.add_argument(
        "--scene-id",
        default=None,
        help="Restrict to records for this exact scene_id (then --index selects within).",
    )
    p.add_argument(
        "--ambiguity",
        type=int,
        default=None,
        help="Optional filter: keep only statements with this ambiguity level (e.g. 1).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed — matches the benchmark driver's ordering.",
    )

    # Pipeline config
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "best.pt",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--proposer-provider", default="openai")
    p.add_argument("--proposer-model", default="gpt-5.2")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "cache",
    )

    # Rendering
    p.add_argument(
        "--resolution",
        type=float,
        default=0.25,
        help="Voxel size in metres for the density overlay.",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Samples drawn from the GMM for the density estimate.",
    )
    p.add_argument(
        "--heatmap-alpha",
        type=float,
        default=0.35,
    )
    p.add_argument(
        "--background-alpha",
        type=float,
        default=0.55,
        help="Opacity of the semantic backdrop. <1 mutes objects so the density pops.",
    )
    p.add_argument(
        "--hide-target",
        action="store_true",
        help="Hide the GT location bullseye marker (shown by default).",
    )
    p.add_argument(
        "--no-legend",
        action="store_true",
        help="Suppress the density colorbar + anchor/GT/mode legend.",
    )
    p.add_argument(
        "--full-color-backdrop",
        action="store_true",
        help="Paint every object by its NYU40 palette colour (classic mode) "
        "instead of the default grayscale-except-anchors backdrop.",
    )

    # Config / IO
    p.add_argument(
        "--config-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "cfgs",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override the training config's data_root.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "sample_pipeline_bev.png",
    )
    return p.parse_args()


def _select_record(records, args: argparse.Namespace):
    if args.scene_id is not None:
        records = [r for r in records if r.scene.scene_id == args.scene_id]
        if not records:
            raise SystemExit(f"No records found for scene_id={args.scene_id!r}.")

    if args.ambiguity is not None:
        records = [r for r in records if r.statement.ambiguity == args.ambiguity]
        if not records:
            raise SystemExit(f"No records with ambiguity={args.ambiguity}.")

    if args.random:
        rng = random.Random()  # system entropy
        return rng.choice(records)

    rng = random.Random(args.seed)
    rng.shuffle(records)
    if args.index >= len(records):
        raise SystemExit(
            f"--index {args.index} out of range (only {len(records)} records)."
        )
    return records[args.index]


def main() -> None:
    args = _parse_args()

    # 1) Load records for the chosen dataset + split (matches training's split layout).
    records = load_training_split_records(
        config_dir=args.config_dir,
        split=args.split,
        data_root_override=args.data_root,
        datasets_override=[args.dataset],
    )
    if not records:
        raise SystemExit(f"No records found for dataset={args.dataset!r}, split={args.split!r}.")

    record = _select_record(records, args)
    print(
        f"Selected: dataset={record.scene.dataset}  scene_id={record.scene.scene_id}  "
        f"target_object_id={record.statement.target_object_id}  "
        f"ambiguity={record.statement.ambiguity}"
    )

    # 2) Materialize the SpatialQuery (pointcloud + object_split loaded).
    queries = records_to_queries([record], desc="render", load_pointclouds=True)
    if not queries:
        raise SystemExit("Failed to materialize SpatialQuery from record.")
    query = queries[0]
    print(f"utterance: {query.language!r}")

    # Report rooms whose label is outside VALID_REGION_LABELS — these are dropped
    # from the BEV bbox overlay and from the GT scene ontology.
    invalid_rooms = [
        (r.id, r.label)
        for r in query.scene_graph.regions
        if r.label.lower().strip() not in VALID_REGION_LABELS
    ]
    if invalid_rooms:
        print(f"Rooms excluded (label not in valid ontology): {len(invalid_rooms)}")
        for rid, label in invalid_rooms:
            print(f"  region {rid}: {label!r}")
    else:
        print("All rooms have labels in the valid ontology.")

    # 3) Build the full pipeline and predict.
    proposer = LLMProposer(
        provider=args.proposer_provider,
        model=args.proposer_model,
        cache_dir=str(args.cache_dir / "proposer"),
    )
    predictor = LSSGaussianPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    approach = GMMApproach(proposer=proposer, predictor=predictor)
    gmm = approach.predict(query)

    print(
        f"GMM: K={len(gmm.weights)} components  "
        f"weights={[f'{float(w):.2f}' for w in gmm.weights]}"
    )

    # 4) Render BEV with the GMM heatmap overlay.
    fig = render_bev_with_gmm_overlay(
        query=query,
        gmm_result=gmm,
        n_samples=args.n_samples,
        resolution=args.resolution,
        heatmap_alpha=args.heatmap_alpha,
        show_target=not args.hide_target,
        seed=args.seed,
        include_legend=not args.no_legend,
        background_alpha=args.background_alpha,
        full_color_backdrop=args.full_color_backdrop,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"\nSaved -> {args.output}  ({args.output.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
