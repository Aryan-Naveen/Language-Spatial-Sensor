#!/usr/bin/env python3
"""Render a sample BEV PNG using the same settings the VLM predictor uses.

Saves to ``sample_bev.png`` at the repo root so you can eyeball what gets
shipped to GPT-5.2's vision API. Helpful for verifying the resolution
tradeoff vs. token cost.

Usage::

    python scripts/sample_bev.py                       # defaults: first query from val_seen
    python scripts/sample_bev.py --resolution 0.05     # higher-res for comparison
    python scripts/sample_bev.py --index 3             # pick the 4th query
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmark import load_training_split_records, records_to_queries  # noqa: E402
from viz.bev import render_bev  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "cfgs",
    )
    p.add_argument("--split", default="val_seen", choices=["val_seen", "val_unseen"])
    p.add_argument(
        "--resolution",
        type=float,
        default=0.10,
        help="BEV grid cell size in metres. The VLM predictor now defaults to 0.10.",
    )
    p.add_argument(
        "--index",
        type=int,
        default=0,
        help="Which no-ambiguity record to render (after deterministic shuffle).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed — matches the benchmark driver's ordering.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "sample_bev.png",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override the training config's data_root.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Override the training config's dataset list.",
    )
    return p.parse_args()


def main() -> None:
    import random

    args = _parse_args()

    records = load_training_split_records(
        config_dir=args.config_dir,
        split=args.split,
        data_root_override=args.data_root,
        datasets_override=args.datasets,
    )
    records = [r for r in records if r.statement.ambiguity == 1]
    rng = random.Random(args.seed)
    rng.shuffle(records)
    if not records:
        raise SystemExit("No ambiguity==1 records found.")
    if args.index >= len(records):
        raise SystemExit(
            f"--index {args.index} out of range (only {len(records)} records)."
        )

    queries = records_to_queries(
        [records[args.index]],
        desc=f"render[{args.index}]",
        load_pointclouds=True,
    )
    query = queries[0]
    anchor_ids = set(query.gt_anchor_object_ids or [])

    print(f"scene_id:       {query.scene_id}")
    print(f"utterance:      {query.language!r}")
    print(f"anchor_ids:     {sorted(anchor_ids)}")
    print(f"anchor_room_id: {query.gt_anchor_room_id}")
    print(f"resolution:     {args.resolution} m")

    fig = render_bev(
        query=query,
        resolution=args.resolution,
        anchor_highlight=True,
        highlight_object_ids=anchor_ids,
        show_target=False,  # match what VLMGaussianPredictor sends to GPT
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)

    size_kb = args.output.stat().st_size / 1024
    print(f"\nSaved -> {args.output}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
