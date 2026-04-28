"""Table 2 driver: ambiguous-setting benchmark (no GT grounding).

Compares four approaches, broken down by the ``ambiguity`` metadata field the
VLA-3D dataset attaches to each statement.  Ambiguity levels at or above
:data:`AMBIGUITY_TOP_THRESHOLD` are collapsed into a single
:data:`AMBIGUITY_TOP_LABEL` bucket so the long tail of highly-ambiguous
statements doesn't fragment the table.

Approaches:
    * ``framework``      — LLMProposer + LSSGaussianPredictor (current LSS pipeline).
    * ``straight_llm``   — LLMFullGMMPredictor (no proposer; LLM emits a GMM in one shot).
    * ``scaffolded_llm`` — LLMProposer + LLMGaussianPredictor.
    * ``scaffolded_vlm`` — LLMProposer + VLMGaussianPredictor.

Sampling:
    Pass ``--min-per-ambiguity N`` to guarantee at least N records per
    ambiguity bucket (``AMBIGUITY_BUCKETS``).  If a bucket has fewer than
    N, all of its records are included.  ``--max-samples`` then caps the
    grand total.

Run::

    python -m evaluation.run_benchmark_ambig \\
        --checkpoint checkpoints/best.pt \\
        --min-per-ambiguity 30
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from evaluation.approach import GMMApproach
from evaluation.benchmark import (
    load_training_split_records,
    print_by_ambiguity_table,
    records_to_queries,
    run_benchmark,
)
from language_spatial_sensor.pipeline.distribution_predictor import (
    LLMFullGMMPredictor,
    LLMGaussianPredictor,
    LSSGaussianPredictor,
    VLMGaussianPredictor,
)
from language_spatial_sensor.pipeline.proposer import LLMProposer, GroundTruthProposer


# ---------------------------------------------------------------------------
# Bucketing helpers
# ---------------------------------------------------------------------------

# Ambiguity levels at or above this threshold collapse into the top bucket.
AMBIGUITY_TOP_THRESHOLD = 5
AMBIGUITY_TOP_LABEL = f"{AMBIGUITY_TOP_THRESHOLD}+"
AMBIGUITY_BUCKETS = (
    [str(i) for i in range(1, AMBIGUITY_TOP_THRESHOLD)] + [AMBIGUITY_TOP_LABEL]
)


def bucket_ambiguity(level: int) -> str:
    """Map a raw ambiguity integer to one of ``AMBIGUITY_BUCKETS``."""
    level = int(level)
    return str(level) if level < AMBIGUITY_TOP_THRESHOLD else AMBIGUITY_TOP_LABEL


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--config-dir",
        type=Path,
        default=Path("experiments/cfgs"),
        help="Hydra config root — splits are loaded from <config-dir>/data/vla3d.yaml "
        "so the eval set is byte-identical to training.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override the data_root in the training config. Does not affect which records are kept.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Override the dataset list from the training config. WARNING: "
        "changing this from training will change which records appear in the split.",
    )
    p.add_argument("--split", default="val_seen", choices=["val_seen", "val_unseen"])
    p.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Cap on the total number of test queries.",
    )
    p.add_argument(
        "--min-per-ambiguity",
        type=int,
        default=None,
        help="If set, take at least this many records from each ambiguity bucket "
        f"(buckets: {AMBIGUITY_BUCKETS}).  Buckets with fewer records contribute everything they have.  "
        "Total is then capped at --max-samples.",
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--proposer-provider", default="openai")
    p.add_argument("--proposer-model", default="gpt-5.2")
    p.add_argument("--llm-provider", default="openai")
    p.add_argument("--llm-model", default="gpt-5.2")
    p.add_argument("--vlm-provider", default="ollama")
    p.add_argument("--vlm-model", default="qwen2.5vl:32b")
    p.add_argument("--cache-dir", type=Path, default=Path("cache"))

    p.add_argument(
        "--coverage",
        type=float,
        default=0.9,
        help="Conformal target coverage. Must match the value used when "
        "calibrating via run_benchmark_no_ambig (so the tail weight 1-coverage "
        "is consistent with the cached Mahalanobis quantile q).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-csv", type=Path, default=Path("results/table2_by_ambig.csv"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Record selection
# ---------------------------------------------------------------------------

def _select_records(
    records,
    min_per_ambiguity: int | None,
    max_samples: int,
    rng: random.Random,
) -> list:
    """Bucket records by ambiguity, optionally guarantee ≥N per bucket."""
    by_bucket: dict[str, list] = defaultdict(list)
    for r in records:
        by_bucket[bucket_ambiguity(r.statement.ambiguity)].append(r)

    # Shuffle inside each bucket deterministically.
    for bucket in by_bucket.values():
        rng.shuffle(bucket)

    if min_per_ambiguity is None:
        all_records = [r for bucket in by_bucket.values() for r in bucket]
        rng.shuffle(all_records)
        return all_records[:max_samples]

    picked: list = []
    underfull: list[tuple[str, int]] = []
    print(f"Selection plan (target {min_per_ambiguity} per bucket):")
    for label in AMBIGUITY_BUCKETS:
        bucket = by_bucket.get(label, [])
        take = bucket[:min_per_ambiguity]
        picked.extend(take)
        marker = "  UNDERFULL" if len(take) < min_per_ambiguity else ""
        print(f"  ambiguity={label:>4}: took {len(take):>4}/{len(bucket):<4}{marker}")
        if len(take) < min_per_ambiguity:
            underfull.append((label, len(take)))
    if underfull:
        print(
            f"\n⚠️  {len(underfull)} bucket(s) have fewer than {min_per_ambiguity} records: "
            + ", ".join(f"{lbl}={n}" for lbl, n in underfull)
        )
    rng.shuffle(picked)
    return picked[:max_samples]


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def _write_csv(results: dict, path: Path) -> None:
    import csv

    from evaluation.benchmark import ambiguity_sort_key

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "approach", "ambiguity", "n",
            "rmse_mean", "rmse_std",
            "nll_mean", "nll_std",
            "anees_median", "anees_q25", "anees_q75", "anees_iqr",
            "anees_min_median", "anees_min_q25", "anees_min_q75", "anees_min_iqr",
            "anees_w_mean", "anees_w_std",
        ])
        for name, m in results.items():
            if "error" in m:
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
                w.writerow([
                    name,
                    lvl,
                    count,
                    f"{r.get('mean', 0):.6f}",
                    f"{r.get('std', 0):.6f}",
                    f"{n.get('mean', 0):.6f}",
                    f"{n.get('std', 0):.6f}",
                    f"{e.get('median', 0):.6f}",
                    f"{e.get('q25', 0):.6f}",
                    f"{e.get('q75', 0):.6f}",
                    f"{e.get('iqr', 0):.6f}",
                    f"{emin.get('median', 0):.6f}",
                    f"{emin.get('q25', 0):.6f}",
                    f"{emin.get('q75', 0):.6f}",
                    f"{emin.get('iqr', 0):.6f}",
                    f"{ew.get('mean', 0):.6f}",
                    f"{ew.get('std', 0):.6f}",
                ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # 1) Load records (metadata only — no pointclouds yet) and stratify.
    all_records = load_training_split_records(
        config_dir=args.config_dir,
        split=args.split,
        data_root_override=args.data_root,
        datasets_override=args.datasets,
    )
    print(f"Loaded {len(all_records)} records from {args.split}")

    # Summary of ambiguity distribution across the full split.
    bucket_counts: dict[str, int] = defaultdict(int)
    for r in all_records:
        bucket_counts[bucket_ambiguity(r.statement.ambiguity)] += 1
    print("Ambiguity distribution (full split):")
    for label in AMBIGUITY_BUCKETS:
        print(f"  {label:>4}: {bucket_counts.get(label, 0)}")

    rng = random.Random(args.seed)
    selected = _select_records(
        all_records, args.min_per_ambiguity, args.max_samples, rng,
    )
    print(f"Selected {len(selected)} test records")

    # Confirm the split before loading pointclouds + hitting the API.
    if args.min_per_ambiguity is not None:
        reply = input("\nProceed with this split? [y/N]: ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted.")
            return

    # 2) Materialize queries; rewrite metadata so by_ambiguity keys are buckets.
    queries = records_to_queries(selected, desc="queries", load_pointclouds=True)
    for q in queries:
        raw = int(q.metadata.get("ambiguity", 0))
        q.metadata["ambiguity_raw"] = raw
        q.metadata["ambiguity"] = bucket_ambiguity(raw)

    # 3) Shared components.
    proposer = LLMProposer(
        provider=args.proposer_provider,
        model=args.proposer_model,
        cache_dir=str(args.cache_dir / "proposer"),
    )
    gt_proposer = GroundTruthProposer()
    lss_pred = LSSGaussianPredictor(args.checkpoint, device=args.device)
    llm_pred = LLMGaussianPredictor(
        provider=args.llm_provider,
        model=args.llm_model,
        cache_dir=str(args.cache_dir / "llm_gauss"),
    )
    vlm_pred = VLMGaussianPredictor(
        provider=args.vlm_provider,
        model=args.vlm_model,
        cache_dir=str(args.cache_dir / "vlm_gauss"),
    )
    full_gmm_pred = LLMFullGMMPredictor(
        provider=args.llm_provider,
        model=args.llm_model,
        cache_dir=str(args.cache_dir / "llm_full_gmm"),
    )

    approaches = {
        "framework": GMMApproach(proposer, lss_pred),
        "straight_llm": GMMApproach(None, full_gmm_pred),
        "scaffolded_llm": GMMApproach(proposer, llm_pred),
        "scaffolded_vlm": GMMApproach(proposer, vlm_pred),
    }

    # 4) Run and report.
    results = run_benchmark(approaches, queries)

    print("\n=== Table 2: Ambiguous-setting benchmark (per-ambiguity bucket) ===")
    print_by_ambiguity_table(results)
    _write_csv(results, args.output_csv)
    print(f"Saved -> {args.output_csv}")


if __name__ == "__main__":
    main()

