"""Table 1 driver: no-ambiguity benchmark on val_seen AND val_unseen.

All approaches use :class:`GroundTruthProposer` so grounding is fixed.  We
compare how different distribution predictors (LSS, LLM, VLM, and each of
them calibrated via conformal superposition) score the *same* anchor/region.

Filter: only ``ambiguity == 1`` queries.  With higher-ambiguity queries the
"ground truth" anchor set is not unique, which would confound the
predictor-level comparison we're trying to do here.

Two test sets, same size:
    * val_seen:   same scenes the model validated on during training.
    * val_unseen: scenes held out entirely (different buildings/meshes).
Conformal wrappers are calibrated **once** on val_seen (disjoint from both
test sets) so both tables use the same calibrated ``q`` — this is the
correct way to evaluate distribution shift: we measure how the calibration
generalises to unseen scenes rather than re-fitting it.

Run::

    python -m evaluation.run_benchmark_no_ambig \\
        --checkpoint checkpoints/best.pt \\
        --max-samples 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.approach import GMMApproach
from evaluation.benchmark import (
    load_training_split_records,
    print_by_relation_table,
    print_no_ambiguity_table,
    records_to_queries,
    run_benchmark,
)
from language_spatial_sensor.pipeline.conformal import ConformalSuperimposed
from language_spatial_sensor.pipeline.distribution_predictor import (
    LLMGaussianPredictor,
    LSSGaussianPredictor,
    VLMGaussianPredictor,
)
from language_spatial_sensor.pipeline.proposer import GroundTruthProposer


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
    p.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Test-set size for each of val_seen and val_unseen.",
    )
    p.add_argument(
        "--samples-per-relation",
        type=int,
        default=None,
        help="If set, stratify each test split by relation: take up to this many "
        "records per relation. Caps the total at max-samples.",
    )
    p.add_argument(
        "--calib-max-samples",
        type=int,
        default=200,
        help="Calibration-set size for conformal wrappers. Drawn from val_seen, "
        "disjoint from both test sets.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip-seen",
        action="store_true",
        help="Skip the val_seen test table (calibration still runs on val_seen).",
    )
    p.add_argument(
        "--skip-unseen",
        action="store_true",
        help="Skip the val_unseen test table.",
    )

    p.add_argument("--llm-provider", default="openai")
    p.add_argument("--llm-model", default="gpt-5.2")
    p.add_argument("--vlm-provider", default="ollama")
    p.add_argument("--vlm-model", default="qwen2.5vl:32b")
    p.add_argument("--cache-dir", type=Path, default=Path("cache"))

    p.add_argument("--coverage", type=float, default=0.9)
    p.add_argument("--device", default="cuda")

    p.add_argument(
        "--output-csv-seen",
        type=Path,
        default=Path("results/table1_no_ambig_seen.csv"),
    )
    p.add_argument(
        "--output-csv-seen-by-relation",
        type=Path,
        default=Path("results/table1_no_ambig_seen_by_relation.csv"),
    )
    p.add_argument(
        "--output-csv-unseen",
        type=Path,
        default=Path("results/table1_no_ambig_unseen.csv"),
    )
    p.add_argument(
        "--output-csv-unseen-by-relation",
        type=Path,
        default=Path("results/table1_no_ambig_unseen_by_relation.csv"),
    )
    return p.parse_args()


def _select_test_records(records, samples_per_relation, max_samples, rng):
    """Pick up to *max_samples* records, optionally stratified by relation."""
    if samples_per_relation is not None:
        from collections import defaultdict
        by_rel: dict[str, list] = defaultdict(list)
        for r in records:
            by_rel[r.statement.relation].append(r)
        picked: list = []
        for rel in sorted(by_rel.keys()):
            bucket = by_rel[rel][:samples_per_relation]
            picked.extend(bucket)
            print(f"  relation {rel!r}: took {len(bucket)}/{len(by_rel[rel])}")
        rng.shuffle(picked)
        return picked[:max_samples]
    return records[:max_samples]


def _run_and_report(label: str, approaches, queries, out_csv: Path, out_csv_by_rel: Path) -> None:
    print(f"\n=== Table 1 [{label}]: No-ambiguity benchmark (RMSE / NLL / ANEES) ===")
    results = run_benchmark(approaches, queries)
    print_no_ambiguity_table(results)
    _write_csv(results, out_csv)
    print(f"Saved -> {out_csv}")

    print(f"\n=== Per-relation breakdown [{label}] ===")
    print_by_relation_table(results)
    _write_csv_by_relation(results, out_csv_by_rel)
    print(f"Saved -> {out_csv_by_rel}")


def _write_csv(results: dict, path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "approach", "n",
            "rmse_mean", "rmse_std", "rmse_median",
            "nll_mean", "nll_std", "nll_median",
            "anees", "nees_std", "nees_median",
        ])
        for name, m in results.items():
            o = m.get("overall", {})
            w.writerow([
                name,
                o.get("n", 0),
                f"{o.get('rmse_mean', 0):.6f}",
                f"{o.get('rmse_std', 0):.6f}",
                f"{o.get('rmse_median', 0):.6f}",
                f"{o.get('nll_mean', 0):.6f}",
                f"{o.get('nll_std', 0):.6f}",
                f"{o.get('nll_median', 0):.6f}",
                f"{o.get('anees', 0):.6f}",
                f"{o.get('nees_std', 0):.6f}",
                f"{o.get('nees_median', 0):.6f}",
            ])


def _write_csv_by_relation(results: dict, path: Path) -> None:
    """Long-format CSV: one row per (approach, relation)."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "approach", "relation", "n",
            "rmse_mean", "rmse_std", "rmse_median",
            "nll_mean", "nll_std", "nll_median",
            "anees", "nees_std", "nees_median",
        ])
        for name, m in results.items():
            if "error" in m:
                continue
            br = m.get("by_relation", {})
            relations = set()
            for metric_name in ("rmse", "nll", "nees"):
                relations |= set(br.get(metric_name, {}).keys())
            for rel in sorted(relations, key=str):
                r = br.get("rmse", {}).get(rel, {})
                n = br.get("nll", {}).get(rel, {})
                e = br.get("nees", {}).get(rel, {})
                count = r.get("count", n.get("count", e.get("count", 0)))
                if count == 0:
                    continue
                w.writerow([
                    name, rel, count,
                    f"{r.get('mean', 0):.6f}",
                    f"{r.get('std', 0):.6f}",
                    f"{r.get('median', 0):.6f}",
                    f"{n.get('mean', 0):.6f}",
                    f"{n.get('std', 0):.6f}",
                    f"{n.get('median', 0):.6f}",
                    f"{e.get('mean', 0):.6f}",
                    f"{e.get('std', 0):.6f}",
                    f"{e.get('median', 0):.6f}",
                ])


def main() -> None:
    import random

    args = _parse_args()

    # 1) val_seen records — source of both the calibration set and the
    # val_seen test set. Load once, shuffle once, then split:
    #   [0 : calib_max_samples)      → calibration
    #   [calib_max_samples : ...)    → val_seen test candidates
    seen_all_records = load_training_split_records(
        config_dir=args.config_dir,
        split="val_seen",
        data_root_override=args.data_root,
        datasets_override=args.datasets,
    )
    seen_no_ambig = [r for r in seen_all_records if r.statement.ambiguity == 1]
    print(
        f"val_seen: kept {len(seen_no_ambig)}/{len(seen_all_records)} "
        f"records with ambiguity==1"
    )

    rng = random.Random(args.seed)
    rng.shuffle(seen_no_ambig)
    calib_records = seen_no_ambig[: args.calib_max_samples]
    seen_remaining = seen_no_ambig[args.calib_max_samples :]
    seen_test_records = _select_test_records(
        seen_remaining, args.samples_per_relation, args.max_samples, rng,
    )
    print(f"Calibration: {len(calib_records)}  val_seen test: {len(seen_test_records)}")

    # 2) val_unseen records — independent held-out scenes. Sample same size.
    unseen_test_records: list = []
    if not args.skip_unseen:
        unseen_all_records = load_training_split_records(
            config_dir=args.config_dir,
            split="val_unseen",
            data_root_override=args.data_root,
            datasets_override=args.datasets,
        )
        unseen_no_ambig = [r for r in unseen_all_records if r.statement.ambiguity == 1]
        print(
            f"val_unseen: kept {len(unseen_no_ambig)}/{len(unseen_all_records)} "
            f"records with ambiguity==1"
        )
        rng_u = random.Random(args.seed + 100)
        rng_u.shuffle(unseen_no_ambig)
        unseen_test_records = _select_test_records(
            unseen_no_ambig, args.samples_per_relation, args.max_samples, rng_u,
        )
        print(f"val_unseen test: {len(unseen_test_records)}")

    # 3) Materialise queries. Only the records we actually benchmark get
    # pointclouds loaded — not the whole split.
    calib = records_to_queries(calib_records, desc="calib", load_pointclouds=True)
    seen_test = (
        records_to_queries(seen_test_records, desc="seen_test", load_pointclouds=True)
        if (seen_test_records and not args.skip_seen)
        else []
    )
    unseen_test = (
        records_to_queries(unseen_test_records, desc="unseen_test", load_pointclouds=True)
        if unseen_test_records
        else []
    )

    # 2) Build predictors (shared across rows).
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

    # 3) Calibrate conformal wrappers on the calibration split.
    gt_proposer = GroundTruthProposer()
    lss_ckpt_tag = args.checkpoint.stem
    lss_conformal = ConformalSuperimposed(
        base=lss_pred,
        proposer=gt_proposer,
        coverage=args.coverage,
        predictor_name=f"lss_{lss_ckpt_tag}",
        cache_dir=str(args.cache_dir / "conformal"),
    )
    llm_conformal = ConformalSuperimposed(
        base=llm_pred,
        proposer=gt_proposer,
        coverage=args.coverage,
        predictor_name=f"llm_{args.llm_model}",
        cache_dir=str(args.cache_dir / "conformal"),
    )
    vlm_conformal = ConformalSuperimposed(
        base=vlm_pred,
        proposer=gt_proposer,
        coverage=args.coverage,
        predictor_name=f"vlm_{args.vlm_model}",
        cache_dir=str(args.cache_dir / "conformal"),
    )
    lss_conformal.calibrate(calib)
    llm_conformal.calibrate(calib)
    vlm_conformal.calibrate(calib)

    # 4) Assemble approaches.
    approaches = {
        "gt_lss": GMMApproach(gt_proposer, lss_pred),
        "gt_lss_conformal": GMMApproach(gt_proposer, lss_conformal),
        "gt_llm": GMMApproach(gt_proposer, llm_pred),
        "gt_llm_conformal": GMMApproach(gt_proposer, llm_conformal),
        "gt_vlm": GMMApproach(gt_proposer, vlm_pred),
        "gt_vlm_conformal": GMMApproach(gt_proposer, vlm_conformal),
    }

    # 5) Run and report — once per test set, using the same calibrated approaches.
    if seen_test:
        _run_and_report(
            "val_seen", approaches, seen_test,
            args.output_csv_seen, args.output_csv_seen_by_relation,
        )
    else:
        print("\n(skipping val_seen)")

    if unseen_test:
        _run_and_report(
            "val_unseen", approaches, unseen_test,
            args.output_csv_unseen, args.output_csv_unseen_by_relation,
        )
    else:
        print("\n(skipping val_unseen)")


if __name__ == "__main__":
    main()
