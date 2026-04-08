"""EvalRunner: run the LanguageSensorPipeline over evaluation splits.

Writes per-sample JSONL files and per-split metrics.json broken down by:
    (1) overall
    (2) by_relation
    (3) by_ambiguity

Output layout::

    output_dir/
    ├── config.yaml          # written by the calling script
    ├── val_seen/
    │   ├── results.jsonl
    │   └── metrics.json
    └── val_unseen/
        ├── results.jsonl
        └── metrics.json

Typical usage::

    runner = EvalRunner(pipeline, output_dir=Path("outputs/benchmarks/run1"))
    runner.run_all(splits)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from data.vla3d.splits import DataSplits, SplitRecord
from language_spatial_sensor.core.transforms import build_spatial_query
from language_spatial_sensor.eval.metrics import (
    aggregate_metrics,
    breakdown_by,
    cdf_likelihood,
    mahalanobis_dist,
    proposer_mrr,
    proposer_precision,
    proposer_recall,
)
from language_spatial_sensor.pipeline.language_sensor import (
    GMMPrediction,
    LanguageSensorPipeline,
)

logger = logging.getLogger(__name__)

_TIMING_STAGES = ["scene_load_ms", "ollama_http_ms", "proposer_ms", "model_ms", "unaccounted_ms", "metrics_ms", "total_ms"]


class TimingAccumulator:
    """Thread-safe accumulator for per-stage wall-clock timings.

    Prints a summary table to stderr every ``report_every`` completions and
    once more at the end of the split.
    """

    def __init__(self, report_every: int = 25) -> None:
        self._lock        = threading.Lock()
        self._samples:    dict[str, list[float]] = defaultdict(list)
        self._n           = 0
        self.report_every = report_every

    def record(self, timings: dict[str, float]) -> None:
        with self._lock:
            for k, v in timings.items():
                self._samples[k].append(v)
            self._n += 1
            self._print_one(timings)
            if self._n % self.report_every == 0:
                self._print(prefix=f"[Timing avg @ {self._n} samples]")

    def summary(self, split_name: str) -> None:
        with self._lock:
            self._print(prefix=f"[Timing final — {split_name} ({self._n} samples)]")

    def _print_one(self, timings: dict[str, float]) -> None:
        parts = "  ".join(
            f"{s.replace('_ms','')}: {timings[s]:>6.0f}ms"
            for s in _TIMING_STAGES
            if s in timings
        )
        tqdm.write(f"  [{self._n:>5}]  {parts}")

    def _print(self, prefix: str) -> None:
        if not self._samples:
            return
        col = 22
        lines = [f"\n{prefix}"]
        lines.append(f"  {'stage':<{col}}  {'mean':>8}  {'p50':>8}  {'p95':>8}  {'max':>8}")
        lines.append("  " + "-" * (col + 36))
        for stage in _TIMING_STAGES:
            vals = self._samples.get(stage)
            if not vals:
                continue
            arr = np.array(vals)
            lines.append(
                f"  {stage:<{col}}  {arr.mean():>7.0f}ms"
                f"  {np.median(arr):>7.0f}ms"
                f"  {np.percentile(arr, 95):>7.0f}ms"
                f"  {arr.max():>7.0f}ms"
            )
        total = self._samples.get("total_ms")
        if total:
            throughput = 1000.0 / np.mean(total)
            lines.append(f"\n  throughput  {throughput:.2f} samples/s")
        tqdm.write("\n".join(lines))


class EvalRunner:
    """Run a LanguageSensorPipeline over one or more evaluation splits.

    Args:
        pipeline:   Fully constructed LanguageSensorPipeline.
        output_dir: Root directory for benchmark outputs.
        load_pc:    If True (default), load point cloud + object split for each
                    scene.  Set False to skip loading point clouds (faster if the
                    pipeline does not use the point cloud).
    """

    def __init__(
        self,
        pipeline: LanguageSensorPipeline,
        output_dir: str | Path,
        load_pc: bool = True,
        num_workers: int = 4,
    ) -> None:
        self.pipeline    = pipeline
        self.output_dir  = Path(output_dir)
        self.load_pc     = load_pc
        self.num_workers = num_workers

    # ── Public API ────────────────────────────────────────────────────────────

    def run_all(self, splits: DataSplits) -> None:
        """Run val_seen and val_unseen splits; skip train."""
        self.run_split(splits.val_seen,   "val_seen")
        self.run_split(splits.val_unseen, "val_unseen")

    def run_split(
        self,
        split_records: list[SplitRecord],
        split_name: str,
    ) -> list[dict[str, Any]]:
        """Evaluate all records in a split; write results.jsonl and metrics.json.

        Returns the list of per-sample result dicts.
        """
        split_dir = self.output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        results_path = split_dir / "results.jsonl"
        metrics_path = split_dir / "metrics.json"

        logger.info("Evaluating split '%s' (%d samples) …", split_name, len(split_records))

        timing = TimingAccumulator(report_every=25)

        records: list[dict[str, Any]] = []
        with open(results_path, "w") as f, \
             ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {
                pool.submit(self._eval_one, rec, timing): rec
                for rec in split_records
            }
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result is None:
                    continue
                records.append(result)
                f.write(json.dumps(result, default=_json_default) + "\n")

        timing.summary(split_name)

        metrics = _build_metrics(records)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=_json_default)

        logger.info(
            "Split '%s': %d samples evaluated. "
            "proposer_recall=%.3f  cdf_joint=%.3f  ece=%.3f",
            split_name,
            len(records),
            metrics["overall"].get("proposer_recall", float("nan")),
            metrics["overall"].get("cdf_joint_mean", float("nan")),
            metrics["overall"].get("ece", float("nan")),
        )
        return records

    # ── Private helpers ───────────────────────────────────────────────────────

    def _eval_one(
        self,
        split_rec: SplitRecord,
        timing: TimingAccumulator,
    ) -> dict[str, Any] | None:
        """Evaluate a single SplitRecord; return result dict or None on error."""
        t_total = time.perf_counter()
        scene     = split_rec.scene
        statement = split_rec.statement

        # ── Stage 1: scene loading ────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            scene_graph = scene.load_scene_graph()
            if self.load_pc:
                pcd = scene.load_pointcloud()
                points       = np.asarray(pcd.points, dtype=np.float32)
                object_split = scene.load_object_split()
            else:
                points       = None
                object_split = None
            query = build_spatial_query(
                scene_id     = scene.scene_id,
                scene_graph  = scene_graph,
                statement    = statement,
                points       = points,
                object_split = object_split,
            )
        except Exception as exc:
            logger.warning("Failed to build SpatialQuery for scene '%s': %s", scene.scene_id, exc)
            return None
        scene_load_ms = (time.perf_counter() - t0) * 1000

        # ── Stage 2: pipeline (proposer + model) ──────────────────────────────
        t_pipeline = time.perf_counter()
        try:
            gmm: GMMPrediction = self.pipeline.run(query)
        except Exception as exc:
            logger.warning(
                "Pipeline failed for scene '%s', utterance '%s': %s",
                scene.scene_id, statement.text, exc,
            )
            return None

        # ── Stage 3: metrics ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        proposals   = gmm.proposals
        gt_xyz      = query.target_xyz
        target_bbox = query.target_bbox

        prec = proposer_precision(proposals, query)
        rec  = proposer_recall(proposals, query)
        mrr  = proposer_mrr(proposals, query)

        if target_bbox is not None:
            from language_spatial_sensor.eval.metrics import _to_aabb6
            cdf_vals = cdf_likelihood(gmm, _to_aabb6(target_bbox))
        else:
            eps = np.full(3, 0.1, dtype=np.float32)
            cdf_vals = cdf_likelihood(gmm, np.concatenate([gt_xyz - eps, gt_xyz + eps]))

        maha = mahalanobis_dist(gmm, gt_xyz)
        metrics_ms = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - t_total) * 1000

        pipeline_ms = (time.perf_counter() - t_pipeline) * 1000
        unaccounted_ms = pipeline_ms - gmm.timings.get("proposer_ms", 0.0) - gmm.timings.get("model_ms", 0.0)

        timing.record({
            "scene_load_ms":   scene_load_ms,
            "ollama_http_ms":  gmm.timings.get("ollama_http_ms", 0.0),
            "proposer_ms":     gmm.timings.get("proposer_ms", 0.0),
            "model_ms":        gmm.timings.get("model_ms", 0.0),
            "unaccounted_ms":  max(unaccounted_ms, 0.0),
            "metrics_ms":      metrics_ms,
            "total_ms":        total_ms,
        })

        return {
            "scene_id":            scene.scene_id,
            "utterance":           statement.text,
            "relation":            statement.relation,
            "ambiguity":           statement.ambiguity,
            "gt_xyz":              gt_xyz.tolist(),
            "target_bbox":         target_bbox.tolist() if target_bbox is not None else None,
            "n_proposals":         len(proposals),
            "proposer_precision":  prec,
            "proposer_recall":     rec,
            "proposer_mrr":        mrr,
            **cdf_vals,
            "mahalanobis_dist":    maha,
            "timings_ms":          {k: round(v, 1) for k, v in gmm.timings.items()},
            "proposals": [
                {
                    "utterance":  p.utterance,
                    "relation":   p.relation,
                    "region_id":  p.region_id,
                    "anchor_ids": p.anchor_ids,
                    "confidence": p.confidence,
                }
                for p in proposals
            ],
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the three-level metrics dict written to metrics.json."""
    return {
        "overall":      aggregate_metrics(records),
        "by_relation":  breakdown_by(records, "relation"),
        "by_ambiguity": breakdown_by(records, "ambiguity"),
    }


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy scalars/arrays."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
