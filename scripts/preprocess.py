"""Preprocess VLA-3D data into cached CachedSample tensors.

Run once before training to avoid re-running bbox normalisation and CLIP
lookups every epoch.  Outputs are saved as individual .pt files under the
configured cache directory, grouped by split.

Usage::

    # From the lss/ root directory:
    python scripts/preprocess.py

    # Override data root or cache directory:
    python scripts/preprocess.py data.data_root=/path/to/vla3d cache.dir=/fast/ssd/cache

    # Force-rebuild even if cache already exists:
    python scripts/preprocess.py cache.force_rebuild=true

The script groups statements by scene so each scene's point cloud and scene
graph are loaded only once.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

# Ensure lss/ is on the path when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.vla3d.splits import DataSplits, SplitRecord, build_splits
from language_spatial_sensor.core.schema import SpatialQuery
from language_spatial_sensor.core.transforms import build_spatial_query
from language_spatial_sensor.pipeline.tensorizer import Tensorizer, build_clip_label_map


# ── Helpers ───────────────────────────────────────────────────────────────────

def _warn_low_disk_space(path: Path, min_free_gb: float = 1.0) -> None:
    """Print a clear warning if the cache volume has little free space."""
    try:
        usage = shutil.disk_usage(path.resolve())
    except OSError:
        return
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        print(
            f"  [warn] Only {free_gb:.2f} GiB free on volume containing {path}. "
            "torch.save will fail when the disk is full (often as basic_ios::clear / zip pos errors). "
            "Free space or set cache.dir= to another filesystem."
        )


def _atomic_torch_save(obj: object, path: Path) -> None:
    """Write via a temp file in the same directory, then replace (avoids half-written .pt)."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        suffix=".pt.tmp", dir=str(path.parent), text=False
    )
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(obj, f)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _collect_labels(records: list[SplitRecord]) -> set[str]:
    """First pass: gather all unique obj.label strings across scenes."""
    labels: set[str] = set()
    seen: set[str] = set()

    by_scene: dict = defaultdict(list)
    for r in records:
        by_scene[r.scene.scene_id].append(r)

    for scene_id, recs in tqdm(by_scene.items(), desc="  collecting labels"):
        if scene_id in seen:
            continue
        seen.add(scene_id)
        try:
            sg = recs[0].scene.load_scene_graph()
            for obj in sg.objects:
                labels.add(obj.label)
        except Exception as e:
            print(f"  [warn] Could not load scene graph for {scene_id}: {e}")

    return labels


def _batched_tensorize_save(
    tensorizer: Tensorizer,
    queries: list[SpatialQuery],
    chunk_meta: list[tuple[int, SplitRecord]],
    scene_id: str,
    cache_split_dir: Path,
    manifest: list[str],
) -> int:
    """Run tensorize_batch (or per-item fallback) and save .pt files. Returns failure count."""
    failed = 0
    if not queries:
        return failed
    try:
        outputs = tensorizer.tensorize_batch(queries)
    except Exception as batch_err:
        print(
            f"  [warn] tensorize_batch failed for scene {scene_id} ({batch_err}); "
            "falling back to per-item tensorize."
        )
        outputs = []
        for q in queries:
            try:
                outputs.append(tensorizer.tensorize(q))
            except Exception as e:
                print(f"  [warn] Failed to tensorize {scene_id}: {e}")
                outputs.append(None)

    for (global_idx, _), output in zip(chunk_meta, outputs):
        if output is None:
            failed += 1
            continue
        if bool(output.obj_padding_mask.all().item()):
            print(
                f"  [preprocess] all-padded object slots (no context objects in tensorized sample): "
                f"scene_id={scene_id} global_idx={global_idx}"
            )
        fname = f"{scene_id}_{global_idx:07d}.pt"
        _atomic_torch_save(output, cache_split_dir / fname)
        manifest.append(fname)
    return failed


def _process_split(
    split_name: str,
    records: list[SplitRecord],
    cache_split_dir: Path,
    tensorizer: Tensorizer,
    force_rebuild: bool,
    tensorize_batch_size: int,
) -> None:
    """Tensorize all records in one split and save .pt files + manifest."""
    manifest_path = cache_split_dir / "manifest.json"
    if manifest_path.exists() and not force_rebuild:
        print(f"  [{split_name}] cache exists — skipping (use force_rebuild=true to override)")
        return

    cache_split_dir.mkdir(parents=True, exist_ok=True)

    # Group records by scene to load PC / scene graph once per scene
    by_scene: dict[str, list[tuple[int, SplitRecord]]] = defaultdict(list)
    for global_idx, record in enumerate(records):
        by_scene[record.scene.scene_id].append((global_idx, record))

    manifest: list[str] = []
    failed  = 0

    for scene_id, indexed_records in tqdm(by_scene.items(), desc=f"  [{split_name}]"):
        scene = indexed_records[0][1].scene
        try:
            sg = scene.load_scene_graph()
        except Exception as e:
            print(f"  [warn] Skipping scene {scene_id}: {e}")
            failed += len(indexed_records)
            continue

        bs = max(1, int(tensorize_batch_size))
        for chunk_start in range(0, len(indexed_records), bs):
            chunk = indexed_records[chunk_start : chunk_start + bs]
            queries: list = []
            chunk_meta: list[tuple[int, SplitRecord]] = []
            for global_idx, record in chunk:
                try:
                    q = build_spatial_query(
                        scene_id, sg, record.statement
                    )
                    queries.append(q)
                    chunk_meta.append((global_idx, record))
                except Exception as e:
                    print(f"  [warn] Failed to build query {scene_id} stmt {global_idx}: {e}")
                    failed += 1

            failed += _batched_tensorize_save(
                tensorizer, queries, chunk_meta, scene_id, cache_split_dir, manifest
            )

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"  [{split_name}] saved {len(manifest):,} samples "
        f"({failed} failed) → {cache_split_dir}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(
    config_path="../experiments/cfgs",
    config_name="train",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    cache_dir = Path(cfg.cache.dir)
    print(f"Cache directory: {cache_dir}")
    _warn_low_disk_space(cache_dir)

    # ── 1. Build splits ───────────────────────────────────────────────────────
    print("Building data splits...")
    splits: DataSplits = build_splits(cfg.data)
    print(splits.summary())

    # ── 2. Collect all object labels for CLIP embedding map ───────────────────
    all_records = (
        splits.train + splits.val_seen + splits.val_unseen
    )
    print(f"Collecting object labels from {len(all_records):,} records...")
    all_labels = _collect_labels(all_records)
    print(f"Found {len(all_labels):,} unique object labels. Building CLIP map...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_map = build_clip_label_map(list(all_labels), device=device)
    print(f"CLIP map built for {len(clip_map):,} labels.")

    # Save the label map alongside the cache for reproducibility
    cache_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(clip_map, cache_dir / "clip_label_map.pt")

    # ── 3. Initialise tensorizer ──────────────────────────────────────────────
    model_cfg = cfg.model
    tensorizer = Tensorizer(
        max_objects=model_cfg.max_objects,
        clip_embedding_map=clip_map,
        clip_dim=model_cfg.clip_dim,
    )

    # ── 4. Process each split ─────────────────────────────────────────────────
    split_map = {
        "train":      splits.train,
        "val_seen":   splits.val_seen,
        "val_unseen": splits.val_unseen,
    }

    for split_name, records in split_map.items():
        if not records:
            print(f"  [{split_name}] empty — skipping")
            continue
        _process_split(
            split_name,
            records,
            cache_dir / split_name,
            tensorizer,
            force_rebuild=cfg.cache.force_rebuild,
            tensorize_batch_size=cfg.cache.tensorize_batch_size,
        )

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
