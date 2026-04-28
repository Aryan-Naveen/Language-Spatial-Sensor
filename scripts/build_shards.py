#!/usr/bin/env python3
"""Convert the per-sample ``.pt`` cache into WebDataset tar shards.

FastFile streaming of 1.4M tiny ``.pt`` files pays one S3 GET per sample (~50 ms
each), turning an I/O-bound training loop into a network-latency loop. Packing
samples into ~500 MB tar shards amortises that cost: one S3 GET fetches ~1000
samples at full S3 throughput (~500 MB/s), dropping per-sample overhead by
roughly three orders of magnitude.

Output layout (per split)::

    out_dir/<split>/
        shard-000000.tar
        shard-000001.tar
        ...
        shards.json          # {"shards": [...], "n_samples": N, "samples_per_shard": K}

Each tar member is named ``<original_fname_stem>.sample``. The payload is
exactly what ``torch.save`` would have written for a ``CachedSample`` object —
no schema change, so a sharded cache and a ``.pt`` cache are interchangeable
from the training loop's point of view.

Usage::

    # All three splits:
    python scripts/build_shards.py --cache-dir cache --out-dir cache_shards

    # One split (e.g. just train):
    python scripts/build_shards.py --cache-dir cache/train --out-dir cache_shards/train --single-split
"""
from __future__ import annotations

import argparse
import json
import tarfile
import time
from pathlib import Path


DEFAULT_SAMPLES_PER_SHARD = 1024  # ~300 MB at ~300 KB/sample — good S3 GET size


def _shard_split(
    split_src: Path,
    split_dst: Path,
    samples_per_shard: int,
) -> None:
    """Pack one split directory (manifest.json + *.pt) into tar shards."""
    manifest_path = split_src / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {split_src}")

    with open(manifest_path) as f:
        filenames: list[str] = json.load(f)

    split_dst.mkdir(parents=True, exist_ok=True)

    shard_names: list[str] = []
    n_samples = len(filenames)
    n_shards  = (n_samples + samples_per_shard - 1) // samples_per_shard

    t0 = time.time()
    for shard_idx in range(n_shards):
        start = shard_idx * samples_per_shard
        end   = min(start + samples_per_shard, n_samples)
        shard_name = f"shard-{shard_idx:06d}.tar"
        shard_path = split_dst / shard_name
        tmp_path   = shard_path.with_suffix(".tar.tmp")

        with tarfile.open(tmp_path, mode="w") as tar:
            for fname in filenames[start:end]:
                src = split_src / fname
                if not src.is_file():
                    raise FileNotFoundError(f"Missing sample: {src}")
                # Rename: "<stem>.pt" -> "<stem>.sample" so webdataset treats
                # the extension as a generic raw-bytes field and doesn't try
                # to auto-decode as a torch model.
                stem = fname[:-3] if fname.endswith(".pt") else fname
                info = tar.gettarinfo(str(src), arcname=f"{stem}.sample")
                with open(src, "rb") as fsrc:
                    tar.addfile(info, fsrc)

        tmp_path.rename(shard_path)
        shard_names.append(shard_name)

        elapsed = time.time() - t0
        rate    = (shard_idx + 1) / max(elapsed, 1e-6)
        eta     = (n_shards - (shard_idx + 1)) / max(rate, 1e-6)
        print(
            f"  [{split_src.name}] shard {shard_idx + 1}/{n_shards}  "
            f"({end}/{n_samples} samples)  "
            f"rate={rate:.2f} shards/s  eta={eta / 60:.1f} min",
            flush=True,
        )

    index = {
        "shards":            shard_names,
        "n_samples":         n_samples,
        "samples_per_shard": samples_per_shard,
    }
    with open(split_dst / "shards.json", "w") as f:
        json.dump(index, f, indent=2)

    total_bytes = sum((split_dst / s).stat().st_size for s in shard_names)
    print(
        f"  [{split_src.name}] DONE: {n_shards} shards, "
        f"{total_bytes / 1e9:.2f} GB total, "
        f"avg {total_bytes / n_shards / 1e6:.1f} MB/shard",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path,
                        help="Input cache dir (with train/, val_seen/, val_unseen/) "
                             "or a single split dir when --single-split is set.")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Output directory for shards (parallel layout).")
    parser.add_argument("--samples-per-shard", type=int,
                        default=DEFAULT_SAMPLES_PER_SHARD,
                        help=f"Samples packed per tar shard (default {DEFAULT_SAMPLES_PER_SHARD}).")
    parser.add_argument("--single-split", action="store_true",
                        help="Treat --cache-dir as a single split (no train/ val_seen/ nesting).")
    parser.add_argument("--splits", nargs="+",
                        default=["train", "val_seen", "val_unseen"],
                        help="Splits to shard (default: train val_seen val_unseen).")
    args = parser.parse_args()

    if args.single_split:
        _shard_split(args.cache_dir, args.out_dir, args.samples_per_shard)
        return

    for split in args.splits:
        src = args.cache_dir / split
        if not src.is_dir():
            print(f"[skip] {split}: {src} does not exist")
            continue
        print(f"[{split}] sharding {src} → {args.out_dir / split}")
        _shard_split(src, args.out_dir / split, args.samples_per_shard)

    print(f"\nAll done. Upload:\n  aws s3 sync {args.out_dir}/ s3://$BUCKET/$PREFIX/")


if __name__ == "__main__":
    main()
