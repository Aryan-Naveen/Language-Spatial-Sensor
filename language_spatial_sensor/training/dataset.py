"""PyTorch Dataset over pre-cached TensorizerOutput tensors.

The dataset expects a cache directory built by scripts/preprocess.py:

    cache_dir/
    ├── train/
    │   ├── manifest.json         # ordered list of filenames
    │   └── *.pt                  # individual TensorizerOutput files
    ├── val_seen/
    │   ├── manifest.json
    │   └── *.pt
    └── val_unseen/
        ├── manifest.json
        └── *.pt

Usage::

    from language_spatial_sensor.training.dataset import CachedLSSDataset, collate_fn

    train_ds = CachedLSSDataset("cache/train", augmentations=augs)
    loader   = DataLoader(train_ds, batch_size=32, collate_fn=collate_fn, shuffle=True)
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import fields, replace
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer

from language_spatial_sensor.core.schema import CachedSample, TensorizerOutput

# Matches filenames written by preprocess.py: {scene_id}_{global_idx:07d}.pt
_FNAME_RE = re.compile(r'^(.+)_\d{7}\.pt$')


def _coerce_target_bbox_aabb6(sample: CachedSample) -> CachedSample:
    """Ensure ``target_bbox_world`` is (6,) AABB; convert 8×3 corners (24) if needed."""
    tb = sample.target_bbox_world.reshape(-1)
    if tb.numel() == 6:
        return replace(sample, target_bbox_world=tb.reshape(6).clone())
    if tb.numel() == 24:
        corners = tb.reshape(8, 3)
        cmin = corners.min(dim=0).values
        cmax = corners.max(dim=0).values
        new_tb = torch.cat([cmin, cmax]).to(
            dtype=sample.target_bbox_world.dtype,
            device=sample.target_bbox_world.device,
        )
        return replace(sample, target_bbox_world=new_tb)
    raise ValueError(
        "target_bbox_world must have 6 (AABB) or 24 (8 corners) values, "
        f"got shape {tuple(sample.target_bbox_world.shape)}"
    )


class CachedLSSDataset(Dataset):
    """Load pre-cached CachedSample tensors and optionally apply augmentations.

    Args:
        cache_dir:      Directory containing manifest.json and *.pt files.
        augmentations:  Optional callable (e.g. Compose([...])) applied to each
                        loaded item before returning.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        augmentations=None,
    ) -> None:
        self.cache_dir     = Path(cache_dir)
        self.augmentations = augmentations

        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest.json found at {manifest_path}. "
                "Run scripts/preprocess.py first."
            )

        with open(manifest_path) as f:
            self._files: list[str] = json.load(f)

    def __len__(self) -> int:
        return len(self._files)

    def get_scene_id(self, idx: int) -> str:
        """Return the scene_id embedded in the cached filename at position ``idx``."""
        fname = self._files[idx]
        m = _FNAME_RE.match(fname)
        if m is None:
            raise ValueError(f"Cannot parse scene_id from cached filename: {fname!r}")
        return m.group(1)

    def __getitem__(self, idx: int) -> CachedSample:
        path = self.cache_dir / self._files[idx]
        item: CachedSample = torch.load(path, weights_only=False)
        item = _coerce_target_bbox_aabb6(item)

        if self.augmentations is not None:
            item = self.augmentations(item)

        return item


class ShardedLSSDataset(IterableDataset):
    """WebDataset-backed streaming reader over tar-sharded cache.

    Each shard holds ~1024 pickled ``CachedSample`` objects as ``.sample`` tar
    members. One S3 GET fetches one shard, amortising ~500 ms of network
    overhead across ~1024 samples — roughly 1000× fewer round-trips than the
    per-sample ``.pt`` layout for the same training budget.

    Args:
        cache_dir:      Directory containing ``shards.json`` + ``shard-*.tar``.
        augmentations:  Optional callable applied to each loaded item.
        shuffle_buffer: Reservoir size for in-flight sample shuffle. Set to 0
                        for no shuffle (validation). 8000 is a good training
                        default — large enough to mix across many shards, small
                        enough to fit in RAM with many workers.
        shardshuffle:   If True, shuffle the shard list each epoch. Required
                        for proper training-set randomisation.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        augmentations=None,
        shuffle_buffer: int = 0,
        shardshuffle: bool = False,
    ) -> None:
        super().__init__()
        self.cache_dir      = Path(cache_dir)
        self.augmentations  = augmentations
        self.shuffle_buffer = shuffle_buffer
        self.shardshuffle   = shardshuffle

        index_path = self.cache_dir / "shards.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"No shards.json at {index_path}. "
                "Build shards first with scripts/build_shards.py."
            )
        with open(index_path) as f:
            index = json.load(f)

        self._shard_files: list[str] = index["shards"]
        self._n_samples:   int       = int(index["n_samples"])

    def __len__(self) -> int:
        return self._n_samples

    def __iter__(self):
        # Lazy import: webdataset is a training-container dep; local .pt paths
        # don't need it.
        import webdataset as wds

        urls = [str(self.cache_dir / name) for name in self._shard_files]

        # webdataset ≥0.2.86 requires an int or False — True is deprecated.
        # Passing the full shard count means "shuffle across all shards each epoch".
        shard_shuffle_n = len(urls) if self.shardshuffle else False

        pipeline = wds.WebDataset(
            urls,
            shardshuffle=shard_shuffle_n,
            nodesplitter=wds.split_by_node,
        )
        if self.shuffle_buffer > 0:
            pipeline = pipeline.shuffle(self.shuffle_buffer)

        for record in pipeline:
            raw = record.get("sample")
            if raw is None:
                # Corrupt shard / unexpected member — skip with a warning.
                continue
            item: CachedSample = torch.load(
                io.BytesIO(raw), weights_only=False, map_location="cpu"
            )
            item = _coerce_target_bbox_aabb6(item)
            if self.augmentations is not None:
                item = self.augmentations(item)
            yield item


def is_sharded_cache(cache_dir: str | Path) -> bool:
    """True if the given split directory holds tar shards instead of .pt files."""
    return (Path(cache_dir) / "shards.json").is_file()


class CollateFn:
    """Collate a list of CachedSample into a batched TensorizerOutput.

    Tokenization runs here so the tokenizer lives only in the training process,
    not in preprocessing.

    Args:
        tokenizer_name: HuggingFace tokenizer name (should match TextEncoder).
        max_text_len:   Padding/truncation length for BERT tokens.
    """

    def __init__(self, tokenizer_name: str = "bert-base-uncased", max_text_len: int = 64) -> None:
        self.tokenizer    = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_text_len = max_text_len

    def __call__(self, batch: list[CachedSample]) -> TensorizerOutput:
        enc = self.tokenizer(
            [item.language for item in batch],
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tensor_fields = [f.name for f in fields(CachedSample) if f.name != "language"]
        return TensorizerOutput(
            text_input_ids=enc["input_ids"],
            text_attention_mask=enc["attention_mask"],
            **{name: torch.stack([getattr(item, name) for item in batch]) for name in tensor_fields},
        )
