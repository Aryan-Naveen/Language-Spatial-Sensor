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

import json
from dataclasses import fields, replace
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from language_spatial_sensor.core.schema import CachedSample, TensorizerOutput


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

    def __getitem__(self, idx: int) -> CachedSample:
        path = self.cache_dir / self._files[idx]
        item: CachedSample = torch.load(path, weights_only=False)
        item = _coerce_target_bbox_aabb6(item)

        if self.augmentations is not None:
            item = self.augmentations(item)

        return item


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
