"""Dataset for the 3D-ViSTA grounding baseline.

Loads pre-cached .pt files produced by preprocess.py.

Named ``grounding_dataset`` (not ``dataset``) so ``.../vista_grounding`` on
``sys.path`` does not shadow 3D-VisTA's top-level ``dataset`` package.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


class ViSTAGroundingDataset(Dataset):
    """Loads cached per-sample .pt dicts produced by vista_grounding/preprocess.py.

    Each sample is a dict with keys:
        scene_id, language            — identity for BEV reload
        obj_pcds   (O, P, 6)
        obj_locs   (O, 6)
        obj_masks  (O,) bool
        txt_ids    (L,)
        txt_masks  (L,)
        target_xyz_world   (3,)
        target_bbox_world  (6,)
        coord_scale        (3,)
        coord_shift        (3,)
    """

    def __init__(self, cache_split_dir: Path) -> None:
        manifest_path = Path(cache_split_dir) / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. "
                "Run baselines/vista_grounding/preprocess.py first."
            )
        with open(manifest_path) as f:
            self._fnames: list[str] = json.load(f)
        self._dir = Path(cache_split_dir)

    def __len__(self) -> int:
        return len(self._fnames)

    def __getitem__(self, idx: int) -> dict:
        sample = torch.load(self._dir / self._fnames[idx], weights_only=True)
        tb = sample["target_bbox_world"].reshape(-1)
        if tb.numel() == 24:
            corners = tb.reshape(8, 3)
            sample["target_bbox_world"] = torch.cat(
                [corners.min(dim=0).values, corners.max(dim=0).values]
            )
        return sample

    def get_scene_id(self, idx: int) -> str:
        """Return the scene_id for index ``idx`` without loading the full sample."""
        # scene_id is encoded in the filename: "{scene_id}_{global_idx:07d}.pt"
        fname = self._fnames[idx]
        return "_".join(fname.split("_")[:-1])


def vista_collate_fn(batch: list[dict]) -> dict:
    """Collate a list of sample dicts into a batch dict.

    All tensor values are stacked along dim 0.
    String values (scene_id, language) are kept as lists.
    """
    out: dict = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], Tensor):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values   # list of strings
    return out
