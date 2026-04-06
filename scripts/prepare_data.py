"""
Data preprocessing entrypoint.

Usage
-----
From the repo root (src/lss/):

    python scripts/prepare_data.py

Override any config value on the CLI:

    python scripts/prepare_data.py \
        data.data_root=/data/VLA3D \
        data.datasets=[HM3D] \
        data.splits.val_unseen_scene_frac=0.2 \
        data.splits.seed=0
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Make the repo root importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.vla3d.splits import build_splits


@hydra.main(
    config_path="../experiments/cfgs",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    splits = build_splits(cfg.data)
    print(f"[prepare_data] Split summary: {splits.summary()}")


if __name__ == "__main__":
    main()
