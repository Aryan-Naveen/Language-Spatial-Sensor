"""End-to-end LSS pipeline evaluation script.

Loads val_seen and val_unseen splits, runs the full
OllamaProposer → LSSModel → GMM pipeline on each sample,
and writes per-sample JSONL + aggregate metrics.json to the output directory.

Usage (from src/lss/)::

    python scripts/eval_pipeline.py

Override config values::

    python scripts/eval_pipeline.py \\
        checkpoint=checkpoints/best.pt \\
        ollama.model=qwen2.5:32b \\
        data.datasets=[HM3D]

Override the output directory (same as training pattern)::

    python scripts/eval_pipeline.py \\
        hydra.run.dir=outputs/benchmarks/lss_v1
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

# Make the repo root importable when running as a script
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from data.vla3d.splits import build_splits
from baselines.lss_pipeline import build_pipeline
from language_spatial_sensor.eval.runner import EvalRunner

log = logging.getLogger(__name__)


@hydra.main(
    config_path="../experiments/cfgs",
    config_name="eval",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Hydra has already changed cwd to hydra.run.dir; Path(".") is the
    # benchmark output directory.  The checkpoint path is resolved relative
    # to the original launch directory so relative paths work correctly.
    orig_cwd   = Path(get_original_cwd())
    output_dir = Path(".")

    # ── Build splits ──────────────────────────────────────────────────────────
    # Uses the same seed / fractions as training → identical val_seen /
    # val_unseen partitions.  Change data.datasets or splits.* to vary.
    log.info("Building data splits …")
    splits = build_splits(cfg.data)
    log.info("Splits: %s", splits.summary())

    # ── Build pipeline ────────────────────────────────────────────────────────
    # Resolve checkpoint path relative to original cwd before Hydra moved us.
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    ckpt_path = str(orig_cwd / cfg_dict["checkpoint"])
    cfg_dict["checkpoint"] = ckpt_path

    log.info("Loading pipeline (checkpoint: %s) …", ckpt_path)
    pipeline = build_pipeline(cfg_dict)

    # ── Run evaluation ────────────────────────────────────────────────────────
    runner = EvalRunner(
        pipeline     = pipeline,
        output_dir   = output_dir,
        load_pc      = bool(cfg.get("load_pc", True)),
        num_workers  = int(cfg.get("num_workers", 4)),
    )
    runner.run_all(splits)

    log.info("Results written to %s", output_dir.resolve())


if __name__ == "__main__":
    main()
