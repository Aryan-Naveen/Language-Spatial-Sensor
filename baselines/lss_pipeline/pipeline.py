"""LSS pipeline baseline entry point.

Instantiates a LanguageSensorPipeline from a config dict and exposes it via
``build_pipeline``.  Future baselines in ``baselines/`` follow the same
interface so eval scripts can swap them with a single config change.

Expected config keys (all optional — fall back to sensible defaults)::

    checkpoint:      str   path to .pt checkpoint
    clip_model:      str   CLIP variant, e.g. "ViT-B/32"
    device:          str   "cuda" or "cpu"
    max_objects:     int   object sequence length (must match training)
    max_text_len:    int   text token length (must match training)
    ollama:
      model:         str   Ollama model tag
      base_url:      str   Ollama API base URL
      max_proposals: int   max proposals requested from LLM
      timeout:       int   HTTP timeout in seconds

Usage::

    from baselines.lss_pipeline import build_pipeline

    pipeline = build_pipeline(cfg)
    gmm = pipeline.run(query)
"""

from __future__ import annotations

from typing import Any

from language_spatial_sensor.pipeline.inference import LSSInference
from language_spatial_sensor.pipeline.language_sensor import LanguageSensorPipeline
from language_spatial_sensor.proposer.ollama import OllamaProposer


def build_pipeline(cfg: dict[str, Any]) -> LanguageSensorPipeline:
    """Construct a LanguageSensorPipeline from a flat config dict.

    Args:
        cfg: Configuration dictionary (typically from Hydra / OmegaConf).
             Keys described in module docstring.

    Returns:
        A ready-to-use LanguageSensorPipeline instance.
    """
    ollama_cfg = cfg.get("ollama", {})

    proposer = OllamaProposer(
        model         = ollama_cfg.get("model",         "qwen2.5:32b"),
        base_url      = ollama_cfg.get("base_url",      "http://localhost:11434"),
        max_proposals = ollama_cfg.get("max_proposals", 8),
        timeout       = ollama_cfg.get("timeout",       120),
        num_predict   = ollama_cfg.get("num_predict",   400),
    )

    model = LSSInference(
        ckpt_path       = cfg["checkpoint"],
        clip_model_name = cfg.get("clip_model",    "ViT-B/32"),
        device          = cfg.get("device",        "cuda"),
        max_objects     = cfg.get("max_objects",   100),
        max_text_len    = cfg.get("max_text_len",  64),
    )

    return LanguageSensorPipeline(proposer=proposer, model=model)
