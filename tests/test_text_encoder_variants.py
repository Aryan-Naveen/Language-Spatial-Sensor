"""Smoke test: TextEncoder + CollateFn handle every text_model in the sweep.

Verifies the fixes that let the Optuna sweep sample any of the six HF
backbones without a shape mismatch in the projection layer or a tokenizer
error in the collator. Downloads ~2 GB of HF weights on first run; re-uses
``HF_HOME`` on subsequent runs.

Run directly::

    pytest tests/test_text_encoder_variants.py -v

Or per-model::

    pytest tests/test_text_encoder_variants.py -k roberta -v
"""

from __future__ import annotations

import pytest
import torch

from language_spatial_sensor.models.components.encoders import TextEncoder
from language_spatial_sensor.models.config import LSSConfig
from language_spatial_sensor.training.dataset import CollateFn


# Keep in sync with training/sagemaker/optuna_sweep.py::_suggest_hparams
SWEEP_TEXT_MODELS = [
    "bert-base-uncased",
    "bert-large-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]


@pytest.mark.parametrize("text_model", SWEEP_TEXT_MODELS)
def test_text_encoder_forward_shape(text_model: str) -> None:
    """TextEncoder must output (B, hidden_dim) regardless of backbone size."""
    cfg = LSSConfig(
        text_model=text_model,
        hidden_dim=64,
        max_text_len=16,
        freeze_text=False,
    )
    encoder = TextEncoder(cfg).eval()

    B, L = 2, cfg.max_text_len
    input_ids = torch.randint(low=5, high=100, size=(B, L), dtype=torch.long)
    attention_mask = torch.ones(B, L, dtype=torch.long)
    attention_mask[0, L // 2 :] = 0   # exercise the padding path in mean pool

    with torch.no_grad():
        out = encoder(input_ids, attention_mask)

    assert out.shape == (B, cfg.hidden_dim)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("text_model", SWEEP_TEXT_MODELS)
def test_text_encoder_pool_mode_matches_family(text_model: str) -> None:
    """sentence-transformers → mean; everything else → CLS."""
    cfg = LSSConfig(text_model=text_model, hidden_dim=64, max_text_len=16)
    encoder = TextEncoder(cfg)
    expected = "mean" if text_model.startswith("sentence-transformers/") else "cls"
    assert encoder.pool_mode == expected


@pytest.mark.parametrize("text_model", SWEEP_TEXT_MODELS)
def test_text_encoder_grads_flow(text_model: str) -> None:
    """Backward pass must produce finite grads for the projection + backbone."""
    cfg = LSSConfig(
        text_model=text_model,
        hidden_dim=64,
        max_text_len=16,
        freeze_text=False,
        dropout=0.0,
    )
    encoder = TextEncoder(cfg).train()

    B, L = 2, cfg.max_text_len
    input_ids = torch.randint(low=5, high=100, size=(B, L), dtype=torch.long)
    attention_mask = torch.ones(B, L, dtype=torch.long)

    out = encoder(input_ids, attention_mask).sum()
    out.backward()

    assert encoder.proj.weight.grad is not None
    assert torch.isfinite(encoder.proj.weight.grad).all()


@pytest.mark.parametrize("text_model", SWEEP_TEXT_MODELS)
def test_collate_fn_token_shapes(text_model: str) -> None:
    """CollateFn must emit (B, max_text_len) ids/mask for any tokenizer."""
    from dataclasses import fields

    from language_spatial_sensor.core.schema import CachedSample

    max_text_len = 16
    B = 3
    collate = CollateFn(tokenizer_name=text_model, max_text_len=max_text_len)

    # Build minimal CachedSample stubs — CollateFn only touches `language` and
    # stacks the remaining tensor fields, so zero tensors suffice.
    tensor_fields = {f.name: f for f in fields(CachedSample) if f.name != "language"}

    def _zeros_like_field(name: str) -> torch.Tensor:
        shapes = {
            "obj_clip_features": (150, 512),
            "obj_bboxes": (150, 6),
            "obj_is_anchor": (150,),
            "obj_padding_mask": (150,),
            "coord_shift": (3,),
            "coord_scale": (3,),
            "target_xyz_world": (3,),
            "target_bbox_world": (6,),
        }
        shape = shapes[name]
        if name in ("obj_is_anchor", "obj_padding_mask"):
            return torch.zeros(shape, dtype=torch.bool)
        return torch.zeros(shape, dtype=torch.float32)

    batch = [
        CachedSample(
            language=f"the red chair is next to the table {i}",
            **{name: _zeros_like_field(name) for name in tensor_fields},
        )
        for i in range(B)
    ]

    out = collate(batch)
    assert out.text_input_ids.shape == (B, max_text_len)
    assert out.text_attention_mask.shape == (B, max_text_len)
    assert out.text_attention_mask.dtype in (torch.long, torch.int64, torch.int32)
