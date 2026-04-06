from dataclasses import dataclass


@dataclass
class LSSConfig:
    """Flat config for the full LSS model.

    Every field is a primitive so Optuna can sample directly:
        cfg = LSSConfig(
            hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512]),
            pooling_type = trial.suggest_categorical("pooling_type", ["mean", "max", "attention"]),
            ...
        )

    YAML equivalent lives at experiments/cfgs/model/lss.yaml.
    """

    # ── Dimensions ────────────────────────────────────────────────────────────
    hidden_dim: int = 256        # shared D across all transformer layers
    clip_dim: int = 512          # CLIP ViT-B/32 output dim (frozen)
    bert_dim: int = 768          # BERT base hidden dim

    # ── Text encoder ──────────────────────────────────────────────────────────
    text_model: str = "bert-base-uncased"
    freeze_text: bool = False    # fine-tune BERT by default

    # ── Spatial relation MLP ──────────────────────────────────────────────────
    # calc_pairwise_locs produces 12 geometric features; optionally project to a
    # larger dim before feeding as spatial bias.
    spatial_relation_dim: int = 12   # raw geometric feature count (fixed by formula)
    spatial_mlp_hidden: int = 64     # hidden dim of the relation projection MLP

    # ── Backbone (3D-SceneSpatial encoder) ───────────────────────────────────
    backbone_type: str = "scene_spatial"     # key into BACKBONE_REGISTRY
    num_spatial_layers: int = 3

    # ── Global fusion transformer ─────────────────────────────────────────────
    num_fusion_layers: int = 3
    num_heads: int = 8
    ffn_dim: int = 1024              # feed-forward dim in all transformer layers

    # ── Pooling ───────────────────────────────────────────────────────────────
    pooling_type: str = "mean"       # key into POOLING_REGISTRY

    # ── Output head ───────────────────────────────────────────────────────────
    head_type: str = "gaussian_cholesky"   # key into HEAD_REGISTRY

    # ── Regularisation ────────────────────────────────────────────────────────
    dropout: float = 0.3

    # ── FiLM conditioning ─────────────────────────────────────────────────────
    use_film: bool = True          # condition pooled embedding on coord_scale/coord_shift
    film_hidden_dim: int = 64      # hidden dim of FiLM MLP (6 → film_hidden_dim → 2*hidden_dim)

    # ── Sequence limits ───────────────────────────────────────────────────────
    max_objects: int = 100        # N — sequences are padded/truncated to this length
    max_text_len: int = 64       # L — BERT token limit
