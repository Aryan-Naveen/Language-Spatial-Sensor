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
    # calc_pairwise_locs output last dim must match: mlp → 12; center/vertical_bottom → 1, 4, or 5.
    spatial_relation_dim: int = 12
    spatial_mlp_hidden: int = 64     # hidden dim of the relation projection MLP
    pairwise_rel_type: str = "mlp"  # "mlp" | "center" | "vertical_bottom"
    spatial_pairwise_dist_norm: bool = True  # ignored when pairwise_rel_type == "mlp"
    # Language-conditioned spatial bias: FiLM the spatial-MLP hidden layer with text CLS.
    # Lets the pairwise bias depend on the query (e.g. "above" up-weights vertical pairs).
    condition_spatial_on_text: bool = False

    # ── Anchor-centric object embedding ──────────────────────────────────────
    # Inject per-object (centre − anchor-centroid) delta as an additive hidden embedding.
    # Encourages the model to reason in relative coordinates w.r.t. referenced anchors.
    use_anchor_centric_coords: bool = False

    # ── Backbone (3D-SceneSpatial encoder) ───────────────────────────────────
    backbone_type: str = "scene_spatial"     # key into BACKBONE_REGISTRY
    num_spatial_layers: int = 3

    # ── Global fusion transformer ─────────────────────────────────────────────
    num_fusion_layers: int = 3
    num_heads: int = 8
    ffn_dim: int = 1024              # feed-forward dim in all transformer layers

    # ── Pooling ───────────────────────────────────────────────────────────────
    # "mean" | "max" | "attention" | "query_token"
    # query_token: DETR-style learnable [Q_target] token prepended to the fusion
    # sequence; its post-fusion state is used as the context vector.
    pooling_type: str = "mean"

    # ── Output head ───────────────────────────────────────────────────────────
    head_type: str = "gaussian_cholesky"   # key into HEAD_REGISTRY

    # Diagonal Gaussian head (head_type == "gaussian_diagonal") — ignored otherwise
    head_hidden_sizes: tuple[int, ...] = (128, 128)
    head_use_layernorm: bool = False
    head_use_tanh: bool = False       # if True, mu_region = tanh(raw) / 2
    head_min_sigma: float = 0.0     # floor added before softplus σ (region frame)
    head_film_from_scale: bool = False  # extra FiLM(γ,β) from coord_scale inside head
    head_dropout: float = 0.15        # dropout inside gaussian_diagonal μ/σ MLPs only

    # ── Regularisation ────────────────────────────────────────────────────────
    dropout: float = 0.3

    # ── FiLM conditioning ─────────────────────────────────────────────────────
    use_film: bool = True          # condition pooled embedding on coord_scale/coord_shift
    film_hidden_dim: int = 64      # hidden dim of FiLM MLP (6 → film_hidden_dim → 2*hidden_dim)

    # ── Sequence limits ───────────────────────────────────────────────────────
    max_objects: int = 100        # N — sequences are padded/truncated to this length
    max_text_len: int = 64       # L — BERT token limit
