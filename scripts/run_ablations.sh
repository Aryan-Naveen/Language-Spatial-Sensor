#!/usr/bin/env bash
# Macro architecture ablations for LSS. Trains 5 variants sequentially and
# logs them to a separate W&B project so they don't pollute the main one.
#
# Usage (from anywhere):
#     bash scripts/run_ablations.sh
#
# Pass extra Hydra overrides after the script name; they apply to every run:
#     bash scripts/run_ablations.sh training.epochs=20 training.batch_size=128

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LSS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${LSS_ROOT}"

PROJECT="lss-ablations"
TS="$(date +%Y%m%d_%H%M%S)"
EXTRA=("$@")  # caller-supplied Hydra overrides applied to every run

run_ablation() {
    local name="$1"; shift
    local run_name="${TS}_${name}"
    echo
    echo "============================================================"
    echo "  Ablation: ${name}"
    echo "  Run name: ${run_name}"
    echo "============================================================"
    python train.py \
        wandb.project="${PROJECT}" \
        wandb.name="${run_name}" \
        "wandb.tags=[ablation,${name}]" \
        checkpoint.dir="checkpoints/ablations/${run_name}" \
        hydra.run.dir="outputs/ablations/${run_name}" \
        "$@" \
        "${EXTRA[@]}"
}

# (1) No spatial backbone — bypass ViSTA; identity pass-through over object
#     features. Tests whether the explicit pairwise spatial bias matters or
#     whether the fusion transformer alone can learn geometry from bbox tokens.
# run_ablation no_spatial_backbone \
#     model.backbone_type=identity

# (2) No global fusion — drop the GlobalFusionTransformer; pool object features
#     alone and add the text CLS to the pooled vector. query_token pooling is
#     incompatible with this, so we fall back to mean.
run_ablation no_global_fusion \
    model.use_global_fusion=false \
    model.pooling_type=mean

# (3) No FiLM region conditioning. Tests whether the head's coord_scale/shift
#     affine de-normalization carries enough region-frame signal on its own.
run_ablation no_film \
    model.use_film=false

# (4) No anchor-centric embedding. Pairs with the existing "no anchor flag"
#     ablation: this one removes the relative coordinate, that one removes
#     the role tag.
run_ablation no_anchor_centric \
    model.use_anchor_centric_coords=false

# (5) No 3D geometry — zero out obj_bboxes so spatial features and anchor-
#     centric deltas collapse to constants. Symmetric to the "no language"
#     baseline; isolates the contribution of the geometry channel.
run_ablation no_geometry \
    model.zero_bboxes=true

echo
echo "All ablations finished. W&B project: ${PROJECT}"
