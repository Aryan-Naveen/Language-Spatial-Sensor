"""Training entry point for the Language Spatial Sensor model.

Single trial::

    python train.py

Override any config value::

    python train.py training.lr=3e-4 model.pooling_type=attention

Optuna hyperparameter sweep (requires hydra-optuna-sweeper)::

    python train.py --multirun \\
        training.lr=1e-4,3e-4,1e-3 \\
        model.hidden_dim=128,256,512

    # Or with continuous distributions (enable the sweeper block in train.yaml):
    python train.py --multirun hydra/sweeper=optuna

WandB is enabled by default; disable with::

    python train.py wandb.enabled=false
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

# wandb is optional — the sweep disables it, and the container image skips
# the install to keep start-up fast. All call sites guard with `wandb is not None`.
try:
    import wandb
except ImportError:
    wandb = None
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# Ensure lss/ root is on sys.path when invoked from a subdirectory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in training processes
import matplotlib.pyplot as plt

from language_spatial_sensor.models.components.heads import (
    GaussianPrediction,
    marginal_cdf_at_gt,
)
from language_spatial_sensor.models.config import LSSConfig
from language_spatial_sensor.models.lss_model import LSSModel
from language_spatial_sensor.models.registry import LOSS_REGISTRY

# Side-effect import: ensure loss functions are registered.
import language_spatial_sensor.models.components.heads  # noqa: F401
from language_spatial_sensor.pipeline.augmentations import (
    Compose,
    RandomMaskObjects,
    RandomObjectJitter,
    RandomSceneRotation,
)
from language_spatial_sensor.training.dataset import (
    CachedLSSDataset,
    CollateFn,
    ShardedLSSDataset,
    is_sharded_cache,
)
from viz.bev import render_bev_with_sample_overlay


# ── Utilities ─────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_augmentations(cfg: DictConfig):
    """Build the training augmentation pipeline from config."""
    transforms = []
    if cfg.mask_prob > 0:
        transforms.append(RandomMaskObjects(mask_prob=cfg.mask_prob))
    if cfg.rotate:
        transforms.append(RandomSceneRotation(max_angle_deg=cfg.rotate_max_deg))
    if cfg.jitter_std > 0:
        transforms.append(RandomObjectJitter(std=cfg.jitter_std))
    return Compose(transforms) if transforms else None


def build_model(cfg: DictConfig) -> LSSModel:
    """Instantiate LSSModel from the model config node."""
    model_cfg = LSSConfig(**OmegaConf.to_container(cfg, resolve=True))
    return LSSModel(model_cfg)


def build_optimizer(model: LSSModel, cfg: DictConfig):
    """AdamW with a lower LR for the pre-trained BERT encoder."""
    bert_params  = list(model.text_enc.bert.parameters())
    bert_ids     = {id(p) for p in bert_params}
    other_params = [p for p in model.parameters() if id(p) not in bert_ids]

    return torch.optim.AdamW(
        [
            {"params": bert_params,  "lr": cfg.lr * cfg.bert_lr_scale},
            {"params": other_params, "lr": cfg.lr},
        ],
        weight_decay=cfg.weight_decay,
    )


def _compute_loss(tr, pred: GaussianPrediction, batch) -> torch.Tensor:
    """Dispatch to the configured loss function."""
    if tr.loss_type == "bbox_cdf":
        return LOSS_REGISTRY.build(
            "bbox_cdf",
            pred,
            batch.target_bbox_world,
            batch.coord_scale,
            lambda_dist=tr.bbox_cdf_lambda_dist,
            eps=tr.bbox_cdf_eps,
        )
    elif tr.loss_type == "center_nll":
        return LOSS_REGISTRY.build(
            "center_nll",
            pred,
            batch.target_xyz_world,
            lambda_l1=tr.center_nll_lambda_l1,
            lambda_mahal=tr.center_nll_lambda_mahal,
            lambda_vol=tr.center_nll_lambda_vol,
        )
    else:
        raise ValueError(f"Unknown loss_type: {tr.loss_type}")


@torch.no_grad()
def evaluate(
    model: LSSModel,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype,
    tr,
) -> dict[str, float]:
    """Run validation and return a dict of metrics."""
    model.eval()
    total_loss = 0.0
    n = 0

    # Collect per-sample metrics for percentile computation.
    all_dist: list[torch.Tensor] = []
    all_mahal: list[torch.Tensor] = []
    all_nll: list[torch.Tensor] = []
    all_quad: list[torch.Tensor] = []      # diffᵀ Σ⁻¹ diff  (Mahalanobis²)
    all_log_det: list[torch.Tensor] = []   # log|Σ|           (variance volume)
    all_min_sigma: list[torch.Tensor] = [] # min(diag(L))     (variance collapse proxy)
    all_inside: list[torch.Tensor] = []
    all_gt_cdf_axis: list[torch.Tensor] = []
    all_gt_cdf_prod: list[torch.Tensor] = []
    # For conformal superset volumes.
    all_marginal_sigma: list[torch.Tensor] = []

    conformal = getattr(tr, "conformal_superset", False)
    max_val_steps = int(getattr(tr, "max_val_steps", 0) or 0)

    for step, batch in enumerate(loader):
        if max_val_steps > 0 and step >= max_val_steps:
            break
        batch = _to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            pred: GaussianPrediction = _forward(model, batch)
            loss = _compute_loss(tr, pred, batch)

        B = pred.mu.shape[0]
        total_loss += loss.item() * B
        n += B

        pred_f = GaussianPrediction(mu=pred.mu.float(), L=pred.L.float())
        axis_mass, mass_prod = marginal_cdf_at_gt(pred_f, batch.target_bbox_world.float())

        mu_f = pred_f.mu
        target_f = batch.target_xyz_world.float()
        diff = target_f - mu_f
        dist = diff.norm(dim=-1)

        # Mahalanobis distance via triangular solve.
        L_f = pred_f.L
        z = torch.linalg.solve_triangular(
            L_f, diff.unsqueeze(-1), upper=False,
        ).squeeze(-1)
        quad  = (z * z).sum(dim=-1).clamp(min=1e-12)        # (B,) = diffᵀΣ⁻¹diff
        mahal = quad.sqrt()

        # Multivariate Gaussian NLL at GT target centre (matches gmm_nll_at_gt):
        #   -log N(y; μ, Σ) = 0.5 (log|Σ| + (y-μ)ᵀ Σ⁻¹ (y-μ) + k log 2π),  k=3
        log_det = 2.0 * L_f.diagonal(dim1=-2, dim2=-1).clamp(min=1e-6).log().sum(dim=-1)
        nll     = 0.5 * (log_det + quad + 3.0 * math.log(2.0 * math.pi))

        bbox = batch.target_bbox_world.float()
        inside = (mu_f >= bbox[:, :3]).all(dim=-1) & (mu_f <= bbox[:, 3:]).all(dim=-1)

        # Per-sample diag of L (region-frame σ lower bound) → flags variance collapse.
        L_diag    = L_f.diagonal(dim1=-2, dim2=-1).clamp(min=1e-12)   # (B, 3)
        min_sigma = L_diag.min(dim=-1).values                         # (B,)

        all_dist.append(dist.cpu())
        all_mahal.append(mahal.cpu())
        all_nll.append(nll.cpu())
        all_quad.append(quad.cpu())
        all_log_det.append(log_det.cpu())
        all_min_sigma.append(min_sigma.cpu())
        all_inside.append(inside.cpu())
        all_gt_cdf_axis.append(axis_mass.cpu())
        all_gt_cdf_prod.append(mass_prod.cpu())

        if conformal:
            marginal_var = (L_f ** 2).sum(dim=-1)              # (B, 3)
            marginal_sigma = marginal_var.clamp(min=1e-12).sqrt()
            all_marginal_sigma.append(marginal_sigma.cpu())

    model.train()

    # Concatenate per-sample tensors.
    cat_dist      = torch.cat(all_dist)
    cat_mahal     = torch.cat(all_mahal)
    cat_nll       = torch.cat(all_nll)
    cat_quad      = torch.cat(all_quad)
    cat_log_det   = torch.cat(all_log_det)
    cat_min_sigma = torch.cat(all_min_sigma)
    cat_inside    = torch.cat(all_inside).float()
    cat_cdf_axis  = torch.cat(all_gt_cdf_axis)
    cat_cdf_prod  = torch.cat(all_gt_cdf_prod)

    def _iqr(t: torch.Tensor) -> float:
        return (t.quantile(0.75) - t.quantile(0.25)).item()

    denom = max(n, 1)
    metrics: dict[str, float] = {
        "loss":                       total_loss / denom,
        "mean_dist":                  cat_dist.mean().item(),
        "dist_p90":                   cat_dist.quantile(0.9).item(),
        "acc":                        cat_inside.mean().item(),
        "mahal_mean":                 cat_mahal.mean().item(),
        "mahal_p90":                  cat_mahal.quantile(0.9).item(),
        # ── NLL and its two additive components (log|Σ| + quad; constant 3·log2π omitted).
        # Compare nll_iqr vs log_det_iqr vs quad_iqr to see which term dominates the spread.
        "nll_mean":                   cat_nll.mean().item(),
        "nll_median":                 cat_nll.median().item(),
        "nll_iqr":                    _iqr(cat_nll),
        "nll_p90":                    cat_nll.quantile(0.9).item(),
        "quad_mean":                  cat_quad.mean().item(),
        "quad_iqr":                   _iqr(cat_quad),
        "quad_p90":                   cat_quad.quantile(0.9).item(),
        "log_det_mean":               cat_log_det.mean().item(),
        "log_det_iqr":                _iqr(cat_log_det),
        "log_det_p10":                cat_log_det.quantile(0.1).item(),  # most collapsed Σ
        # Per-sample min σ across axes — tracks variance collapse directly.
        "min_sigma_mean":             cat_min_sigma.mean().item(),
        "min_sigma_p10":              cat_min_sigma.quantile(0.1).item(),
        "gt_bbox_marginal_mass_mean": cat_cdf_axis.mean().item(),
        "gt_bbox_marginal_prod_mean": cat_cdf_prod.mean().item(),
        "gt_bbox_marginal_prod_p90":  cat_cdf_prod.quantile(0.9).item(),
    }

    if conformal:
        cat_marginal_sigma = torch.cat(all_marginal_sigma)     # (N, 3)

        coverage = getattr(tr, "conformal_coverage", 0.9)
        q = cat_mahal.quantile(coverage).item()

        # Conformal ellipsoid volume: V = (4/3) pi q^3 sqrt(det Sigma)
        sqrt_det = (0.5 * cat_log_det).exp()
        ellipsoid_vol = (4.0 / 3.0) * math.pi * (q ** 3) * sqrt_det

        # AABB superset volume: V = prod_i(2 q sigma_i) = (2q)^3 prod(sigma_i)
        superset_vol = ((2.0 * q) ** 3) * cat_marginal_sigma.prod(dim=-1)

        metrics.update({
            "conformal_q":                  q,
            "conformal_ellipsoid_vol_mean": ellipsoid_vol.mean().item(),
            "conformal_ellipsoid_vol_p90":  ellipsoid_vol.quantile(0.9).item(),
            "conformal_superset_vol_mean":  superset_vol.mean().item(),
            "conformal_superset_vol_p90":   superset_vol.quantile(0.9).item(),
        })

    return metrics


def _to_device(batch, device: torch.device):
    """Move all tensor fields of a TensorizerOutput to device."""
    from dataclasses import fields, replace
    from language_spatial_sensor.core.schema import TensorizerOutput
    return replace(batch, **{
        f.name: getattr(batch, f.name).to(device)
        for f in fields(TensorizerOutput)
    })


def _forward(model: LSSModel, batch) -> GaussianPrediction:
    """Single call site for the model forward pass."""
    return model(
        batch.text_input_ids,
        batch.text_attention_mask,
        batch.obj_clip_features,
        batch.obj_bboxes,
        batch.obj_is_anchor,
        batch.obj_padding_mask,
        batch.coord_scale,
        batch.coord_shift,
    )


def _resolve_precision(precision: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def _load_spatial_query(
    scene_id: str,
    language: str,
    data_root: Path,
    datasets: list[str],
):
    """Reload raw VLA3D data for a single scene and return a SpatialQuery.

    Searches each configured dataset directory for ``scene_id``, then matches
    the statement by exact language text.  Raises if the scene or statement
    cannot be found.
    """
    from data.vla3d.dataset import VLA3DScene

    _CANONICAL = {
        "3rscan": "3RScan", "arkitscenes": "ARKitScenes", "hm3d": "HM3D",
        "matterport": "Matterport", "scannet": "Scannet", "unity": "Unity",
    }

    scene = None
    for ds_name in datasets:
        scene_path = data_root / _CANONICAL.get(ds_name.lower(), ds_name) / scene_id
        if scene_path.exists():
            scene = VLA3DScene(scene_path)
            break
    if scene is None:
        raise FileNotFoundError(
            f"Scene '{scene_id}' not found under {data_root} in datasets {datasets}"
        )

    from language_spatial_sensor.core.transforms import build_spatial_query

    sg         = scene.load_scene_graph()
    statements = scene.load_statements(sg)
    stmt       = next((s for s in statements if s.text == language), None)
    if stmt is None:
        raise ValueError(f"Statement not found in scene '{scene_id}': {language!r}")

    pcd          = scene.load_pointcloud()
    points       = np.asarray(pcd.points, dtype=np.float32)
    object_split = scene.load_object_split()

    return build_spatial_query(scene_id, sg, stmt, points=points, object_split=object_split)


@torch.no_grad()
def generate_bev_plots(
    model: LSSModel,
    dataset: CachedLSSDataset,
    collate_fn: CollateFn,
    device: torch.device,
    autocast_dtype: torch.dtype,
    n_scenes: int,
    n_samples: int,
    viz_seed: int,
    data_root: Path,
    datasets: list[str],
    conformal_q: float | None = None,
) -> list[tuple[str, plt.Figure]]:
    """Run inference on N seeded val_seen scenes and return BEV + sample-density figures.

    The same ``viz_seed`` always selects the same scenes so plots are epoch-comparable.
    ``n_samples`` points are drawn from the predicted Gaussian and passed to
    ``render_bev_with_sample_overlay`` as raw (K, 3) XYZ — swap this sampling call
    for a diffusion-transformer sampler and everything downstream is unchanged.

    When ``conformal_q`` is provided, the Mahalanobis conformal ellipsoid at that
    threshold is projected onto the X-Y plane and drawn on each BEV figure.

    Returns:
        List of (language_query, figure) pairs; scenes that fail to load are skipped.
    """
    from matplotlib.patches import Ellipse as MplEllipse

    rng     = np.random.RandomState(viz_seed)
    indices = rng.choice(len(dataset), size=min(n_scenes, len(dataset)), replace=False)

    model.eval()
    figs: list[tuple[str, plt.Figure]] = []

    for idx in indices:
        sample   = dataset[int(idx)]
        scene_id = dataset.get_scene_id(int(idx))

        # Load raw point cloud + scene graph to build the ground-truth BEV
        try:
            query = _load_spatial_query(scene_id, sample.language, data_root, datasets)
        except Exception as e:
            print(f"[viz] Skipping scene {scene_id}: {e}")
            continue

        # Run model inference
        batch = collate_fn([sample])
        batch = _to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            pred: GaussianPrediction = _forward(model, batch)

        # Sample n_samples points from the Gaussian (batch item 0)
        mu   = pred.mu[0].float()   # (3,)
        L    = pred.L[0].float()    # (3, 3) lower-triangular
        dist = torch.distributions.MultivariateNormal(mu, scale_tril=L)
        samples_xyz = dist.sample((n_samples,)).cpu().numpy()  # (n_samples, 3)

        fig = render_bev_with_sample_overlay(query, samples_xyz=samples_xyz, resolution=0.25)

        # Overlay conformal ellipsoid projected onto the X-Y plane.
        if conformal_q is not None:
            Sigma = (L @ L.T).cpu().numpy()                    # (3, 3)
            Sigma_xy = Sigma[:2, :2]                           # marginal X-Y covariance
            eigvals, eigvecs = np.linalg.eigh(Sigma_xy)
            # Semi-axis lengths = q * sqrt(eigenvalue)
            width  = 2.0 * conformal_q * np.sqrt(eigvals[1])  # major
            height = 2.0 * conformal_q * np.sqrt(eigvals[0])  # minor
            angle  = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))

            mu_np = mu.cpu().numpy()
            ax = fig.axes[0]
            ellipse = MplEllipse(
                xy=(mu_np[0], mu_np[1]),
                width=width, height=height, angle=angle,
                edgecolor="cyan", facecolor="none",
                linewidth=1.5, linestyle="--", zorder=6,
            )
            ax.add_patch(ellipse)

        figs.append((sample.language, fig))

    model.train()
    return figs


# ── Training loop ─────────────────────────────────────────────────────────────

def run_training(cfg: DictConfig, trial=None) -> float:
    """Train the model and return the best val_seen loss (for Optuna).

    If ``trial`` is provided (an ``optuna.Trial``), intermediate ``val_loss`` is
    reported after every validation epoch via ``trial.report`` and the trial is
    pruned if the sampler's pruner requests it (raises ``optuna.TrialPruned``).
    """
    tr = cfg.training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = _resolve_precision(tr.precision)

    seed_everything(tr.seed)

    # ── Flash attention flags ──────────────────────────────────────────────────
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    # ── Datasets ───────────────────────────────────────────────────────────────
    cache_dir  = Path(cfg.cache.dir)
    train_augs = build_augmentations(cfg.augmentation)

    # Auto-detect shard layout: tar shards stream ~1000× faster than per-sample
    # .pt files over FastFile. Map-style .pt path stays for local runs that
    # need random access (e.g. BEV viz).
    use_shards = is_sharded_cache(cache_dir / "train")
    print(f"[data] cache format: {'tar-shards' if use_shards else 'per-sample .pt'}", flush=True)

    if use_shards:
        train_ds = ShardedLSSDataset(
            cache_dir / "train",
            augmentations=train_augs,
            shuffle_buffer=8000,   # ~8 shards of mixing
            shardshuffle=True,
        )
        val_seen_ds = ShardedLSSDataset(cache_dir / "val_seen", augmentations=None)
    else:
        train_ds    = CachedLSSDataset(cache_dir / "train",     augmentations=train_augs)
        val_seen_ds = CachedLSSDataset(cache_dir / "val_seen",  augmentations=None)

    # val_unseen is optional — skip gracefully if the split doesn't exist yet
    val_unseen_ds: CachedLSSDataset | ShardedLSSDataset | None = None
    try:
        if use_shards:
            val_unseen_ds = ShardedLSSDataset(cache_dir / "val_unseen", augmentations=None)
        else:
            val_unseen_ds = CachedLSSDataset(cache_dir / "val_unseen", augmentations=None)
    except FileNotFoundError:
        print("[warn] val_unseen split not found — skipping unseen evaluation.")

    collate_fn = CollateFn(
        tokenizer_name=cfg.model.text_model,
        max_text_len=cfg.model.max_text_len,
    )

    # IterableDataset forbids shuffle=True; shuffle happens inside the
    # dataset (shard-level + in-worker buffer).
    train_loader = DataLoader(
        train_ds,
        batch_size=tr.batch_size,
        shuffle=False if use_shards else True,
        num_workers=tr.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(tr.num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_seen_ds,
        batch_size=tr.batch_size,
        shuffle=False,
        num_workers=tr.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(tr.num_workers > 0),
    )
    val_unseen_loader: DataLoader | None = None
    if val_unseen_ds is not None:
        val_unseen_loader = DataLoader(
            val_unseen_ds,
            batch_size=tr.batch_size,
            shuffle=False,
            num_workers=tr.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(tr.num_workers > 0),
        )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = build_model(cfg.model).to(device)

    # torch.compile for additional fusion / kernel optimisation.
    # Disabled via `training.disable_compile=True` for Optuna sweeps: running
    # many varied model topologies + text backbones through torch.compile in a
    # single process accumulates inductor/CUDA state and eventually SIGSEGVs
    # inside a plain Linear forward after a few trials.
    if (
        device.type == "cuda"
        and hasattr(torch, "compile")
        and not bool(getattr(tr, "disable_compile", False))
    ):
        model = torch.compile(model, dynamic=False)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = build_optimizer(model, tr)
    total_steps = len(train_loader) * tr.epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=tr.warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=(autocast_dtype == torch.float16))

    # ── W&B ────────────────────────────────────────────────────────────────────
    use_wandb = cfg.wandb.enabled and wandb is not None
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity or None,
            name=cfg.wandb.name or None,
            tags=list(cfg.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )

    # ── Checkpoint dir ─────────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.checkpoint.dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    global_step   = 0

    # ── Training epochs ────────────────────────────────────────────────────────
    max_train_steps = int(getattr(tr, "max_train_steps_per_epoch", 0) or 0)
    full_len = len(train_loader)
    pbar_total = min(max_train_steps, full_len) if max_train_steps > 0 else full_len
    for epoch in range(1, tr.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_step = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{tr.epochs}",
            leave=False,
            total=pbar_total,
        )
        for batch in pbar:
            if max_train_steps > 0 and epoch_step >= max_train_steps:
                break
            epoch_step += 1
            batch = _to_device(batch, device)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                pred: GaussianPrediction = _forward(model, batch)
                loss = _compute_loss(tr, pred, batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), tr.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            global_step += 1

            pbar.set_postfix(loss=f"{loss_val:.4f}")

            if use_wandb and global_step % 50 == 0:
                # param group 0 = BERT (lr = base * bert_lr_scale), group 1 = rest.
                # Log the main-branch lr so freezing BERT (scale=0) doesn't zero the plot.
                last_lrs = scheduler.get_last_lr()
                wandb.log({
                    "train/loss":    loss_val,
                    "train/lr":      last_lrs[1],
                    "train/lr_bert": last_lrs[0],
                    "step":          global_step,
                })

        steps_this_epoch = epoch_step if max_train_steps > 0 else len(train_loader)
        avg_train_loss = epoch_loss / max(steps_this_epoch, 1)

        # ── Validation ─────────────────────────────────────────────────────────
        if epoch % tr.val_every == 0:
            eval_kwargs = dict(
                device=device,
                autocast_dtype=autocast_dtype,
                tr=tr,
            )

            # val_seen
            val_metrics = evaluate(model, val_loader, **eval_kwargs)
            val_loss = val_metrics["loss"]

            # val_unseen (optional)
            val_unseen_metrics: dict[str, float] | None = None
            if val_unseen_loader is not None:
                val_unseen_metrics = evaluate(model, val_unseen_loader, **eval_kwargs)

            # ── Console output ──────────────────────────────────────────────
            seen_str = (
                f"[epoch {epoch:3d}] train={avg_train_loss:.4f}  "
                f"seen_loss={val_loss:.4f}  seen_acc={val_metrics['acc']:.3f}  "
                f"seen_dist={val_metrics['mean_dist']:.3f}m(p90={val_metrics['dist_p90']:.3f})  "
                f"seen_mahal={val_metrics['mahal_mean']:.3f}(p90={val_metrics['mahal_p90']:.3f})  "
                f"seen_nll={val_metrics['nll_mean']:.2f}(iqr={val_metrics['nll_iqr']:.2f})  "
                f"seen_prod={val_metrics['gt_bbox_marginal_prod_mean']:.4f}"
            )
            diag_str = (
                f"{'':>12}"
                f"seen_quad={val_metrics['quad_mean']:.2f}(iqr={val_metrics['quad_iqr']:.2f}, p90={val_metrics['quad_p90']:.2f})  "
                f"seen_logdet={val_metrics['log_det_mean']:.2f}(iqr={val_metrics['log_det_iqr']:.2f}, p10={val_metrics['log_det_p10']:.2f})  "
                f"seen_minσ={val_metrics['min_sigma_mean']:.3f}(p10={val_metrics['min_sigma_p10']:.3f})"
            )
            print(seen_str)
            print(diag_str)
            if "conformal_q" in val_metrics:
                conf_str = (
                    f"{'':>12}"
                    f"conformal_q={val_metrics['conformal_q']:.3f}  "
                    f"ellipsoid_vol={val_metrics['conformal_ellipsoid_vol_mean']:.3f}"
                    f"(p90={val_metrics['conformal_ellipsoid_vol_p90']:.3f})  "
                    f"superset_vol={val_metrics['conformal_superset_vol_mean']:.3f}"
                    f"(p90={val_metrics['conformal_superset_vol_p90']:.3f})"
                )
                print(conf_str)
            if val_unseen_metrics is not None:
                unseen_str = (
                    f"{'':>12}"
                    f"unseen_loss={val_unseen_metrics['loss']:.4f}  "
                    f"unseen_acc={val_unseen_metrics['acc']:.3f}  "
                    f"unseen_dist={val_unseen_metrics['mean_dist']:.3f}m(p90={val_unseen_metrics['dist_p90']:.3f})  "
                    f"unseen_mahal={val_unseen_metrics['mahal_mean']:.3f}(p90={val_unseen_metrics['mahal_p90']:.3f})  "
                    f"unseen_nll={val_unseen_metrics['nll_mean']:.2f}(iqr={val_unseen_metrics['nll_iqr']:.2f})  "
                    f"unseen_prod={val_unseen_metrics['gt_bbox_marginal_prod_mean']:.4f}"
                )
                print(unseen_str)

            # ── W&B logging ─────────────────────────────────────────────────
            if use_wandb:
                log_dict: dict = {
                    "epoch":            epoch,
                    "train/epoch_loss": avg_train_loss,
                }
                for k, v in val_metrics.items():
                    log_dict[f"val_seen/{k}"] = v
                if val_unseen_metrics is not None:
                    for k, v in val_unseen_metrics.items():
                        log_dict[f"val_unseen/{k}"] = v
                wandb.log(log_dict)

            # ── BEV visualisation (val_seen only) ───────────────────────────
            # generate_bev_plots random-indexes the dataset — only map-style
            # CachedLSSDataset supports that. Sharded runs skip viz silently.
            if epoch % cfg.viz.plot_every == 0 and not use_shards:
                figs = generate_bev_plots(
                    model, val_seen_ds, collate_fn, device, autocast_dtype,
                    n_scenes=cfg.viz.n_scenes,
                    n_samples=cfg.viz.n_samples,
                    viz_seed=cfg.viz.seed,
                    data_root=Path(cfg.data.data_root),
                    datasets=list(cfg.data.datasets),
                    conformal_q=val_metrics.get("conformal_q"),
                )
                if use_wandb:
                    wandb.log({
                        f"viz/val_seen/scene_{i}": wandb.Image(fig)
                        for i, (_, fig) in enumerate(figs)
                    })
                for _, fig in figs:
                    plt.close(fig)

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch":      epoch,
                        "model":      model.state_dict(),
                        "optimizer":  optimizer.state_dict(),
                        "val_loss":   val_loss,
                        "cfg":        OmegaConf.to_container(cfg, resolve=True),
                    },
                    ckpt_dir / "best.pt",
                )

            # Optuna: persist intermediate value + honour the pruner.
            # Done after save so a pruned trial still has its best checkpoint.
            if trial is not None:
                import optuna
                trial.report(val_loss, step=epoch)
                # Persist per-epoch metrics to the DB as user attrs. W&B is
                # disabled during sweeps, so without this the train-loss curve
                # is only in CloudWatch logs and vanishes with the container.
                # Append to a list so `trial.user_attrs["train_loss_curve"]`
                # becomes the full per-epoch history.
                train_curve = list(trial.user_attrs.get("train_loss_curve", []))
                train_curve.append(float(avg_train_loss))
                trial.set_user_attr("train_loss_curve", train_curve)

                val_curve = list(trial.user_attrs.get("val_loss_curve", []))
                val_curve.append(float(val_loss))
                trial.set_user_attr("val_loss_curve", val_curve)

                if val_unseen_metrics is not None:
                    val_unseen_curve = list(trial.user_attrs.get("val_unseen_loss_curve", []))
                    val_unseen_curve.append(float(val_unseen_metrics["loss"]))
                    trial.set_user_attr("val_unseen_loss_curve", val_unseen_curve)

                if trial.should_prune():
                    if use_wandb:
                        wandb.finish()
                    raise optuna.TrialPruned(
                        f"Pruned at epoch {epoch} (val_loss={val_loss:.4f})"
                    )

        # Periodic checkpoint
        if epoch % cfg.checkpoint.save_every == 0:
            torch.save(
                {
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg":       OmegaConf.to_container(cfg, resolve=True),
                },
                ckpt_dir / f"epoch_{epoch:04d}.pt",
            )

    if use_wandb:
        wandb.finish()

    report_to = OmegaConf.select(cfg, "report_to", default=None)
    if report_to:
        Path(report_to).write_text(f"{float(best_val_loss)}\n")

    return best_val_loss


# ── Hydra entry point ─────────────────────────────────────────────────────────

@hydra.main(
    config_path="experiments/cfgs",
    config_name="train",
    version_base="1.3",
)
def main(cfg: DictConfig) -> float | None:
    """Train the model.  Returns best val loss so Optuna can minimise it."""
    return run_training(cfg)


if __name__ == "__main__":
    main()
