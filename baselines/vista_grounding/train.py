"""Training entry point for the 3D-ViSTA grounding baseline.

Single run::

    python baselines/vista_grounding/train.py

Override config values::

    python baselines/vista_grounding/train.py training.lr=3e-4 model.num_components=8

Disable W&B::

    python baselines/vista_grounding/train.py wandb.enabled=false
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# ── sys.path: make lss/ root importable ───────────────────────────────────────
_LSS_ROOT = Path(__file__).resolve().parents[2]
if str(_LSS_ROOT) not in sys.path:
    sys.path.insert(0, str(_LSS_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines.common.gmm import gmm_nll_loss
from baselines.common.evaluate import evaluate, generate_bev_plots
from baselines.vista_grounding.grounding_dataset import (
    ViSTAGroundingDataset,
    vista_collate_fn,
)
from baselines.vista_grounding.vista_model import ViSTAGroundingModel


# ── Utilities ─────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: DictConfig) -> ViSTAGroundingModel:
    m = cfg.model
    return ViSTAGroundingModel(
        hidden_size        = m.vista_hidden_size,
        num_components     = m.num_components,
        num_text_layers    = m.num_text_layers,
        num_spatial_layers = m.num_spatial_layers,
        spatial_dim        = m.spatial_dim,
        dim_loc            = m.dim_loc,
        gmm_hidden_dim     = m.gmm_hidden_dim,
        min_sigma          = m.min_sigma,
        dropout            = m.dropout,
        vista_ckpt_path    = m.vista_ckpt_path,
        freeze_layers      = m.freeze_layers,
    )


def build_optimizer(model: ViSTAGroundingModel, cfg: DictConfig):
    tr = cfg.training
    groups = model.parameter_groups(
        lr=tr.lr,
        lang_lr_scale=tr.lang_lr_scale,
        backbone_lr_scale=tr.backbone_lr_scale,
    )
    return torch.optim.AdamW(groups, weight_decay=tr.weight_decay)


def _resolve_dtype(precision: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def run_training(cfg: DictConfig) -> float:
    """Train the model and return the best val_seen loss (for Optuna)."""
    tr     = cfg.training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = _resolve_dtype(tr.precision)

    seed_everything(tr.seed)

    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    # ── Datasets ───────────────────────────────────────────────────────────────
    cache_dir = Path(cfg.cache.dir)

    train_ds    = ViSTAGroundingDataset(cache_dir / "train")
    val_seen_ds = ViSTAGroundingDataset(cache_dir / "val_seen")

    val_unseen_ds = None
    try:
        val_unseen_ds = ViSTAGroundingDataset(cache_dir / "val_unseen")
    except FileNotFoundError:
        print("[warn] val_unseen split not found — skipping unseen evaluation.")

    train_loader = DataLoader(
        train_ds,
        batch_size=tr.batch_size,
        shuffle=True,
        num_workers=tr.num_workers,
        collate_fn=vista_collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(tr.num_workers > 0),
    )
    val_loader = DataLoader(
        val_seen_ds,
        batch_size=tr.batch_size * 2,
        shuffle=False,
        num_workers=tr.num_workers,
        collate_fn=vista_collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(tr.num_workers > 0),
    )
    val_unseen_loader = None
    if val_unseen_ds is not None:
        val_unseen_loader = DataLoader(
            val_unseen_ds,
            batch_size=tr.batch_size * 2,
            shuffle=False,
            num_workers=tr.num_workers,
            collate_fn=vista_collate_fn,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(tr.num_workers > 0),
        )

    # ── Model + optimiser ──────────────────────────────────────────────────────
    model = build_model(cfg).to(device)

    if (
        device.type == "cuda"
        and getattr(cfg.training, "compile_model", False)
        and hasattr(torch, "compile")
    ):
        model = torch.compile(model)

    optimizer = build_optimizer(model, cfg)
    total_steps = len(train_loader) * tr.epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=tr.warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=(dtype == torch.float16))

    # ── W&B ────────────────────────────────────────────────────────────────────
    use_wandb = cfg.wandb.enabled and wandb is not None
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity") or None,
            name=cfg.wandb.get("name") or None,
            tags=list(cfg.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit="finish_previous",
        )

    # ── Checkpoint dir ─────────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.checkpoint.dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Loss kwargs (passed through to gmm_nll_loss) ───────────────────────────
    loss_kwargs = {
        "lambda_dist":  tr.gmm_lambda_dist,
        "entropy_reg":  tr.gmm_entropy_reg,
    }

    best_val_loss = float("inf")
    global_step   = 0

    # ── Training epochs ────────────────────────────────────────────────────────
    for epoch in range(1, tr.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{tr.epochs}", leave=False)
        for batch in pbar:
            batch = _to_device(batch, device)

            with torch.autocast(device_type=device.type, dtype=dtype):
                pred = model(**batch)
                loss = gmm_nll_loss(pred, batch["target_xyz_world"], **loss_kwargs)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), tr.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            loss_val    = loss.item()
            epoch_loss += loss_val
            global_step += 1

            pbar.set_postfix(loss=f"{loss_val:.4f}")

            if use_wandb and global_step % 50 == 0:
                wandb.log({
                    "train/loss": loss_val,
                    "train/lr":   scheduler.get_last_lr()[0],
                    "step":       global_step,
                })

        avg_train_loss = epoch_loss / max(len(train_loader), 1)

        # ── Validation ─────────────────────────────────────────────────────────
        if epoch % tr.val_every == 0:
            eval_kwargs = dict(
                device=device,
                autocast_dtype=dtype,
                loss_fn=gmm_nll_loss,
                loss_kwargs=loss_kwargs,
            )

            val_metrics = evaluate(model, val_loader, **eval_kwargs)
            val_loss    = val_metrics["loss"]

            val_unseen_metrics = None
            if val_unseen_loader is not None:
                val_unseen_metrics = evaluate(model, val_unseen_loader, **eval_kwargs)

            # Console
            print(
                f"[epoch {epoch:3d}] train={avg_train_loss:.4f}  "
                f"seen_loss={val_loss:.4f}  seen_acc={val_metrics['acc']:.3f}  "
                f"seen_dist={val_metrics['mean_dist']:.3f}m  "
                f"seen_mpm={val_metrics['gt_bbox_marginal_mass_mean']:.3f}  "
                f"seen_prod={val_metrics['gt_bbox_marginal_prod_mean']:.4f}"
            )
            if val_unseen_metrics is not None:
                print(
                    f"{'':>12}"
                    f"unseen_loss={val_unseen_metrics['loss']:.4f}  "
                    f"unseen_acc={val_unseen_metrics['acc']:.3f}  "
                    f"unseen_dist={val_unseen_metrics['mean_dist']:.3f}m  "
                    f"unseen_mpm={val_unseen_metrics['gt_bbox_marginal_mass_mean']:.3f}  "
                    f"unseen_prod={val_unseen_metrics['gt_bbox_marginal_prod_mean']:.4f}"
                )

            # W&B logging
            if use_wandb:
                log_dict: dict = {
                    "epoch":                epoch,
                    "train/epoch_loss":     avg_train_loss,
                    "val_seen/loss":        val_loss,
                    "val_seen/acc":         val_metrics["acc"],
                    "val_seen/mean_dist":   val_metrics["mean_dist"],
                    "val_seen/gt_bbox_marginal_mass_mean": val_metrics["gt_bbox_marginal_mass_mean"],
                    "val_seen/gt_bbox_marginal_prod_mean": val_metrics["gt_bbox_marginal_prod_mean"],
                }
                if val_unseen_metrics is not None:
                    log_dict.update({
                        "val_unseen/loss":      val_unseen_metrics["loss"],
                        "val_unseen/acc":       val_unseen_metrics["acc"],
                        "val_unseen/mean_dist": val_unseen_metrics["mean_dist"],
                        "val_unseen/gt_bbox_marginal_mass_mean": val_unseen_metrics["gt_bbox_marginal_mass_mean"],
                        "val_unseen/gt_bbox_marginal_prod_mean": val_unseen_metrics["gt_bbox_marginal_prod_mean"],
                    })
                wandb.log(log_dict)

            # BEV visualisation
            if epoch % cfg.viz.plot_every == 0:
                figs = generate_bev_plots(
                    model=model,
                    dataset=val_seen_ds,
                    device=device,
                    autocast_dtype=dtype,
                    collate_fn=vista_collate_fn,
                    n_scenes=cfg.viz.n_scenes,
                    n_samples=cfg.viz.n_samples,
                    viz_seed=cfg.viz.seed,
                    data_root=Path(cfg.data.data_root),
                    datasets=list(cfg.data.datasets),
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
                        "epoch":     epoch,
                        "model":     model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_loss":  val_loss,
                        "cfg":       OmegaConf.to_container(cfg, resolve=True),
                    },
                    ckpt_dir / "best.pt",
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

    return best_val_loss


# ── Hydra entry point ─────────────────────────────────────────────────────────

@hydra.main(
    config_path=".",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> float | None:
    return run_training(cfg)


if __name__ == "__main__":
    main()
