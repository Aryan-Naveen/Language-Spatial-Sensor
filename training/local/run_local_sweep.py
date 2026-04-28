#!/usr/bin/env python3
"""Minimal local Optuna sweep.

Each trial runs in its own subprocess (fresh Python, fresh CUDA context) —
the only reliable way to avoid torch/CUDA state accumulation crashes across
many varied model topologies and text backbones in one long-lived process.

Usage::

    python -m training.local.run_local_sweep --stage1 --epochs 30 --n-trials 50
    python -m training.local.run_local_sweep --study-name <name>   # resume

Study state persists in ./optuna.db. Inspect with::

    optuna-dashboard sqlite:///optuna.db
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import optuna


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent  # lss/


def _suggest(trial: optuna.Trial) -> dict[str, object]:
    """Sample one configuration. Returns Hydra dotted-key overrides."""
    pairwise = trial.suggest_categorical(
        "pairwise_rel_type",
        ["mlp", "center", "vertical_bottom", "topological", "geometric_algebra"],
    )
    if pairwise == "mlp":
        spatial_dim = 12
    elif pairwise in ("topological", "geometric_algebra"):
        spatial_dim = 7
    else:
        spatial_dim = trial.suggest_categorical("spatial_relation_dim", [1, 4, 5])

    cond_text = trial.suggest_categorical("condition_spatial_on_text", [True, False])
    cond_type = (
        trial.suggest_categorical(
            "spatial_conditioning_type", ["film", "gate", "film_gate"]
        )
        if cond_text
        else "film"
    )

    return {
        "model.pairwise_rel_type": pairwise,
        "model.spatial_relation_dim": spatial_dim,
        "model.use_anchor_centric_coords": trial.suggest_categorical(
            "use_anchor_centric_coords", [True, False]
        ),
        "model.pooling_type": trial.suggest_categorical(
            "pooling_type", ["mean", "max", "attention", "query_token"]
        ),
        "model.use_film": trial.suggest_categorical("use_film", [True, False]),
        "model.condition_spatial_on_text": cond_text,
        "model.spatial_conditioning_type": cond_type,
        "model.text_model": trial.suggest_categorical(
            "text_model",
            [
                "bert-base-uncased",
                "distilbert-base-uncased",
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
            ],
        ),
        "model.hidden_dim": trial.suggest_categorical(
            "hidden_dim", [128, 256, 384, 512]
        ),
        "model.num_fusion_layers": trial.suggest_int("num_fusion_layers", 2, 5),
        "model.num_spatial_layers": trial.suggest_int("num_spatial_layers", 1, 4),
        "model.num_heads": trial.suggest_categorical("num_heads", [4, 8, 16]),
        "model.ffn_dim": trial.suggest_categorical("ffn_dim", [512, 1024, 2048]),
        "model.dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "model.head_dropout": trial.suggest_float("head_dropout", 0.0, 0.3),
        "training.lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "training.weight_decay": trial.suggest_float(
            "weight_decay", 1e-4, 1e-1, log=True
        ),
        "training.warmup_steps": trial.suggest_int(
            "warmup_steps", 0, 2000, step=250
        ),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--train-steps-per-epoch", type=int, default=0)
    p.add_argument("--val-steps", type=int, default=0)
    p.add_argument(
        "--stage1",
        action="store_true",
        help="Shortcut: --train-steps-per-epoch 500 --val-steps 50",
    )
    p.add_argument("--cache-dir", type=Path, default=_repo_root() / "cache_shards")
    p.add_argument("--db", type=Path, default=_repo_root() / "optuna.db")
    p.add_argument(
        "--study-name",
        type=str,
        default=f"lss_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Reuse an existing name to resume a prior sweep.",
    )
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.stage1:
        args.train_steps_per_epoch = args.train_steps_per_epoch or 500
        args.val_steps = args.val_steps or 50

    root = _repo_root()
    cache_dir = args.cache_dir.resolve()
    if not (cache_dir / "train").is_dir():
        raise SystemExit(f"Cache not found under {cache_dir}")

    db_path = args.db.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_root = root / "checkpoints" / args.study_name
    ckpt_root.mkdir(parents=True, exist_ok=True)

    fixed = {
        "training.epochs": args.epochs,
        "training.batch_size": args.batch_size,
        "training.num_workers": args.num_workers,
        "training.max_train_steps_per_epoch": args.train_steps_per_epoch,
        "training.max_val_steps": args.val_steps,
        "wandb.enabled": False,
        "viz.plot_every": 10**9,
        "cache.dir": str(cache_dir),
        "checkpoint.save_every": 10**9,
    }

    print(f"study:   {args.study_name}")
    print(f"db:      {db_path}")
    print(f"trials:  {args.n_trials}  epochs/trial: {args.epochs}")
    print()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{db_path}",
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=42),
    )

    def objective(trial: optuna.Trial) -> float:
        trial_ckpt = ckpt_root / f"trial_{trial.number:04d}"
        trial_ckpt.mkdir(parents=True, exist_ok=True)
        report_path = trial_ckpt / "val_loss.txt"

        overrides = _suggest(trial)
        overrides.update(fixed)
        overrides["checkpoint.dir"] = str(trial_ckpt)
        overrides["report_to"] = str(report_path)

        cli = [f"{k}={v}" for k, v in overrides.items()]
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": f"{root}:{os.environ.get('PYTHONPATH', '').strip(':')}",
        }

        rc = subprocess.call(
            [sys.executable, "-u", "train.py", *cli],
            cwd=str(root),
            env=env,
        )
        if rc != 0 or not report_path.exists():
            print(f"[trial {trial.number}] failed (rc={rc})", flush=True)
            return float("inf")
        return float(report_path.read_text().strip())

    study.optimize(objective, n_trials=args.n_trials)

    print()
    print(f"best value:  {study.best_value:.4f}")
    print(f"best params: {study.best_params}")


if __name__ == "__main__":
    main()
