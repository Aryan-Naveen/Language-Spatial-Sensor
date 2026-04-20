"""Optuna hyperparameter sweep driver for LSS on a multi-GPU instance.

Architecture
------------
One `optuna_sweep.py` process is spawned per GPU (see `train_lss_sagemaker.py`).
Each process pins itself to a single GPU via `CUDA_VISIBLE_DEVICES` and runs
`study.optimize(..., n_trials=PER_WORKER)`. All workers share a single SQLite
study at `${OPTUNA_DB_PATH}` (typically on the SageMaker checkpoint volume,
which syncs to S3 continuously).

The objective trains the LSS model via the existing `run_training(cfg)` entry
point in `train.py` after overriding the sampled hyperparameters onto the
Hydra config. It returns the best `val_seen/loss` to Optuna.

Callers set these env vars:
    OPTUNA_DB_PATH        absolute path to optuna.db (shared across workers)
    OPTUNA_STUDY_NAME     study name (same for every worker)
    OPTUNA_N_TRIALS       trials per worker
    OPTUNA_EPOCHS         epochs per trial (short budget recommended, e.g. 5–8)
    OPTUNA_BATCH_SIZE     per-GPU batch size (default 128)
    OPTUNA_NUM_WORKERS    DataLoader workers per trial (default 4)
    OPTUNA_CACHE_DIR      path to the LSS tensor cache (train/, val_seen/, …)
    CUDA_VISIBLE_DEVICES  set by the launcher; each worker sees exactly one GPU
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import optuna
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


# ── Hyperparameter search space ───────────────────────────────────────────────

def _suggest_hparams(trial: optuna.Trial) -> dict:
    """Sample one hyperparameter configuration. Edit here to tune the sweep."""

    # Categorical features requested by the user.
    model = {
        "pairwise_rel_type":         trial.suggest_categorical(
            "pairwise_rel_type", ["mlp", "center", "vertical_bottom"]),
        "use_anchor_centric_coords": trial.suggest_categorical(
            "use_anchor_centric_coords", [True, False]),
        "pooling_type":              trial.suggest_categorical(
            "pooling_type", ["mean", "max", "attention", "query_token"]),
        "use_film":                  trial.suggest_categorical(
            "use_film", [True, False]),
        "condition_spatial_on_text": trial.suggest_categorical(
            "condition_spatial_on_text", [True, False]),
        # Only CLS-token-friendly BERT variants (you are pooling CLS only).
        "text_model":                trial.suggest_categorical("text_model", [
            "bert-base-uncased",
            "bert-large-uncased",
            "distilbert-base-uncased",
            "roberta-base",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ]),

        # Network size.
        "hidden_dim":          trial.suggest_categorical("hidden_dim", [128, 256, 384, 512]),
        "num_fusion_layers":   trial.suggest_int("num_fusion_layers", 2, 5),
        "num_spatial_layers":  trial.suggest_int("num_spatial_layers", 1, 4),
        "num_heads":           trial.suggest_categorical("num_heads", [4, 8, 16]),
        "ffn_dim":             trial.suggest_categorical("ffn_dim", [512, 1024, 2048]),
        "dropout":             trial.suggest_float("dropout", 0.1, 0.4),
        "head_dropout":        trial.suggest_float("head_dropout", 0.0, 0.3),
    }

    training = {
        "lr":                       trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay":             trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "bert_lr_scale":            trial.suggest_categorical("bert_lr_scale", [0.0, 0.1]),
        "warmup_steps":             trial.suggest_int("warmup_steps", 0, 2000, step=250),
        "center_nll_lambda_mahal":  trial.suggest_float("center_nll_lambda_mahal", 0.0, 1.0),
        "center_nll_lambda_vol":    trial.suggest_float("center_nll_lambda_vol", 0.0, 0.5),
    }

    return {"model": model, "training": training}


# ── Objective ─────────────────────────────────────────────────────────────────

def _apply_overrides(cfg, suggestions: dict, fixed_overrides: dict):
    """Merge sampled + fixed overrides into the Hydra config in-place."""
    for section, values in suggestions.items():
        for k, v in values.items():
            OmegaConf.update(cfg, f"{section}.{k}", v, merge=False)
    for dotted, v in fixed_overrides.items():
        OmegaConf.update(cfg, dotted, v, merge=False)


def _make_objective(cfg_dir: Path, fixed_overrides: dict):
    """Build an Optuna objective that trains LSS once per trial."""
    # Import run_training lazily inside the objective so initialization errors
    # show up as a trial-level exception rather than at module import time.
    def objective(trial: optuna.Trial) -> float:
        from train import run_training  # type: ignore

        suggestions = _suggest_hparams(trial)

        # Hydra's `initialize_config_dir` must use an absolute path.
        with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.3"):
            cfg = compose(config_name="train")

        _apply_overrides(cfg, suggestions, fixed_overrides)

        # Give each trial its own checkpoint subdir (same volume as optuna.db).
        trial_ckpt = Path(fixed_overrides["checkpoint.dir"]) / f"trial_{trial.number:04d}"
        OmegaConf.update(cfg, "checkpoint.dir", str(trial_ckpt), merge=False)
        trial_ckpt.mkdir(parents=True, exist_ok=True)

        try:
            best_val_loss = run_training(cfg, trial=trial)
        except optuna.TrialPruned:
            # Let Optuna record the trial as pruned (not failed).
            raise
        except Exception as exc:
            print(f"[trial {trial.number}] FAILED: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()
            # Mark as failed so Optuna can continue; return +inf sentinel.
            return float("inf")

        return float(best_val_loss)

    return objective


# ── Entry point (one process per GPU) ─────────────────────────────────────────

def main() -> None:
    db_path    = os.environ["OPTUNA_DB_PATH"]
    study_name = os.environ["OPTUNA_STUDY_NAME"]
    n_trials   = int(os.environ["OPTUNA_N_TRIALS"])
    epochs     = int(os.environ.get("OPTUNA_EPOCHS", "6"))
    batch_size = int(os.environ.get("OPTUNA_BATCH_SIZE", "128"))
    num_workers = int(os.environ.get("OPTUNA_NUM_WORKERS", "4"))
    cache_dir  = os.environ["OPTUNA_CACHE_DIR"]

    # Workers see one GPU each — keep DataLoader workers modest to share RAM.
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    repo_root = Path(__file__).resolve().parent.parent.parent  # lss/
    cfg_dir   = repo_root / "experiments" / "cfgs"
    sys.path.insert(0, str(repo_root))

    # Fixed, non-searched overrides for every trial.
    fixed_overrides = {
        "training.epochs":      epochs,
        "training.batch_size":  batch_size,
        "training.num_workers": num_workers,
        "training.val_every":   1,
        # No W&B — Optuna DB is the only store of results.
        "wandb.enabled":        False,
        # Disable BEV visualisation so trials don't touch raw VLA-3D scenes.
        "viz.plot_every":       10**9,
        # Point training at the mounted S3 cache.
        "cache.dir":            cache_dir,
        # Trial-specific checkpoint dir — overridden inside the objective.
        "checkpoint.dir":       str(Path(db_path).parent / "trial_ckpts"),
        # Small save_every so periodic checkpoints don't bloat the volume.
        "checkpoint.save_every": 10**9,
    }

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        heartbeat_interval=60,
        grace_period=120,
        # SQLite + many concurrent writers needs WAL mode + longer timeout.
        engine_kwargs={
            "connect_args": {"timeout": 60.0, "check_same_thread": False},
        },
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )

    print(
        f"[gpu={gpu_id}] study={study_name!r} db={db_path!r} "
        f"n_trials={n_trials} epochs={epochs} bs={batch_size}",
        flush=True,
    )

    study.optimize(
        _make_objective(cfg_dir, fixed_overrides),
        n_trials=n_trials,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    print(f"[gpu={gpu_id}] finished. best_value={study.best_value:.4f}", flush=True)


if __name__ == "__main__":
    main()
