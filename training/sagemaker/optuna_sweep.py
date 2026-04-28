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

import faulthandler
import os
import sqlite3
import sys
import threading
import time
import traceback
from pathlib import Path

# Enable BEFORE heavy imports so a hang during `import optuna`/`import hydra`
# or CUDA init still gets a stack trace dumped to stderr every 5 minutes.
# Without this, a silent import-time block produces zero bytes of log output.
faulthandler.enable()
faulthandler.dump_traceback_later(timeout=300, repeat=True)

_gpu_id_env = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
print(f"[gpu={_gpu_id_env}] worker booted — importing deps", flush=True)

import optuna
print(f"[gpu={_gpu_id_env}] imported optuna {optuna.__version__}", flush=True)
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
print(f"[gpu={_gpu_id_env}] imported hydra + omegaconf", flush=True)


# ── Hyperparameter search space ───────────────────────────────────────────────

def _suggest_hparams(trial: optuna.Trial) -> dict:
    """Sample one hyperparameter configuration. Edit here to tune the sweep."""

    # `pairwise_rel_type` constrains `spatial_relation_dim`:
    #   "mlp"                          → always 12 (concat of two 6-dim bboxes)
    #   "center" / "vertical_bottom"   → 1, 4, or 5 (see calc_pairwise_locs)
    #   "topological" / "geometric_algebra" → always 7
    pairwise_rel_type = trial.suggest_categorical(
        "pairwise_rel_type",
        ["mlp", "center", "vertical_bottom", "topological", "geometric_algebra"],
    )
    if pairwise_rel_type == "mlp":
        spatial_relation_dim = 12
    elif pairwise_rel_type in ("topological", "geometric_algebra"):
        spatial_relation_dim = 7
    else:
        spatial_relation_dim = trial.suggest_categorical(
            "spatial_relation_dim", [1, 4, 5])

    # Text conditioning on the spatial MLP. Only sample the conditioning kind
    # when conditioning is enabled — keeps the search space small.
    condition_spatial_on_text = trial.suggest_categorical(
        "condition_spatial_on_text", [True, False])
    if condition_spatial_on_text:
        spatial_conditioning_type = trial.suggest_categorical(
            "spatial_conditioning_type", ["film", "gate", "film_gate"])
    else:
        spatial_conditioning_type = "film"  # unused; kept for config completeness

    # Categorical features requested by the user.
    model = {
        "pairwise_rel_type":         pairwise_rel_type,
        "spatial_relation_dim":      spatial_relation_dim,
        "use_anchor_centric_coords": trial.suggest_categorical(
            "use_anchor_centric_coords", [True, False]),
        "pooling_type":              trial.suggest_categorical(
            "pooling_type", ["mean", "max", "attention", "query_token"]),
        "use_film":                  trial.suggest_categorical(
            "use_film", [True, False]),
        "condition_spatial_on_text": condition_spatial_on_text,
        "spatial_conditioning_type": spatial_conditioning_type,
        # Only CLS-token-friendly BERT variants (you are pooling CLS only).
        "text_model":                trial.suggest_categorical("text_model", [
            "bert-base-uncased",
            "distilbert-base-uncased",
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
        "warmup_steps":             trial.suggest_int("warmup_steps", 0, 2000, step=250),
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


_DATALOADER_CRASH_MARKERS = (
    # Canonical PyTorch message when a DataLoader worker dies.
    "DataLoader worker (pid",
    "worker is killed by signal",
    "exited unexpectedly",
    # FastFile / mmap-level faults sometimes surface as the OS signal.
    "Segmentation fault",
    "Bus error",
)

# If any of these appear in the exception message, the CUDA context on this
# worker is corrupted and every subsequent trial will fail the same way
# ("misaligned address", etc) — sticky until the process dies.  We kill the
# worker with os._exit so SageMaker marks it dead and the other 3 workers
# stay healthy, instead of one poisoned worker chewing through the trial
# budget with inf results.
_CUDA_POISONED_MARKERS = (
    "CUDA error: misaligned address",
    "CUDA error: an illegal memory access",
    "CUDA error: unspecified launch failure",
    "CUDA error: device-side assert triggered",
    "CUDA error: invalid configuration argument",
    "CUDA driver error",
)


def _is_dataloader_crash(exc: BaseException) -> bool:
    """True if *exc* looks like a DataLoader-worker-died RuntimeError.

    These are almost always transient and retrying with ``num_workers=0``
    (synchronous loading) sidesteps the crash entirely.
    """
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc)
    return any(marker in msg for marker in _DATALOADER_CRASH_MARKERS)


def _is_cuda_context_poisoned(exc: BaseException) -> bool:
    """True if *exc* indicates an irrecoverable CUDA context corruption."""
    msg = str(exc)
    return any(marker in msg for marker in _CUDA_POISONED_MARKERS)


def _make_objective(cfg_dir: Path, fixed_overrides: dict):
    """Build an Optuna objective that trains LSS once per trial."""
    # Import run_training lazily inside the objective so initialization errors
    # show up as a trial-level exception rather than at module import time.
    def objective(trial: optuna.Trial) -> float:
        import gc
        import torch

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

        # Attempt 1: run with the original (worker-heavy) config.
        # Attempt 2: on a DataLoader worker crash, fall back to num_workers=0.
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                best_val_loss = run_training(cfg, trial=trial)
                if attempt > 1:
                    print(
                        f"[trial {trial.number}] succeeded on attempt {attempt} "
                        f"(num_workers={int(cfg.training.num_workers)})",
                        flush=True,
                    )
                return float(best_val_loss)

            except optuna.TrialPruned:
                raise

            except Exception as exc:
                # CUDA context corruption is sticky: every future trial on this
                # worker will fail the same way.  Kill the worker hard so the
                # other 3 workers on the node keep running — don't chew through
                # the trial budget with inf results.
                if _is_cuda_context_poisoned(exc):
                    print(
                        f"[trial {trial.number}] FATAL CUDA error on this worker: {exc!s}. "
                        f"Killing worker so other GPUs stay productive.",
                        file=sys.stderr,
                        flush=True,
                    )
                    traceback.print_exc()
                    sys.stderr.flush()
                    sys.stdout.flush()
                    os._exit(42)  # bypasses Optuna teardown; SageMaker sees a dead worker

                if attempt < max_attempts and _is_dataloader_crash(exc):
                    print(
                        f"[trial {trial.number}] attempt {attempt} hit DataLoader "
                        f"crash: {exc!s}. Retrying with num_workers=0.",
                        file=sys.stderr,
                        flush=True,
                    )
                    # Release workers / GPU memory before the retry.
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    OmegaConf.update(cfg, "training.num_workers", 0, merge=False)
                    continue

                print(
                    f"[trial {trial.number}] FAILED (attempt {attempt}/{max_attempts}): {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc()
                return float("inf")

        # Unreachable — loop always either returns or raises.
        return float("inf")

    return objective


# ── Periodic DB snapshot ─────────────────────────────────────────────────────
#
# SageMaker's `aws s3 sync` can't reliably re-upload a live SQLite file:
# connections are held open, the main .db is constantly being rewritten, and
# sync captures a single point-in-time copy shortly after create_study() and
# then skips the file forever. Fix: periodically call sqlite3's online
# `.backup()` API to produce a closed, consistent copy at a different path.
# That path IS sync-friendly, so S3 always has a recent view of study state.


def _snapshot_loop(db_path: str, snapshot_path: str, interval: int = 60) -> None:
    """Daemon: every ``interval`` seconds, write an atomic DB snapshot."""
    tmp_path = snapshot_path + ".tmp"
    while True:
        time.sleep(interval)
        try:
            # `.backup()` uses SQLite's online backup API — acquires a shared
            # lock on the source, streams pages to the dest, releases. Safe to
            # run while the study is actively being written.
            src = sqlite3.connect(db_path)
            dst = sqlite3.connect(tmp_path)
            try:
                src.backup(dst)
            finally:
                dst.close()
                src.close()
            os.replace(tmp_path, snapshot_path)  # atomic on POSIX
        except Exception as e:
            # Never kill the worker over a snapshot hiccup; log and retry.
            print(f"[snapshot] failed: {e}", flush=True)


def _start_snapshot_daemon(db_path: str) -> None:
    """Launch the snapshot daemon once, from worker 0 only."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "?") != "0":
        return
    snapshot_path = str(Path(db_path).with_name("optuna_snapshot.db"))
    t = threading.Thread(
        target=_snapshot_loop,
        args=(db_path, snapshot_path),
        daemon=True,
        name="optuna-snapshot",
    )
    t.start()
    print(f"[snapshot] daemon started — {snapshot_path} refreshed every 60 s", flush=True)


# ── Entry point (one process per GPU) ─────────────────────────────────────────

def main() -> None:
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[gpu={gpu_id}] entered main()", flush=True)
    db_path    = os.environ["OPTUNA_DB_PATH"]
    study_name = os.environ["OPTUNA_STUDY_NAME"]
    n_trials   = int(os.environ["OPTUNA_N_TRIALS"])
    epochs     = int(os.environ.get("OPTUNA_EPOCHS", "6"))
    batch_size = int(os.environ.get("OPTUNA_BATCH_SIZE", "128"))
    num_workers = int(os.environ.get("OPTUNA_NUM_WORKERS", "4"))
    cache_dir  = os.environ["OPTUNA_CACHE_DIR"]
    # Optional subset-per-epoch caps for cheap Stage-1 sweeps.
    max_train_steps = int(os.environ.get("OPTUNA_MAX_TRAIN_STEPS", "0"))
    max_val_steps   = int(os.environ.get("OPTUNA_MAX_VAL_STEPS", "0"))

    repo_root = Path(__file__).resolve().parent.parent.parent  # lss/
    cfg_dir   = repo_root / "experiments" / "cfgs"
    sys.path.insert(0, str(repo_root))

    # Fixed, non-searched overrides for every trial.
    fixed_overrides = {
        "training.epochs":      epochs,
        "training.batch_size":  batch_size,
        "training.num_workers": num_workers,
        "training.val_every":   1,
        # Subset caps: 0 = no cap (full epoch). For Stage-1 architecture
        # sweeps set these to small numbers so each trial finishes in minutes.
        "training.max_train_steps_per_epoch": max_train_steps,
        "training.max_val_steps":             max_val_steps,
        # Sweep runs many architectures in one process; torch.compile's
        # inductor state accumulates across trials and SIGSEGVs.
        "training.disable_compile":           True,
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

    print(f"[gpu={gpu_id}] opening RDBStorage at {db_path}", flush=True)
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        heartbeat_interval=60,
        grace_period=120,
        # SQLite + many concurrent writers needs WAL mode + longer timeout.
        engine_kwargs={
            "connect_args": {"timeout": 60.0, "check_same_thread": False},
        },
    )

    # Per-worker seed offset: each worker process constructs its own
    # TPESampler, and TPE's first `n_startup_trials` draws come from the
    # sampler's seeded random fallback. Identical seeds across workers →
    # identical first trials, wasting 4× the compute. Offset by GPU id so
    # each worker explores a different slice of the search space.
    try:
        sampler_seed = 42 + int(gpu_id)
    except (TypeError, ValueError):
        sampler_seed = 42

    print(
        f"[gpu={gpu_id}] calling create_study(study_name={study_name!r}, "
        f"sampler_seed={sampler_seed})",
        flush=True,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=sampler_seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )

    print(
        f"[gpu={gpu_id}] study={study_name!r} db={db_path!r} "
        f"n_trials={n_trials} epochs={epochs} bs={batch_size}",
        flush=True,
    )

    _start_snapshot_daemon(db_path)

    study.optimize(
        _make_objective(cfg_dir, fixed_overrides),
        n_trials=n_trials,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    print(f"[gpu={gpu_id}] finished. best_value={study.best_value:.4f}", flush=True)


if __name__ == "__main__":
    main()
