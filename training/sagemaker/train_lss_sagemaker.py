"""SageMaker training entry for the LSS Optuna hyperparameter sweep.

Invoked by SageMaker after it installs training/sagemaker/requirements.txt.
Spawns one `optuna_sweep.py` worker per GPU, each pinned to a unique
`CUDA_VISIBLE_DEVICES`, all writing to a single `optuna.db` on the checkpoint
volume. SageMaker continuously syncs `/opt/ml/checkpoints/` ↔
`${SM_CHECKPOINT_S3_URI}` so the DB can be pulled at any time.

Input channel `cache` must be mounted at SM_CHANNEL_CACHE with the structure:
    cache/
    ├── train/           (manifest.json + *.pt)
    ├── val_seen/
    └── val_unseen/

Hyperparameters merged from:
    1. DEFAULT_HYPERPARAMETERS (this file)
    2. CLI args (SageMaker `--key value` contract)
    3. SM_HPS env var (JSON dict injected by SageMaker)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent  # lss/


def _argv_to_hps(argv: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            body = a[2:]
            if "=" in body:
                k, v = body.split("=", 1)
                out[k.replace("-", "_")] = v
                i += 1
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[body.replace("-", "_")] = argv[i + 1]
                i += 2
            else:
                out[body.replace("-", "_")] = "1"
                i += 1
        else:
            i += 1
    return out


def _merge_hyperparameters() -> dict[str, str]:
    hps = dict(DEFAULT_HYPERPARAMETERS)
    hps.update(_argv_to_hps(sys.argv[1:]))
    raw = os.environ.get("SM_HPS", "{}")
    try:
        user = json.loads(raw)
        if isinstance(user, dict):
            hps.update({k: str(v) for k, v in user.items()})
    except json.JSONDecodeError:
        pass
    return hps


def _find_cache_root(channel: Path) -> Path:
    """Locate the directory that contains train/ val_seen/ val_unseen/."""
    candidates = [channel, channel / "cache"]
    for p in candidates:
        if (p / "train").is_dir() and (p / "val_seen").is_dir():
            return p
    raise FileNotFoundError(
        f"Could not find LSS tensor cache (train/ + val_seen/) under {channel}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def _num_gpus(hps: dict[str, str]) -> int:
    raw = hps.get("nproc_per_node", "")
    if raw.strip() not in ("", "0"):
        return max(1, int(raw))
    import torch
    return max(1, int(torch.cuda.device_count()))


DEFAULT_HYPERPARAMETERS: dict[str, str] = {
    "n_trials_total": "64",       # split evenly across GPUs
    "epochs":         "6",        # per trial — keep short for sweep breadth
    "batch_size":     "128",      # per GPU
    "num_workers":    "4",        # DataLoader workers per trial
    "study_name":     f"lss_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "nproc_per_node": "",         # empty = auto-detect
}


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    root = _repo_root()
    os.chdir(root)
    os.environ["PYTHONPATH"] = f"{root}:{os.environ.get('PYTHONPATH', '').strip(':')}"

    if "SM_CHANNEL_CACHE" not in os.environ:
        raise RuntimeError("SM_CHANNEL_CACHE is not set — add a 'cache' input channel.")

    hps = _merge_hyperparameters()

    # ── Locate mounted cache ──────────────────────────────────────────────────
    channel    = Path(os.environ["SM_CHANNEL_CACHE"])
    cache_root = _find_cache_root(channel)
    print(f"Cache root: {cache_root}", flush=True)

    # ── Register the lss package for imports (train.py uses local imports) ──
    subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, '.'); import train; print('train.py importable')"],
        check=True, cwd=str(root),
    )

    # ── Optuna DB lives on the SageMaker checkpoint volume (syncs to S3) ────
    # If CheckpointConfig is set, `/opt/ml/checkpoints` is kept in sync with
    # the configured S3 URI every ~seconds.
    ckpt_dir = Path(os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    db_path = ckpt_dir / "optuna.db"
    print(f"Optuna DB path (auto-synced to S3): {db_path}", flush=True)

    n_gpus  = _num_gpus(hps)
    n_total = int(hps["n_trials_total"])
    # Distribute trials evenly across workers (any remainder goes to worker 0).
    base = n_total // n_gpus
    extra = n_total - base * n_gpus
    per_worker = [base + (1 if i < extra else 0) for i in range(n_gpus)]
    print(f"Launching {n_gpus} workers: trials per worker = {per_worker}", flush=True)

    common_env = {
        **os.environ,
        "OPTUNA_DB_PATH":    str(db_path),
        "OPTUNA_STUDY_NAME": hps["study_name"],
        "OPTUNA_EPOCHS":     hps["epochs"],
        "OPTUNA_BATCH_SIZE": hps["batch_size"],
        "OPTUNA_NUM_WORKERS": hps["num_workers"],
        "OPTUNA_CACHE_DIR":  str(cache_root),
        "PYTHONUNBUFFERED":  "1",
    }

    procs: list[subprocess.Popen] = []
    sweep_script = root / "training" / "sagemaker" / "optuna_sweep.py"
    log_dir = ckpt_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for gpu_id in range(n_gpus):
        env = dict(common_env)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["OPTUNA_N_TRIALS"]      = str(per_worker[gpu_id])
        log_path = log_dir / f"worker_gpu{gpu_id}.log"
        logf = open(log_path, "ab", buffering=0)
        print(f"  → spawning worker gpu={gpu_id} log={log_path}", flush=True)
        procs.append(subprocess.Popen(
            [sys.executable, str(sweep_script)],
            env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=str(root),
        ))
        # Stagger a bit so all workers don't hit `create_study` simultaneously.
        time.sleep(2)

    failed = 0
    for gpu_id, p in enumerate(procs):
        rc = p.wait()
        print(f"[gpu={gpu_id}] exit={rc}", flush=True)
        if rc != 0:
            failed += 1

    print(f"\nSweep complete. workers_failed={failed}/{n_gpus}", flush=True)
    print(f"Final optuna.db: {db_path}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
