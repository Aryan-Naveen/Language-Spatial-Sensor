#!/usr/bin/env python3
"""Submit the LSS Optuna hyperparameter sweep to Amazon SageMaker.

Run from your laptop with AWS credentials configured:
    python training/sagemaker/submit_lss_sweep.py

BEFORE RUNNING
--------------
1. pip install -r training/sagemaker/submit_requirements.txt
2. aws configure  (or use an IAM role with sagemaker + s3 access)
3. Upload the tensor cache to S3:
       aws s3 sync cache/ s3://{BUCKET_NAME}/{DATA_PREFIX}/
   Expected S3 layout:
       {DATA_PREFIX}/train/       (manifest.json + *.pt)
       {DATA_PREFIX}/val_seen/
       {DATA_PREFIX}/val_unseen/
4. Edit BUCKET_NAME / DATA_PREFIX / SAGEMAKER_ROLE below (or set env vars).
5. python training/sagemaker/submit_lss_sweep.py

OPTUNA DB ACCESS
----------------
The sweep writes `optuna.db` to the SageMaker checkpoint volume, which is
continuously synced to `s3://{BUCKET_NAME}/{CKPT_PREFIX}/{JOB}/optuna.db`.
Pull it at any time with:
    aws s3 cp s3://{BUCKET_NAME}/{CKPT_PREFIX}/<JOB>/optuna.db ./optuna.db

ENV OVERRIDES
-------------
LSS_BUCKET            S3 bucket name
LSS_DATA_PREFIX       S3 prefix for the tensor cache
LSS_CKPT_PREFIX       S3 prefix for the checkpoint volume (holds optuna.db)
LSS_SAGEMAKER_ROLE    IAM execution role ARN
LSS_INSTANCE_TYPE     SageMaker instance (default ml.g6e.8xlarge = 4×L40S)
LSS_TRAINING_IMAGE    Full ECR URI override (skip DLC auto-resolve)
LSS_MAX_RUNTIME       Wall-clock cap in seconds (default 259200 = 72h)
LSS_N_TRIALS          Total trials across all workers (default 64)
LSS_TRIAL_EPOCHS      Epochs per trial (default 6)
LSS_TRIAL_BATCH_SIZE  Per-GPU batch size per trial (default 128)
"""
from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

# Same sys.path hygiene as LangSteer: prevent training/ from shadowing `sagemaker`.
_repo_root = Path(__file__).resolve().parent.parent.parent  # lss/
for _rm in (str(_repo_root), str(_repo_root / "training"), ".", ""):
    while _rm in sys.path:
        sys.path.remove(_rm)

# Optional collections compat shim (Python 3.10+) — reuse LangSteer's if present.
_compat = _repo_root.parent / "LangSteer" / "utils" / "collections_compat.py"
if _compat.exists():
    _spec = importlib.util.spec_from_file_location("_collections_compat", _compat)
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)

try:
    from sagemaker.train.model_trainer import ModelTrainer
    from sagemaker.train.configs import (
        Compute,
        InputData,
        OutputDataConfig,
        CheckpointConfig,
        SourceCode,
        StoppingCondition,
    )
    from sagemaker.core import image_uris
    from sagemaker.core.helper.session_helper import Session
except ImportError as exc:
    raise ImportError(
        "Install SageMaker SDK: pip install -r training/sagemaker/submit_requirements.txt"
    ) from exc


# =============================================================================
# CONFIGURATION — edit here or override via env vars above
# =============================================================================

BUCKET_NAME    = "lang-map-lss"#os.environ.get("LSS_BUCKET",         "calvin-abcd-dataset-bucket")
DATA_PREFIX    = os.environ.get("LSS_DATA_PREFIX",    "lss/cache")
CKPT_PREFIX    = os.environ.get("LSS_CKPT_PREFIX",    "lss/sweeps")
SAGEMAKER_ROLE = os.environ.get("LSS_SAGEMAKER_ROLE",
                                "arn:aws:iam::317694661330:role/SageMakerExecutionRole")
INSTANCE_TYPE  = "ml.g6e.8xlarge"  # 4× L40S
TRAINING_IMAGE = os.environ.get("LSS_TRAINING_IMAGE", "").strip()
MAX_RUNTIME    = int(os.environ.get("LSS_MAX_RUNTIME", "259200"))        # 72h

# =============================================================================
# Sweep hyperparameters (all values are strings)
# =============================================================================

STUDY_NAME = f"lss_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

HYPERPARAMETERS = {
    "n_trials_total":  os.environ.get("LSS_N_TRIALS",       "64"),
    "epochs":          os.environ.get("LSS_TRIAL_EPOCHS",   "24"),
    "batch_size":      os.environ.get("LSS_TRIAL_BATCH_SIZE", "128"),
    "num_workers":     "4",
    "study_name":      STUDY_NAME,
    "nproc_per_node":  "",  # auto-detect
}

REPO_ROOT = _repo_root

IGNORE_PATTERNS = [
    ".git", "__pycache__", "*.pyc",
    ".env", ".venv", "venv",
    ".DS_Store",
    "cache",           # 468 GB — must stay out of the source bundle
    "outputs", "checkpoints", "wandb", "multirun",
    "*.ipynb", ".ipynb_checkpoints",
]


def main() -> None:
    sess   = Session()
    region = sess.boto_region_name
    stamp  = datetime.now().strftime("%Y%m%d%H%M%S")
    job    = f"lss-sweep-{stamp}"

    data_uri  = f"s3://{BUCKET_NAME}/{DATA_PREFIX}".rstrip("/") + "/"
    s3_output = f"s3://{BUCKET_NAME}/{CKPT_PREFIX.strip('/')}"       # model artifacts
    s3_ckpt   = f"{s3_output}/{job}"                                  # optuna.db lives here

    if TRAINING_IMAGE:
        training_image = TRAINING_IMAGE
    else:
        training_image = image_uris.retrieve(
            framework="pytorch", region=region,
            version="2.4.0", py_version="py311",
            instance_type=INSTANCE_TYPE, image_scope="training",
        )

    print(f"Role:              {SAGEMAKER_ROLE}")
    print(f"Region:            {region}")
    print(f"Image:             {training_image}")
    print(f"Instance:          {INSTANCE_TYPE}")
    print(f"Input cache (S3):  {data_uri}")
    print(f"Checkpoints (S3):  {s3_ckpt}/   ← optuna.db lands here")
    print(f"Artifacts (S3):    {s3_output}/{job}/output/model.tar.gz")
    print(f"Study name:        {STUDY_NAME}")
    print(f"Trials total:      {HYPERPARAMETERS['n_trials_total']}")
    print(f"Epochs / trial:    {HYPERPARAMETERS['epochs']}")

    source_code = SourceCode(
        source_dir=str(REPO_ROOT),
        entry_script="training/sagemaker/train_lss_sagemaker.py",
        requirements="training/sagemaker/requirements.txt",
        ignore_patterns=IGNORE_PATTERNS,
    )

    trainer = ModelTrainer(
        training_image=training_image,
        source_code=source_code,
        compute=Compute(instance_type=INSTANCE_TYPE, instance_count=1),
        role=SAGEMAKER_ROLE,
        sagemaker_session=sess,
        hyperparameters=HYPERPARAMETERS,
        base_job_name=job,
        output_data_config=OutputDataConfig(s3_output_path=s3_output),
        # Continuously syncs /opt/ml/checkpoints ↔ S3 — where optuna.db lives.
        checkpoint_config=CheckpointConfig(
            s3_uri=s3_ckpt,
            local_path="/opt/ml/checkpoints",
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=MAX_RUNTIME),
    )

    print("\nSubmitting sweep job...")
    trainer.train(
        input_data_config=[InputData(
            channel_name="cache",
            data_source=data_uri,
            # FastFile mode streams files on demand — essential for the 1.4M-file cache.
            input_mode="FastFile",
        )],
    )

    tj   = trainer._latest_training_job
    name = getattr(tj, "training_job_name", None) or str(tj)
    print("\n" + "=" * 72)
    print(f"Job submitted:   {name}")
    print(f"Optuna DB:       s3://{BUCKET_NAME}/{CKPT_PREFIX}/{name}/optuna.db")
    print(f"Pull with:       aws s3 cp s3://{BUCKET_NAME}/{CKPT_PREFIX}/{name}/optuna.db ./optuna.db")
    print(f"Console:         https://console.aws.amazon.com/sagemaker/home?region={region}#/training-jobs/{name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
