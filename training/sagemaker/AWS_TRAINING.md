# LSS — AWS SageMaker Optuna Hyperparameter Sweep

Runs an Optuna HPO sweep for the Language Spatial Sensor on **`ml.g6e.12xlarge`**
(4× NVIDIA L40S, 32 vCPU, 256 GB RAM) — the same instance used by LangSteer's
`submit_langsteer_training.py`. One Optuna worker per GPU, all writing to a
single `optuna.db`. No W&B.

## Architecture at a glance

```
SageMaker job (ml.g6e.12xlarge, 4× L40S)
└─ train_lss_sagemaker.py
   ├─ spawns 4 × optuna_sweep.py (one per GPU)
   │    ├─ samples hparams via TPE
   │    ├─ calls run_training(cfg) in train.py
   │    └─ returns best val_seen/loss to the shared study
   │
   └─ /opt/ml/checkpoints/optuna.db   ─────┐
                                            │ continuously synced
          s3://$BUCKET/$CKPT_PREFIX/$JOB/ ──┘  (every few seconds)
```

`CheckpointConfig` is SageMaker's native "keep this dir in sync with S3"
mechanism — **you can pull `optuna.db` at any time while the job is running.**

## Files in this directory

| File | Runs where | Purpose |
|------|------------|---------|
| `submit_lss_sweep.py`          | laptop    | Submits the SageMaker job |
| `submit_requirements.txt`      | laptop    | `sagemaker>=3.0`, `boto3` |
| `train_lss_sagemaker.py`       | container | Entry point — spawns workers |
| `optuna_sweep.py`              | container | One process per GPU; runs the study |
| `requirements.txt`             | container | Installed on top of the PyTorch 2.4 DLC |
| `AWS_TRAINING.md`              | —         | This doc |

## What gets searched

Edit the search space in [`optuna_sweep.py::_suggest_hparams`](optuna_sweep.py). Defaults:

**Categorical** (your list):
- `model.pairwise_rel_type` ∈ {mlp, center, vertical_bottom}
- `model.use_anchor_centric_coords` ∈ {True, False}
- `model.pooling_type` ∈ {mean, max, attention, query_token}
- `model.use_film` ∈ {True, False}
- `model.condition_spatial_on_text` ∈ {True, False}
- `model.text_model` ∈ {bert-base-uncased, bert-large-uncased, distilbert-base-uncased, roberta-base, sentence-transformers/all-MiniLM-L6-v2, sentence-transformers/all-mpnet-base-v2}

**Network size:**
- `model.hidden_dim` ∈ {128, 256, 384, 512}
- `model.num_fusion_layers` ∈ [2, 5]
- `model.num_spatial_layers` ∈ [1, 4]
- `model.num_heads` ∈ {4, 8, 16}
- `model.ffn_dim` ∈ {512, 1024, 2048}
- `model.dropout` ∈ [0.1, 0.4]
- `model.head_dropout` ∈ [0.0, 0.3]

**Optimisation:**
- `training.lr` ∈ [1e-5, 1e-3] (log)
- `training.weight_decay` ∈ [1e-4, 1e-1] (log)
- `training.bert_lr_scale` ∈ {0.0, 0.1}
- `training.warmup_steps` ∈ [0, 2000] (step 250)
- `training.center_nll_lambda_mahal` ∈ [0.0, 1.0]
- `training.center_nll_lambda_vol` ∈ [0.0, 0.5]

**Sampler:** TPE (multivariate, grouped). **Pruner:** MedianPruner.
**Objective:** minimise best `val_seen/loss` over the trial's epochs.

---

## 1 — Build and upload the tar-sharded cache to S3

The training code reads pre-tensorized samples from a local cache built by
[`scripts/preprocess.py`](../../scripts/preprocess.py). `preprocess.py` emits
one `.pt` file per sample (1.4M small files, ~412 GB for train). That layout
is fine on local NVMe but **catastrophic on SageMaker FastFile** — each
sample becomes one S3 GET, and training becomes network-latency-bound at
~2.7 s/step.

**Fix:** pack samples into ~1000-sample tar shards with
[`scripts/build_shards.py`](../../scripts/build_shards.py) before uploading.
One S3 GET per shard amortises network overhead across ~1000 samples and
gets per-step time to near-GPU-bound (~0.1 s/step). `train.py` auto-detects
the shard layout via a `shards.json` index, so there are no code toggles.

### a) Build shards locally (or on EC2)

```bash
# Run on the machine that already has cache/ built by preprocess.py.
python scripts/build_shards.py \
    --cache-dir cache \
    --out-dir   cache_shards
```

Output layout:

```
cache_shards/
├── train/       shard-000000.tar … shard-00136X.tar + shards.json   (~412 GB)
├── val_seen/    shard-000000.tar … + shards.json                    (~22 GB)
└── val_unseen/  shard-000000.tar … + shards.json                    (~35 GB)
```

Approx. runtime: ~20-40 min on a local NVMe machine (pure I/O copy, no
decode). Per-shard progress + ETA prints to stdout.

### b) Upload shards to S3

```bash
export BUCKET=lang-map-lss          # your bucket
aws configure set default.s3.max_concurrent_requests 64
aws configure set default.s3.max_queue_size 10000

aws s3 sync cache_shards/ s3://$BUCKET/lss/cache_shards/
```

Only ~1500 tar objects total (vs 1.4M `.pt` files) — this upload finishes
in ~30-60 min over a decent connection instead of many hours.

### c) Point the sweep at the sharded prefix

Set `LSS_DATA_PREFIX=lss/cache_shards` before submitting (or edit
`DATA_PREFIX` in [submit_lss_sweep.py](submit_lss_sweep.py)).

```bash
export LSS_DATA_PREFIX=lss/cache_shards
python training/sagemaker/submit_lss_sweep.py
```

### Why FastFile mode (still)

`submit_lss_sweep.py` keeps `input_mode="FastFile"`. With tar shards each
GET is ~500 MB of sequential bytes — exactly what FastFile's FUSE layer is
good at streaming. `File` mode would download all 468 GB to EBS up front
(~30 min stall every job) for no further speedup once shards are in use.

### Local runs (unchanged)

`train.py` auto-detects the cache format per split — if the split directory
has `shards.json`, it uses the streaming reader; otherwise it falls back to
the original `.pt` map-style dataset. Local smoke tests, BEV viz, and every
prior workflow continue to work against a `.pt` cache. Shard only when you
need to ship the cache to SageMaker.

---

## 2 — One-time AWS setup

```bash
# Laptop (one time):
pip install -r training/sagemaker/submit_requirements.txt
aws configure                    # or rely on an IAM role

# If you haven't used ml.g6e.12xlarge before, request quota in AWS Console →
# Service Quotas → SageMaker → "ml.g6e.12xlarge for training".
```

Verify the SageMaker execution role has: `AmazonSageMakerFullAccess` + S3
access to your bucket. LangSteer already uses
`arn:aws:iam::317694661330:role/SageMakerExecutionRole` — same account, same
role works here.

---

## 3 — Edit constants in `submit_lss_sweep.py` (or export env vars)

```python
BUCKET_NAME    = "calvin-abcd-dataset-bucket"                         # your bucket
DATA_PREFIX    = "lss/cache"                                         # where you synced the cache
CKPT_PREFIX    = "lss/sweeps"                                        # optuna.db lands here
SAGEMAKER_ROLE = "arn:aws:iam::317694661330:role/SageMakerExecutionRole"
INSTANCE_TYPE  = "ml.g6e.12xlarge"
```

Everything above is overridable via env vars (`LSS_BUCKET`, `LSS_DATA_PREFIX`,
`LSS_CKPT_PREFIX`, `LSS_SAGEMAKER_ROLE`, `LSS_INSTANCE_TYPE`,
`LSS_N_TRIALS`, `LSS_TRIAL_EPOCHS`, `LSS_TRIAL_BATCH_SIZE`,
`LSS_MAX_RUNTIME`).

---

## 4 — Submit the sweep

```bash
python training/sagemaker/submit_lss_sweep.py
```

You'll see:

```
Role:              arn:aws:iam::.../SageMakerExecutionRole
Image:             763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.4.0-gpu-py311-...
Instance:          ml.g6e.12xlarge
Input cache (S3):  s3://calvin-abcd-dataset-bucket/lss/cache/
Checkpoints (S3):  s3://calvin-abcd-dataset-bucket/lss/sweeps/lss-sweep-YYYYMMDDHHMMSS/
Study name:        lss_sweep_YYYYMMDD_HHMMSS
Trials total:      64
Epochs / trial:    6
...
Job submitted:   lss-sweep-YYYYMMDDHHMMSS
Optuna DB:       s3://calvin-abcd-dataset-bucket/lss/sweeps/lss-sweep-YYYYMMDDHHMMSS/optuna.db
Pull with:       aws s3 cp s3://calvin-abcd-dataset-bucket/lss/sweeps/lss-sweep-YYYYMMDDHHMMSS/optuna.db ./optuna.db
Console:         https://console.aws.amazon.com/sagemaker/...
```

The script returns immediately after submission. Monitor the job in the AWS
Console or via `aws sagemaker describe-training-job --training-job-name $JOB`.

---

## 5 — Download `optuna_snapshot.db` whenever you want

SageMaker's `CheckpointConfig` can't reliably re-upload a live SQLite file —
the workers hold the connection open and sync skips it after the first pass.
To work around this, `optuna_sweep.py` runs a daemon on worker 0 that calls
SQLite's online `.backup()` every 60 s to produce `optuna_snapshot.db` — an
atomic, closed, sync-friendly copy. Always pull the snapshot, not the live
DB.

```bash
STAMP=20260420180000       # from the submit output
aws s3 cp s3://$BUCKET/lss/sweeps/ckpts/$STAMP/optuna_snapshot.db ./optuna.db

# Inspect locally
optuna-dashboard sqlite:///./optuna.db
# or programmatically:
python - <<'EOF'
import optuna
study = optuna.load_study(study_name="lss_sweep_...", storage="sqlite:///./optuna.db")
print(f"trials: {len(study.trials)}, best: {study.best_value:.4f}")
print(study.best_params)
EOF
```

Worker logs are also synced (under `$CKPT_PREFIX/$JOB/logs/worker_gpu{0..3}.log`):

```bash
aws s3 cp s3://$BUCKET/lss/sweeps/$JOB/logs/worker_gpu0.log - | tail -200
```

---

## 6 — What happens inside the container

1. SageMaker runs `pip install -r training/sagemaker/requirements.txt` on top
   of the PyTorch 2.4 + Python 3.11 DLC.
2. `train_lss_sagemaker.py`:
   - Sets `PYTHONPATH` to the `lss/` root.
   - Finds the mounted cache at `$SM_CHANNEL_CACHE` (auto-detects the
     `train/`+`val_seen/`+`val_unseen/` layout).
   - Reads the number of GPUs and evenly splits `n_trials_total` across them.
   - Spawns `optuna_sweep.py` once per GPU with `CUDA_VISIBLE_DEVICES=i`.
3. Each `optuna_sweep.py` process:
   - Opens the shared SQLite study at `/opt/ml/checkpoints/optuna.db`
     (SQLite in WAL mode handles 4 concurrent writers fine).
   - Samples hparams, composes the Hydra config, overrides them onto
     `model.*` and `training.*`, points `cache.dir` at the mounted S3 cache,
     disables W&B + BEV viz, sets a small `epochs` budget.
   - Calls `run_training(cfg)` from `train.py`. The per-trial best val loss is
     returned to Optuna.
4. When all workers exit, the final `optuna.db` plus
   `/opt/ml/checkpoints/logs/` and `/opt/ml/model/` are in S3.

---

## 7 — Tuning the sweep

| What you want to change | Where |
|-------------------------|-------|
| Search space (add/remove hparams, change ranges) | [`optuna_sweep.py::_suggest_hparams`](optuna_sweep.py) |
| Sampler / pruner | bottom of `optuna_sweep.py::main` |
| Trials per worker, per-GPU batch, epochs | `HYPERPARAMETERS` in [`submit_lss_sweep.py`](submit_lss_sweep.py), or env vars `LSS_N_TRIALS` / `LSS_TRIAL_EPOCHS` / `LSS_TRIAL_BATCH_SIZE` |
| Instance type / GPU count | `LSS_INSTANCE_TYPE` env var; auto-detects GPU count |
| Wall-clock cap | `LSS_MAX_RUNTIME` (default 72 h) |

### Rough budget

At `epochs=6`, `batch_size=128`, ~1.4M train samples ⇒ ~65k steps/epoch
≈ ~25 min/epoch on one L40S (rough — depends on `text_model` and
`num_fusion_layers`). **One trial ≈ 2.5 h.** With 4 workers running in
parallel, 64 trials ≈ **~40 h wall-clock**. Adjust `epochs` or `batch_size`
up or down based on how many hparams you want to evaluate.

### Warm-starting / resuming

The study is created with `load_if_exists=True`. To resume after a crash (or
to add more trials) simply submit again with the **same `study_name`** and
the same `$CKPT_PREFIX/$JOB/` S3 path — or copy the existing `optuna.db`
into a new `local_path` for the next run.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ResourceLimitExceeded` on submit | AWS Console → Service Quotas → SageMaker → request `ml.g6e.12xlarge` quota |
| `FileNotFoundError: ... train/` | Verify `aws s3 ls s3://$BUCKET/lss/cache/train/` shows `manifest.json` + `*.pt`. Re-check `DATA_PREFIX`. |
| SQLite `database is locked` | Rare with 4 workers + WAL. If it persists, reduce `nproc_per_node` to 2 or swap `RDBStorage` for `JournalFileStorage` in `optuna_sweep.py`. |
| DataLoader workers OOM | Drop `OPTUNA_NUM_WORKERS` from 4 → 2. Per-GPU we have 256 GB / 4 GPUs = ~64 GB RAM headroom. |
| DLC image not found | Set `LSS_TRAINING_IMAGE` to a known ECR URI (check LangSteer's submit_langsteer_training.py output for a working URI). |
---

## Local smoke test (before you submit)

Run these on your laptop to catch bugs cheaper than a SageMaker job.

### a) Verify every `text_model` loads + pools correctly

```bash
pytest tests/test_text_encoder_variants.py -v
```

Parametrized across all six HF backbones the sweep can sample. Checks: auto
hidden-size derivation, per-family pooling (CLS vs. mean + L2 norm),
gradient flow, and tokenizer shape compatibility. First run downloads
~2 GB of HF weights into `$HF_HOME`; subsequent runs are instant.

### b) Run one real Optuna trial end-to-end

Requires the `cache/` directory built by `scripts/preprocess.py`. One trial
takes ~1–2 min on a single GPU with these tiny overrides:

```bash
cd /path/to/lss
OPTUNA_DB_PATH=/tmp/smoke_optuna.db \
OPTUNA_STUDY_NAME=smoke \
OPTUNA_N_TRIALS=1 \
OPTUNA_EPOCHS=1 \
OPTUNA_BATCH_SIZE=8 \
OPTUNA_NUM_WORKERS=0 \
OPTUNA_CACHE_DIR=$(pwd)/cache \
CUDA_VISIBLE_DEVICES=0 \
python training/sagemaker/optuna_sweep.py
```

Expected output ends with `[gpu=0] finished. best_value=...`. This exercises
the full path: hparam sampling → Hydra overrides → tokenizer + model build →
training loop → DB write. Delete `/tmp/smoke_optuna.db` between runs to start
fresh.

To force a specific `text_model` for the trial, temporarily edit the
relevant `trial.suggest_categorical` in `optuna_sweep.py::_suggest_hparams`
to a single-element list, or wrap it in `os.environ.get("SMOKE_TEXT_MODEL")`.

---

## Environment variable reference

| Var | Default | Used by |
|-----|---------|---------|
| `LSS_BUCKET` | `calvin-abcd-dataset-bucket` | submit |
| `LSS_DATA_PREFIX` | `lss/cache` | submit |
| `LSS_CKPT_PREFIX` | `lss/sweeps` | submit |
| `LSS_SAGEMAKER_ROLE` | see script | submit |
| `LSS_INSTANCE_TYPE` | `ml.g6e.12xlarge` | submit |
| `LSS_TRAINING_IMAGE` | *(DLC auto-resolve)* | submit |
| `LSS_MAX_RUNTIME` | `259200` (72h) | submit |
| `LSS_N_TRIALS` | `64` | submit → container |
| `LSS_TRIAL_EPOCHS` | `6` | submit → container |
| `LSS_TRIAL_BATCH_SIZE` | `128` | submit → container |
| `OPTUNA_DB_PATH` | `/opt/ml/checkpoints/optuna.db` | container (set by entry) |
| `OPTUNA_STUDY_NAME` | from `HYPERPARAMETERS['study_name']` | container |
