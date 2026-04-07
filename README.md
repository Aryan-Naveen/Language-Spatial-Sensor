# Language-Spatial-Sensor (LSS)

Language-grounded 3D spatial sensor that predicts a continuous probability distribution over target object locations given a natural-language query and a scene point cloud. The repository also contains a 3D-ViSTA finetuning baseline that outputs a Gaussian Mixture Model (GMM) over 3D positions.

All commands are run from the `lss/` root unless noted otherwise.

---

## Prerequisites

- Python 3.10+
- CUDA GPU (strongly recommended)
- VLA-3D dataset at a local path (default: `/home/aryannav/mit/data/VLA-3D/VLA-3D_dataset`)
- For the ViSTA baseline only: 3D-ViSTA pretrained checkpoint (`pretrain.pth`) and compiled PointNet++ CUDA extensions (see [ViSTA baseline](#4-finetune-3d-vista))

---

## 1. Preprocess data for the LSS model

The preprocessor builds a split cache of tensorized samples (CLIP embeddings, object bboxes, target labels) from the raw VLA-3D dataset.

```bash
python scripts/preprocess.py
```

Key overrides:

| Override | Default | Description |
|---|---|---|
| `data.data_root` | `/home/aryannav/mit/data/VLA-3D/VLA-3D_dataset` | Path to raw VLA-3D dataset |
| `data.datasets` | `[Unity]` | Datasets to include (any of `3RScan`, `ARKitScenes`, `HM3D`, `Matterport`, `Scannet`, `Unity`) |
| `cache.dir` | `cache` | Output directory for preprocessed tensors |
| `cache.force_rebuild` | `false` | Rebuild cache even if it already exists |
| `cache.num_workers` | `4` | Parallel workers |

Example with overrides:

```bash
python scripts/preprocess.py \
  data.data_root=/path/to/VLA-3D_dataset \
  data.datasets="[Scannet,Unity]" \
  cache.dir=/fast/ssd/lss_cache \
  cache.force_rebuild=true
```

**Output layout:**

```
cache/
├── train/
│   ├── manifest.json
│   └── <scene_id>_<idx>.pt
├── val_seen/
│   ├── manifest.json
│   └── <scene_id>_<idx>.pt
├── val_unseen/
│   ├── manifest.json
│   └── <scene_id>_<idx>.pt
└── clip_label_map.pt          # precomputed CLIP label embeddings
```

---

## 2. Train the LSS model

Training uses a Hydra config stack under `experiments/cfgs/`. The core configs are `train.yaml` (top-level), `data/vla3d.yaml`, and `model/lss.yaml`.

```bash
python train.py
```

Key overrides:

| Override | Default | Description |
|---|---|---|
| `cache.dir` | `cache` | Preprocessed cache from step 1 |
| `training.epochs` | `50` | Number of training epochs |
| `training.batch_size` | `64` | Batch size |
| `training.lr` | `1e-4` | Base learning rate |
| `training.bert_lr_scale` | `0.1` | LR multiplier for BERT encoder |
| `training.precision` | `bf16` | Mixed precision (`bf16` / `fp16` / `fp32`) |
| `model.num_spatial_layers` | `3` | Spatial transformer layers |
| `model.head_type` | `gaussian_diagonal` | Output head (`gaussian_diagonal` / `gaussian_cholesky`) |
| `model.freeze_text` | `false` | Freeze BERT weights |
| `checkpoint.dir` | `checkpoints` | Where to save checkpoints |
| `wandb.enabled` | `true` | Toggle WandB logging |

Example with overrides:

```bash
python train.py \
  cache.dir=/fast/ssd/lss_cache \
  training.lr=3e-4 \
  training.batch_size=32 \
  model.head_type=gaussian_cholesky \
  wandb.enabled=false
```

**Outputs:**

```
checkpoints/
├── best.pt          # best validation-loss checkpoint
└── epoch_<N>.pt     # periodic checkpoints (every checkpoint.save_every epochs)

outputs/<timestamp>/
└── .hydra/config.yaml   # full resolved config for the run
```

---

## 3. Preprocess data for 3D-ViSTA

The ViSTA preprocessor produces per-sample point clouds (per-object PCDs sampled from the scene), tokenized text, and grounding targets. It uses its own config at `baselines/vista_grounding/config.yaml`.

```bash
python baselines/vista_grounding/preprocess.py \
  cache.dir=baselines_cache/vista
```

> **Note on paths:** The config default for `cache.dir` is `../../baselines_cache/vista`, which is relative to the script's own directory — not `lss/`. Always pass `cache.dir` explicitly when running from `lss/` to avoid writing two levels above the repo root.

Key overrides:

| Override | Recommended | Description |
|---|---|---|
| `cache.dir` | `baselines_cache/vista` | Output directory (always set explicitly) |
| `data.data_root` | `/home/aryannav/mit/data/VLA-3D/VLA-3D_dataset` | Path to raw VLA-3D dataset |
| `data.datasets` | `[unity]` | Datasets to include |
| `cache.force_rebuild` | `false` | Force full rebuild |
| `model.max_objects` | `80` | Max objects per scene |
| `model.max_text_len` | `50` | Max token length |

Example with overrides:

```bash
python baselines/vista_grounding/preprocess.py \
  cache.dir=baselines_cache/vista \
  data.data_root=/path/to/VLA-3D_dataset \
  cache.force_rebuild=true
```

**Output layout:**

```
baselines_cache/vista/
├── train/
│   ├── manifest.json
│   └── <scene_id>_<idx>.pt    # obj_pcds, obj_locs, obj_masks, txt_ids,
│                               # txt_masks, target_xyz_world, target_bbox_world
├── val_seen/
│   ├── manifest.json
│   └── <scene_id>_<idx>.pt
└── val_unseen/
    ├── manifest.json
    └── <scene_id>_<idx>.pt
```

---

## 4. Finetune 3D-ViSTA

The ViSTA baseline loads a pretrained 3D-ViSTA backbone and attaches a GMM prediction head, then finetunes end-to-end on VLA-3D.

### PointNet++ CUDA extensions

PointNet++ must be compiled for your GPU before training. Set `TORCH_CUDA_ARCH_LIST` to match your GPU's SM version (e.g. `8.9` for RTX 4090, `9.0` for H100, `12.0` for RTX 5090):

```bash
export TORCH_CUDA_ARCH_LIST="8.9"   # adjust for your GPU
cd baselines/3dvista/model/vision/pointnet2
pip install --no-build-isolation -e .
cd -
```

### Pretrained checkpoint

Download the 3D-ViSTA pretrained weights and place them at:

```
checkpoints/3dvista/pretrain.pth
```

### Training

```bash
python baselines/vista_grounding/train.py \
  model.vista_ckpt_path=checkpoints/3dvista/pretrain.pth \
  cache.dir=baselines_cache/vista \
  checkpoint.dir=checkpoints/vista
```

> **Note on paths:** Like the preprocessor, `cache.dir` and `checkpoint.dir` defaults are relative to the script's directory (`../../…`). Always pass them explicitly when running from `lss/`.

Key overrides:

| Override | Recommended | Description |
|---|---|---|
| `model.vista_ckpt_path` | `checkpoints/3dvista/pretrain.pth` | Path to `pretrain.pth`; omit to train from scratch |
| `cache.dir` | `baselines_cache/vista` | Preprocessed cache from step 3 (always set explicitly) |
| `checkpoint.dir` | `checkpoints/vista` | Where to save checkpoints (always set explicitly) |
| `training.epochs` | `50` | Number of epochs |
| `training.batch_size` | `16` | Batch size (smaller than LSS due to per-object PCDs) |
| `training.lr` | `1e-4` | Base learning rate |
| `training.backbone_lr_scale` | `0.1` | LR multiplier for ViSTA backbone |
| `training.lang_lr_scale` | `0.1` | LR multiplier for BERT encoder |
| `training.precision` | `bf16` | Mixed precision |
| `model.num_components` | `5` | GMM mixture components K |
| `model.freeze_layers` | `0` | `0` = full finetune; `-1` = freeze all ViSTA; `N` = freeze first N spatial layers |
| `checkpoint.dir` | `checkpoints/vista` | Where to save checkpoints |
| `wandb.enabled` | `true` | Toggle WandB logging |
| `wandb.tags` | `[vista, gmm]` | WandB run tags |

Example — freeze backbone, higher LR on head:

```bash
python baselines/vista_grounding/train.py \
  model.vista_ckpt_path=checkpoints/3dvista/pretrain.pth \
  model.freeze_layers=-1 \
  model.num_components=8 \
  training.lr=5e-4 \
  training.batch_size=8 \
  wandb.name=vista-frozen-backbone
```

**Outputs:**

```
checkpoints/vista/
├── best.pt
└── epoch_<N>.pt
```

---

## Config reference

| File | Purpose |
|---|---|
| `experiments/cfgs/train.yaml` | Top-level LSS training config |
| `experiments/cfgs/data/vla3d.yaml` | Data splits and dataset selection |
| `experiments/cfgs/model/lss.yaml` | LSS model architecture |
| `baselines/vista_grounding/config.yaml` | All ViSTA preprocessing and training config |
