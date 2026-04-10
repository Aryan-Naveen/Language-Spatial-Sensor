# %% [markdown]
# # Language Spatial Sensor — Pipeline Demo
#
# This notebook demonstrates the end-to-end pipeline:
# 1. Load a scene and build a scene graph
# 2. Initialise the sensor with a trained checkpoint + LLM proposer
# 3. Query with a natural-language utterance → GMM → sampled points
# 4. Visualise the GMM on a BEV occupancy grid
# 5. Run the benchmark to compare proposer approaches

# %% Setup
import sys
from pathlib import Path

# Ensure lss/ root is on sys.path
LSS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LSS_ROOT))

import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Load a scene from VLA-3D

# %%
from data.vla3d.dataset import VLA3DScene

DATA_ROOT = Path("/home/aryannav/mit/data/VLA-3D/VLA-3D_dataset")
CHECKPOINT = LSS_ROOT / "checkpoints" / "best.pt"  # adjust to your path


# Pick any scene
scene = VLA3DScene(DATA_ROOT / "Matterport" / "17DRP5sb8fy")  # example scene
sg = scene.load_scene_graph()
pcd = scene.load_pointcloud()
points = np.asarray(pcd.points, dtype=np.float32)
obj_split = scene.load_object_split()

print(f"Scene: {scene.scene_id}")
print(f"  Regions: {len(sg.regions)}")
print(f"  Objects: {len(sg.objects)}")
print(f"  Points:  {len(points):,}")

# %% [markdown]
# ## 2. Initialise the Language Spatial Sensor

# %%
from language_spatial_sensor.pipeline.language_sensor import LanguageSpatialSensor
from language_spatial_sensor.pipeline.proposer import LLMProposer, GroundTruthProposer

# --- Option A: Ollama (local, fast) ---
proposer = LLMProposer(
    provider="ollama",
    model="qwen2.5:32b",          # or "qwen2.5:16b" for smaller model
    cache_dir="cache/proposer",  # cache LLM responses to disk
    verbose=True,                # print LLM responses
)

# --- Option B: OpenAI ---
# proposer = LLMProposer(provider="openai", model="gpt-4o-mini")

# --- Option C: Ground-truth oracle (for comparison) ---
# proposer = GroundTruthProposer()

sensor = LanguageSpatialSensor(
    checkpoint_path=CHECKPOINT,
    proposer=proposer,
    device="cuda",
    clip_label_map='/home/aryannav/mit/research/langmap/src/lss/cache/clip_label_map.pt'
)

# %% [markdown]
# ## 3. Query with a language utterance

# %%
utterance = "there is a lamp that is near the curtain"
print(utterance)
result = sensor.predict(
    scene_graph=sg,
    utterance=utterance,
    scene_id=scene.scene_id,
    pc=points,
    object_split=obj_split,
)

# Inspect the GMM
print(f"Number of Gaussian components: {len(result.weights)}")
for k, (g, w) in enumerate(zip(result.groundings, result.weights)):
    mu = result.mus[k].numpy()
    print(f"  Component {k}: weight={float(w):.3f}  "
          f"anchors={g.anchor_object_ids}  "
          f"mu=[{mu[0]:.2f}, {mu[1]:.2f}, {mu[2]:.2f}]")

# Sample 1000 points
samples = result.sample(1000)
print(f"\nSampled {len(samples)} points, shape={samples.shape}")
print(f"GMM weighted mean: {result.mean}")

# %% [markdown]
# ## 4. Visualise on BEV

# %%
from language_spatial_sensor.core.transforms import build_spatial_query
from viz.bev import render_bev_with_gmm_overlay

# For visualization we need a SpatialQuery (which has the point cloud).
# If we have GT info (from a statement), use build_spatial_query.
# For pure inference, we can construct a minimal query:

stmts = scene.load_statements(sg)
stmt = next((s for s in stmts if "lamp" in s.text.lower()), stmts[0])
query = build_spatial_query(scene.scene_id, sg, stmt, points, obj_split)

fig = render_bev_with_gmm_overlay(query, result, n_samples=5000, resolution=0.25)
fig.suptitle(f'"{utterance}"', fontsize=10, y=1.02)
plt.show()

# %% [markdown]
# ## 5. Benchmark: compare proposer approaches
#
# This section shows how to run a systematic comparison across approaches.

# %%
from evaluation.benchmark import load_mini_val, run_benchmark, print_comparison_table

# Load a small validation set for quick iteration
queries = load_mini_val(
    data_root=DATA_ROOT,
    datasets=["Unity", "3RScan"],
    n=50,  # mini-val: 50 samples
)

# Define approaches to compare
approaches = {
    "gt_oracle": LanguageSpatialSensor(CHECKPOINT, GroundTruthProposer()),
    "ollama_qwen": LanguageSpatialSensor(
        CHECKPOINT,
        LLMProposer(provider="ollama", model="qwen2.5:32b", cache_dir="cache/proposer"),
    ),
}

# Run the benchmark
results = run_benchmark(approaches, queries)

# Print comparison table
print("\n")
print_comparison_table(results)

# %% [markdown]
# ## 6. Inspect per-relation and per-ambiguity breakdowns

# %%
import json

for name, m in results.items():
    if "error" in m:
        continue
    print(f"\n=== {name} ===")
    print("\nBy relation (CDF):")
    for rel, stats in m["by_relation"]["cdf"].items():
        print(f"  {rel:>10}: mean={stats['mean']:.4f}  std={stats['std']:.4f}  n={stats['count']}")
    print("\nBy ambiguity (RMSE):")
    for amb, stats in m["by_ambiguity"]["rmse"].items():
        print(f"  ambig={amb}: mean={stats['mean']:.3f}m  std={stats['std']:.3f}  n={stats['count']}")
