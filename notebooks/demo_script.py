#!/usr/bin/env python3

"""
Language Spatial Sensor — Pipeline Demo (Script Version)

Steps:
1. Load a scene and build a scene graph
2. Initialise the sensor with a trained checkpoint + LLM proposer
3. Query with a natural-language utterance → GMM → sampled points
4. Visualise the GMM on a BEV occupancy grid
5. Run the benchmark to compare proposer approaches
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def setup_paths():
    """Ensure lss root is on sys.path."""
    LSS_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(LSS_ROOT))
    return LSS_ROOT


def load_scene(DATA_ROOT):
    from data.vla3d.dataset import VLA3DScene

    scene = VLA3DScene(DATA_ROOT / "Matterport" / "17DRP5sb8fy")
    sg = scene.load_scene_graph()
    pcd = scene.load_pointcloud()
    points = np.asarray(pcd.points, dtype=np.float32)
    obj_split = scene.load_object_split()

    print(f"Scene: {scene.scene_id}")
    print(f"  Regions: {len(sg.regions)}")
    print(f"  Objects: {len(sg.objects)}")
    print(f"  Points:  {len(points):,}")

    return scene, sg, points, obj_split


def init_sensor(LSS_ROOT, CHECKPOINT):
    from language_spatial_sensor.pipeline.language_sensor import LanguageSpatialSensor
    from language_spatial_sensor.pipeline.proposer import LLMProposer

    proposer = LLMProposer(
        provider="openai",
        model="gpt-5.2",
        cache_dir="cache/proposer",
        verbose=True,
    )

    sensor = LanguageSpatialSensor(
        checkpoint_path=CHECKPOINT,
        proposer=proposer,
        device="cuda",
        clip_label_map=str(
            LSS_ROOT / "cache" / "clip_label_map.pt"
        ),
    )

    return sensor


def run_query(sensor, scene, sg, points, obj_split):
    utterance = "there is a lamp."
    print("\nUtterance:", utterance)

    result = sensor.predict(
        scene_graph=sg,
        utterance=utterance,
        scene_id=scene.scene_id,
        pc=points,
        object_split=obj_split,
    )

    print(f"\nNumber of Gaussian components: {len(result.weights)}")
    for k, (g, w) in enumerate(zip(result.groundings, result.weights)):
        mu = result.mus[k].numpy()
        print(
            f"  Component {k}: weight={float(w):.3f} "
            f"anchors={g.anchor_object_ids} "
            f"mu=[{mu[0]:.2f}, {mu[1]:.2f}, {mu[2]:.2f}]"
        )

    samples = result.sample(1000)
    print(f"\nSampled {len(samples)} points, shape={samples.shape}")
    print(f"GMM weighted mean: {result.mean}")

    return result


def visualize(scene, sg, points, obj_split, result, utterance):
    from language_spatial_sensor.core.transforms import build_spatial_query
    from viz.bev import render_bev_with_gmm_overlay

    stmts = scene.load_statements(sg)
    stmt = next((s for s in stmts if "lamp" in s.text.lower()), stmts[0])

    query = build_spatial_query(
        scene.scene_id, sg, stmt, points, obj_split
    )
    query.language = utterance

    fig = render_bev_with_gmm_overlay(
        query, result, n_samples=5000, resolution=0.25
    )
    plt.show()


def run_benchmark(DATA_ROOT, CHECKPOINT):
    from evaluation.benchmark import (
        load_mini_val,
        run_benchmark,
        print_comparison_table,
    )
    from language_spatial_sensor.pipeline.language_sensor import LanguageSpatialSensor
    from language_spatial_sensor.pipeline.proposer import (
        LLMProposer,
        GroundTruthProposer,
    )

    queries = load_mini_val(
        data_root=DATA_ROOT,
        datasets=["Unity", "3RScan"],
        n=50,
    )
    LSS_ROOT = setup_paths()
    openai_proposer = LLMProposer(
        provider="openai",
        model="gpt-5.2",
        cache_dir="cache/proposer",
        verbose=True,
    )
    approaches = {
        "gt_oracle": LanguageSpatialSensor(
            CHECKPOINT, GroundTruthProposer(),
            clip_label_map=str(
                LSS_ROOT / "cache" / "clip_label_map.pt"
            ),            
        ),
        "ollama_qwen": LanguageSpatialSensor(
            CHECKPOINT,
            LLMProposer(
                provider="ollama",
                model="qwen2.5:32b",
                cache_dir="cache/proposer",
            ),
            clip_label_map=str(
                LSS_ROOT / "cache" / "clip_label_map.pt"
            ),            
        ),
        "openai": LanguageSpatialSensor(
            CHECKPOINT,
            openai_proposer,
            clip_label_map=str(
                LSS_ROOT / "cache" / "clip_label_map.pt"
            ),            
        ),        
    }

    results = run_benchmark(approaches, queries)

    print("\n")
    print_comparison_table(results)

    for name, m in results.items():
        if "error" in m:
            continue

        print(f"\n=== {name} ===")

        print("\nBy relation (CDF):")
        for rel, stats in m["by_relation"]["cdf"].items():
            print(
                f"  {rel:>10}: mean={stats['mean']:.4f} "
                f"std={stats['std']:.4f} "
                f"n={stats['count']}"
            )

        print("\nBy ambiguity (RMSE):")
        for amb, stats in m["by_ambiguity"]["rmse"].items():
            print(
                f"  ambig={amb}: mean={stats['mean']:.3f}m "
                f"std={stats['std']:.3f} "
                f"n={stats['count']}"
            )


def main():
    LSS_ROOT = setup_paths()

    DATA_ROOT = Path("/home/aryannav/mit/data/VLA-3D/VLA-3D_dataset")
    CHECKPOINT = LSS_ROOT / "checkpoints" / "best.pt"

    # scene, sg, points, obj_split = load_scene(DATA_ROOT)
    # sensor = init_sensor(LSS_ROOT, CHECKPOINT)

    # result = run_query(sensor, scene, sg, points, obj_split)

    # visualize(
    #     scene,
    #     sg,
    #     points,
    #     obj_split,
    #     result,
    #     "there is a lamp.",
    # )

    run_benchmark(DATA_ROOT, CHECKPOINT)


if __name__ == "__main__":
    main()