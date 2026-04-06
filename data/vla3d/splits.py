"""
Train / val-seen / val-unseen split construction for VLA-3D.

Split strategy
--------------
* Scene-level split first:
    - A random fraction (val_unseen_scene_frac) of all scenes across the
      selected datasets is withheld entirely → val_unseen.
    - The remaining scenes are "seen" scenes, shared by train and val_seen.

* Statement-level split within seen scenes:
    - val_seen_stmt_frac of each scene's statements → val_seen.
    - The rest → train.

* All statements from unseen scenes → val_unseen.

* Region-ambiguity splits (seen / unseen) contain synthetic statements of the
  form "there is a <object> in the <region>", generated via
  VLA3DScene.generate_region_statements().
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig
from tqdm import tqdm

from data.vla3d.dataset import VLA3D, VLA3DScene
from language_spatial_sensor.core.ontology import VALID_NYU40_LABELS, VALID_RELATIONS
from language_spatial_sensor.core.schema import ReferentialStatement


@dataclass
class SplitRecord:
    scene: VLA3DScene
    statement: ReferentialStatement


@dataclass
class DataSplits:
    train: list[SplitRecord] = field(default_factory=list)
    val_seen: list[SplitRecord] = field(default_factory=list)
    val_unseen: list[SplitRecord] = field(default_factory=list)
    val_seen_region_ambiguity: list[SplitRecord] = field(default_factory=list)
    val_unseen_region_ambiguity: list[SplitRecord] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"train={len(self.train):,}  "
            f"val_seen={len(self.val_seen):,}  "
            f"val_unseen={len(self.val_unseen):,}  "
            f"val_seen_region_ambiguity={len(self.val_seen_region_ambiguity):,}  "
            f"val_unseen_region_ambiguity={len(self.val_unseen_region_ambiguity):,}"
        )


def _filter_statements(
    scene_graph,
    stmts: list[ReferentialStatement],
) -> list[ReferentialStatement]:
    """Keep only statements where:
    - relation is in VALID_RELATIONS
    - target object's nyu40_label is in VALID_NYU40_LABELS
    - every anchor object's nyu40_label is in VALID_NYU40_LABELS
    """
    obj_label: dict[int, str] = {
        obj.id: str(obj.metadata.get("nyu40_label", "")).lower().strip()
        for obj in scene_graph.objects
    }

    kept = []
    for s in stmts:
        if s.relation not in VALID_RELATIONS:
            continue
        if obj_label.get(s.target_object_id, "") not in VALID_NYU40_LABELS:
            continue
        anchors = s.anchor_object_id or []
        if any(obj_label.get(aid, "") not in VALID_NYU40_LABELS for aid in anchors):
            continue
        kept.append(s)
    return kept


def build_splits(cfg: DictConfig) -> DataSplits:
    """Build all splits from a resolved Hydra DictConfig.

    Args:
        cfg: The ``data`` config node (i.e. ``cfg.data`` from the top-level
             Hydra config).  Expected keys: ``data_root``, ``datasets``,
             ``splits.seed``, ``splits.val_unseen_scene_frac``,
             ``splits.val_seen_stmt_frac``.
    """
    vla3d = VLA3D(Path(cfg.data_root))
    split_cfg = cfg.splits

    rng = random.Random(split_cfg.seed)

    # --- collect all scenes across selected datasets ----------------------
    all_scenes: list[VLA3DScene] = []
    for name in cfg.datasets:
        try:
            dataset = vla3d.get_dataset(name)
        except (FileNotFoundError, ValueError) as e:
            print(f"[splits] Skipping dataset '{name}': {e}")
            continue
        all_scenes.extend(dataset.scenes())

    if not all_scenes:
        raise RuntimeError("No scenes found. Check data_root and datasets config.")

    # --- scene-level split ------------------------------------------------
    shuffled = all_scenes[:]
    rng.shuffle(shuffled)

    n_unseen = max(1, round(len(shuffled) * split_cfg.val_unseen_scene_frac))
    unseen_scenes = shuffled[-n_unseen:]
    seen_scenes = shuffled[:-n_unseen]

    print(
        f"[splits] {len(all_scenes)} total scenes → "
        f"{len(seen_scenes)} seen, {len(unseen_scenes)} unseen"
    )

    # --- build records ----------------------------------------------------
    splits = DataSplits()

    for scene in tqdm(seen_scenes):
        try:
            sg = scene.load_scene_graph()
            stmts = _filter_statements(sg, scene.load_statements(sg))
            region_stmts = scene.generate_region_statements(sg)
        except Exception as e:
            print(f"[splits] Could not load statements for {scene.scene_id}: {e}")
            continue

        rng.shuffle(stmts)
        n_val = max(1, round(len(stmts) * split_cfg.val_seen_stmt_frac)) if stmts else 0
        splits.val_seen.extend(SplitRecord(scene, s) for s in stmts[:n_val])
        splits.train.extend(SplitRecord(scene, s) for s in stmts[n_val:])
        splits.val_seen_region_ambiguity.extend(SplitRecord(scene, s) for s in region_stmts)

    for scene in tqdm(unseen_scenes):
        try:
            sg = scene.load_scene_graph()
            stmts = _filter_statements(sg, scene.load_statements(sg))
            region_stmts = scene.generate_region_statements(sg)
        except Exception as e:
            print(f"[splits] Could not load statements for {scene.scene_id}: {e}")
            continue

        splits.val_unseen.extend(SplitRecord(scene, s) for s in stmts)
        splits.val_unseen_region_ambiguity.extend(SplitRecord(scene, s) for s in region_stmts)

    return splits
