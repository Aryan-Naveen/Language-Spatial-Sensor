"""End-to-end Language Spatial Sensor pipeline.

Usage::

    from language_spatial_sensor.pipeline.language_sensor import LanguageSpatialSensor
    from language_spatial_sensor.pipeline.proposer import LLMProposer

    sensor = LanguageSpatialSensor(
        checkpoint_path="checkpoints/best.pt",
        proposer=LLMProposer(provider="ollama", model="qwen2.5:32b"),
    )
    result = sensor.predict(scene_graph, "there is a lamp on the desk")
    samples = result.sample(1000)   # (1000, 3) np.ndarray in world frame
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch

from language_spatial_sensor.core.schema import (
    CachedSample,
    Grounding,
    GroundedQuery,
    SceneGraph,
    SpatialQuery,
    TensorizerOutput,
)
from language_spatial_sensor.models.components.heads import GaussianPrediction
from language_spatial_sensor.models.config import LSSConfig
from language_spatial_sensor.models.lss_model import LSSModel
from language_spatial_sensor.pipeline.proposer import Proposer
from language_spatial_sensor.pipeline.tensorizer import Tensorizer, build_clip_label_map
from language_spatial_sensor.training.dataset import CollateFn


# ---------------------------------------------------------------------------
# GMM result container
# ---------------------------------------------------------------------------

@dataclass
class GMMResult:
    """Gaussian Mixture Model output from the pipeline."""

    groundings: list[Grounding]
    mus: torch.Tensor       # (K, 3) component means, world frame
    Ls: torch.Tensor        # (K, 3, 3) Cholesky factors, world frame
    weights: torch.Tensor   # (K,) normalized mixture weights

    # ---- sampling ----------------------------------------------------------

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        """Draw *n* points from the GMM. Returns (n, 3) float32 array."""
        rng = np.random.RandomState(seed)
        counts = rng.multinomial(n, self.weights.numpy())
        parts: list[torch.Tensor] = []
        for k, c in enumerate(counts):
            if c == 0:
                continue
            dist = torch.distributions.MultivariateNormal(
                self.mus[k], scale_tril=self.Ls[k],
            )
            parts.append(dist.sample((int(c),)))
        return torch.cat(parts, dim=0).numpy() if parts else np.zeros((0, 3), dtype=np.float32)

    # ---- densities ---------------------------------------------------------

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log-density of the GMM at point(s) *x*.  x: (*, 3) -> (*)."""
        log_w = torch.log(self.weights.clamp(min=1e-12))
        parts = []
        for k in range(len(self.weights)):
            dist = torch.distributions.MultivariateNormal(
                self.mus[k], scale_tril=self.Ls[k],
            )
            parts.append(dist.log_prob(x))           # (*)
        return torch.logsumexp(torch.stack(parts, dim=-1) + log_w, dim=-1)

    def cdf_bbox(self, bbox: np.ndarray) -> float:
        """Probability mass inside an AABB [xmin,ymin,zmin,xmax,ymax,zmax]."""
        bbox_t = torch.as_tensor(bbox, dtype=torch.float32)
        bbox_min, bbox_max = bbox_t[:3], bbox_t[3:]
        total = 0.0
        for k in range(len(self.weights)):
            sigma = self.Ls[k].diagonal().clamp(min=1e-8)
            mu = self.mus[k]
            z_lo = (bbox_min - mu) / (sigma * math.sqrt(2.0))
            z_hi = (bbox_max - mu) / (sigma * math.sqrt(2.0))
            p_axis = 0.5 * (torch.erf(z_hi) - torch.erf(z_lo))
            total += float(self.weights[k]) * float(p_axis.prod())
        return total

    @property
    def mean(self) -> np.ndarray:
        """Weighted mean of the GMM components."""
        return (self.weights.unsqueeze(-1) * self.mus).sum(0).numpy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_device(batch: TensorizerOutput, device: torch.device) -> TensorizerOutput:
    return replace(batch, **{
        f.name: getattr(batch, f.name).to(device)
        for f in fields(TensorizerOutput)
    })


def _clean_state_dict(sd: dict) -> dict:
    """Strip torch.compile / DataParallel prefixes from state dict keys."""
    cleaned = {}
    for k, v in sd.items():
        k = k.replace("_orig_mod.", "").replace("module.", "")
        cleaned[k] = v
    return cleaned


def _resolve_clip_map(
    clip_label_map,
    checkpoint_path: Path,
) -> dict[str, np.ndarray]:
    """Resolve CLIP label map from various input types."""
    if isinstance(clip_label_map, dict):
        return clip_label_map
    if isinstance(clip_label_map, (str, Path)):
        return torch.load(clip_label_map, weights_only=False)
    # Try to find alongside checkpoint
    for candidate in [
        checkpoint_path.parent / "clip_label_map.pt",
        checkpoint_path.parent.parent / "clip_label_map.pt",
    ]:
        if candidate.exists():
            return torch.load(candidate, weights_only=False)
    raise FileNotFoundError(
        "Could not find clip_label_map.pt. Pass it explicitly via clip_label_map=."
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class LanguageSpatialSensor:
    """Utterance + scene graph -> spatial distribution (GMM) -> sampled points.

    Args:
        checkpoint_path: Path to a training checkpoint (``best.pt``).
        proposer:        Any :class:`Proposer` subclass (GroundTruthProposer, LLMProposer, ...).
        clip_label_map:  Dict, path to .pt file, or ``None`` (auto-discovered near checkpoint).
        device:          ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        proposer: Proposer,
        clip_label_map: dict[str, np.ndarray] | str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.proposer = proposer

        # Load checkpoint
        ckpt_path = Path(checkpoint_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_cfg_dict = ckpt["cfg"]["model"]
        self.cfg = LSSConfig(**model_cfg_dict)

        self.model = LSSModel(self.cfg)
        self.model.load_state_dict(_clean_state_dict(ckpt["model"]))
        self.model.to(self.device).eval()

        # Tensorizer + collate
        clip_map = _resolve_clip_map(clip_label_map, ckpt_path)
        self.tensorizer = Tensorizer(
            max_objects=self.cfg.max_objects,
            clip_embedding_map=clip_map,
            clip_dim=self.cfg.clip_dim,
        )
        self.collate_fn = CollateFn(
            tokenizer_name=self.cfg.text_model,
            max_text_len=self.cfg.max_text_len,
        )

    # ---- public API --------------------------------------------------------

    def predict(
        self,
        scene_graph: SceneGraph,
        utterance: str,
        scene_id: str = "inference",
        pc: np.ndarray | None = None,
        object_split: np.ndarray | None = None,
        target_xyz: np.ndarray | None = None,
        target_bbox: np.ndarray | None = None,
    ) -> GMMResult:
        """Run the full pipeline: proposer -> model -> GMM.

        Args:
            scene_graph:  Scene context (objects + regions).
            utterance:    Natural-language placement description.
            scene_id:     Optional scene identifier (used for caching).
            pc:           Point cloud (only needed for GroundTruthProposer eval).
            object_split: Per-point object IDs (only needed for GT eval).
            target_xyz:   GT target position (only for evaluation).
            target_bbox:  GT target AABB (only for evaluation).

        Returns:
            :class:`GMMResult` with K Gaussian components weighted by proposer confidence.
        """
        _pc = pc if pc is not None else np.zeros((1, 3), dtype=np.float32)
        _split = object_split if object_split is not None else np.zeros(1, dtype=np.int64)

        query = SpatialQuery(
            scene_id=scene_id,
            scene_graph=scene_graph,
            pc=_pc,
            object_split=_split,
            language=utterance,
            target_xyz=target_xyz if target_xyz is not None else np.zeros(3, dtype=np.float32),
            target_bbox=target_bbox,
        )

        # -- proposer --
        groundings = self.proposer.propose(query)
        if not groundings:
            raise RuntimeError("Proposer returned no groundings")

        # -- tensorize --
        grounded_queries = [
            GroundedQuery(
                scene_id=scene_id,
                scene_graph=scene_graph,
                pc=_pc,
                object_split=_split,
                grounding=g,
                target_xyz=target_xyz,
                target_bbox=target_bbox,
            )
            for g in groundings
        ]
        samples = self.tensorizer.tensorize_grounded_batch(grounded_queries)
        batch = self.collate_fn(samples)
        batch = _to_device(batch, self.device)

        # -- model forward --
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        ):
            pred: GaussianPrediction = self.model(
                batch.text_input_ids,
                batch.text_attention_mask,
                batch.obj_clip_features,
                batch.obj_bboxes,
                batch.obj_is_anchor,
                batch.obj_padding_mask,
                batch.coord_scale,
                batch.coord_shift,
            )

        # -- build GMM --
        weights = np.array([g.confidence for g in groundings], dtype=np.float32)
        weights = weights / weights.sum()

        return GMMResult(
            groundings=groundings,
            mus=pred.mu.float().cpu(),
            Ls=pred.L.float().cpu(),
            weights=torch.from_numpy(weights),
        )

    def sample(
        self,
        scene_graph: SceneGraph,
        utterance: str,
        n_samples: int = 1000,
        **kwargs,
    ) -> np.ndarray:
        """Convenience: predict + sample.  Returns (n_samples, 3) array."""
        return self.predict(scene_graph, utterance, **kwargs).sample(n_samples)
