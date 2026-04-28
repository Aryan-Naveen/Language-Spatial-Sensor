"""End-to-end Language Spatial Sensor pipeline.

Thin convenience wrapper around
``(Proposer, LSSGaussianPredictor)`` → GMM.  New benchmarks should prefer
:class:`evaluation.approach.GMMApproach`, which accepts any
:class:`DistributionPredictor` (LLM, VLM, learned model, ...).  This class
stays for backwards compatibility with existing call sites / notebooks.

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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from language_spatial_sensor.core.schema import (
    Grounding,
    SceneGraph,
    SpatialQuery,
)
from language_spatial_sensor.pipeline.distribution_predictor import (
    DistributionPredictor,
    LSSGaussianPredictor,
)
from language_spatial_sensor.pipeline.proposer import Proposer


# ---------------------------------------------------------------------------
# Conformal tail component
# ---------------------------------------------------------------------------

@dataclass
class ConformalComponent:
    """Uniform-over-ellipsoid tail, superimposed on a GMM.

    The ellipsoid is ``{x : (x - mu)ᵀ Σ⁻¹ (x - mu) ≤ q²}`` with
    ``Σ = L @ Lᵀ``.  Used by :class:`ConformalSuperimposed` to give
    LLM/VLM predictors the same calibrated coverage guarantee LSS has.

    Attributes:
        mu:     (3,) ellipsoid centre.
        L:      (3, 3) Cholesky factor of the ellipsoid shape matrix.
        q:      Mahalanobis radius (from the held-out calibration quantile).
        weight: Mixture weight of the uniform tail (Gaussian share = 1 - weight).
    """
    mu: torch.Tensor      # (3,)
    L: torch.Tensor       # (3, 3)
    q: float
    weight: float


# ---------------------------------------------------------------------------
# GMM result container
# ---------------------------------------------------------------------------

@dataclass
class GMMResult:
    """Gaussian Mixture Model output from the pipeline.

    Optionally carries a :class:`ConformalComponent` tail — a
    uniform-over-Mahalanobis-ellipsoid component superimposed on the GMM
    to give calibrated coverage.
    """

    groundings: list[Grounding]
    mus: torch.Tensor       # (K, 3) component means, world frame
    Ls: torch.Tensor        # (K, 3, 3) Cholesky factors, world frame
    weights: torch.Tensor   # (K,) normalized mixture weights (Gaussian components)
    conformal: ConformalComponent | None = None

    # ---- sampling ----------------------------------------------------------

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        """Draw *n* points from the mixture. Returns (n, 3) float32 array."""
        rng = np.random.RandomState(seed)

        if self.conformal is None:
            gauss_n = n
            uniform_n = 0
        else:
            uniform_n = int(rng.binomial(n, self.conformal.weight))
            gauss_n = n - uniform_n

        parts: list[torch.Tensor] = []

        if gauss_n > 0:
            counts = rng.multinomial(gauss_n, self.weights.numpy())
            for k, c in enumerate(counts):
                if c == 0:
                    continue
                dist = torch.distributions.MultivariateNormal(
                    self.mus[k], scale_tril=self.Ls[k],
                )
                parts.append(dist.sample((int(c),)))

        if uniform_n > 0:
            parts.append(self._sample_uniform_ellipsoid(uniform_n, rng))

        return torch.cat(parts, dim=0).numpy() if parts else np.zeros((0, 3), dtype=np.float32)

    def _sample_uniform_ellipsoid(self, n: int, rng: np.random.RandomState) -> torch.Tensor:
        """Draw n points uniformly from the conformal ellipsoid."""
        c = self.conformal
        # Sample uniformly in the unit ball: direction ~ sphere, radius = U^(1/3).
        z = rng.standard_normal(size=(n, 3)).astype(np.float32)
        z /= np.linalg.norm(z, axis=-1, keepdims=True).clip(min=1e-12)
        r = rng.uniform(size=(n, 1)).astype(np.float32) ** (1.0 / 3.0)
        unit = z * r                                      # (n, 3) in unit ball
        unit_t = torch.from_numpy(unit)
        # Map into the ellipsoid: x = mu + q * L @ u
        return c.mu + c.q * (unit_t @ c.L.T)

    # ---- densities ---------------------------------------------------------

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log-density at point(s) *x*. x: (*, 3) -> (*)."""
        log_w = torch.log(self.weights.clamp(min=1e-12))
        parts = []
        for k in range(len(self.weights)):
            dist = torch.distributions.MultivariateNormal(
                self.mus[k], scale_tril=self.Ls[k],
            )
            parts.append(dist.log_prob(x))
        gauss_log_prob = torch.logsumexp(torch.stack(parts, dim=-1) + log_w, dim=-1)

        if self.conformal is None:
            return gauss_log_prob

        c = self.conformal
        diff = (x - c.mu).unsqueeze(-1)                                         # (*, 3, 1)
        z = torch.linalg.solve_triangular(c.L, diff, upper=False).squeeze(-1)   # (*, 3)
        inside = (z * z).sum(dim=-1) <= (c.q ** 2)

        log_det_L = c.L.diagonal().clamp(min=1e-8).log().sum()
        vol = (4.0 / 3.0) * math.pi * (c.q ** 3) * float(torch.exp(log_det_L))
        uniform_log_prob = torch.full_like(
            gauss_log_prob,
            float(-math.inf),
        )
        uniform_log_prob = torch.where(
            inside,
            torch.tensor(math.log(max(1.0 / vol, 1e-30)), dtype=gauss_log_prob.dtype),
            uniform_log_prob,
        )

        w_g = math.log(max(1.0 - c.weight, 1e-12))
        w_u = math.log(max(c.weight, 1e-12))
        return torch.logsumexp(
            torch.stack([gauss_log_prob + w_g, uniform_log_prob + w_u], dim=-1),
            dim=-1,
        )

    def cdf_bbox(self, bbox: np.ndarray) -> float:
        """Probability mass inside an AABB [xmin,ymin,zmin,xmax,ymax,zmax]."""
        bbox_t = torch.as_tensor(bbox, dtype=torch.float32)
        bbox_min, bbox_max = bbox_t[:3], bbox_t[3:]
        gauss_total = 0.0
        for k in range(len(self.weights)):
            sigma = self.Ls[k].diagonal().clamp(min=1e-8)
            mu = self.mus[k]
            z_lo = (bbox_min - mu) / (sigma * math.sqrt(2.0))
            z_hi = (bbox_max - mu) / (sigma * math.sqrt(2.0))
            p_axis = 0.5 * (torch.erf(z_hi) - torch.erf(z_lo))
            gauss_total += float(self.weights[k]) * float(p_axis.prod())

        if self.conformal is None:
            return gauss_total

        c = self.conformal
        # Monte-Carlo estimate of uniform-ellipsoid mass in the AABB.
        samples = self._sample_uniform_ellipsoid(
            4096, np.random.RandomState(0),
        ).numpy()
        inside = (
            (samples >= bbox_t[:3].numpy()).all(-1)
            & (samples <= bbox_t[3:].numpy()).all(-1)
        )
        uniform_total = float(inside.mean())
        return (1.0 - c.weight) * gauss_total + c.weight * uniform_total

    @property
    def mean(self) -> np.ndarray:
        """Weighted mean of the full mixture (including conformal tail)."""
        gauss_mean = (self.weights.unsqueeze(-1) * self.mus).sum(0)
        if self.conformal is None:
            return gauss_mean.numpy()
        c = self.conformal
        return ((1.0 - c.weight) * gauss_mean + c.weight * c.mu).numpy()


# ---------------------------------------------------------------------------
# Main pipeline (thin wrapper around LSSGaussianPredictor)
# ---------------------------------------------------------------------------

class LanguageSpatialSensor:
    """Utterance + scene graph -> spatial distribution (GMM) -> sampled points.

    Convenience wrapper: ``(Proposer, LSSGaussianPredictor)``.  For benchmarks
    that swap the distribution predictor too (LLM, VLM, ...), use
    :class:`evaluation.approach.GMMApproach` instead.

    Args:
        checkpoint_path: Path to a training checkpoint (``best.pt``).
        proposer:        Any :class:`Proposer` subclass.
        clip_label_map:  Dict, path to .pt file, or ``None`` (auto-discovered).
        device:          ``"cuda"`` or ``"cpu"``.
        predictor:       Optional pre-built :class:`DistributionPredictor`.
                         If provided, ``checkpoint_path``/``clip_label_map``/``device``
                         are ignored.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        proposer: Proposer | None = None,
        clip_label_map: dict[str, np.ndarray] | str | Path | None = None,
        device: str = "cuda",
        predictor: DistributionPredictor | None = None,
    ) -> None:
        if proposer is None:
            raise ValueError("LanguageSpatialSensor requires a proposer")
        self.proposer = proposer

        if predictor is not None:
            self.predictor = predictor
        else:
            if checkpoint_path is None:
                raise ValueError(
                    "LanguageSpatialSensor: either pass a predictor or a checkpoint_path"
                )
            self.predictor = LSSGaussianPredictor(
                checkpoint_path=checkpoint_path,
                clip_label_map=clip_label_map,
                device=device,
            )

    @property
    def device(self) -> torch.device:
        return getattr(self.predictor, "device", torch.device("cpu"))

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
        gt_anchor_object_ids: list[int] | None = None,
        gt_anchor_room_id: int | None = None,
    ) -> GMMResult:
        """Run the full pipeline: proposer -> predictor -> GMM.

        Returns:
            :class:`GMMResult` with K Gaussian components weighted by
            proposer confidence.
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
            gt_anchor_object_ids=gt_anchor_object_ids,
            gt_anchor_room_id=gt_anchor_room_id,
        )

        groundings = self.proposer.propose(query)
        if not groundings:
            raise RuntimeError("Proposer returned no groundings")

        mus, Ls = self.predictor.predict(query, groundings)

        weights = np.array([g.confidence for g in groundings], dtype=np.float32)
        weights = weights / weights.sum()

        return GMMResult(
            groundings=groundings,
            mus=mus,
            Ls=Ls,
            weights=torch.from_numpy(weights),
        )

    def sample(
        self,
        scene_graph: SceneGraph,
        utterance: str,
        n_samples: int = 1000,
        **kwargs,
    ) -> np.ndarray:
        """Convenience: predict + sample. Returns (n_samples, 3) array."""
        return self.predict(scene_graph, utterance, **kwargs).sample(n_samples)
