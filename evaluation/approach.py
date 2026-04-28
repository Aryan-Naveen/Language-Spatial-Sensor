"""GMMApproach: orchestrator for benchmark rows.

Each benchmark row (e.g. "gt_lss", "scaffolded_vlm", "straight_llm") is one
:class:`GMMApproach`.  It bundles:

- A :class:`Proposer` (or ``None`` for straight-shot predictors that emit
  a full GMM in one call).
- A :class:`DistributionPredictor` (learned LSS, LLM, VLM, ...) *or* a
  :class:`FullGMMPredictor`-style object that exposes ``.predict(query) -> GMMResult``.

The approach handles:

1. Running the proposer to get K groundings.
2. Passing groundings through the predictor to get K Gaussians.
3. Assembling those into a :class:`GMMResult`, weighted by proposer confidence.
4. Optionally attaching a conformal tail (via
   :class:`language_spatial_sensor.pipeline.conformal.ConformalSuperimposed`).

Usage::

    from evaluation.approach import GMMApproach
    from language_spatial_sensor.pipeline.proposer import GroundTruthProposer
    from language_spatial_sensor.pipeline.distribution_predictor import LSSGaussianPredictor

    approach = GMMApproach(
        proposer=GroundTruthProposer(),
        predictor=LSSGaussianPredictor("best.pt"),
    )
    result = approach.predict(query)   # GMMResult
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from language_spatial_sensor.core.schema import SpatialQuery
from language_spatial_sensor.pipeline.conformal import ConformalSuperimposed
from language_spatial_sensor.pipeline.distribution_predictor import DistributionPredictor
from language_spatial_sensor.pipeline.language_sensor import GMMResult
from language_spatial_sensor.pipeline.proposer import Proposer


class FullGMMPredictor(Protocol):
    """Duck-type for straight-shot predictors that emit a full GMM directly."""

    def predict(self, query: SpatialQuery) -> GMMResult: ...


class GMMApproach:
    """One benchmark row: ``(Proposer, DistributionPredictor)`` -> GMMResult.

    Args:
        proposer:  :class:`Proposer` that enumerates groundings.  Set to
                   ``None`` when ``predictor`` is a :class:`FullGMMPredictor`
                   that returns a full GMM in one call.
        predictor: Either a :class:`DistributionPredictor` (per-grounding
                   Gaussian) or a :class:`FullGMMPredictor` (full mixture).
                   May also be a :class:`ConformalSuperimposed` wrapper,
                   in which case its conformal tail is attached to the
                   returned :class:`GMMResult`.
    """

    def __init__(
        self,
        proposer: Proposer | None,
        predictor: DistributionPredictor | FullGMMPredictor | ConformalSuperimposed,
    ) -> None:
        self.proposer = proposer
        self.predictor = predictor

    def predict(self, query: SpatialQuery) -> GMMResult:
        # Straight-shot: the predictor owns grounding enumeration and returns a GMM.
        if self.proposer is None:
            if not hasattr(self.predictor, "predict"):
                raise TypeError(
                    "GMMApproach with proposer=None requires a predictor with .predict(query) -> GMMResult"
                )
            return self.predictor.predict(query)  # type: ignore[return-value]

        groundings = self.proposer.propose(query)
        if not groundings:
            raise RuntimeError(f"Proposer returned no groundings for {query.scene_id}")

        # Conformal wrapper returns (mus, Ls, ConformalComponent); others return (mus, Ls).
        if isinstance(self.predictor, ConformalSuperimposed):
            mus, Ls, conformal = self.predictor.predict(query, groundings)
        else:
            mus, Ls = self.predictor.predict(query, groundings)  # type: ignore[assignment]
            conformal = None

        weights = np.asarray([g.confidence for g in groundings], dtype=np.float32)
        total = weights.sum()
        weights = weights / total if total > 0 else np.full_like(weights, 1.0 / len(weights))

        return GMMResult(
            groundings=groundings,
            mus=mus,
            Ls=Ls,
            weights=torch.from_numpy(weights),
            conformal=conformal,
        )
