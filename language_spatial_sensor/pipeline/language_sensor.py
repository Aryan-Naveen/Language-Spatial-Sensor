"""LanguageSensorPipeline: end-to-end inference from SpatialQuery → GMMPrediction.

Flow:
  1. OllamaProposer generates K proposals from the utterance + masked scene graph
  2. LSSInference runs each proposal through the trained model → GaussianPrediction
  3. Proposals are combined into a GMM weighted by proposal confidence

Typical usage::

    pipeline = LanguageSensorPipeline(proposer, model)
    gmm = pipeline.run(query)
    # gmm.means:   (K, 3)  Gaussian centres in world frame
    # gmm.Ls:      (K, 3, 3) Cholesky factors
    # gmm.weights: (K,)    non-negative, sum to 1
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from language_spatial_sensor.core.schema import SpatialQuery
from language_spatial_sensor.proposer.ollama import OllamaProposer
from language_spatial_sensor.proposer.schema import Proposal
from language_spatial_sensor.pipeline.inference import LSSInference

logger = logging.getLogger(__name__)


@dataclass
class GMMPrediction:
    """A Gaussian Mixture Model prediction over 3-D target position.

    Attributes:
        means:     (K, 3)   centre of each Gaussian component in world frame.
        Ls:        (K, 3, 3) lower-triangular Cholesky factors (Σ_k = L_k @ L_kᵀ).
        weights:   (K,)     non-negative mixture weights, summing to 1.
        proposals: list of K Proposal objects (one per component), in weight order.
        timings:   Per-stage wall-clock times in milliseconds:
                     proposer_ms  — total time in OllamaProposer (HTTP + parse)
                     ollama_http_ms — just the HTTP round-trip
                     model_ms     — total LSS model inference across all proposals
                     model_ms_per_proposal — model_ms / n_proposals
    """

    means:     np.ndarray        # (K, 3)
    Ls:        np.ndarray        # (K, 3, 3)
    weights:   np.ndarray        # (K,)
    proposals: list[Proposal]
    timings:   dict[str, float] = field(default_factory=dict)


class LanguageSensorPipeline:
    """End-to-end LSS pipeline: utterance + masked scene graph → GMMPrediction.

    Args:
        proposer: OllamaProposer instance.
        model:    LSSInference instance (loaded checkpoint).
    """

    def __init__(
        self,
        proposer: OllamaProposer,
        model: LSSInference,
    ) -> None:
        self.proposer = proposer
        self.model    = model

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, query: SpatialQuery) -> GMMPrediction:
        """Run the full pipeline on a single SpatialQuery.

        Returns a GMMPrediction. If the proposer returns no valid proposals,
        falls back to a single uninformative component centred at the scene mean.
        """
        t0 = time.perf_counter()
        proposals, ollama_http_ms = self.proposer.generate_proposals(
            query.language, query.scene_graph
        )
        proposer_ms = (time.perf_counter() - t0) * 1000

        if not proposals:
            logger.warning(
                "No proposals generated for scene '%s' — using scene-mean fallback",
                query.scene_id,
            )
            return self._fallback_prediction(
                query,
                timings={"proposer_ms": proposer_ms, "ollama_http_ms": ollama_http_ms, "model_ms": 0.0},
            )

        means_list:   list[np.ndarray] = []
        Ls_list:      list[np.ndarray] = []
        confs:        list[float]      = []
        model_ms      = 0.0

        for proposal in proposals:
            try:
                t1 = time.perf_counter()
                pred = self.model.predict(proposal, query)
                model_ms += (time.perf_counter() - t1) * 1000
            except Exception as exc:
                logger.warning(
                    "Model inference failed for proposal '%s': %s",
                    proposal.utterance, exc,
                )
                continue
            means_list.append(pred.mu.numpy())        # (3,)
            Ls_list.append(pred.L.numpy())            # (3, 3)
            confs.append(float(proposal.confidence))

        if not means_list:
            logger.warning(
                "All model inference calls failed for scene '%s' — using fallback",
                query.scene_id,
            )
            return self._fallback_prediction(
                query,
                timings={"proposer_ms": proposer_ms, "ollama_http_ms": ollama_http_ms, "model_ms": model_ms},
            )

        n = len(means_list)
        valid_proposals = proposals[:n]
        weights = _softmax(np.array(confs, dtype=np.float32))

        return GMMPrediction(
            means     = np.stack(means_list, axis=0),
            Ls        = np.stack(Ls_list,    axis=0),
            weights   = weights,
            proposals = valid_proposals,
            timings   = {
                "proposer_ms":           proposer_ms,
                "ollama_http_ms":        ollama_http_ms,
                "model_ms":              model_ms,
                "model_ms_per_proposal": model_ms / n if n else 0.0,
            },
        )

    # ── Fallback ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_prediction(
        query: SpatialQuery,
        timings: dict[str, float] | None = None,
    ) -> GMMPrediction:
        """Single isotropic Gaussian centred at the mean of all object positions."""
        positions = np.array(
            [obj.position for obj in query.scene_graph.objects], dtype=np.float32
        )
        if positions.size > 0:
            centre = positions.mean(axis=0)
        else:
            centre = np.zeros(3, dtype=np.float32)

        # Large isotropic uncertainty (3 m std dev per axis)
        L = np.eye(3, dtype=np.float32) * 3.0

        return GMMPrediction(
            means     = centre[np.newaxis],
            Ls        = L[np.newaxis],
            weights   = np.array([1.0], dtype=np.float32),
            proposals = [],
            timings   = timings or {},
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
