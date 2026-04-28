"""Per-predictor conformal superposition.

Mirrors the conformal pattern in :mod:`train` (Mahalanobis-ellipsoid superset
at the target coverage level) but applied to any
:class:`DistributionPredictor` — giving LLM/VLM predictors the same
calibrated coverage guarantee LSS enjoys, so the per-predictor output can
be compared fairly against LSS's conformal output.

Workflow::

    calib_queries = load_eval_queries(..., split="val_seen", ...)   # held out from test
    base = LLMGaussianPredictor(...)
    calibrated = ConformalSuperimposed(
        base=base,
        proposer=GroundTruthProposer(),
        coverage=0.9,
    )
    calibrated.calibrate(calib_queries)    # one-time pass; persists q to disk

    mus, Ls, conformal = calibrated.predict(query, groundings)
    # caller assembles GMMResult(mus, Ls, weights, conformal=conformal)

The calibration step collects the Mahalanobis residual
:math:`m_i = \\sqrt{(y_i-\\mu_i)^{\\mathsf T} \\Sigma_i^{-1} (y_i-\\mu_i)}`
on every calibration query and sets ``q = quantile(m, coverage)``.  At
inference, we attach a :class:`ConformalComponent` carrying ``(μ, L, q, α)``;
``GMMResult`` handles sampling and density accordingly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from language_spatial_sensor.core.schema import Grounding, SpatialQuery
from language_spatial_sensor.pipeline.distribution_predictor import DistributionPredictor
from language_spatial_sensor.pipeline.language_sensor import ConformalComponent
from language_spatial_sensor.pipeline.proposer import Proposer


@dataclass
class ConformalCalibration:
    """Persisted calibration state."""
    q: float
    coverage: float
    n_calibration: int
    predictor_name: str

    def to_dict(self) -> dict:
        return {
            "q": float(self.q),
            "coverage": float(self.coverage),
            "n_calibration": int(self.n_calibration),
            "predictor_name": self.predictor_name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConformalCalibration":
        return cls(
            q=float(d["q"]),
            coverage=float(d["coverage"]),
            n_calibration=int(d["n_calibration"]),
            predictor_name=str(d["predictor_name"]),
        )


class ConformalSuperimposed:
    """Wraps a :class:`DistributionPredictor` with a conformal-uniform tail.

    The wrapped predictor is used to score groundings exactly as before;
    on top of that, every output carries a :class:`ConformalComponent`
    uniform over the Mahalanobis-``q²`` ellipsoid.  The uniform weight
    :math:`\\alpha = 1 - \\text{coverage}` by construction: the Gaussian
    carries ``coverage`` of the mass, the uniform ellipsoid the rest.

    Args:
        base:           Underlying :class:`DistributionPredictor` to wrap.
        proposer:       Proposer used during calibration (typically
                        :class:`GroundTruthProposer` so calibration is on
                        the predictor's native accuracy, not the proposer's).
        coverage:       Target marginal coverage (e.g. 0.9).
        predictor_name: Tag used for the on-disk cache filename.
        cache_dir:      Directory to persist the calibration state.
    """

    def __init__(
        self,
        base: DistributionPredictor,
        proposer: Proposer,
        coverage: float = 0.9,
        predictor_name: str = "predictor",
        cache_dir: str | Path | None = "cache/conformal",
    ) -> None:
        self.base = base
        self.proposer = proposer
        self.coverage = float(coverage)
        self.predictor_name = predictor_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calibration: ConformalCalibration | None = None
        self._try_load_cache()

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _cache_path(self) -> Path | None:
        if not self.cache_dir:
            return None
        safe = self.predictor_name.replace("/", "_")
        return self.cache_dir / f"{safe}.json"

    def _try_load_cache(self) -> None:
        p = self._cache_path()
        if p and p.exists():
            try:
                self.calibration = ConformalCalibration.from_dict(
                    json.loads(p.read_text())
                )
            except Exception:
                self.calibration = None

    def _save_cache(self) -> None:
        p = self._cache_path()
        if p and self.calibration is not None:
            p.write_text(json.dumps(self.calibration.to_dict(), indent=2))

    def calibrate(
        self,
        queries: list[SpatialQuery],
        force: bool = False,
        verbose: bool = True,
    ) -> ConformalCalibration:
        """Compute the Mahalanobis quantile on a held-out split.

        Runs ``self.proposer`` + ``self.base`` on every query, collects
        Mahalanobis residuals of GT vs the **best** grounding's Gaussian
        (highest proposer confidence), and stores the ``coverage`` quantile.
        """
        if self.calibration is not None and not force:
            if verbose:
                print(
                    f"[conformal] using cached q={self.calibration.q:.3f} "
                    f"(n={self.calibration.n_calibration}, cov={self.calibration.coverage})"
                )
            return self.calibration

        residuals: list[float] = []
        n_skipped = 0
        n_empty_groundings = 0
        pbar = tqdm(
            queries,
            desc=f"[conformal] calibrate {self.predictor_name}",
            disable=not verbose,
            unit="q",
        )
        for q in pbar:
            try:
                groundings = self.proposer.propose(q)
                if not groundings:
                    n_empty_groundings += 1
                    pbar.set_postfix(ok=len(residuals), skip=n_skipped, empty=n_empty_groundings)
                    continue
                mus, Ls = self.base.predict(q, groundings[:1])  # best grounding only
                mu = mus[0]
                L = Ls[0]
                y = torch.as_tensor(q.target_xyz, dtype=torch.float32)
                diff = (y - mu).unsqueeze(-1)
                z = torch.linalg.solve_triangular(L, diff, upper=False).squeeze(-1)
                m = float((z * z).sum().clamp(min=1e-12).sqrt())
                if np.isfinite(m):
                    residuals.append(m)
                pbar.set_postfix(ok=len(residuals), skip=n_skipped, empty=n_empty_groundings)
            except Exception as e:
                n_skipped += 1
                pbar.set_postfix(ok=len(residuals), skip=n_skipped, empty=n_empty_groundings)
                if verbose:
                    tqdm.write(f"[conformal] skip {q.scene_id}: {e}")

        if not residuals:
            raise RuntimeError(
                f"Calibration produced no residuals "
                f"(processed {len(queries)}, skipped {n_skipped}, empty groundings {n_empty_groundings})"
            )

        q_val = float(np.quantile(np.asarray(residuals), self.coverage))
        self.calibration = ConformalCalibration(
            q=q_val,
            coverage=self.coverage,
            n_calibration=len(residuals),
            predictor_name=self.predictor_name,
        )
        self._save_cache()
        if verbose:
            print(
                f"[conformal] {self.predictor_name}: q={q_val:.3f} "
                f"from n={len(residuals)} residuals at coverage={self.coverage}"
            )
        return self.calibration

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        query: SpatialQuery,
        groundings: list[Grounding],
    ) -> tuple[torch.Tensor, torch.Tensor, ConformalComponent]:
        """Score groundings and attach a conformal tail.

        Returns:
            ``(mus, Ls, conformal)`` — ``mus`` ``(K, 3)``, ``Ls`` ``(K, 3, 3)``,
            plus a :class:`ConformalComponent` centred on the top-confidence
            grounding's Gaussian.  Caller composes the final :class:`GMMResult`.
        """
        if self.calibration is None:
            raise RuntimeError(
                "ConformalSuperimposed.predict called before calibrate(); "
                "run .calibrate(calibration_queries) first or pre-populate the cache."
            )

        mus, Ls = self.base.predict(query, groundings)

        # Anchor the conformal ellipsoid on the top-confidence grounding.
        top_idx = int(np.argmax([g.confidence for g in groundings]))
        conformal = ConformalComponent(
            mu=mus[top_idx].clone(),
            L=Ls[top_idx].clone(),
            q=float(self.calibration.q),
            weight=float(1.0 - self.coverage),
        )
        return mus, Ls, conformal
