"""DistributionPredictor: per-grounding (mu, L) prediction.

Parallel to :class:`Proposer`.  A Proposer decides *which* groundings to score;
a DistributionPredictor scores *one grounding at a time* (or a batch of them)
into a single Gaussian ``(mu, L)`` in world frame.

Typical inference loop (see :class:`evaluation.approach.GMMApproach`)::

    groundings = proposer.propose(query)
    mus, Ls    = predictor.predict(query, groundings)   # (K,3), (K,3,3)
    weights    = softmax([g.confidence for g in groundings])
    gmm        = GMMResult(groundings, mus, Ls, weights)

Implementations:
    - :class:`LSSGaussianPredictor` — the learned LSS model.
    - ``LLMGaussianPredictor``      — GPT-5.2 text predictor (separate module).
    - ``VLMGaussianPredictor``      — Ollama VLM + BEV image (separate module).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

from language_spatial_sensor.core.schema import (
    Grounding,
    GroundedQuery,
    SpatialQuery,
    TensorizerOutput,
)
from language_spatial_sensor.models.components.heads import GaussianPrediction
from language_spatial_sensor.models.config import LSSConfig
from language_spatial_sensor.models.lss_model import LSSModel
from language_spatial_sensor.pipeline.tensorizer import Tensorizer
from language_spatial_sensor.training.dataset import CollateFn


class DistributionPredictor(ABC):
    """Abstract per-grounding distribution predictor.

    Subclasses score *K* groundings into a stack of Gaussians ``(mus, Ls)``
    in world frame.  Implementations may batch internally (e.g. the learned
    LSS model) or loop (e.g. calling an LLM once per grounding).
    """

    @abstractmethod
    def predict(
        self,
        query: SpatialQuery,
        groundings: list[Grounding],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score *groundings* for *query*.

        Args:
            query:      Scene context (scene_graph, pc, object_split, language, ...).
            groundings: K groundings to score.

        Returns:
            ``(mus, Ls)``:
                - ``mus``: ``(K, 3)`` float32 CPU tensor, means in world frame.
                - ``Ls``:  ``(K, 3, 3)`` float32 CPU tensor, lower-triangular
                  Cholesky factors with ``Σ = L @ Lᵀ``, world frame.
        """
        ...


# ---------------------------------------------------------------------------
# Helpers shared with LanguageSpatialSensor
# ---------------------------------------------------------------------------

def _to_device(batch: TensorizerOutput, device: torch.device) -> TensorizerOutput:
    return replace(batch, **{
        f.name: getattr(batch, f.name).to(device)
        for f in fields(TensorizerOutput)
    })


def _clean_state_dict(sd: dict) -> dict:
    cleaned = {}
    for k, v in sd.items():
        k = k.replace("_orig_mod.", "").replace("module.", "")
        cleaned[k] = v
    return cleaned


def _resolve_clip_map(
    clip_label_map,
    checkpoint_path: Path,
) -> dict[str, np.ndarray]:
    return torch.load('/home/aryannav/mit/research/langmap/src/lss/cache/clip_label_map.pt', weights_only=False)
    # if isinstance(clip_label_map, dict):
    #     return clip_label_map
    # if isinstance(clip_label_map, (str, Path)):
    #     return torch.load(clip_label_map, weights_only=False)
    # for candidate in [
    #     checkpoint_path.parent / "clip_label_map.pt",
    #     checkpoint_path.parent.parent / "clip_label_map.pt",
    # ]:
    #     if candidate.exists():
    #         return torch.load(candidate, weights_only=False)
    # raise FileNotFoundError(
    #     "Could not find clip_label_map.pt. Pass it explicitly via clip_label_map=."
    # )


# ---------------------------------------------------------------------------
# LSSGaussianPredictor
# ---------------------------------------------------------------------------

class LSSGaussianPredictor(DistributionPredictor):
    """Learned LSS model as a per-grounding Gaussian predictor.

    Loads a training checkpoint (``best.pt``) and runs the same forward pass
    used by :class:`LanguageSpatialSensor`, but with the proposer / GMM-assembly
    logic stripped out.  Batches all K groundings through one forward call.

    Args:
        checkpoint_path: Path to a training checkpoint.
        clip_label_map:  Dict, path to .pt file, or ``None`` (auto-discovered
                         alongside the checkpoint).
        device:          ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        clip_label_map: dict[str, np.ndarray] | str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)

        ckpt_path = Path(checkpoint_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_cfg_dict = ckpt["cfg"]["model"]
        self.cfg = LSSConfig(**model_cfg_dict)

        self.model = LSSModel(self.cfg)
        self.model.load_state_dict(_clean_state_dict(ckpt["model"]))
        self.model.to(self.device).eval()

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

    def predict(
        self,
        query: SpatialQuery,
        groundings: list[Grounding],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not groundings:
            raise ValueError("LSSGaussianPredictor.predict called with no groundings")

        grounded_queries = [
            GroundedQuery(
                scene_id=query.scene_id,
                scene_graph=query.scene_graph,
                pc=query.pc,
                object_split=query.object_split,
                grounding=g,
                target_xyz=query.target_xyz,
                target_bbox=query.target_bbox,
            )
            for g in groundings
        ]
        samples = self.tensorizer.tensorize_grounded_batch(grounded_queries)
        batch = self.collate_fn(samples)
        batch = _to_device(batch, self.device)

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

        return pred.mu.float().cpu(), pred.L.float().cpu()


# ---------------------------------------------------------------------------
# LLM / VLM shared helpers
# ---------------------------------------------------------------------------

_MIN_EIGENVAL = 0.01   # m²; floor for PSD projection of LLM/VLM-returned Σ
_DEFAULT_SIGMA = 0.5   # m; fallback isotropic σ if parsing fails


def _project_to_psd_cholesky(
    sigma_raw: list[list[float]] | np.ndarray,
    min_eigval: float = _MIN_EIGENVAL,
) -> torch.Tensor:
    """Project a (possibly non-PSD) 3x3 matrix onto the nearest PSD and
    return its lower-triangular Cholesky factor.
    """
    S = np.asarray(sigma_raw, dtype=np.float64).reshape(3, 3)
    # Symmetrize — LLMs often emit rounded entries that break symmetry.
    S = 0.5 * (S + S.T)
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.clip(eigvals, min_eigval, None)
    S_psd = (eigvecs * eigvals) @ eigvecs.T
    # One more symmetrize to kill round-off.
    S_psd = 0.5 * (S_psd + S_psd.T)
    L = np.linalg.cholesky(S_psd).astype(np.float32)
    return torch.from_numpy(L)


def _fallback_L(scale: float = _DEFAULT_SIGMA) -> torch.Tensor:
    return torch.eye(3, dtype=torch.float32) * scale


def _parse_mu_sigma(obj: dict) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Parse ``{"mu": [...], "sigma": [[...]...]}`` -> ``(mu, L)``.

    Returns ``None`` if the payload is unusable (caller should fall back).
    """
    mu_raw = obj.get("mu") or obj.get("mean") or obj.get("center")
    sig_raw = obj.get("sigma") or obj.get("cov") or obj.get("covariance")
    if mu_raw is None or sig_raw is None:
        return None
    try:
        mu_arr = np.asarray(mu_raw, dtype=np.float32).reshape(3)
    except Exception:
        return None
    # sigma may be a 3x3 matrix OR a length-3 diagonal vector.
    try:
        sig_np = np.asarray(sig_raw, dtype=np.float64)
    except Exception:
        return None
    if sig_np.shape == (3,):
        sig_np = np.diag(np.clip(sig_np, _MIN_EIGENVAL, None))
    if sig_np.shape != (3, 3):
        return None
    try:
        L = _project_to_psd_cholesky(sig_np)
    except np.linalg.LinAlgError:
        return None
    return torch.from_numpy(mu_arr), L


def _grounding_summary(grounding: Grounding, sg) -> str:
    """Human-readable summary of a grounding for the LLM prompt."""
    id_to_label = {obj.id: obj.label for obj in sg.objects}
    region_map = {r.id: r.label for r in sg.regions}
    anchors = [f"{aid}:{id_to_label.get(aid, '?')}" for aid in grounding.anchor_object_ids]
    region = region_map.get(grounding.anchor_room_id, f"region_{grounding.anchor_room_id}")
    return (
        f'region_id={grounding.anchor_room_id} ("{region}"), '
        f'anchor_object_ids=[{", ".join(anchors)}]'
    )


# ---------------------------------------------------------------------------
# LLMGaussianPredictor (text-only)
# ---------------------------------------------------------------------------

_GAUSS_SYSTEM = """You are a spatial reasoning module that predicts the 3D location of an object described by a natural-language utterance.

Given:
- A scene graph: objects (with world-frame centres) grouped by region.
- An utterance introducing a new (target) object.
- A grounding: the region and anchor object(s) the utterance refers to (assumed correct).

Output a 3D Gaussian distribution N(μ, Σ) over the TARGET object's world-frame position.

Format strictly as JSON:
{
  "mu":    [x, y, z],                                    # world-frame centre, in metres
  "sigma": [[s11,s12,s13],[s12,s22,s23],[s13,s23,s33]],  # symmetric PSD covariance, in m²
  "explanation": "..."                                    # one short sentence
}

Guidance for Σ:
- Use tighter σ (e.g. 0.05–0.2 m along each axis) for precise relations like "on", "in".
- Use looser σ (e.g. 0.5–2.0 m) for vague relations like "near", "around".
- Σ must be symmetric and positive semi-definite.
- Use anchor object centres + typical placement priors to decide μ."""


_GAUSS_USER_TEMPLATE = """Scene Graph:
{scene_json}

Utterance:
"{utterance}"

Grounding (assumed correct):
{grounding_desc}

Return ONE JSON object with keys "mu", "sigma", "explanation"."""


class LLMGaussianPredictor(DistributionPredictor):
    """GPT-style text LLM as a per-grounding Gaussian predictor.

    Prompts the LLM with scene graph + utterance + grounding and parses
    ``{mu, sigma, explanation}`` into a :class:`GaussianPrediction`-style
    output.  Non-PSD Σ are projected onto the nearest PSD.  Responses are
    disk-cached by ``(scene_json, utterance, grounding, model)``.

    Args:
        provider:    ``"openai"`` or ``"ollama"``.
        model:       Model name (e.g. ``"gpt-5.2"`` or ``"qwen2.5:32b"``).
        base_url:    API endpoint override.  Defaults to localhost:11434 for Ollama.
        api_key:     API key.  Defaults to ``OPENAI_API_KEY`` env var.
        temperature: Sampling temperature.
        cache_dir:   Optional on-disk cache directory.
        verbose:     Print raw LLM responses.
    """

    OLLAMA_BASE_URL = "http://localhost:11434/v1"

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-5.2",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        cache_dir: str | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        import os

        self._provider = provider.strip().lower()
        self._model = model
        self._temperature = temperature
        self._seed = seed
        self._verbose = verbose

        if self._provider == "ollama":
            self._base_url = base_url or self.OLLAMA_BASE_URL
            self._api_key = api_key or "ollama"
        else:
            self._base_url = base_url
            self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = None

    # -- openai-compatible client ------------------------------------------
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs: dict = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url.rstrip("/")
            self._client = OpenAI(**kwargs)
        return self._client

    def _cache_key(self, scene_json: str, utterance: str, grounding_desc: str) -> str:
        import hashlib
        raw = f"{self._model}|{utterance}|{grounding_desc}|{scene_json}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _call_llm(self, scene_json: str, utterance: str, grounding_desc: str) -> str:
        user_msg = _GAUSS_USER_TEMPLATE.format(
            scene_json=scene_json,
            utterance=utterance,
            grounding_desc=grounding_desc,
        )
        messages = [
            {"role": "system", "content": _GAUSS_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._seed is not None and self._provider == "openai":
            kwargs["seed"] = self._seed

        client = self._get_client()
        resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    def _predict_one(
        self,
        scene_json: str,
        utterance: str,
        grounding: Grounding,
        grounding_desc: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import json

        key = self._cache_key(scene_json, utterance, grounding_desc)
        cache_path = self._cache_dir / f"{key}.json" if self._cache_dir else None

        parsed: dict | None = None
        if cache_path and cache_path.exists():
            try:
                parsed = json.loads(cache_path.read_text())
            except Exception:
                parsed = None

        if parsed is None:
            raw = self._call_llm(scene_json, utterance, grounding_desc)
            if self._verbose:
                print(f"[LLMGaussianPredictor] {utterance!r} // {grounding_desc}\n{raw}\n")
            parsed = _extract_json_object(raw)
            if cache_path and parsed is not None:
                try:
                    cache_path.write_text(json.dumps(parsed, indent=2))
                except Exception:
                    pass

        if parsed is not None:
            result = _parse_mu_sigma(parsed)
            if result is not None:
                return result

        # Fallback: anchor centroid + isotropic σ.
        return _anchor_centroid_fallback(grounding, self._scene_graph)

    def predict(
        self,
        query: SpatialQuery,
        groundings: list[Grounding],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from language_spatial_sensor.pipeline.proposer import _scene_graph_to_json

        scene_json = _scene_graph_to_json(query.scene_graph)
        self._scene_graph = query.scene_graph  # used by fallback in _predict_one

        mus: list[torch.Tensor] = []
        Ls: list[torch.Tensor] = []
        for g in groundings:
            desc = _grounding_summary(g, query.scene_graph)
            mu, L = self._predict_one(scene_json, query.language, g, desc)
            mus.append(mu)
            Ls.append(L)
        return torch.stack(mus, dim=0), torch.stack(Ls, dim=0)


# ---------------------------------------------------------------------------
# Straight-shot LLM → full GMM (no proposer)
# ---------------------------------------------------------------------------

_FULL_GMM_SYSTEM = """You are a spatial reasoning module that predicts the 3D location of a target object described by a natural-language utterance, *without* being told which anchor objects the utterance refers to.

Given:
- A scene graph: objects (with world-frame centres) grouped by region.
- An utterance introducing a new target object.

Output a Gaussian mixture model over the new target object's world-frame position — one component per plausible grounding of the utterance.

Format strictly as JSON:
{
  "components": [
    {
      "mu":    [x, y, z],
      "sigma": [[...3x3...]],
      "weight": 0.5,
      "explanation": "..."
    },
    ...
  ]
}

- Weights must be non-negative and sum to 1.
- Return at least 1 and at most 5 components (one per plausible anchor).
- Σ must be symmetric and positive semi-definite."""


_FULL_GMM_USER_TEMPLATE = """Scene Graph:
{scene_json}

Utterance:
"{utterance}"

Return ONE JSON object with key "components"."""


class LLMFullGMMPredictor:
    """Straight-shot LLM: utterance + scene -> full GMM in a single call.

    No proposer / grounding step.  The LLM must enumerate its own groundings
    internally and emit a mixture directly.  Used as the "straight_llm"
    baseline in Table 2.
    """

    OLLAMA_BASE_URL = "http://localhost:11434/v1"

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-5.2",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        cache_dir: str | None = None,
        seed: int = 42,
        verbose: bool = False,
        max_components: int = 5,
    ) -> None:
        import os

        self._provider = provider.strip().lower()
        self._model = model
        self._temperature = temperature
        self._seed = seed
        self._verbose = verbose
        self._max_components = max_components

        if self._provider == "ollama":
            self._base_url = base_url or self.OLLAMA_BASE_URL
            self._api_key = api_key or "ollama"
        else:
            self._base_url = base_url
            self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs: dict = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url.rstrip("/")
            self._client = OpenAI(**kwargs)
        return self._client

    def _cache_key(self, scene_json: str, utterance: str) -> str:
        import hashlib
        raw = f"full_gmm|{self._model}|{utterance}|{scene_json}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _call_llm(self, scene_json: str, utterance: str) -> str:
        user_msg = _FULL_GMM_USER_TEMPLATE.format(
            scene_json=scene_json, utterance=utterance,
        )
        messages = [
            {"role": "system", "content": _FULL_GMM_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._seed is not None and self._provider == "openai":
            kwargs["seed"] = self._seed

        client = self._get_client()
        resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    def predict(self, query: SpatialQuery):
        """Returns a GMMResult. Imported lazily to avoid a module cycle."""
        import json

        from language_spatial_sensor.pipeline.language_sensor import GMMResult
        from language_spatial_sensor.pipeline.proposer import _scene_graph_to_json

        sg = query.scene_graph
        scene_json = _scene_graph_to_json(sg)

        key = self._cache_key(scene_json, query.language)
        cache_path = self._cache_dir / f"{key}.json" if self._cache_dir else None

        parsed: dict | None = None
        if cache_path and cache_path.exists():
            try:
                parsed = json.loads(cache_path.read_text())
            except Exception:
                parsed = None

        if parsed is None:
            raw = self._call_llm(scene_json, query.language)
            if self._verbose:
                print(f"[LLMFullGMMPredictor] {query.language!r}\n{raw}\n")
            parsed = _extract_json_object(raw)
            if cache_path and parsed is not None:
                try:
                    cache_path.write_text(json.dumps(parsed, indent=2))
                except Exception:
                    pass

        components = (parsed or {}).get("components") or []

        mus: list[torch.Tensor] = []
        Ls: list[torch.Tensor] = []
        weights: list[float] = []
        for comp in components[: self._max_components]:
            mu_L = _parse_mu_sigma(comp) if isinstance(comp, dict) else None
            if mu_L is None:
                continue
            mu, L = mu_L
            w = float(comp.get("weight", 1.0))
            mus.append(mu)
            Ls.append(L)
            weights.append(max(w, 0.0))

        if not mus:
            # Fallback: scene centroid + broad σ.
            centroid = torch.tensor(
                [np.mean([o.position for o in sg.objects], axis=0)
                 if sg.objects else [0.0, 0.0, 0.0]],
                dtype=torch.float32,
            ).reshape(3)
            mus = [centroid]
            Ls = [_fallback_L(scale=1.0)]
            weights = [1.0]

        w = np.asarray(weights, dtype=np.float32)
        w = w / w.sum() if w.sum() > 0 else np.full_like(w, 1.0 / len(w))

        # Synthesize a placeholder Grounding list so downstream code has something
        # to carry — these straight-shot components have no anchor binding.
        groundings = [
            Grounding(
                anchor_room_id=(sg.regions[0].id if sg.regions else 0),
                anchor_object_ids=[],
                language=query.language,
                confidence=float(w_i),
            )
            for w_i in w
        ]

        return GMMResult(
            groundings=groundings,
            mus=torch.stack(mus, dim=0),
            Ls=torch.stack(Ls, dim=0),
            weights=torch.from_numpy(w),
        )


# ---------------------------------------------------------------------------
# VLMGaussianPredictor (Ollama VLM + BEV image)
# ---------------------------------------------------------------------------

_VLM_SYSTEM = """You are a spatial reasoning module that predicts the 3D location of a target object described by a natural-language utterance.

You are given:
- A Bird's-Eye-View (BEV) image of the region, with anchor object(s) highlighted.
- A natural-language utterance introducing the target object.
- A text description of the anchors and region.

Output a 3D Gaussian N(μ, Σ) over the target object's world-frame position.

The BEV image uses world-frame (x, y) coordinates on the image axes; z (height) must be inferred from context (floor, table-top, etc.).

Format strictly as JSON:
{
  "mu":    [x, y, z],                                    # world frame, metres
  "sigma": [[s11,s12,s13],[s12,s22,s23],[s13,s23,s33]],  # symmetric PSD, m²
  "explanation": "..."
}"""


_VLM_USER_TEMPLATE = """Utterance:
"{utterance}"

Grounding (assumed correct):
{grounding_desc}

Return ONE JSON object with keys "mu", "sigma", "explanation"."""


class VLMGaussianPredictor(DistributionPredictor):
    """Vision-language model as a per-grounding Gaussian predictor.

    Renders a BEV with anchors highlighted via :func:`viz.bev.render_bev`,
    base64-encodes the PNG, and sends it to an Ollama VLM endpoint along
    with the utterance and a textual grounding description.  Parses
    ``{mu, sigma, explanation}`` from the response (same schema as
    :class:`LLMGaussianPredictor`).

    Args:
        provider:    ``"ollama"`` or ``"openai"``.
        model:       Vision model name (e.g. ``"qwen2.5vl:32b"``).
        base_url:    API endpoint override.
        api_key:     API key (Ollama uses sentinel ``"ollama"``).
        temperature: Sampling temperature.
        cache_dir:   Optional on-disk cache directory.
        bev_px:      BEV image resolution.
        verbose:     Print raw VLM responses.
    """

    OLLAMA_BASE_URL = "http://localhost:11434/v1"

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2.5vl:32b",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        cache_dir: str | None = None,
        seed: int = 42,
        bev_resolution: float = 0.10,
        verbose: bool = False,
    ) -> None:
        import os

        self._provider = provider.strip().lower()
        self._model = model
        self._temperature = temperature
        self._seed = seed
        self._bev_resolution = bev_resolution
        self._verbose = verbose

        if self._provider == "ollama":
            self._base_url = base_url or self.OLLAMA_BASE_URL
            self._api_key = api_key or "ollama"
        else:
            self._base_url = base_url
            self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs: dict = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url.rstrip("/")
            self._client = OpenAI(**kwargs)
        return self._client

    def _cache_key(
        self, utterance: str, grounding_desc: str, bev_hash: str,
    ) -> str:
        import hashlib
        raw = f"vlm|{self._model}|{utterance}|{grounding_desc}|{bev_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _render_bev_png(self, query: SpatialQuery, grounding: Grounding) -> bytes:
        """Render BEV with the grounding's anchors highlighted; return PNG bytes."""
        import io

        from viz.bev import render_bev

        import matplotlib.pyplot as plt

        fig = render_bev(
            query=query,
            resolution=self._bev_resolution,
            anchor_highlight=True,
            highlight_object_ids=set(grounding.anchor_object_ids),
            show_target=False,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        return buf.getvalue()

    def _call_vlm(
        self,
        png_bytes: bytes,
        utterance: str,
        grounding_desc: str,
    ) -> str:
        import base64

        b64 = base64.b64encode(png_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        user_msg = _VLM_USER_TEMPLATE.format(
            utterance=utterance, grounding_desc=grounding_desc,
        )
        messages = [
            {"role": "system", "content": _VLM_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_msg},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._seed is not None and self._provider == "openai":
            kwargs["seed"] = self._seed

        client = self._get_client()
        resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    def _predict_one(
        self,
        query: SpatialQuery,
        grounding: Grounding,
        grounding_desc: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import hashlib
        import json

        png_bytes = self._render_bev_png(query, grounding)
        bev_hash = hashlib.sha256(png_bytes).hexdigest()[:16]

        key = self._cache_key(query.language, grounding_desc, bev_hash)
        cache_path = self._cache_dir / f"{key}.json" if self._cache_dir else None

        parsed: dict | None = None
        if cache_path and cache_path.exists():
            try:
                parsed = json.loads(cache_path.read_text())
            except Exception:
                parsed = None

        if parsed is None:
            raw = self._call_vlm(png_bytes, query.language, grounding_desc)
            if self._verbose:
                print(f"[VLMGaussianPredictor] {query.language!r} // {grounding_desc}\n{raw}\n")
            parsed = _extract_json_object(raw)
            if cache_path and parsed is not None:
                try:
                    cache_path.write_text(json.dumps(parsed, indent=2))
                except Exception:
                    pass

        if parsed is not None:
            result = _parse_mu_sigma(parsed)
            if result is not None:
                return result

        return _anchor_centroid_fallback(grounding, query.scene_graph)

    def predict(
        self,
        query: SpatialQuery,
        groundings: list[Grounding],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mus: list[torch.Tensor] = []
        Ls: list[torch.Tensor] = []
        for g in groundings:
            desc = _grounding_summary(g, query.scene_graph)
            mu, L = self._predict_one(query, g, desc)
            mus.append(mu)
            Ls.append(L)
        return torch.stack(mus, dim=0), torch.stack(Ls, dim=0)


# ---------------------------------------------------------------------------
# Shared JSON-extraction + fallback helpers
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> dict | None:
    """Pull the first balanced-brace JSON object out of free-form text."""
    import json

    text = text.strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None
    try:
        return json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        return None


def _anchor_centroid_fallback(grounding: Grounding, sg) -> tuple[torch.Tensor, torch.Tensor]:
    """Last-resort: place μ at anchor centroid (or region centre) with σ=0.5m."""
    id_to_pos = {obj.id: np.asarray(obj.position, dtype=np.float32) for obj in sg.objects}
    anchor_positions = [id_to_pos[aid] for aid in grounding.anchor_object_ids if aid in id_to_pos]
    if anchor_positions:
        mu = torch.from_numpy(np.mean(anchor_positions, axis=0).astype(np.float32))
    else:
        region_map = {r.id: np.asarray(r.position, dtype=np.float32) for r in sg.regions}
        region_pos = region_map.get(grounding.anchor_room_id)
        mu = torch.from_numpy(region_pos.astype(np.float32)) if region_pos is not None \
            else torch.zeros(3, dtype=torch.float32)
    return mu, _fallback_L()
