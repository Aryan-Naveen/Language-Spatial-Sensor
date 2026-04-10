"""Proposer: generates a set of Grounding hypotheses from a SpatialQuery.

A Proposer takes a scene (SpatialQuery with gt_anchor_* fields absent or ignored)
and produces a ranked list of Grounding hypotheses that the tensorizer and LSS model
will score.  Each Grounding pins a specific room, a set of anchor objects, and a
(possibly rephrased) language utterance.

Typical inference loop::

    proposer  = MyProposer(...)
    tensorizer = Tensorizer(clip_embedding_map=label_map)

    groundings: list[Grounding] = proposer.propose(query)

    grounded_queries = [
        GroundedQuery(
            scene_id=query.scene_id,
            scene_graph=query.scene_graph,
            pc=query.pc,
            object_split=query.object_split,
            grounding=g,
            target_xyz=query.target_xyz,   # kept for eval; None at pure inference
            target_bbox=query.target_bbox,
        )
        for g in groundings
    ]

    samples  = tensorizer.tensorize_grounded_batch(grounded_queries)
    # → feed samples into CollateFn / LSSModel to get K GaussianPredictions
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

from language_spatial_sensor.core.schema import Grounding, SceneGraph, SpatialQuery


class Proposer(ABC):
    """Abstract base class for all Proposers.

    Subclasses implement :meth:`propose` to enumerate Grounding hypotheses for a
    given query.  The number and quality of hypotheses is entirely up to the
    subclass — the base class imposes no constraints beyond the return type.
    """

    @abstractmethod
    def propose(self, query: SpatialQuery) -> list[Grounding]:
        """Generate Grounding hypotheses for *query*.

        Args:
            query: Scene context.  The ``gt_anchor_*`` fields are intentionally
                   ignored — the proposer must derive groundings from the scene
                   graph and language alone.

        Returns:
            A non-empty list of :class:`Grounding` objects, ordered from most-
            to least-preferred if the proposer has a natural ranking.
        """
        ...


class GroundTruthProposer(Proposer):
    """Proposer that returns a single Grounding built from the query's gt fields.

    Useful as a sanity-check / oracle baseline: plug this into the inference
    loop and the model should perform at (or near) its training-time upper bound.
    """

    def propose(self, query: SpatialQuery) -> list[Grounding]:
        if query.gt_anchor_room_id is None:
            raise ValueError(
                f"GroundTruthProposer requires gt_anchor_room_id "
                f"(scene '{query.scene_id}')"
            )
        return [
            Grounding(
                anchor_room_id=query.gt_anchor_room_id,
                anchor_object_ids=list(query.gt_anchor_object_ids or []),
                language=query.language,
            )
        ]


# ---------------------------------------------------------------------------
# LLM-based proposer
# ---------------------------------------------------------------------------

_SYSTEM = """You are a spatial reasoning module.

You are given:

A scene graph describing a region and its objects.

A natural language utterance introducing a new object.

Your task is to generate K grounded spatial hypotheses that explain how the new object could relate to existing objects in the scene.

A hypothesis must include:

utterance: A fully specified sentence of the form
"there is a <target object> <relation> <existing object(s)>". Infer the target object from the provided utterance.

referred_object_id: A list of object ids the relation is grounded to. Use exactly one id for relations like "near", "on", "against", "above", "below". Use exactly two ids for "between" (both target objects, in any order).

region_id: The region id (integer) of the referred object(s). Same region for all referred objects in that hypothesis.

confidence: A number between 0 and 1 representing plausibility given the scene.

Only use objects that exist in the scene graph.
Do NOT invent objects.
Output MUST be valid JSON.
Return exactly K hypotheses."""

_USER_TEMPLATE = """Scene Graph:
{scene_json}

Utterance:
"{utterance}"

Number of hypotheses (K):
{k_placeholder}

Instructions:
- Generate K diverse and plausible spatial hypotheses.
- Use only relations such as: near, between, above, below, in, or on. Avoid all other relations.
- referred_object_id must always be a list: one object id for single-object relations (e.g. "near the desk"), exactly two object ids for "between" (e.g. "between the desk and the wall" -> ["id1", "id2"]).
- Use only objects present in the scene graph.
- Confidence should reflect spatial plausibility and typical region layout priors.
- Output format (referred_object_id is always a list; region_id is a single integer):

{{
  "hypotheses": [
    {{
      "utterance": "...",
      "referred_object_id": ["<object_id>"],
      "region_id": 0,
      "confidence": 0.8
    }}
  ]
}}"""

_K_DYNAMIC = (
    "as many as appropriate (at least 1, up to 5). "
    "Generate the minimum number of hypotheses necessary to disambiguate the utterance."
)

# Few-shot examples
_FS_OFFICE_SCENE = """{
  "Office": {
    "objects": {
      "0": {"semantics": "desk", "center": [0.0, -1.0, 0.4], "volume": 2.5, "object_id": "0"},
      "1": {"semantics": "monitor", "center": [0.1, -1.0, 0.9], "volume": 0.04, "object_id": "1"},
      "2": {"semantics": "keyboard", "center": [0.1, -0.85, 0.7], "volume": 0.003, "object_id": "2"},
      "3": {"semantics": "chair", "center": [0.8, -0.6, 0.5], "volume": 0.3, "object_id": "3"},
      "4": {"semantics": "wall", "center": [0.0, -2.5, 1.5], "volume": 6.0, "object_id": "4"},
      "5": {"semantics": "chair", "center": [1.8, -1.6, 0.5], "volume": 0.3, "object_id": "5"}
    },
    "region_id": 0
  }
}"""

_FS_BEDROOM_SCENE = """{
  "Bedroom": {
    "objects": {
      "0": {"semantics": "nightstand", "center": [-0.8, -1.1, 0.6], "volume": 0.35, "object_id": "0"},
      "1": {"semantics": "bed", "center": [0.0, -1.2, 0.5], "volume": 3.5, "object_id": "1"},
      "2": {"semantics": "nightstand", "center": [0.8, -1.1, 0.6], "volume": 0.35, "object_id": "2"},
      "3": {"semantics": "lamp", "center": [0.8, -1.1, 1.1], "volume": 0.05, "object_id": "3"},
      "4": {"semantics": "dresser", "center": [-1.2, -0.4, 0.9], "volume": 1.8, "object_id": "4"},
      "5": {"semantics": "chair", "center": [1.4, -0.3, 0.5], "volume": 0.35, "object_id": "5"},
      "6": {"semantics": "window", "center": [0.0, -2.5, 1.4], "volume": 3.0, "object_id": "6"}
    },
    "region_id": 0
  }
}"""

_FS_USER_1 = _USER_TEMPLATE.format(
    scene_json=_FS_OFFICE_SCENE,
    utterance="there is a mouse on the desk",
    k_placeholder=_K_DYNAMIC,
)
_FS_ASST_1 = """{
  "hypotheses": [
    {"utterance": "there is a mouse on the desk", "referred_object_id": ["0"], "region_id": 0, "confidence": 1.0}
  ]
}"""

_FS_USER_2 = _USER_TEMPLATE.format(
    scene_json=_FS_BEDROOM_SCENE,
    utterance="there is a bag on the nightstand",
    k_placeholder=_K_DYNAMIC,
)
_FS_ASST_2 = """{
  "hypotheses": [
    {"utterance": "there is a bag on the nightstand", "referred_object_id": ["0"], "region_id": 0, "confidence": 0.88},
    {"utterance": "there is a bag on the nightstand", "referred_object_id": ["2"], "region_id": 0, "confidence": 0.72}
  ]
}"""


def _scene_graph_to_json(sg: SceneGraph) -> str:
    """Serialize a SceneGraph into JSON for the LLM prompt.

    Groups objects by region and includes semantics, center, volume, and
    object_id — matching the format used in the few-shot examples.
    """
    import numpy as np

    region_map = {r.id: r for r in sg.regions}
    by_region: dict[int, list] = defaultdict(list)
    for obj in sg.objects:
        rid = obj.metadata.get("region_id")
        if rid is not None:
            by_region[int(rid)].append(obj)

    out: dict[str, Any] = {}
    for rid, objects in by_region.items():
        region = region_map.get(rid)
        region_name = region.label if region else f"region_{rid}"
        obj_map: dict[str, Any] = {}
        for obj in objects:
            center = [round(c, 2) for c in obj.position]
            vol = obj.metadata.get("volume")
            if vol is None and obj.bbox is not None:
                corners = np.array(obj.bbox, dtype=np.float32).reshape(8, 3)
                size = corners.max(axis=0) - corners.min(axis=0)
                vol = float(np.prod(size))
            obj_map[str(obj.id)] = {
                "semantics": obj.label,
                "center": center,
                "volume": round(float(vol), 3) if vol else 0.0,
                "object_id": str(obj.id),
            }
        out[region_name] = {"objects": obj_map, "region_id": rid}
    return json.dumps(out, indent=2)


def _parse_llm_response(text: str) -> list[dict[str, Any]]:
    """Extract hypotheses from an LLM JSON response."""
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return []
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
        return []
    try:
        data = json.loads(text[start:end])
        raw = data.get("hypotheses", [])
        if not isinstance(raw, list):
            return []
        out = []
        for h in raw:
            if not isinstance(h, dict):
                continue
            ref_ids = h.get("referred_object_id", h.get("referred_object_ids", []))
            if isinstance(ref_ids, (str, int)):
                ref_ids = [ref_ids]
            out.append({
                "referred_object_id": [str(x) for x in ref_ids],
                "region_id": int(h.get("region_id", 0)),
                "confidence": float(h.get("confidence", 0.5)),
            })
        return out
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def _cache_key(scene_json: str, utterance: str, model: str) -> str:
    raw = f"{model}|{utterance}|{scene_json}"
    return hashlib.sha256(raw.encode()).hexdigest()


class LLMProposer(Proposer):
    """LLM-based proposer: scene + utterance -> K grounding hypotheses.

    Supports OpenAI-compatible APIs (OpenAI, Ollama local models).
    Responses are cached to disk when *cache_dir* is set.

    Args:
        provider:    ``"ollama"`` or ``"openai"``.
        model:       Model name, e.g. ``"qwen2.5:32b"`` or ``"gpt-4o-mini"``.
        base_url:    API endpoint override.  Defaults to localhost:11434 for Ollama.
        api_key:     API key.  Defaults to ``OPENAI_API_KEY`` env var for OpenAI.
        temperature: Sampling temperature.  0 for deterministic.
        cache_dir:   Optional disk cache directory.
        seed:        Random seed for reproducibility (OpenAI only).
        verbose:     Print LLM responses to stdout.
    """

    OLLAMA_BASE_URL = "http://localhost:11434/v1"

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2.5:32b",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        cache_dir: str | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
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

        self._client = None  # lazy init

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url.rstrip("/")
            self._client = OpenAI(**kwargs)
        return self._client

    def _call_llm(self, scene_json: str, utterance: str) -> str:
        """Call the LLM with few-shot prompt and return raw response."""
        user_msg = _USER_TEMPLATE.format(
            scene_json=scene_json,
            utterance=utterance,
            k_placeholder=_K_DYNAMIC,
        )
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _FS_USER_1},
            {"role": "assistant", "content": _FS_ASST_1},
            {"role": "user", "content": _FS_USER_2},
            {"role": "assistant", "content": _FS_ASST_2},
            {"role": "user", "content": user_msg},
        ]
        print(messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if self._seed is not None and self._provider == "openai":
            kwargs["seed"] = self._seed

        client = self._get_client()
        resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    def propose(self, query: SpatialQuery) -> list[Grounding]:
        sg = query.scene_graph
        utterance = query.language
        scene_json = _scene_graph_to_json(sg)

        # Check cache
        key = _cache_key(scene_json, utterance, self._model)
        hypotheses: list[dict] | None = None
        cache_path = None
        if self._cache_dir:
            cache_path = self._cache_dir / f"{key}.json"
            if cache_path.exists():
                try:
                    with open(cache_path) as f:
                        hypotheses = json.load(f).get("hypotheses", [])
                except Exception:
                    pass

        # Call LLM if not cached
        if hypotheses is None:
            raw = self._call_llm(scene_json, utterance)
            if self._verbose:
                print(f"[LLMProposer] utterance={utterance!r}\n{raw}\n")
            hypotheses = _parse_llm_response(raw)
            if cache_path and hypotheses:
                try:
                    with open(cache_path, "w") as f:
                        json.dump({"hypotheses": hypotheses}, f, indent=2)
                except Exception:
                    pass

        if not hypotheses:
            # Fallback: ground to first region, no anchors
            first_region = sg.regions[0].id if sg.regions else 0
            return [Grounding(
                anchor_room_id=first_region,
                anchor_object_ids=[],
                language=utterance,
                confidence=0.1,
            )]

        # Validate object IDs against scene graph
        valid_ids = {obj.id for obj in sg.objects}
        obj_to_region = {
            obj.id: int(obj.metadata["region_id"])
            for obj in sg.objects
            if "region_id" in obj.metadata
        }

        groundings: list[Grounding] = []
        for h in hypotheses:
            ref_ids = [int(x) for x in h["referred_object_id"] if int(x) in valid_ids]
            if not ref_ids:
                continue
            # Resolve region from the anchor objects
            region_id = h.get("region_id", 0)
            if ref_ids:
                region_id = obj_to_region.get(ref_ids[0], region_id)
            groundings.append(Grounding(
                anchor_room_id=region_id,
                anchor_object_ids=ref_ids,
                language=utterance,
                confidence=h.get("confidence", 0.5),
            ))

        if not groundings:
            first_region = sg.regions[0].id if sg.regions else 0
            return [Grounding(
                anchor_room_id=first_region,
                anchor_object_ids=[],
                language=utterance,
                confidence=0.1,
            )]

        # Sort by confidence descending
        groundings.sort(key=lambda g: g.confidence, reverse=True)

        if self._verbose:
            id_to_label = {obj.id: obj.label for obj in sg.objects}
            print(f"[LLMProposer] {len(groundings)} grounding(s) for {utterance!r}:")
            for k, g in enumerate(groundings):
                anchor_semantics = [
                    f"{aid}:{id_to_label.get(aid, '?')}" for aid in g.anchor_object_ids
                ]
                print(f"  [{k}] conf={g.confidence:.2f}  anchors=[{', '.join(anchor_semantics)}]  region={g.anchor_room_id}")
            print()

        return groundings

