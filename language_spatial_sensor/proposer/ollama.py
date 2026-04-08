"""OllamaProposer: LLM-based spatial proposal generator.

Serializes the masked scene graph to text, prompts Qwen 2.5 32B (via Ollama)
with the original utterance, and parses a structured JSON list of Proposal
objects — each naming a specific relation, region, and anchor object(s).

Typical usage::

    proposer = OllamaProposer(model="qwen2.5:32b")
    proposals = proposer.generate_proposals(utterance, scene_graph)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import requests
from pydantic import ValidationError

from language_spatial_sensor.core.schema import ObjectInfo, RegionInfo, SceneGraph
from .schema import Proposal

logger = logging.getLogger(__name__)


def _scene_hash(scene_graph: SceneGraph) -> str:
    """Stable hash of scene object IDs — used as cache key."""
    ids = tuple(sorted(obj.id for obj in scene_graph.objects))
    return hashlib.md5(str(ids).encode()).hexdigest()[:12]

# ── Scene graph serializer ────────────────────────────────────────────────────

def serialize_scene_graph(scene_graph: SceneGraph) -> str:
    """Convert a SceneGraph to a compact text block for LLM consumption.

    Groups objects by region. Each region block lists its objects with id,
    label, and rounded position. Objects whose region_id is missing are
    grouped under an "unknown" region.

    Example output::

        Region 3 (bedroom) [size: 4.2 x 3.1 x 2.8 m]:
          obj 12: chair at (1.2, 0.5, 0.4)
          obj 14: desk  at (1.8, 0.5, 0.4)
        Region 5 (living room) [size: 6.0 x 4.0 x 2.8 m]:
          obj 21: sofa  at (3.1, 2.0, 0.4)
    """
    # Build region lookup
    region_by_id: dict[int, RegionInfo] = {r.id: r for r in scene_graph.regions}

    # Group objects by region
    region_objects: dict[int | str, list[ObjectInfo]] = {}
    for obj in scene_graph.objects:
        rid = obj.metadata.get("region_id", "unknown")
        region_objects.setdefault(rid, []).append(obj)

    lines: list[str] = []
    # Emit known regions first, in id order
    sorted_rids: list[int | str] = sorted(
        (rid for rid in region_objects if rid != "unknown"),
        key=lambda x: (0, x),
    )
    if "unknown" in region_objects:
        sorted_rids.append("unknown")

    for rid in sorted_rids:
        objs = region_objects[rid]
        if rid == "unknown":
            header = "Region unknown (no region assigned):"
        else:
            region = region_by_id.get(rid)  # type: ignore[arg-type]
            if region is not None:
                size_str = _region_size_str(region)
                header = f"Region {rid} ({region.label}){size_str}:"
            else:
                header = f"Region {rid}:"
        lines.append(header)
        for obj in objs:
            pos = obj.position
            lines.append(
                f"  obj {obj.id}: {obj.label} at "
                f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            )

    return "\n".join(lines)


def _region_size_str(region: RegionInfo) -> str:
    """Return ' [size: X x Y x Z m]' if bbox metadata is available."""
    m = region.metadata
    keys_min = ["bbox_x_min", "bbox_y_min", "bbox_z_min"]
    keys_max = ["bbox_x_max", "bbox_y_max", "bbox_z_max"]
    if not all(k in m for k in keys_min + keys_max):
        return ""
    sizes = [
        abs(m[kmax] - m[kmin])
        for kmin, kmax in zip(keys_min, keys_max)
    ]
    return f" [size: {sizes[0]:.1f} x {sizes[1]:.1f} x {sizes[2]:.1f} m]"


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a spatial reasoning module. Given a scene graph and an utterance describing a \
missing object, output the minimum number of spatial proposals needed to cover all \
plausible interpretations — no more, no less.

Rules:
- Each proposal must use exactly ONE relation from: between, near, in, on, above, below
- "between" requires exactly 2 anchor_ids; all other relations require exactly 1
- region_id and anchor_ids must reference IDs present in the scene graph
- Confidence reflects semantic plausibility (0.0 = impossible, 1.0 = certain)
- Output ONLY a JSON array — no markdown, no explanation, no extra keys

Each element: {"utterance": "...", "relation": "...", "region_id": 0, "anchor_ids": [0], "confidence": 0.0}
"""


# ── OllamaProposer ────────────────────────────────────────────────────────────

class OllamaProposer:
    """Generate spatial proposals by querying an Ollama-hosted LLM.

    Args:
        model:        Ollama model tag, e.g. "qwen2.5:32b".
        base_url:     Base URL of the Ollama API server.
        max_proposals: Maximum number of proposals to request from the LLM.
        timeout:      HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "qwen2.5:32b",
        base_url: str = "http://localhost:11434",
        max_proposals: int = 8,
        timeout: int = 120,
        num_predict: int = 512,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_proposals = max_proposals
        self.timeout = timeout
        self.num_predict = num_predict

        # Pre-build the set of valid object/region IDs for quick validation
        self._valid_obj_ids: set[int] = set()
        self._valid_region_ids: set[int] = set()

        # In-memory response cache: (utterance, scene_hash) → list[Proposal]
        self._cache: dict[tuple[str, str], list[Proposal]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_proposals(
        self,
        utterance: str,
        scene_graph: SceneGraph,
        max_retries: int = 1,
    ) -> tuple[list[Proposal], float]:
        """Return (proposals, http_ms) for the given utterance.

        http_ms is the raw Ollama HTTP round-trip time, or 0.0 on a cache hit.
        Results are cached in memory keyed on (utterance, scene_hash).
        If the LLM returns no valid proposals, retries up to max_retries times
        with an explicit reminder before giving up and returning ([], 0.0).
        """
        self._valid_obj_ids    = {obj.id for obj in scene_graph.objects}
        self._valid_region_ids = {r.id   for r in scene_graph.regions}

        cache_key = (utterance, _scene_hash(scene_graph))
        if cache_key in self._cache:
            return self._cache[cache_key], 0.0

        scene_text = serialize_scene_graph(scene_graph)
        user_message = self._build_user_message(utterance, scene_text)

        total_http_ms = 0.0
        for attempt in range(1 + max_retries):
            try:
                raw_json, http_ms = self._call_ollama(user_message)
            except Exception as exc:
                logger.warning("Ollama call failed (attempt %d): %s", attempt + 1, exc)
                break
            total_http_ms += http_ms
            proposals = self._parse_and_validate(raw_json)
            if proposals:
                self._cache[cache_key] = proposals
                return proposals, total_http_ms
            if attempt < max_retries:
                logger.warning(
                    "No valid proposals on attempt %d — retrying with reminder.",
                    attempt + 1,
                )
                user_message = self._build_retry_message(utterance, scene_text)

        logger.warning("No valid proposals after %d attempt(s) for: %s", 1 + max_retries, utterance)
        return [], total_http_ms

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_user_message(self, utterance: str, scene_text: str) -> str:
        return (
            f"Scene graph:\n{scene_text}\n\n"
            f'Utterance: "{utterance}"\n\n'
            f"Generate as many proposals as needed to cover all plausible anchor "
            f"interpretations — at least 1, at most {self.max_proposals}. "
            f"Omit redundant or implausible proposals. JSON array only."
        )

    def _build_retry_message(self, utterance: str, scene_text: str) -> str:
        return (
            f"Scene graph:\n{scene_text}\n\n"
            f'Utterance: "{utterance}"\n\n'
            f"Your previous response produced no valid proposals. "
            f"You MUST return a JSON array with at least 1 proposal using only "
            f"object IDs and region IDs present in the scene graph above. "
            f"JSON array only — no explanation."
        )

    def _call_ollama(self, user_message: str) -> str:
        """POST to Ollama and return the generated text.

        Tries /api/chat first (Ollama ≥ 0.1.14); falls back to /api/generate
        for older versions.
        """
        # Try /api/chat first
        chat_payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            "stream": False,
            "format": "json",
            "keep_alive": -1,   # keep model loaded in VRAM between calls
            "options": {
                "temperature": 0.2,
                "num_predict": self.num_predict,  # cap output tokens
            },
        }
        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=chat_payload,
            timeout=self.timeout,
        )
        if resp.status_code == 404:
            return self._call_generate(user_message)
        resp.raise_for_status()
        http_ms = (time.perf_counter() - t0) * 1000
        return resp.json()["message"]["content"], http_ms

    def _call_generate(self, user_message: str) -> tuple[str, float]:
        """POST to /api/generate (Ollama < 0.1.14 fallback)."""
        prompt = f"{_SYSTEM_PROMPT}\n\n{user_message}"
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": -1,
            "options": {
                "temperature": 0.2,
                "num_predict": self.num_predict,
            },
        }
        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        http_ms = (time.perf_counter() - t0) * 1000
        return resp.json()["response"], http_ms

    def _parse_and_validate(self, raw: str) -> list[Proposal]:
        """Parse the LLM's JSON output into a list of validated Proposal objects."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse LLM JSON response: %s\nRaw: %.200s", exc, raw)
            return []

        # The model should return a JSON array; handle both array and {"proposals": [...]}
        if isinstance(parsed, dict):
            # Try common wrapper keys
            for key in ("proposals", "results", "items"):
                if key in parsed and isinstance(parsed[key], list):
                    parsed = parsed[key]
                    break
            else:
                # Single-proposal dict
                parsed = [parsed]

        if not isinstance(parsed, list):
            logger.warning("LLM returned unexpected JSON type: %s", type(parsed))
            return []

        proposals: list[Proposal] = []
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue
            try:
                proposal = Proposal.model_validate(item)
            except ValidationError as exc:
                logger.debug("Proposal %d failed validation: %s", i, exc)
                continue

            # Filter out proposals that reference non-existent IDs
            if proposal.region_id not in self._valid_region_ids:
                logger.debug(
                    "Proposal %d dropped: region_id %d not in scene graph",
                    i, proposal.region_id,
                )
                continue
            if not all(aid in self._valid_obj_ids for aid in proposal.anchor_ids):
                logger.debug(
                    "Proposal %d dropped: anchor_ids %s not all in scene graph",
                    i, proposal.anchor_ids,
                )
                continue

            proposals.append(proposal)

        proposals.sort(key=lambda p: p.confidence, reverse=True)
        return proposals[: self.max_proposals]
