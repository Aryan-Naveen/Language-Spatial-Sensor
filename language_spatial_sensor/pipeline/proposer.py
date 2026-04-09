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

from abc import ABC, abstractmethod

from language_spatial_sensor.core.schema import Grounding, SpatialQuery


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

