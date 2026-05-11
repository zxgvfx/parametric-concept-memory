"""Tier-D cook wrappers for the space-domain heads (paper §6.2)."""
from __future__ import annotations

from pcm.concept_graph import ConceptGraph
from pcm.graph_eval import GraphEvaluator
from pcm.heads.cook_factory import (
    MLPBackbone,
    copy_three_linears,
    make_cook_subgraph,
)

from ._config import FACET_DIST, FACET_MOVE
from .heads import DistanceHead, MoveHead


__all__ = ["build_move_cook", "build_dist_cook"]


def build_move_cook(
    cg: ConceptGraph,
    head: MoveHead,
    *,
    node_id: str = "muscle.cook.space_move",
    backbone_key: str = "space_move_mlp",
    evaluator: GraphEvaluator | None = None,
) -> tuple[str, MLPBackbone, GraphEvaluator]:
    """Cook wrapper for :class:`MoveHead` (5-class direction)."""
    backbone = MLPBackbone(
        in_dim=2 * head.facet_dim,
        hidden_dim=head.fc1.out_features,
        out_dim=head.fc3.out_features,
    )
    copy_three_linears(head.fc1, head.fc2, head.fc3, backbone)
    nid, ev = make_cook_subgraph(
        cg,
        node_id=node_id,
        inputs=["ids_a", "ids_b"],
        facet_collapses=[("ids_a", FACET_MOVE), ("ids_b", FACET_MOVE)],
        backbone=backbone,
        backbone_key=backbone_key,
        evaluator=evaluator,
    )
    return nid, backbone, ev


def build_dist_cook(
    cg: ConceptGraph,
    head: DistanceHead,
    *,
    node_id: str = "muscle.cook.space_dist",
    backbone_key: str = "space_dist_mlp",
    evaluator: GraphEvaluator | None = None,
) -> tuple[str, MLPBackbone, GraphEvaluator]:
    """Cook wrapper for :class:`DistanceHead` (9-class L1)."""
    backbone = MLPBackbone(
        in_dim=2 * head.facet_dim,
        hidden_dim=head.fc1.out_features,
        out_dim=head.fc3.out_features,
    )
    copy_three_linears(head.fc1, head.fc2, head.fc3, backbone)
    nid, ev = make_cook_subgraph(
        cg,
        node_id=node_id,
        inputs=["ids_a", "ids_b"],
        facet_collapses=[("ids_a", FACET_DIST), ("ids_b", FACET_DIST)],
        backbone=backbone,
        backbone_key=backbone_key,
        evaluator=evaluator,
    )
    return nid, backbone, ev
