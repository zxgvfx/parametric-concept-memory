"""Tier-D cook wrappers for the color-domain heads (paper §5).

Bit-identical Tier-D equivalents of :class:`ColorMixingHead` and
:class:`ColorAdjacencyHead`, both built on the generic
:func:`pcm.heads.cook_factory.make_cook_subgraph`.
"""
from __future__ import annotations

from pcm.concept_graph import ConceptGraph
from pcm.graph_eval import GraphEvaluator
from pcm.heads.cook_factory import (
    MLPBackbone,
    copy_three_linears,
    make_cook_subgraph,
)

from ._config import ADJ_DIM, BIAS_DIM, EMBED_DIM, FACET_ADJ, FACET_MIX
from .heads import ColorAdjacencyHead, ColorMixingHead


__all__ = ["build_mix_cook", "build_adj_cook"]


def build_mix_cook(
    cg: ConceptGraph,
    head: ColorMixingHead,
    *,
    node_id: str = "muscle.cook.color_mix",
    backbone_key: str = "color_mix_mlp",
    evaluator: GraphEvaluator | None = None,
) -> tuple[str, MLPBackbone, GraphEvaluator]:
    """Cook wrapper for :class:`ColorMixingHead` (no extras)."""
    backbone = MLPBackbone(
        in_dim=2 * head.bias_dim,
        hidden_dim=head.embed_dim,
        out_dim=head.embed_dim,
    )
    copy_three_linears(head.fc1, head.fc2, head.fc3, backbone)
    nid, ev = make_cook_subgraph(
        cg,
        node_id=node_id,
        inputs=["ids_a", "ids_b"],
        facet_collapses=[("ids_a", FACET_MIX), ("ids_b", FACET_MIX)],
        backbone=backbone,
        backbone_key=backbone_key,
        evaluator=evaluator,
    )
    return nid, backbone, ev


def build_adj_cook(
    cg: ConceptGraph,
    head: ColorAdjacencyHead,
    *,
    node_id: str = "muscle.cook.color_adj",
    backbone_key: str = "color_adj_mlp",
    evaluator: GraphEvaluator | None = None,
) -> tuple[str, MLPBackbone, GraphEvaluator]:
    """Cook wrapper for :class:`ColorAdjacencyHead` (3-class)."""
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
        facet_collapses=[("ids_a", FACET_ADJ), ("ids_b", FACET_ADJ)],
        backbone=backbone,
        backbone_key=backbone_key,
        evaluator=evaluator,
    )
    return nid, backbone, ev
