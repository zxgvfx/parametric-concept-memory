"""Tier-D cook wrapper for the phoneme :class:`_SingleInputHead` (paper §6.3).

All three phoneme heads (voicing / manner / place) share the same
:class:`_SingleInputHead` class with different ``(caller, facet,
facet_dim, n_classes)`` instantiations, so a single factory covers
all three.
"""
from __future__ import annotations

from pcm.concept_graph import ConceptGraph
from pcm.graph_eval import GraphEvaluator
from pcm.heads.cook_factory import (
    MLPBackbone,
    copy_three_linears,
    make_cook_subgraph,
)

from .heads import _SingleInputHead


__all__ = ["build_single_input_cook"]


def build_single_input_cook(
    cg: ConceptGraph,
    head: _SingleInputHead,
    *,
    node_id: str | None = None,
    backbone_key: str | None = None,
    evaluator: GraphEvaluator | None = None,
) -> tuple[str, MLPBackbone, GraphEvaluator]:
    """Cook wrapper for :class:`_SingleInputHead`.

    Defaults derive ``node_id`` and ``backbone_key`` from the head's
    ``caller`` so the three phoneme axes (Voicing / Manner / Place) get
    distinct subgraph nodes when wrapped in succession.
    """
    nid_default = f"muscle.cook.phoneme_{head.caller.lower()}"
    key_default = f"phoneme_{head.caller.lower()}_mlp"
    backbone = MLPBackbone(
        in_dim=head.facet_dim,
        hidden_dim=head.fc1.out_features,
        out_dim=head.fc3.out_features,
    )
    copy_three_linears(head.fc1, head.fc2, head.fc3, backbone)
    nid, ev = make_cook_subgraph(
        cg,
        node_id=node_id or nid_default,
        inputs=["ids"],
        facet_collapses=[("ids", head.facet)],
        backbone=backbone,
        backbone_key=backbone_key or key_default,
        evaluator=evaluator,
    )
    return nid, backbone, ev
