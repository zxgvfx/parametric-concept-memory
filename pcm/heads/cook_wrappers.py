"""cook_wrappers.py - Tier-D cook wrappers for every Tier-A muscle head.

One ``build_<head>_cook(cg, head)`` factory per concrete head, all on top
of :mod:`pcm.heads.cook_factory`:

- Number domain (paper §4 / §B):

  * :func:`build_arith_v2_cook` - wraps :class:`ArithmeticHeadV2`.
  * :func:`build_comparison_cook` - wraps :class:`ComparisonHead`.
  * :func:`build_numerosity_classifier_cook` - wraps
    :class:`NumerosityClassifier` (single-input head).

The remaining 5 heads live inside per-domain experiment packages
(:mod:`experiments.color_concept_study.heads`,
:mod:`experiments.space_concept_study.heads`,
:mod:`experiments.phoneme_concept_study.heads`); their cook wrappers
live next to them in
:mod:`experiments.cook_four_domain` to avoid the core library reaching
into experiment-level code.

Every wrapper is **D1 bit-identical** when its ``copy_three_linears``
mirrors the Tier-A head's fc1/fc2/fc3: the cook output equals the
Tier-A direct forward output to the last bit (verified by
``tests/test_cook_all_heads.py``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

from ..graph_eval import GraphEvaluator
from .arithmetic_head_v2 import ArithmeticHeadV2
from .comparison_head import ComparisonHead
from .cook_factory import MLPBackbone, copy_three_linears, make_cook_subgraph
from .numerosity_classifier import NumerosityClassifier


if TYPE_CHECKING:
    from ..concept_graph import ConceptGraph


__all__ = [
    "build_arith_v2_cook",
    "build_comparison_cook",
    "build_numerosity_classifier_cook",
]


def _backbone_on_head_device(head: nn.Module, backbone: MLPBackbone) -> MLPBackbone:
    """Move ``backbone`` onto the same device as ``head`` (best-effort).

    Tier-A heads may live on CUDA after ``head.to(device)``; the freshly
    allocated :class:`MLPBackbone` in our wrappers is on CPU by default,
    which makes the immediately-following :func:`copy_three_linears` fail
    with a device-mismatch error. We sniff ``head``'s parameter device
    and align ``backbone`` before the copy.
    """
    try:
        dev = next(head.parameters()).device
    except StopIteration:
        return backbone
    return backbone.to(dev)


# ─── Number domain ────────────────────────────────────────────────────


def build_arith_v2_cook(
    cg: "ConceptGraph",
    head: ArithmeticHeadV2,
    *,
    node_id: str = "muscle.cook.arith_v2",
    backbone_key: str = "arith_v2_mlp",
    evaluator: GraphEvaluator | None = None,
) -> tuple[str, MLPBackbone, GraphEvaluator]:
    """Cook wrapper for :class:`ArithmeticHeadV2`.

    Subgraph::

        ba  = collapse_facet("arithmetic_bias", @ids_a)
        bb  = collapse_facet("arithmetic_bias", @ids_b)
        out = invoke_module(arith_v2_mlp, @ba, @bb, @op_onehot)
    """
    backbone = MLPBackbone(
        in_dim=2 * head.bias_dim + 2,
        hidden_dim=head.embed_dim,
        out_dim=head.embed_dim,
    )
    backbone = _backbone_on_head_device(head, backbone)
    copy_three_linears(head.fc1, head.fc2, head.fc3, backbone)
    nid, ev = make_cook_subgraph(
        cg,
        node_id=node_id,
        inputs=["ids_a", "ids_b", "op_onehot"],
        facet_collapses=[
            ("ids_a", "arithmetic_bias"),
            ("ids_b", "arithmetic_bias"),
        ],
        extra_inputs=["op_onehot"],
        backbone=backbone,
        backbone_key=backbone_key,
        evaluator=evaluator,
    )
    return nid, backbone, ev


def build_comparison_cook(
    cg: "ConceptGraph",
    head: ComparisonHead,
    *,
    node_id: str = "muscle.cook.comparison",
    backbone_key: str = "comparison_mlp",
    evaluator: GraphEvaluator | None = None,
) -> tuple[str, MLPBackbone, GraphEvaluator]:
    """Cook wrapper for :class:`ComparisonHead` (no extra inputs)."""
    backbone = MLPBackbone(
        in_dim=2 * head.facet_dim,
        hidden_dim=head.hidden_dim,
        out_dim=3,
    )
    backbone = _backbone_on_head_device(head, backbone)
    copy_three_linears(head.fc1, head.fc2, head.fc3, backbone)
    nid, ev = make_cook_subgraph(
        cg,
        node_id=node_id,
        inputs=["ids_a", "ids_b"],
        facet_collapses=[
            ("ids_a", "ordinal_offset"),
            ("ids_b", "ordinal_offset"),
        ],
        backbone=backbone,
        backbone_key=backbone_key,
        evaluator=evaluator,
    )
    return nid, backbone, ev


def build_numerosity_classifier_cook(
    cg: "ConceptGraph",
    head: NumerosityClassifier,
    *,
    node_id: str = "muscle.cook.numerosity_classifier",
    backbone_key: str = "numerosity_classifier_mlp",
    evaluator: GraphEvaluator | None = None,
) -> tuple[str, MLPBackbone, GraphEvaluator]:
    """Cook wrapper for :class:`NumerosityClassifier` (single-input head)."""
    backbone = MLPBackbone(
        in_dim=head.facet_dim,
        hidden_dim=head.hidden_dim,
        out_dim=head.n_classes,
    )
    backbone = _backbone_on_head_device(head, backbone)
    copy_three_linears(head.fc1, head.fc2, head.fc3, backbone)
    nid, ev = make_cook_subgraph(
        cg,
        node_id=node_id,
        inputs=["ids"],
        facet_collapses=[("ids", "identity_prototype")],
        backbone=backbone,
        backbone_key=backbone_key,
        evaluator=evaluator,
    )
    return nid, backbone, ev
