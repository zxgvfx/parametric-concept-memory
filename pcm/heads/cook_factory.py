"""cook_factory.py — Generic Tier-D cook-subgraph factory + reusable backbones.

Every Tier-A head in the project follows the same shape:

    forward(ids_1, ids_2, ..., extras..., cg) -> tensor:
        bias_i = cg.collapse_batch(caller_i, facet_i, ids_i, shape_i)
        x = torch.cat([bias_1, ..., extras...], dim=-1)
        return mlp_3layer(x)

That's exactly the ``muscle.collapse_facet × N + muscle.invoke_module × 1``
cook subgraph from ``docs/PCM_NODE_AS_FUNCTION_DESIGN.md`` §2.1. This file
provides a *generic* factory so each head needs roughly 10 lines to gain
a Tier-D equivalent (D1 bit-identical) cook entry-point.

Provided primitives:

- :class:`MLPBackbone` — generic 3-layer MLP backbone (matches all Tier-A
  heads' ``fc1``/``fc2``/``fc3`` structure).
- :func:`copy_three_linears` — copy fc1/fc2/fc3 weights from a source head.
- :func:`make_cook_subgraph` — build a ``parametric_muscle_subgraph``
  ConceptNode + register the backbone in the evaluator's module registry.

See :mod:`pcm.heads.cook_wrappers` for the per-head wrappers that pin
this factory to each concrete Tier-A head.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph_eval import GraphEvaluator


if TYPE_CHECKING:
    from ..concept_graph import ConceptGraph


__all__ = [
    "MLPBackbone",
    "HeadAsBackbone",
    "copy_three_linears",
    "make_cook_subgraph",
]


class MLPBackbone(nn.Module):
    """Generic 3-layer MLP (``Linear → ReLU → Linear → ReLU → Linear``).

    Matches the architecture of every Tier-A head in the project:
    ``ArithmeticHeadV2`` / ``ComparisonHead`` / ``NumerosityClassifier``
    / ``ColorMixingHead`` / ``ColorAdjacencyHead`` / ``MoveHead`` /
    ``DistanceHead`` / ``_SingleInputHead``.

    The forward signature is ``(*tensors) -> tensor`` so it can be wired
    into :func:`make_cook_subgraph` without per-head adapters: the cook
    interpreter passes the resolved ``@``-references as positional args.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        x = torch.cat(tensors, dim=-1) if len(tensors) > 1 else tensors[0]
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class HeadAsBackbone(nn.Module):
    """Backbone-shaped view over a Tier-A head's ``fc1``/``fc2``/``fc3``
    (parameter-sharing — no new parameters are created).

    Useful for end-to-end training comparisons where the cook path must
    share parameters with the direct path so RNG state, gradients, and
    optimiser updates stay locked. The wrapped head's ``fc1``/``fc2``/
    ``fc3`` are used directly; ``self.parameters()`` therefore yields
    *the same* tensors as ``head.parameters()`` (sub-module sharing).

    Forward signature mirrors :class:`MLPBackbone` (``(*tensors) ->
    tensor``) so the same ``muscle.invoke_module`` cook op can dispatch
    to either backbone shape interchangeably.
    """

    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        for attr in ("fc1", "fc2", "fc3"):
            if not hasattr(head, attr):
                raise TypeError(
                    f"HeadAsBackbone requires a head with fc1/fc2/fc3; "
                    f"got {type(head).__name__} (missing {attr!r})"
                )
        self.head = head

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        x = torch.cat(tensors, dim=-1) if len(tensors) > 1 else tensors[0]
        h = F.relu(self.head.fc1(x))
        h = F.relu(self.head.fc2(h))
        return self.head.fc3(h)


def copy_three_linears(
    src_fc1: nn.Linear, src_fc2: nn.Linear, src_fc3: nn.Linear,
    dst: MLPBackbone,
) -> None:
    """Copy fc1/fc2/fc3 weights+biases from a source head into the
    backbone. Caller is responsible for shape-matching.
    """
    with torch.no_grad():
        dst.fc1.weight.copy_(src_fc1.weight); dst.fc1.bias.copy_(src_fc1.bias)
        dst.fc2.weight.copy_(src_fc2.weight); dst.fc2.bias.copy_(src_fc2.bias)
        dst.fc3.weight.copy_(src_fc3.weight); dst.fc3.bias.copy_(src_fc3.bias)


def make_cook_subgraph(
    cg: "ConceptGraph",
    *,
    node_id: str,
    inputs: list[str],
    facet_collapses: list[tuple[str, str]],
    extra_inputs: list[str] = (),
    backbone: nn.Module,
    backbone_key: str,
    evaluator: GraphEvaluator | None = None,
    scope: str = "BASE",
) -> tuple[str, GraphEvaluator]:
    """Register a ``parametric_muscle_subgraph`` ConceptNode that:

    1. Calls ``muscle.collapse_facet`` once per ``(input_name, facet)``
       pair in ``facet_collapses``.
    2. Calls ``muscle.invoke_module`` exactly once with the collapsed
       biases + ``extra_inputs`` as positional args, dispatching to the
       module registered under ``backbone_key``.

    Args:
        cg: target graph.
        node_id: subgraph node id (e.g. ``"muscle.cook.arith_v2"``).
        inputs: ordered binding names the caller will pass at cook time.
            Includes both inputs that drive collapses (the ``input_name``
            in ``facet_collapses``) and any ``extra_inputs``.
        facet_collapses: list of ``(input_name, facet)``. The ``input_name``
            must appear in ``inputs``; ``facet`` is the bundle facet to
            collapse via ``muscle.collapse_facet``.
        extra_inputs: ordered binding names that bypass collapse and go
            directly into the backbone (e.g. ``op_onehot``).
        backbone: ``nn.Module`` whose ``forward(*tensors) -> tensor``
            consumes the collapsed biases and extras (in the order they
            appear in ``facet_collapses + extra_inputs``).
        backbone_key: registry key used inside the cook subgraph and the
            evaluator's ``module_registry`` to dispatch ``invoke_module``.
        evaluator: optional pre-existing evaluator; if ``None`` a fresh
            one is created. Multiple wrappers can share an evaluator by
            chaining the returned object.
        scope: scope for the registered ConceptNode (default ``"BASE"``).

    Returns:
        ``(node_id, evaluator)``. The evaluator's ``module_registry``
        always contains ``backbone_key -> backbone`` after this call.
    """
    nodes: list[dict] = []
    bias_ids: list[str] = []
    for cidx, (input_name, facet) in enumerate(facet_collapses):
        bias_id = f"_bias_{cidx}"
        nodes.append({
            "id": bias_id,
            "op": "muscle.collapse_facet",
            "args": [facet, f"@{input_name}"],
        })
        bias_ids.append(bias_id)

    invoke_args: list[str] = [backbone_key]
    invoke_args.extend(f"@{b}" for b in bias_ids)
    invoke_args.extend(f"@{e}" for e in extra_inputs)
    nodes.append({
        "id": "out",
        "op": "muscle.invoke_module",
        "args": invoke_args,
    })

    cg.register_muscle_subgraph(
        node_id=node_id,
        inputs=list(inputs),
        nodes=nodes,
        output="out",
        provenance=f"tier_d:cook:{node_id}",
        scope=scope,
    )

    if evaluator is None:
        evaluator = GraphEvaluator(
            concept_graph=cg, module_registry={backbone_key: backbone}
        )
    else:
        evaluator.module_registry[backbone_key] = backbone
    return node_id, evaluator
