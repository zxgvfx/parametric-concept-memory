"""pcm.graph_eval — Tier-D cook interpreter.

Distilled from
``pcm_agent.cognition.causal.graph.graph_eval`` (391 lines, production)
down to ~250 lines for the parametric-concept-memory repo. Drops the
audit-replay scaffolding and the dual-kind firewall (PCM only ever has
the parametric kind by construction); keeps the `@`-reference resolver,
cycle guard and per-`op_kind` dispatch.

A `parametric_muscle_subgraph` ConceptNode encodes a small computation
as a list of named DNA-op invocations plus dataflow wiring, like a
Houdini node graph. :class:`GraphEvaluator` walks such a node and
produces the output value, **without any Python code beyond the op
implementations registered in** :mod:`pcm.dna_ops`.

Subgraph schema::

    metadata = {
        "inputs":     ["binding_name1", ...],
        "constants":  {"const_4": 4.0, ...},
        "facet_specs": {"cls_weight": [256, 128], ...},
        "nodes": [
            {"id": "ba", "op": "muscle.collapse_facet",
             "args": ["arithmetic_bias", "@concept_ids_a"]},
            ...
        ],
        "output":     "out",
    }

Argument forms:

- ``"@input_name"``  — pulled from caller-supplied ``bindings``;
- ``"@const_name"``  — pulled from ``metadata.constants``;
- ``"@self"``        — the host ConceptNode instance;
- ``"@nodeid"``      — pulled from a previously evaluated node's result;
- anything else      — passed through as a literal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .dna_ops import DNA_OPS, op_kind_of

if TYPE_CHECKING:
    import torch.nn as nn

    from .concept_graph import ConceptGraph

__all__ = [
    "GraphEvaluator",
    "SubgraphEvalError",
    "PARAMETRIC_KIND",
]


PARAMETRIC_KIND = "parametric_muscle_subgraph"
"""The single ConceptNode.kind value Tier-D considers cookable."""


class SubgraphEvalError(RuntimeError):
    """Raised when a subgraph cannot be evaluated.

    Reasons include: unknown op name, missing ``@`` reference, malformed
    ``metadata.nodes`` entry, recursive cook cycle, op_kind context
    violation, and missing required input binding.
    """


@dataclass
class GraphEvaluator:
    """Stateless interpreter for ``parametric_muscle_subgraph`` ConceptNodes.

    Args:
        concept_graph: graph the evaluator looks subgraphs up in. Must
            implement ``concept_graph.concepts[node_id]`` returning a
            :class:`pcm.concept_graph.ConceptNode`.
        op_registry: name -> callable map. Defaults to
            :data:`pcm.dna_ops.DNA_OPS`.
        module_registry: name -> ``nn.Module`` map used by the
            ``muscle.invoke_module`` op. Empty by default.
    """

    concept_graph: "ConceptGraph"
    op_registry: dict[str, Callable[..., Any]] = field(
        default_factory=lambda: dict(DNA_OPS)
    )
    module_registry: dict[str, "nn.Module"] = field(default_factory=dict)
    _active_calls: set[str] = field(default_factory=set, init=False, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def eval(  # noqa: A003 (legacy name, matches pcm-agent for portability)
        self,
        subgraph_node_id: str,
        bindings: dict[str, Any],
        *,
        caller: str = "graph_eval",
        tick: int = 0,
    ) -> Any:
        """Evaluate the named subgraph against ``bindings``.

        ``bindings`` keys must be the strings declared in
        ``metadata.inputs``. Returns the raw output value (whatever the
        ``output`` node produced).

        ``caller`` and ``tick`` are forwarded to context-bound ops
        (``muscle.collapse_facet``, ``muscle.invoke_module``).

        Raises:
            SubgraphEvalError: see class docstring.
        """
        node = self.concept_graph.concepts.get(subgraph_node_id)
        if node is None:
            raise SubgraphEvalError(
                f"subgraph node {subgraph_node_id!r} not found in concept_graph"
            )

        kind = getattr(node, "kind", "data")
        if kind != PARAMETRIC_KIND:
            raise SubgraphEvalError(
                f"subgraph {subgraph_node_id!r}: kind={kind!r} is not "
                f"cookable (expected {PARAMETRIC_KIND!r})"
            )

        if subgraph_node_id in self._active_calls:
            cycle = " -> ".join(list(self._active_calls) + [subgraph_node_id])
            raise SubgraphEvalError(
                f"subgraph cycle detected on {subgraph_node_id!r}: {cycle}"
            )

        meta = getattr(node, "metadata", None) or {}
        declared_inputs = meta.get("inputs") or []
        constants = meta.get("constants") or {}
        node_descs = meta.get("nodes") or []
        output_id = meta.get("output")

        if not isinstance(node_descs, list) or not node_descs:
            raise SubgraphEvalError(
                f"subgraph {subgraph_node_id!r}: metadata.nodes must be "
                "a non-empty list"
            )
        if not output_id or not isinstance(output_id, str):
            raise SubgraphEvalError(
                f"subgraph {subgraph_node_id!r}: metadata.output must be "
                "a non-empty string"
            )

        env: dict[str, Any] = {}
        for k in declared_inputs:
            sk = str(k)
            if sk not in bindings:
                raise SubgraphEvalError(
                    f"subgraph {subgraph_node_id!r}: missing required "
                    f"input binding {sk!r}"
                )
            env[f"@{sk}"] = bindings[sk]
        if isinstance(constants, dict):
            for k, v in constants.items():
                env[f"@{str(k)}"] = v
        env["@self"] = node

        self._active_calls.add(subgraph_node_id)
        try:
            for desc in node_descs:
                self._eval_one(desc, env, subgraph_node_id, caller=caller, tick=tick)
        finally:
            self._active_calls.discard(subgraph_node_id)

        out_key = f"@{output_id}"
        if out_key not in env:
            raise SubgraphEvalError(
                f"subgraph {subgraph_node_id!r}: output node "
                f"{output_id!r} did not produce a value"
            )
        return env[out_key]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_one(
        self,
        desc: Any,
        env: dict[str, Any],
        subgraph_node_id: str,
        *,
        caller: str,
        tick: int,
    ) -> None:
        if not isinstance(desc, dict):
            raise SubgraphEvalError(
                f"subgraph {subgraph_node_id!r}: node entry must be a "
                f"dict, got {type(desc).__name__}"
            )
        n_id = desc.get("id")
        op_name = desc.get("op")
        args_raw = desc.get("args") or []
        if not n_id or not isinstance(n_id, str):
            raise SubgraphEvalError(
                f"subgraph {subgraph_node_id!r}: each node needs a "
                "string 'id'"
            )
        if op_name not in self.op_registry:
            raise SubgraphEvalError(
                f"subgraph {subgraph_node_id!r}: unknown op {op_name!r} "
                f"on node {n_id!r}"
            )

        kind = op_kind_of(op_name)
        resolved = [
            self._resolve_arg(a, env, subgraph_node_id, n_id) for a in args_raw
        ]
        op_fn = self.op_registry[op_name]
        env[f"@{n_id}"] = self._dispatch(
            op_fn, kind, resolved, caller=caller, tick=tick
        )

    def _resolve_arg(
        self,
        a: Any,
        env: dict[str, Any],
        subgraph_node_id: str,
        n_id: str,
    ) -> Any:
        """Recursively resolve ``@``-prefixed references through dicts and
        lists. Literals pass through unchanged.
        """
        if isinstance(a, str) and a.startswith("@"):
            if a not in env:
                raise SubgraphEvalError(
                    f"subgraph {subgraph_node_id!r}: node {n_id!r} "
                    f"references unresolved binding {a!r}"
                )
            return env[a]
        if isinstance(a, dict):
            return {
                k: self._resolve_arg(v, env, subgraph_node_id, n_id)
                for k, v in a.items()
            }
        if isinstance(a, list):
            return [
                self._resolve_arg(v, env, subgraph_node_id, n_id) for v in a
            ]
        return a

    def _dispatch(
        self,
        op_fn: Callable[..., Any],
        op_kind: str,
        args: list[Any],
        *,
        caller: str,
        tick: int,
    ) -> Any:
        if op_kind == "pure":
            return op_fn(*args)
        if op_kind == "collapse":
            return op_fn(
                *args,
                caller=caller,
                tick=tick,
                concept_graph=self.concept_graph,
            )
        if op_kind == "invoke_module":
            return op_fn(
                *args,
                module_registry=self.module_registry,
                caller=caller,
                tick=tick,
            )
        # Defensive default: behave like pure.
        return op_fn(*args)
