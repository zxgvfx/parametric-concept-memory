"""tests/test_cook_subgraph.py - Tier-D evaluator error-surface regressions.

Validates that malformed ``parametric_muscle_subgraph`` ConceptNodes
fail loudly through :class:`pcm.GraphEvaluator` (rather than silently
returning bad outputs):

1. Unknown op name -> :class:`SubgraphEvalError`.
2. Missing input binding at cook time -> :class:`SubgraphEvalError`.
3. ``@``-reference to a non-existent intermediate node ->
   :class:`SubgraphEvalError`.
4. Cycle / self-reference detection (``_active_calls`` guard).
5. Cooking a non-parametric data node ->
   :class:`SubgraphEvalError`.
6. Pure-op kind end-to-end sanity (``add`` / ``mul`` round-trip).

D1 bit-identity for every head (claim D1 from
``docs/PCM_NODE_AS_FUNCTION_DESIGN.md``) is exercised by
``tests/test_cook_all_heads.py`` over all 8 muscle heads (number,
color, space, phoneme), so this file no longer duplicates that
coverage.
"""
from __future__ import annotations

import unittest

from pcm import (
    ConceptGraph,
    GraphEvaluator,
    SubgraphEvalError,
)


def _register_ans(cg: ConceptGraph, n_max: int = 3) -> list[str]:
    cids = []
    for n in range(1, n_max + 1):
        cid = f"concept:ans:{n}"
        cg.register_concept(node_id=cid, label=f"ANS_{n}", scope="BASE",
                            provenance=f"cook-test:n={n}")
        cids.append(cid)
    return cids


class TestEvaluatorErrorSurface(unittest.TestCase):
    """Common malformed-subgraph paths must raise SubgraphEvalError."""

    def setUp(self) -> None:
        self.cg = ConceptGraph(initial_capacity=8)
        _register_ans(self.cg, n_max=3)

    def test_unknown_op_name(self) -> None:
        self.cg.register_muscle_subgraph(
            node_id="muscle.bad_op",
            inputs=["x"],
            nodes=[{"id": "out", "op": "no.such.op", "args": ["@x"]}],
            output="out",
        )
        ev = GraphEvaluator(concept_graph=self.cg)
        with self.assertRaises(SubgraphEvalError):
            ev.eval("muscle.bad_op", bindings={"x": 1})

    def test_missing_input_binding(self) -> None:
        self.cg.register_muscle_subgraph(
            node_id="muscle.missing_input",
            inputs=["x", "y"],
            nodes=[{"id": "out", "op": "add", "args": ["@x", "@y"]}],
            output="out",
        )
        ev = GraphEvaluator(concept_graph=self.cg)
        with self.assertRaises(SubgraphEvalError):
            ev.eval("muscle.missing_input", bindings={"x": 1})

    def test_unresolved_at_reference(self) -> None:
        self.cg.register_muscle_subgraph(
            node_id="muscle.bad_ref",
            inputs=["x"],
            nodes=[
                {"id": "out", "op": "add", "args": ["@x", "@nonexistent"]},
            ],
            output="out",
        )
        ev = GraphEvaluator(concept_graph=self.cg)
        with self.assertRaises(SubgraphEvalError):
            ev.eval("muscle.bad_ref", bindings={"x": 1})

    def test_cycle_self_reference(self) -> None:
        # We cannot easily express a cycle without invoke_subgraph, so we
        # simulate the cycle guard by re-entering the same eval call.
        self.cg.register_muscle_subgraph(
            node_id="muscle.add_xy",
            inputs=["x", "y"],
            nodes=[{"id": "out", "op": "add", "args": ["@x", "@y"]}],
            output="out",
        )
        ev = GraphEvaluator(concept_graph=self.cg)
        ev._active_calls.add("muscle.add_xy")
        with self.assertRaises(SubgraphEvalError):
            ev.eval("muscle.add_xy", bindings={"x": 1, "y": 2})

    def test_kind_must_be_parametric(self) -> None:
        ev = GraphEvaluator(concept_graph=self.cg)
        with self.assertRaises(SubgraphEvalError):
            ev.eval("concept:ans:1", bindings={})

    def test_pure_op_kind_arithmetic(self) -> None:
        """Ops registered as ``pure`` should run fine end-to-end."""
        self.cg.register_muscle_subgraph(
            node_id="muscle.add_then_mul",
            inputs=["x", "y", "z"],
            nodes=[
                {"id": "s", "op": "add", "args": ["@x", "@y"]},
                {"id": "out", "op": "mul", "args": ["@s", "@z"]},
            ],
            output="out",
        )
        ev = GraphEvaluator(concept_graph=self.cg)
        result = ev.eval("muscle.add_then_mul", bindings={"x": 2, "y": 3, "z": 4})
        self.assertEqual(result, 20)


if __name__ == "__main__":
    unittest.main()
