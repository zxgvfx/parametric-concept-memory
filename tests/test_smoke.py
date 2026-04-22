"""Minimal smoke tests — run with `pytest -q` (no pytest dep required;
also works as `python -m unittest`).

Verifies (1) public API imports cleanly, (2) the core collapse
operation preserves attribution, (3) a tiny one-epoch arithmetic
training step backprops gradients into the bundle.
"""
from __future__ import annotations

import unittest

import torch

from pcm import ConceptGraph, ContextualizedConcept, ParamBundle
from pcm.heads import ArithmeticHeadV2


class TestPublicAPI(unittest.TestCase):
    def test_imports(self) -> None:
        from pcm import ConceptGraph as CG  # noqa: F401
        from pcm.heads import (  # noqa: F401
            ArithmeticHead, ArithmeticHeadV2 as V2, ComparisonHead,
            NumerosityClassifier, NumerosityEncoder,
        )


class TestCollapseAttribution(unittest.TestCase):
    def setUp(self) -> None:
        self.cg = ConceptGraph(feat_dim=32)
        for n in range(1, 4):
            self.cg.register_concept(
                node_id=f"concept:test:{n}",
                label=f"T{n}", scope="BASE",
                provenance=f"test:n={n}",
            )

    def test_collapse_returns_ctx_concept(self) -> None:
        c = self.cg.concepts["concept:test:1"]
        cc = c.collapse(caller="M", facet="bias", shape=(8,),
                        tick=0, init="normal_small")
        self.assertIsInstance(cc, ContextualizedConcept)
        self.assertEqual(cc.as_tensor().shape, (8,))

    def test_consumer_registry_records_caller(self) -> None:
        c = self.cg.concepts["concept:test:2"]
        c.collapse(caller="MuscleA", facet="bias", shape=(8,),
                   tick=0, init="normal_small")
        c.collapse(caller="MuscleB", facet="bias", shape=(8,),
                   tick=1, init="normal_small")
        self.assertIn("MuscleA", c.bundle.consumed_by["bias"])
        self.assertIn("MuscleB", c.bundle.consumed_by["bias"])

    def test_bundle_leaves_are_parameters(self) -> None:
        c = self.cg.concepts["concept:test:3"]
        c.collapse(caller="M", facet="bias", shape=(8,),
                   tick=0, init="normal_small")
        self.assertIn("bias", c.bundle.params)
        self.assertTrue(c.bundle.params["bias"].requires_grad)


class TestArithmeticHeadV2BackpropIntoBundle(unittest.TestCase):
    """One-step gradient sanity: after a loss.backward(), the bundle
    tensor of an involved concept has a non-zero gradient."""

    def test_gradient_flows_into_bundle(self) -> None:
        cg = ConceptGraph(feat_dim=128)
        for n in range(1, 4):
            cg.register_concept(
                node_id=f"concept:ans:{n}",
                label=f"ANS_{n}", scope="BASE",
                provenance=f"test:n={n}",
            )
        head = ArithmeticHeadV2(embed_dim=128, bias_dim=64)

        for n in range(1, 4):
            c = cg.concepts[f"concept:ans:{n}"]
            c.collapse(caller="ArithmeticHeadV2", facet="arithmetic_bias",
                       shape=(64,), tick=0, init="normal_small")

        dummy = torch.zeros(2, 128)
        op = torch.zeros(2, 2); op[:, 0] = 1.0  # [add=1, sub=0]
        ids_a = ["concept:ans:1", "concept:ans:2"]
        ids_b = ["concept:ans:2", "concept:ans:1"]
        pred = head(dummy, dummy, op, ids_a, ids_b, cg)
        loss = pred.pow(2).mean()
        loss.backward()

        bundle_param = cg.concepts["concept:ans:1"].bundle.params[
            "arithmetic_bias"
        ]
        self.assertIsNotNone(bundle_param.grad)
        self.assertTrue(bundle_param.grad.abs().sum().item() > 0.0)


if __name__ == "__main__":
    unittest.main()
