"""tests/test_cook_all_heads.py — D1 bit-identity for *every* Tier-A head.

For each muscle in the project, the test:

1. Builds a ConceptGraph + a Tier-A head and warms up the bundle pool.
2. Builds the matching Tier-D cook subgraph via the head's
   ``build_*_cook`` factory (which copies fc1/fc2/fc3 weights).
3. Runs the same B-shaped batch through both paths.
4. Asserts ``torch.equal(direct, cooked)`` (bit-identical).

Heads covered (8 total):

- Number: ArithmeticHeadV2, ComparisonHead, NumerosityClassifier
- Color:  ColorMixingHead, ColorAdjacencyHead
- Space:  MoveHead, DistanceHead
- Phoneme: _SingleInputHead (voicing / manner / place — same class)
"""
from __future__ import annotations

import unittest

import torch

from pcm import ConceptGraph
from pcm.heads import (
    ArithmeticHeadV2,
    ComparisonHead,
    NumerosityClassifier,
    build_arith_v2_cook,
    build_comparison_cook,
    build_numerosity_classifier_cook,
)


def _register_n(cg: ConceptGraph, prefix: str, n: int) -> list[str]:
    cids = []
    for i in range(n):
        cid = f"{prefix}{i}"
        cg.register_concept(node_id=cid, label=cid, scope="BASE",
                            provenance=f"cook-test:{cid}")
        cids.append(cid)
    return cids


# ────────────────────────────────────────────────────────────────────────
# Number domain
# ────────────────────────────────────────────────────────────────────────


class TestNumberHeadsCookBitIdentical(unittest.TestCase):
    """Number-domain heads (paper §4 / §B): cook == direct, bit-perfect."""

    def setUp(self) -> None:
        torch.manual_seed(2026_05_10)
        self.cg = ConceptGraph(initial_capacity=16)
        self.cids = _register_n(self.cg, "concept:ans:", 7)

    def test_arith_v2(self) -> None:
        head = ArithmeticHeadV2(embed_dim=128, bias_dim=64)
        # warm-up + materialise pool
        op = torch.tensor([[1.0, 0.0]] * 7)
        zeros = torch.zeros(7, 128)
        with torch.no_grad():
            head(zeros, zeros, op, self.cids, self.cids, self.cg)
        _, _, ev = build_arith_v2_cook(self.cg, head)

        ids_a = ["concept:ans:1", "concept:ans:3", "concept:ans:5"]
        ids_b = ["concept:ans:2", "concept:ans:1", "concept:ans:4"]
        op_b = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        zeros_b = torch.zeros(3, 128)
        with torch.no_grad():
            y_direct = head(zeros_b, zeros_b, op_b, ids_a, ids_b, self.cg).clone()
            y_cook = ev.eval(
                "muscle.cook.arith_v2",
                bindings={"ids_a": ids_a, "ids_b": ids_b, "op_onehot": op_b},
            )
        self.assertTrue(torch.equal(y_direct, y_cook))

    def test_comparison(self) -> None:
        head = ComparisonHead(embed_dim=128, facet_dim=8, hidden_dim=64)
        with torch.no_grad():
            head(None, None, self.cids, self.cids, self.cg)
        _, _, ev = build_comparison_cook(self.cg, head)

        ids_a = ["concept:ans:1", "concept:ans:5"]
        ids_b = ["concept:ans:5", "concept:ans:1"]
        with torch.no_grad():
            y_direct = head(None, None, ids_a, ids_b, self.cg).clone()
            y_cook = ev.eval(
                "muscle.cook.comparison",
                bindings={"ids_a": ids_a, "ids_b": ids_b},
            )
        self.assertTrue(torch.equal(y_direct, y_cook))

    def test_numerosity_classifier(self) -> None:
        head = NumerosityClassifier(n_classes=7, hidden_dim=32, facet_dim=16)
        with torch.no_grad():
            head(self.cids, self.cg)
        _, _, ev = build_numerosity_classifier_cook(self.cg, head)

        ids = ["concept:ans:0", "concept:ans:3", "concept:ans:6"]
        with torch.no_grad():
            y_direct = head(ids, self.cg).clone()
            y_cook = ev.eval(
                "muscle.cook.numerosity_classifier",
                bindings={"ids": ids},
            )
        self.assertTrue(torch.equal(y_direct, y_cook))


# ────────────────────────────────────────────────────────────────────────
# Color domain
# ────────────────────────────────────────────────────────────────────────


class TestColorHeadsCookBitIdentical(unittest.TestCase):
    """Color-domain heads (paper §5)."""

    def test_color_mix_and_adj(self) -> None:
        from experiments.color_concept_study import (
            ColorAdjacencyHead,
            ColorMixingHead,
        )
        from experiments.color_concept_study.cook import (
            build_adj_cook,
            build_mix_cook,
        )

        torch.manual_seed(2026_05_10)
        cg = ConceptGraph(initial_capacity=16)
        cids = _register_n(cg, "concept:color:", 12)

        head_mix = ColorMixingHead()
        head_adj = ColorAdjacencyHead()
        with torch.no_grad():
            head_mix(cids, cids, cg)
            head_adj(cids, cids, cg)

        _, _, ev = build_mix_cook(cg, head_mix)
        _, _, ev = build_adj_cook(cg, head_adj, evaluator=ev)

        ids_a = ["concept:color:1", "concept:color:5", "concept:color:8"]
        ids_b = ["concept:color:3", "concept:color:7", "concept:color:0"]
        with torch.no_grad():
            y_mix_d = head_mix(ids_a, ids_b, cg).clone()
            y_adj_d = head_adj(ids_a, ids_b, cg).clone()
            y_mix_c = ev.eval(
                "muscle.cook.color_mix",
                bindings={"ids_a": ids_a, "ids_b": ids_b},
            )
            y_adj_c = ev.eval(
                "muscle.cook.color_adj",
                bindings={"ids_a": ids_a, "ids_b": ids_b},
            )
        self.assertTrue(torch.equal(y_mix_d, y_mix_c))
        self.assertTrue(torch.equal(y_adj_d, y_adj_c))


# ────────────────────────────────────────────────────────────────────────
# Space domain
# ────────────────────────────────────────────────────────────────────────


class TestSpaceHeadsCookBitIdentical(unittest.TestCase):
    """Space-domain heads (paper §6.2)."""

    def test_move_and_dist(self) -> None:
        from experiments.space_concept_study import DistanceHead, MoveHead, cid_of
        from experiments.space_concept_study.cook import (
            build_dist_cook,
            build_move_cook,
        )

        torch.manual_seed(2026_05_10)
        cg = ConceptGraph(initial_capacity=64)
        cids = []
        for r in range(5):
            for c in range(5):
                cid = cid_of(r, c)
                cg.register_concept(node_id=cid, label=cid, scope="BASE")
                cids.append(cid)

        head_move = MoveHead()
        head_dist = DistanceHead()
        with torch.no_grad():
            head_move(cids, cids, cg)
            head_dist(cids, cids, cg)

        _, _, ev = build_move_cook(cg, head_move)
        _, _, ev = build_dist_cook(cg, head_dist, evaluator=ev)

        ids_a = [cid_of(0, 0), cid_of(2, 3), cid_of(4, 4)]
        ids_b = [cid_of(0, 1), cid_of(2, 4), cid_of(3, 4)]
        with torch.no_grad():
            y_move_d = head_move(ids_a, ids_b, cg).clone()
            y_dist_d = head_dist(ids_a, ids_b, cg).clone()
            y_move_c = ev.eval(
                "muscle.cook.space_move",
                bindings={"ids_a": ids_a, "ids_b": ids_b},
            )
            y_dist_c = ev.eval(
                "muscle.cook.space_dist",
                bindings={"ids_a": ids_a, "ids_b": ids_b},
            )
        self.assertTrue(torch.equal(y_move_d, y_move_c))
        self.assertTrue(torch.equal(y_dist_d, y_dist_c))


# ────────────────────────────────────────────────────────────────────────
# Phoneme domain (one factory, three heads)
# ────────────────────────────────────────────────────────────────────────


class TestPhonemeHeadsCookBitIdentical(unittest.TestCase):
    """Phoneme-domain heads (paper §6.3) — three axes share one factory."""

    def test_voicing_manner_place(self) -> None:
        from experiments.phoneme_concept_study import (
            cid_of,
            build_manner_head,
            build_place_head,
            build_voicing_head,
        )
        from experiments.phoneme_concept_study.cook import build_single_input_cook

        torch.manual_seed(2026_05_10)
        cg = ConceptGraph(initial_capacity=32)
        cids = [cid_of(i) for i in range(20)]
        for cid in cids:
            cg.register_concept(node_id=cid, label=cid, scope="BASE")

        head_v = build_voicing_head()
        head_m = build_manner_head()
        head_p = build_place_head()
        with torch.no_grad():
            head_v(cids, cg); head_m(cids, cg); head_p(cids, cg)

        nid_v, _, ev = build_single_input_cook(cg, head_v)
        nid_m, _, ev = build_single_input_cook(cg, head_m, evaluator=ev)
        nid_p, _, ev = build_single_input_cook(cg, head_p, evaluator=ev)
        # Three distinct subgraph node ids
        self.assertEqual(len({nid_v, nid_m, nid_p}), 3)

        sample = [cids[0], cids[5], cids[12], cids[18]]
        with torch.no_grad():
            y_v_d = head_v(sample, cg).clone()
            y_m_d = head_m(sample, cg).clone()
            y_p_d = head_p(sample, cg).clone()
            y_v_c = ev.eval(nid_v, bindings={"ids": sample})
            y_m_c = ev.eval(nid_m, bindings={"ids": sample})
            y_p_c = ev.eval(nid_p, bindings={"ids": sample})
        self.assertTrue(torch.equal(y_v_d, y_v_c))
        self.assertTrue(torch.equal(y_m_d, y_m_c))
        self.assertTrue(torch.equal(y_p_d, y_p_c))


if __name__ == "__main__":
    unittest.main()
