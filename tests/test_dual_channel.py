"""tests/test_dual_channel.py — falsifiable unit tests for the
``pcm.dual_channel`` v2 architecture.

Covers:

* Facet pairing (register / lookup / iterate / pool exclusion).
* :func:`collapse_dual_channel` shapes + grad flow on both facets.
* Loss primitives — InfoNCE, arithmetic, successor (with the
  norm penalty that breaks the trivial ``attr=const`` minimum),
  spread.
* :class:`RelativePositionEmbedding` — 1-D / 2-D / 3-D, range
  validation, lookup correctness, gradient propagation.
* :func:`pair_attention_logits` — minimal pair attention smoke.

The tests are deliberately tiny (≤16-d facets, ≤8 concepts) so
the whole file runs in well under 5 s on CPU.
"""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from pcm import ConceptGraph
from pcm.dual_channel import (
    ALiBiRelativePositionBias,
    ATTR_FACET_TEMPLATE,
    RelativePositionEmbedding,
    SinusoidalRelativePositionEmbedding,
    SLOT_FACET_TEMPLATE,
    arithmetic_consistency_loss,
    collapse_dual_channel,
    info_nce_loss,
    is_dual_channel_facet,
    iter_dual_channel_pairs,
    pair_attention_logits,
    register_dual_channel_facet,
    spread_regularizer,
    successor_consistency_loss,
)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _build_cg(n_concepts: int = 6, slot_dim: int = 8, attr_dim: int = 8,
              base: str = "x") -> tuple[ConceptGraph, list[str], str, str]:
    cg = ConceptGraph(initial_capacity=16)
    cids: list[str] = []
    for i in range(n_concepts):
        cid = f"c:{i}"
        cg.register_concept(node_id=cid, label=f"C_{i}", scope="BASE",
                            provenance=f"dc-test:{i}")
        cids.append(cid)
    sf, af = register_dual_channel_facet(
        cg, base, slot_dim=slot_dim, attr_dim=attr_dim,
    )
    return cg, cids, sf, af


# ---------------------------------------------------------------------------
# Facet pairing — DC1.
# ---------------------------------------------------------------------------


class TestDC1FacetPairing(unittest.TestCase):
    def test_dc1a_template_names(self) -> None:
        cg, _, sf, af = _build_cg(base="arith")
        self.assertEqual(sf, SLOT_FACET_TEMPLATE.format(base="arith"))
        self.assertEqual(af, ATTR_FACET_TEMPLATE.format(base="arith"))

    def test_dc1b_pair_lookup(self) -> None:
        cg, _, sf, af = _build_cg(base="m")
        is_pair, base, role = is_dual_channel_facet(cg, sf)
        self.assertTrue(is_pair)
        self.assertEqual(base, "m")
        self.assertEqual(role, "slot")
        is_pair, base, role = is_dual_channel_facet(cg, af)
        self.assertTrue(is_pair)
        self.assertEqual(role, "attr")
        is_pair, _, _ = is_dual_channel_facet(cg, "unrelated")
        self.assertFalse(is_pair)

    def test_dc1c_iterate(self) -> None:
        cg, _, _, _ = _build_cg(base="b")
        register_dual_channel_facet(cg, "b2", slot_dim=4, attr_dim=4)
        pairs = list(iter_dual_channel_pairs(cg))
        self.assertEqual(len(pairs), 2)
        bases = {p[0] for p in pairs}
        self.assertEqual(bases, {"b", "b2"})

    def test_dc1d_attr_facet_excluded_from_sleep(self) -> None:
        cg, _, _, af = _build_cg(base="z")
        excluded = getattr(cg, "_attr_facets_excluded_from_sleep", None)
        self.assertIsNotNone(excluded)
        self.assertIn(af, excluded)


# ---------------------------------------------------------------------------
# Collapse — DC2.
# ---------------------------------------------------------------------------


class TestDC2CollapseDualChannel(unittest.TestCase):
    def test_dc2a_shapes(self) -> None:
        cg, cids, _, _ = _build_cg(slot_dim=8, attr_dim=4)
        slot, attr = collapse_dual_channel(
            cg, caller="t", base_facet="x", concept_ids=cids,
            slot_shape=(8,), attr_shape=(4,), tick=0,
        )
        self.assertEqual(slot.shape, (len(cids), 8))
        self.assertEqual(attr.shape, (len(cids), 4))

    def test_dc2b_grad_flows_both_facets(self) -> None:
        cg, cids, _, _ = _build_cg(slot_dim=4, attr_dim=4)
        slot, attr = collapse_dual_channel(
            cg, caller="t", base_facet="x", concept_ids=cids,
            slot_shape=(4,), attr_shape=(4,), tick=0,
            attr_init="normal",
        )
        loss = slot.pow(2).mean() + attr.pow(2).mean()
        loss.backward()
        slot_pool = cg.bundle_pool["x_slot"]
        attr_pool = cg.bundle_pool["x_attr"]
        self.assertGreater(slot_pool.grad.abs().sum().item(), 0.0)
        self.assertGreater(attr_pool.grad.abs().sum().item(), 0.0)

    def test_dc2c_normalize_attr(self) -> None:
        cg, cids, _, _ = _build_cg(slot_dim=4, attr_dim=4)
        with torch.no_grad():
            collapse_dual_channel(
                cg, caller="warmup", base_facet="x", concept_ids=cids,
                slot_shape=(4,), attr_shape=(4,), tick=0,
                attr_init="normal",
            )
        _, attr = collapse_dual_channel(
            cg, caller="t", base_facet="x", concept_ids=cids,
            slot_shape=(4,), attr_shape=(4,), tick=1,
            normalize_attr=True,
        )
        norms = attr.norm(dim=-1)
        self.assertTrue(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
            f"normalize_attr=True did not produce unit-norm rows: {norms}",
        )


# ---------------------------------------------------------------------------
# Loss primitives — DC3.
# ---------------------------------------------------------------------------


class TestDC3Losses(unittest.TestCase):
    def test_dc3a_info_nce_zero_with_no_positives(self) -> None:
        torch.manual_seed(0)
        attr = torch.randn(4, 8)
        labels = torch.tensor([0, 1, 2, 3])  # all unique → no positives
        loss = info_nce_loss(attr, labels)
        self.assertEqual(float(loss.item()), 0.0)

    def test_dc3b_arithmetic_zero_at_perfect_linear(self) -> None:
        torch.manual_seed(0)
        v = torch.randn(8)
        attr = torch.stack([float(i) * v for i in range(6)])
        ia = torch.tensor([0, 1, 2])
        ib = torch.tensor([1, 2, 3])
        ic = torch.tensor([2, 3, 4])
        id_ = torch.tensor([1, 2, 3])  # b - a + c = d → d = c
        # Actually want a+c-b = d. With attr_i = i*v:
        # (a + c - b)*v = (0+2-1)v = v = attr_1 → d = 1 → matches.
        loss = arithmetic_consistency_loss(attr, ia, ib, ic, id_)
        self.assertLess(float(loss.item()), 1e-6)

    def test_dc3c_successor_norm_penalty_breaks_trivial(self) -> None:
        # All-zero attr table satisfies variance=0 but should be
        # penalised by the norm penalty.
        attr_zero = torch.zeros(5, 4)
        loss_zero = successor_consistency_loss(attr_zero)
        self.assertGreater(float(loss_zero.item()), 0.5)
        # Linear monotone embedding has zero loss (var=0, ‖step‖≈1).
        v = torch.randn(4)
        v = v / v.norm()
        attr_lin = torch.stack([float(i) * v for i in range(5)])
        loss_lin = successor_consistency_loss(attr_lin)
        self.assertLess(float(loss_lin.item()), 1e-3)

    def test_dc3d_spread_regularizer_zero_for_orthogonal(self) -> None:
        torch.manual_seed(1)
        # Orthogonal rows → cosine distance ≈ 1 ≥ target → loss = 0.
        attr = torch.eye(4)
        loss = spread_regularizer(attr, target_min_dist=0.5)
        self.assertEqual(float(loss.item()), 0.0)

    def test_dc3e_spread_regularizer_positive_for_collapsed(self) -> None:
        # All rows equal → cosine distance = 0 → loss > 0.
        attr = torch.ones(4, 4)
        loss = spread_regularizer(attr, target_min_dist=0.5)
        self.assertGreater(float(loss.item()), 0.0)


# ---------------------------------------------------------------------------
# RelativePositionEmbedding — DC4.
# ---------------------------------------------------------------------------


class TestDC4RPE(unittest.TestCase):
    def test_dc4a_1d_lookup(self) -> None:
        torch.manual_seed(0)
        rpe = RelativePositionEmbedding(ranges=[(-3, 3)], embed_dim=4)
        self.assertEqual(rpe.n_displacements, 7)
        idx = torch.tensor([-3, -1, 0, 2, 3])
        out = rpe(idx)
        self.assertEqual(out.shape, (5, 4))

    def test_dc4b_2d_distinct_per_pair(self) -> None:
        torch.manual_seed(0)
        rpe = RelativePositionEmbedding(
            ranges=[(-2, 2), (-2, 2)], embed_dim=8,
        )
        self.assertEqual(rpe.n_displacements, 25)
        # Two different (Δr, Δc) should give different rows.
        out_a = rpe(torch.tensor([0]), torch.tensor([0]))
        out_b = rpe(torch.tensor([1]), torch.tensor([0]))
        self.assertFalse(
            torch.allclose(out_a, out_b),
            "RPE produced identical rows for distinct displacements",
        )

    def test_dc4c_3d_kway(self) -> None:
        torch.manual_seed(0)
        rpe = RelativePositionEmbedding(
            ranges=[(-1, 1), (-1, 1), (-1, 1)], embed_dim=4,
        )
        self.assertEqual(rpe.n_displacements, 27)
        out = rpe(
            torch.tensor([0, 1]),
            torch.tensor([-1, 0]),
            torch.tensor([1, -1]),
        )
        self.assertEqual(out.shape, (2, 4))

    def test_dc4d_invalid_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            RelativePositionEmbedding(ranges=[(3, 1)], embed_dim=4)
        with self.assertRaises(ValueError):
            RelativePositionEmbedding(ranges=[], embed_dim=4)

    def test_dc4e_wrong_arity_raises(self) -> None:
        rpe = RelativePositionEmbedding(
            ranges=[(-1, 1), (-1, 1)], embed_dim=4,
        )
        with self.assertRaises(ValueError):
            rpe(torch.tensor([0]))  # missing second axis

    def test_dc4f_grad_flows(self) -> None:
        rpe = RelativePositionEmbedding(ranges=[(-2, 2)], embed_dim=4)
        out = rpe(torch.tensor([0, 1, -1]))
        loss = out.pow(2).mean()
        loss.backward()
        self.assertGreater(
            rpe.table.weight.grad.abs().sum().item(), 0.0,
        )

    def test_dc4g_sinusoidal_rpe_shape(self) -> None:
        torch.manual_seed(0)
        rpe = SinusoidalRelativePositionEmbedding(n_axes=2, embed_dim=16)
        # n_displacements is -1 sentinel for continuous heads.
        self.assertEqual(rpe.n_displacements, -1)
        out = rpe(torch.tensor([0, 1, -1]), torch.tensor([2, 0, 3]))
        self.assertEqual(out.shape, (3, 16))

    def test_dc4h_sinusoidal_rpe_extrapolation(self) -> None:
        """Sinusoidal basis is well-defined for any Δ regardless
        of train range — output norm should not vanish for large
        OOD displacements (the failure mode of finite lookup)."""
        torch.manual_seed(0)
        rpe = SinusoidalRelativePositionEmbedding(n_axes=1, embed_dim=8)
        small = rpe(torch.tensor([1]))
        large = rpe(torch.tensor([10000]))
        # Both should be unit-magnitude-ish (sin²+cos²=1 per pair).
        self.assertGreater(float(small.norm().item()), 0.5)
        self.assertGreater(float(large.norm().item()), 0.5)

    def test_dc4i_sinusoidal_invalid_dim(self) -> None:
        with self.assertRaises(ValueError):
            SinusoidalRelativePositionEmbedding(n_axes=2, embed_dim=2)
        with self.assertRaises(ValueError):
            SinusoidalRelativePositionEmbedding(n_axes=0, embed_dim=8)

    def test_dc4j_alibi_bias_shape_and_sign(self) -> None:
        torch.manual_seed(0)
        bias = ALiBiRelativePositionBias(n_axes=1)
        out = bias(torch.tensor([0, 1, -3, 5]))
        self.assertEqual(out.shape, (4, 1))
        # ALiBi bias is non-positive, monotonically decreasing in |Δ|.
        flat = out.squeeze(-1)
        self.assertEqual(float(flat[0].item()), 0.0)  # Δ=0 → 0 bias
        self.assertLess(float(flat[2].item()), float(flat[1].item()))
        self.assertLess(float(flat[3].item()), float(flat[2].item()))


# ---------------------------------------------------------------------------
# Pair attention smoke — DC5.
# ---------------------------------------------------------------------------


class TestDC5PairAttention(unittest.TestCase):
    def test_dc5a_shape(self) -> None:
        torch.manual_seed(0)
        D = 8
        a = torch.randn(4, D)
        b = torch.randn(4, D)
        wq = torch.randn(D, D)
        wk = torch.randn(D, D)
        wv = torch.randn(D, D)
        op = torch.randn(D, D)
        out = pair_attention_logits(
            a, b, weight_q=wq, weight_k=wk, weight_v=wv, out_proj=op,
        )
        self.assertEqual(out.shape, (4, D))


# ---------------------------------------------------------------------------
# DC6 — public head exports (pcm.heads.v2_dual_channel).
# ---------------------------------------------------------------------------


class TestDC6PublicHeads(unittest.TestCase):
    def test_dc6a_dual_channel_pair_head_forward(self) -> None:
        from pcm.heads import DualChannelPairHead
        head = DualChannelPairHead(
            n_classes=5, slot_dim=8, attr_dim=8, hidden=16,
        )
        slot_a = torch.randn(4, 8)
        slot_b = torch.randn(4, 8)
        attr_a = torch.randn(4, 8)
        attr_b = torch.randn(4, 8)
        out = head(slot_a, slot_b, attr_a, attr_b)
        self.assertEqual(out.shape, (4, 5))

    def test_dc6b_pair_head_with_rpe(self) -> None:
        from pcm.heads import DualChannelPairHead
        head = DualChannelPairHead(
            n_classes=5, slot_dim=8, attr_dim=8, hidden=16,
            rpe_ranges=[(-3, 3), (-3, 3)], rpe_embed_dim=16,
        )
        sa = torch.randn(4, 8); sb = torch.randn(4, 8)
        aa = torch.randn(4, 8); ab = torch.randn(4, 8)
        dr = torch.tensor([0, 1, -1, 2])
        dc = torch.tensor([1, 0, 2, -1])
        out = head(sa, sb, aa, ab, deltas=(dr, dc))
        self.assertEqual(out.shape, (4, 5))

    def test_dc6c_pair_head_rpe_only(self) -> None:
        from pcm.heads import DualChannelPairHead
        head = DualChannelPairHead(
            n_classes=5, slot_dim=8, attr_dim=8, hidden=16,
            rpe_ranges=[(-3, 3), (-3, 3)], rpe_embed_dim=16,
            use_attr_path=False, use_slot_path=False,
        )
        dr = torch.tensor([0, 1, -1, 2])
        dc = torch.tensor([1, 0, 2, -1])
        out = head(deltas=(dr, dc))
        self.assertEqual(out.shape, (4, 5))

    def test_dc6d_pair_head_rejects_no_path(self) -> None:
        from pcm.heads import DualChannelPairHead
        with self.assertRaises(ValueError):
            DualChannelPairHead(
                n_classes=5, slot_dim=8, attr_dim=8,
                use_attr_path=False, use_slot_path=False,
                rpe_ranges=None,
            )

    def test_dc6e_gate_modes(self) -> None:
        from pcm.heads import DualChannelPairHead
        for mode in ("fixed", "schedule", "learned"):
            head = DualChannelPairHead(
                n_classes=3, slot_dim=4, attr_dim=4, hidden=8,
                gate_mode=mode,
            )
            sa = torch.randn(2, 4); sb = torch.randn(2, 4)
            aa = torch.randn(2, 4); ab = torch.randn(2, 4)
            out = head(sa, sb, aa, ab)
            self.assertEqual(out.shape, (2, 3))
        # Schedule progress
        head = DualChannelPairHead(
            n_classes=3, slot_dim=4, attr_dim=4, gate_mode="schedule",
        )
        head.set_progress(0.0)
        self.assertAlmostEqual(head.current_attn_lambda(), 1.0, places=4)
        head.set_progress(0.5)
        self.assertAlmostEqual(head.current_attn_lambda(), 0.0, places=4)
        head.set_progress(1.0)
        self.assertAlmostEqual(head.current_attn_lambda(), 0.0, places=4)

    def test_dc6f_slot_identity_aux_head(self) -> None:
        from pcm.heads import SlotIdentityAuxHead
        head = SlotIdentityAuxHead(slot_dim=8, n_concepts=10, hidden=16)
        slot = torch.randn(4, 8)
        out = head(slot)
        self.assertEqual(out.shape, (4, 10))

    def test_dc6g_pair_collapse_and_forward(self) -> None:
        from pcm.heads import DualChannelPairHead, pair_collapse_and_forward
        cg, cids, _, _ = _build_cg(slot_dim=8, attr_dim=8)
        head = DualChannelPairHead(
            n_classes=3, slot_dim=8, attr_dim=8, hidden=16,
        )
        # Pair up consecutive cids.
        ids_a = cids[:3]
        ids_b = cids[1:4]
        out = pair_collapse_and_forward(
            head, cg, base_facet="x",
            ids_a=ids_a, ids_b=ids_b,
            slot_shape=(8,), attr_shape=(8,), tick=0,
        )
        self.assertEqual(out.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
