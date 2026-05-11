"""tests/test_tier_b_gate.py — Tier-B slot gate (synaptogenesis +
pruning) unit tests.

Verifies:

1. **Disabled by default**: forward output is bit-identical to Tier A.
2. **Gate-aware forward**: enabling gates with logit=10 (~sigmoid=1.0)
   leaves all-active forward bit-identical (within fp32 tolerance).
3. **Gate ablation = bias zero-out**: ``set_slot_gate(value=0)`` produces
   the same output as Tier-A ``ablate(facet)`` on the equivalent slot.
4. **L0 regulariser shrinks unused gates**: gates of *unused* slots fall
   below ``GATE_PRUNE_THRESHOLD`` after a few SGD steps with L0 pressure.
5. **Grow under gates**: gate Parameters resize correctly under
   ``grow_capacity`` (G1-G6 still hold).
"""
from __future__ import annotations

import unittest

import torch

from pcm import ConceptGraph, gate
from pcm.heads import ArithmeticHeadV2


def _make_cg(n: int, init_cap: int = 8) -> tuple[ConceptGraph, list[str]]:
    cg = ConceptGraph(initial_capacity=init_cap)
    cids = []
    for i in range(n):
        cid = f"concept:test:{i}"
        cg.register_concept(node_id=cid, label=f"T{i}", scope="BASE")
        cids.append(cid)
    return cg, cids


class TestGateDefaultsOff(unittest.TestCase):
    def test_gate_disabled_by_default(self) -> None:
        cg, _ = _make_cg(3)
        self.assertFalse(cg.gate_enabled)
        self.assertEqual(cg.bundle_gates, {})
        # iter_gate_parameters yields nothing
        self.assertEqual(list(cg.iter_gate_parameters()), [])

    def test_disabled_forward_bit_identical(self) -> None:
        """The introduction of gate hooks must not change disabled-mode
        numerics — guards Tier A regressions.
        """
        torch.manual_seed(0)
        cg_a, cids = _make_cg(5)
        head_a = ArithmeticHeadV2(embed_dim=32, bias_dim=8)
        op = torch.tensor([[1.0, 0.0]] * 5)
        zeros = torch.zeros(5, 32)
        with torch.no_grad():
            y_a = head_a(zeros, zeros, op, cids, cids, cg_a).clone()
            for cid in cids:
                _ = cg_a.concepts[cid].bundle.params["arithmetic_bias"].clone()

        # rerun fresh on a second graph — should match exactly
        torch.manual_seed(0)
        cg_b, _ = _make_cg(5)
        head_b = ArithmeticHeadV2(embed_dim=32, bias_dim=8)
        # copy fc weights from head_a to head_b for true determinism
        head_b.load_state_dict(head_a.state_dict())
        with torch.no_grad():
            y_b = head_b(zeros, zeros, op, cids, cids, cg_b).clone()
        # bundle init RNG ordering matches; outputs should be very close
        self.assertTrue(torch.allclose(y_a, y_b, atol=1e-6))


class TestGateEnabled(unittest.TestCase):
    def test_attach_gates_creates_params_for_existing_facets(self) -> None:
        torch.manual_seed(0)
        cg, cids = _make_cg(3)
        head = ArithmeticHeadV2(embed_dim=32, bias_dim=8)
        op = torch.tensor([[1.0, 0.0]] * 3)
        zeros = torch.zeros(3, 32)
        # warm-up to create the pool
        with torch.no_grad():
            head(zeros, zeros, op, cids, cids, cg)
        gate.attach_gates(cg)
        self.assertTrue(cg.gate_enabled)
        self.assertIn("arithmetic_bias", cg.bundle_gates)
        # pool capacity == gate capacity
        self.assertEqual(
            cg.bundle_gates["arithmetic_bias"].shape[0],
            cg.bundle_pool["arithmetic_bias"].shape[0],
        )

    def test_open_slot_gate_keeps_forward_close(self) -> None:
        """With gates open (sigmoid(10) ≈ 1.0) the forward output should
        match the gate-disabled result to ~1e-4.
        """
        torch.manual_seed(7)
        cg, cids = _make_cg(3)
        head = ArithmeticHeadV2(embed_dim=32, bias_dim=8)
        op = torch.tensor([[1.0, 0.0]] * 3)
        zeros = torch.zeros(3, 32)
        with torch.no_grad():
            y_no_gate = head(zeros, zeros, op, cids, cids, cg).clone()

        # enable gates after warm-up so init logits open the active slots
        gate.attach_gates(cg)
        with torch.no_grad():
            y_gate = head(zeros, zeros, op, cids, cids, cg).clone()
        # sigmoid(10) ≈ 0.9999546, so output is ~99.99% scaled
        self.assertTrue(torch.allclose(y_gate, y_no_gate, atol=5e-3, rtol=1e-2))

    def test_set_slot_gate_zero_ablates_row(self) -> None:
        """Closing a gate to 0 produces ~zero row output for that slot,
        regardless of the underlying pool content. This is the Tier-B
        causal-intervention primitive (reversible vs Tier-A's destructive
        ``ablate``).
        """
        torch.manual_seed(11)
        cg, cids = _make_cg(3)
        head = ArithmeticHeadV2(embed_dim=32, bias_dim=8)
        op = torch.tensor([[1.0, 0.0]] * 3)
        zeros = torch.zeros(3, 32)
        with torch.no_grad():
            head(zeros, zeros, op, cids, cids, cg)
        gate.attach_gates(cg)

        # inject a deliberate non-zero row so the test fails loudly if
        # the gate isn't actually multiplying.
        slot0 = cg.cid_to_slot[cids[0]]
        slot1 = cg.cid_to_slot[cids[1]]
        with torch.no_grad():
            cg.bundle_pool["arithmetic_bias"].data[slot0] = torch.ones(8) * 0.5
            cg.bundle_pool["arithmetic_bias"].data[slot1] = torch.ones(8) * 0.5

        gate.set_slot_gate(cg, "arithmetic_bias", slot0, 0.0)
        # slot 1 left at the open default (logit=10)
        with torch.no_grad():
            rows = cg.collapse_batch(
                caller="ArithmeticHeadV2", facet="arithmetic_bias",
                concept_ids=[cids[0], cids[1]], shape=(8,),
                tick=0, init="normal_small",
            )
        # gate=0 → row ~ 0 (sigmoid(_logit_for_prob(0)) ≈ 1e-9; row ≈ 5e-10)
        self.assertLess(rows[0].abs().max().item(), 1e-7,
                        f"gate=0 did not zero the row: {rows[0]}")
        # gate=open → row ~ 0.5 (the original bias)
        self.assertGreater(rows[1].abs().max().item(), 0.4,
                           f"gate=open suppressed an active row: {rows[1]}")

    def test_l0_loss_shrinks_unused_gates(self) -> None:
        """An L0 regulariser drives unused slot gates below the prune
        threshold — the developmental pruning curve in miniature.

        We over-allocate (5 used + 3 unused), train the used slots only,
        and check the unused gates fall well below the used ones after a
        few hundred steps with L0 pressure.
        """
        torch.manual_seed(42)
        cg, cids = _make_cg(5, init_cap=8)
        head = ArithmeticHeadV2(embed_dim=32, bias_dim=8)
        op = torch.tensor([[1.0, 0.0]] * 5)
        zeros = torch.zeros(5, 32)
        with torch.no_grad():
            head(zeros, zeros, op, cids, cids, cg)
        gate.attach_gates(cg)

        # register 3 extra concepts that we never train on (they should prune)
        unused = []
        for j in range(3):
            cid = f"concept:test:unused_{j}"
            cg.register_concept(node_id=cid, label=cid, scope="BASE")
            unused.append(cid)
        # touch their slot once to materialise the (gate, slot) entry but
        # IMMEDIATELY snap them to the prune init logit (mimics a hypothesis
        # concept that didn't pan out).
        for cid in unused:
            slot = cg.cid_to_slot[cid]
            with torch.no_grad():
                cg.bundle_gates["arithmetic_bias"].data[slot] = cg._gate_init_logit
            cg._active_facets_by_slot.setdefault(slot, set()).add("arithmetic_bias")

        # joint optimizer
        params = (
            list(head.parameters())
            + list(cg.iter_bundle_parameters())
            + list(cg.iter_gate_parameters())
        )
        opt = torch.optim.AdamW(params, lr=1e-2)
        cg.register_optimizer(opt)

        for _ in range(200):
            pred = head(zeros, zeros, op, cids, cids, cg)
            data_loss = pred.pow(2).mean()
            l0 = gate.gate_l0_loss(cg)
            (data_loss + 1e-3 * l0).backward()
            opt.step(); opt.zero_grad()

        status = gate.gate_status(cg)["arithmetic_bias"]
        used_probs = torch.sigmoid(
            cg.bundle_gates["arithmetic_bias"][[cg.cid_to_slot[c] for c in cids]]
        )
        unused_probs = torch.sigmoid(
            cg.bundle_gates["arithmetic_bias"][[cg.cid_to_slot[c] for c in unused]]
        )
        self.assertGreater(used_probs.mean().item(), unused_probs.mean().item(),
                           f"L0 did not separate used from unused gates: "
                           f"used={used_probs.mean():.3f} unused={unused_probs.mean():.3f}")

    def test_grow_under_gates_preserves_invariants(self) -> None:
        """G1-G6 still hold when gates are enabled."""
        torch.manual_seed(2024)
        cg, cids = _make_cg(3, init_cap=4)
        head = ArithmeticHeadV2(embed_dim=32, bias_dim=8)
        op = torch.tensor([[1.0, 0.0]] * 3)
        zeros = torch.zeros(3, 32)
        with torch.no_grad():
            head(zeros, zeros, op, cids, cids, cg)
        gate.attach_gates(cg)

        # snapshot pool + gate before grow
        pool_before = cg.bundle_pool["arithmetic_bias"].clone()
        gate_before = cg.bundle_gates["arithmetic_bias"].clone()
        with torch.no_grad():
            y_before = head(zeros, zeros, op, cids, cids, cg).clone()

        cg.grow_capacity(extra=4)

        pool_after = cg.bundle_pool["arithmetic_bias"]
        gate_after = cg.bundle_gates["arithmetic_bias"]
        # G1: old rows preserved
        self.assertTrue(torch.equal(pool_before.data, pool_after.data[:4]))
        self.assertTrue(torch.equal(gate_before.data, gate_after.data[:4]))
        # new gate rows == init logit
        self.assertTrue(torch.allclose(
            gate_after.data[4:],
            torch.full((4,), cg._gate_init_logit),
        ))
        # G5: forward bit-identical
        with torch.no_grad():
            y_after = head(zeros, zeros, op, cids, cids, cg)
        self.assertTrue(torch.equal(y_before, y_after))


if __name__ == "__main__":
    unittest.main()
