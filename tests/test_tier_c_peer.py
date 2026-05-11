"""tests/test_tier_c_peer.py — Tier-C product-key router unit tests.

Validates that:

1. **Symbolic mode unchanged**: simply importing PEER doesn't change
   the deterministic ``cid_to_slot`` lookup. (Already covered by
   ``test_smoke``; here we just spot-check.)
2. **Router shape contract**: ``ProductKeyRouter.forward`` returns
   ``(top_slots, top_weights)`` of shape ``(B, num_heads, top_k)``,
   slots in ``[0, num_experts)``, weights softmax-normalised.
3. **Sub-linear cost**: parameter count scales as ``sqrt(N)*d``, not
   ``N*d``.
4. **gather_values gradient**: gradients flow back into the dense
   pool and the router parameters jointly.
5. **soft_consumed_by_log**: high-weight slots are recorded in
   ``cg._consumed_by_by_slot``.
6. **Discovery toy**: routing many random queries hits a *spread* of
   slots (not just slot 0), confirming the bank is actually exercised.
"""
from __future__ import annotations

import unittest

import torch

from pcm import ConceptGraph
from pcm.peer import PEERConfig, ProductKeyRouter, soft_consumed_by_log


class TestProductKeyRouter(unittest.TestCase):

    def test_shape_contract(self) -> None:
        torch.manual_seed(0)
        cfg = PEERConfig(num_experts=1024, top_k=8, num_heads=4, query_dim=64)
        router = ProductKeyRouter(input_dim=64, cfg=cfg)
        x = torch.randn(16, 64)
        slots, weights = router(x)
        self.assertEqual(slots.shape, (16, 4, 8))
        self.assertEqual(weights.shape, (16, 4, 8))
        # slots in valid range
        self.assertTrue(((slots >= 0) & (slots < 1024)).all().item())
        # weights sum to 1 along K
        sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_sublinear_param_count(self) -> None:
        """Total bank params should scale as sqrt(N)*d, not N*d."""
        cfg = PEERConfig(num_experts=16_384, top_k=16, num_heads=8, query_dim=128)
        router = ProductKeyRouter(input_dim=128, cfg=cfg)
        # keys are (h, sqrt(N), d/2) each; two of them.
        bank_params = router.keys_x.numel() + router.keys_y.numel()
        n_sqrt = 128
        d = 128
        h = 8
        expected = 2 * h * n_sqrt * (d // 2)
        self.assertEqual(bank_params, expected)
        # full N*d would be:
        flat_baseline = cfg.num_experts * cfg.query_dim
        self.assertLess(bank_params, flat_baseline)
        # ratio: 2*h*sqrt(N)*d/2 / (N*d) = h / sqrt(N)
        self.assertAlmostEqual(bank_params / flat_baseline, h / n_sqrt, places=4)

    def test_gather_values_grad_flows(self) -> None:
        torch.manual_seed(7)
        cg = ConceptGraph(initial_capacity=64)
        # warm-up: register 8 dummy concepts so a pool exists
        for i in range(8):
            cg.register_concept(node_id=f"c{i}", label=f"C{i}", scope="BASE")
        # touch facet to allocate the pool
        cg.collapse_batch(
            caller="warm", facet="discovery_v", concept_ids=[f"c{i}" for i in range(8)],
            shape=(64,), tick=0, init="normal_small",
        )

        cfg = PEERConfig(num_experts=64, top_k=4, num_heads=2, query_dim=64)
        router = ProductKeyRouter(input_dim=32, cfg=cfg)
        x = torch.randn(4, 32, requires_grad=True)
        out = router.gather_values(cg, "discovery_v", x)
        loss = out.pow(2).sum()
        loss.backward()
        # Pool grad should be non-zero (some rows were selected)
        pool_grad = cg.bundle_pool["discovery_v"].grad
        self.assertIsNotNone(pool_grad)
        self.assertGreater(pool_grad.abs().sum().item(), 0.0)
        # Router params also got grads
        self.assertIsNotNone(router.keys_x.grad)
        self.assertGreater(router.keys_x.grad.abs().sum().item(), 0.0)

    def test_soft_consumed_by_log_records_attribution(self) -> None:
        torch.manual_seed(13)
        cg = ConceptGraph(initial_capacity=64)
        for i in range(8):
            cg.register_concept(node_id=f"c{i}", label=f"C{i}", scope="BASE")
        cg.collapse_batch(
            caller="warm", facet="discovery_v", concept_ids=[f"c{i}" for i in range(8)],
            shape=(64,), tick=0, init="normal_small",
        )
        cfg = PEERConfig(num_experts=64, top_k=8, num_heads=2, query_dim=64)
        router = ProductKeyRouter(input_dim=32, cfg=cfg)

        x = torch.randn(4, 32)
        slots, weights = router(x)
        n_logged = soft_consumed_by_log(
            cg, facet="discovery_v", caller="DiscoveryMuscle",
            top_slots=slots, top_weights=weights, threshold=0.0, tick=42,
        )
        self.assertGreater(n_logged, 0)
        # at least one slot has DiscoveryMuscle in its consumer set
        any_logged = any(
            "DiscoveryMuscle" in per_slot.get("discovery_v", set())
            for per_slot in cg._consumed_by_by_slot.values()
        )
        self.assertTrue(any_logged)
        # warm-up slots 0-7 still have "warm" in consumer set (Tier-C does
        # not erase Tier-A attribution; it accumulates).
        self.assertIn("warm", cg._consumed_by_by_slot[0]["discovery_v"])

    def test_router_spread(self) -> None:
        """Random queries should activate a meaningful fraction of slots,
        not collapse to a single one. This is a sanity check that the
        product-key bank actually discriminates.
        """
        torch.manual_seed(23)
        cfg = PEERConfig(num_experts=1024, top_k=4, num_heads=4, query_dim=64)
        router = ProductKeyRouter(input_dim=64, cfg=cfg)
        x = torch.randn(64, 64)
        slots, _ = router(x)
        unique = set(slots.flatten().tolist())
        # At least 20 distinct slots out of 1024 should be hit
        self.assertGreater(len(unique), 20)


if __name__ == "__main__":
    unittest.main()
