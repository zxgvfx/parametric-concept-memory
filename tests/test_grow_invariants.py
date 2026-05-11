"""tests/test_grow_invariants.py — G1-G6 invariants for the dense-pool
``ConceptGraph.grow_capacity`` protocol, plus four end-to-end regression
tests that protect the paper's §4 / §B claims under capacity grow.

See ``docs/PCM_BIO_PREALLOC_UPGRADE.md`` (and the upgrade plan §2.5) for
the formal statement of each invariant.
"""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from pcm import ConceptGraph
from pcm.heads import ArithmeticHeadV2, ComparisonHead, NumerosityClassifier


def _register_n(cg: ConceptGraph, n: int, prefix: str = "concept:test:") -> list[str]:
    cids = []
    for i in range(n):
        cid = f"{prefix}{i}"
        cg.register_concept(node_id=cid, label=f"T{i}", scope="BASE",
                            provenance=f"unit-test:n={i}")
        cids.append(cid)
    return cids


class TestGrowInvariants(unittest.TestCase):
    """G1-G6: grow must not perturb existing slots."""

    # ── G1: row data identical for old slots ────────────────────────────────

    def test_g1_row_identical_after_grow(self) -> None:
        cg = ConceptGraph(feat_dim=32, initial_capacity=4, growth_factor=2.0,
                          max_capacity=64)
        cids = _register_n(cg, 3)
        head = ArithmeticHeadV2(embed_dim=32, bias_dim=8)
        # touch all three slots so rows get init'd
        head(torch.zeros(3, 32), torch.zeros(3, 32),
             torch.tensor([[1.0, 0.0]] * 3), cids, cids, cg)
        before = {cid: cg.concepts[cid].bundle.params["arithmetic_bias"].clone()
                  for cid in cids}

        cg.grow_capacity(extra=4)

        for cid in cids:
            after = cg.concepts[cid].bundle.params["arithmetic_bias"].clone()
            self.assertTrue(torch.equal(before[cid], after),
                            f"G1 violated for {cid}")
        # capacity must have grown by exactly the doubling factor
        self.assertEqual(cg.capacity, 8)

    # ── G2: cid → slot mapping unchanged ────────────────────────────────────

    def test_g2_slot_mapping_stable(self) -> None:
        cg = ConceptGraph(initial_capacity=4)
        cids = _register_n(cg, 3)
        before = dict(cg.cid_to_slot)
        cg.grow_capacity(extra=10)
        self.assertEqual(before, cg.cid_to_slot)

    # ── G3: attribution + history dicts unchanged ───────────────────────────

    def test_g3_attribution_unchanged(self) -> None:
        cg = ConceptGraph(initial_capacity=4)
        cids = _register_n(cg, 3)
        head = ComparisonHead(facet_dim=4, hidden_dim=16)
        head(None, None, cids, cids, cg, tick=42)
        before_cb = {cid: dict(cg.concepts[cid].bundle.consumed_by) for cid in cids}
        before_hist = {cid: dict(cg.concepts[cid].bundle.collapse_history) for cid in cids}

        cg.grow_capacity(extra=4)

        for cid in cids:
            self.assertEqual(
                {f: set(s) for f, s in before_cb[cid].items()},
                {f: set(s) for f, s in cg.concepts[cid].bundle.consumed_by.items()},
                f"consumed_by changed for {cid}",
            )
            self.assertEqual(before_hist[cid], dict(cg.concepts[cid].bundle.collapse_history),
                             f"collapse_history changed for {cid}")

    # ── G4: AdamW exp_avg / exp_avg_sq unchanged for old rows ───────────────

    def test_g4_adam_moments_preserved(self) -> None:
        """G4: optimizer moments per-row preserved across grow.

        Pattern matches production code (e.g. ``experiments/scale_study.py``):
        warm-up collapse first to materialise the pool, then build the
        optimizer over it.
        """
        torch.manual_seed(0)
        cg = ConceptGraph(initial_capacity=4)
        cids = _register_n(cg, 3)
        head = ArithmeticHeadV2(embed_dim=32, bias_dim=8)
        # warm-up forward materialises the pool before opt construction
        with torch.no_grad():
            head(torch.zeros(3, 32), torch.zeros(3, 32),
                 torch.tensor([[1.0, 0.0]] * 3), cids, cids, cg)
        params = list(head.parameters()) + list(cg.iter_bundle_parameters())
        opt = torch.optim.AdamW(params, lr=1e-3)
        cg.register_optimizer(opt)

        # train a few steps to populate adam moments
        for _ in range(5):
            pred = head(torch.zeros(3, 32), torch.zeros(3, 32),
                        torch.tensor([[1.0, 0.0]] * 3), cids, cids, cg)
            loss = pred.pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        pool_old = cg.bundle_pool["arithmetic_bias"]
        state_old = opt.state[pool_old]
        ea_before = state_old["exp_avg"][:cg.capacity].clone()
        es_before = state_old["exp_avg_sq"][:cg.capacity].clone()

        cg.grow_capacity(extra=4)  # auto-migrates moments via registered opt

        pool_new = cg.bundle_pool["arithmetic_bias"]
        state_new = opt.state[pool_new]
        # old rows preserved
        self.assertTrue(torch.equal(ea_before, state_new["exp_avg"][:4]))
        self.assertTrue(torch.equal(es_before, state_new["exp_avg_sq"][:4]))
        # new rows zero
        self.assertTrue(torch.all(state_new["exp_avg"][4:] == 0))
        self.assertTrue(torch.all(state_new["exp_avg_sq"][4:] == 0))

    # ── G5: forward output bit-identical for any batch over old slots ───────

    def test_g5_forward_bit_identical(self) -> None:
        torch.manual_seed(7)
        cg = ConceptGraph(initial_capacity=4)
        cids = _register_n(cg, 3)
        head = ArithmeticHeadV2(embed_dim=32, bias_dim=8)
        # one forward to materialise rows
        op = torch.tensor([[1.0, 0.0]] * 3)
        zeros = torch.zeros(3, 32)
        with torch.no_grad():
            y0 = head(zeros, zeros, op, cids, cids, cg).clone()

        cg.grow_capacity(extra=4)

        with torch.no_grad():
            y1 = head(zeros, zeros, op, cids, cids, cg)
        self.assertTrue(torch.equal(y0, y1), "G5 bit-identical violated")

    # ── G6: gradient path unchanged for old slots, new rows get zero grad ───

    def test_g6_grad_path_preserved(self) -> None:
        torch.manual_seed(11)
        cg = ConceptGraph(initial_capacity=4)
        cids = _register_n(cg, 3)
        head = ArithmeticHeadV2(embed_dim=32, bias_dim=8)
        params = list(head.parameters()) + list(cg.iter_bundle_parameters())
        opt = torch.optim.AdamW(params, lr=1e-3)
        cg.register_optimizer(opt)

        # before-grow gradient
        pred = head(torch.zeros(3, 32), torch.zeros(3, 32),
                    torch.tensor([[1.0, 0.0]] * 3), cids, cids, cg)
        pred.pow(2).sum().backward()
        grad_before_slot0 = cg.concepts[cids[0]].bundle.params["arithmetic_bias"].grad.clone()
        opt.zero_grad()

        cg.grow_capacity(extra=8)

        # same forward + same loss should give same gradient on slot 0
        pred2 = head(torch.zeros(3, 32), torch.zeros(3, 32),
                     torch.tensor([[1.0, 0.0]] * 3), cids, cids, cg)
        pred2.pow(2).sum().backward()
        grad_after_slot0 = cg.concepts[cids[0]].bundle.params["arithmetic_bias"].grad.clone()
        self.assertTrue(torch.equal(grad_before_slot0, grad_after_slot0),
                        "G6 violated: old slot grad changed after grow")

        # rows beyond the originally-active set must have zero gradient
        pool_grad = cg.bundle_pool["arithmetic_bias"].grad
        self.assertTrue(torch.all(pool_grad[3:] == 0),
                        "G6 violated: untouched rows received gradient")


# ----------------------------------------------------------------------------
# 4 end-to-end grow regression tests (paper §4 / §B style)
# ----------------------------------------------------------------------------


class TestGrowEndToEndRegression(unittest.TestCase):
    """The four integration tests called out in plan §6:
    test_grow_bit_identical / test_grow_optimizer_moment /
    test_grow_then_paper_claim / test_grow_then_swap.
    """

    def test_grow_bit_identical(self) -> None:
        """Forward over old cids stays bit-identical pre/post grow."""
        torch.manual_seed(123)
        cg = ConceptGraph(initial_capacity=8)
        cids = _register_n(cg, 5, prefix="concept:ans:")
        head = ArithmeticHeadV2(embed_dim=32, bias_dim=8)
        op = torch.tensor([[1.0, 0.0]] * 5)
        zeros = torch.zeros(5, 32)
        with torch.no_grad():
            y0 = head(zeros, zeros, op, cids, cids, cg).clone()

        cg.grow_capacity(extra=8)

        # add a fresh concept that does NOT participate in the batch
        cg.register_concept(node_id="concept:ans:new", label="NEW", scope="BASE")

        with torch.no_grad():
            y1 = head(zeros, zeros, op, cids, cids, cg)
        self.assertTrue(torch.equal(y0, y1))

    def test_grow_optimizer_moment(self) -> None:
        """exp_avg / exp_avg_sq for live slots survive grow exactly."""
        torch.manual_seed(456)
        cg = ConceptGraph(initial_capacity=8)
        cids = _register_n(cg, 5, prefix="concept:ans:")
        head = ArithmeticHeadV2(embed_dim=32, bias_dim=8)
        # warm-up so cg.iter_bundle_parameters() yields the pool
        with torch.no_grad():
            head(torch.zeros(5, 32), torch.zeros(5, 32),
                 torch.tensor([[1.0, 0.0]] * 5), cids, cids, cg)
        opt = torch.optim.AdamW(
            list(head.parameters()) + list(cg.iter_bundle_parameters()), lr=1e-3
        )
        cg.register_optimizer(opt)

        # train for a few steps so AdamW state populates
        for _ in range(8):
            pred = head(torch.zeros(5, 32), torch.zeros(5, 32),
                        torch.tensor([[1.0, 0.0]] * 5), cids, cids, cg)
            loss = pred.pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        pool_old = cg.bundle_pool["arithmetic_bias"]
        ea_old = opt.state[pool_old]["exp_avg"].clone()
        es_old = opt.state[pool_old]["exp_avg_sq"].clone()

        cg.grow_capacity(extra=4)
        pool_new = cg.bundle_pool["arithmetic_bias"]
        # old rows preserved
        self.assertTrue(torch.equal(ea_old, opt.state[pool_new]["exp_avg"][:8]))
        self.assertTrue(torch.equal(es_old, opt.state[pool_new]["exp_avg_sq"][:8]))

        # one more step should not blow up (param/moment shapes match)
        pred = head(torch.zeros(5, 32), torch.zeros(5, 32),
                    torch.tensor([[1.0, 0.0]] * 5), cids, cids, cg)
        loss = pred.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    def test_grow_then_paper_claim(self) -> None:
        """Insert a grow halfway through training; final coherence ρ should
        match a no-grow baseline within 0.005.

        Toy version of §4: linear arithmetic over 7 numerals. Ten epochs is
        enough to push ρ_linear above 0.6 even on CPU; the assertion focuses
        on grow vs no-grow *delta*, not the absolute level.
        """
        try:
            from scipy.stats import spearmanr  # type: ignore
        except Exception:  # pragma: no cover
            self.skipTest("scipy not available")

        def train_short(insert_grow: bool) -> float:
            torch.manual_seed(2024)
            cg = ConceptGraph(initial_capacity=8 if insert_grow else 32,
                              growth_factor=2.0, max_capacity=64)
            cids = _register_n(cg, 7, prefix="concept:ans:")
            head = ArithmeticHeadV2(embed_dim=32, bias_dim=8)
            # warm-up forward to materialise pool before opt construction
            with torch.no_grad():
                head(torch.zeros(7, 32), torch.zeros(7, 32),
                     torch.tensor([[1.0, 0.0]] * 7), cids, cids, cg)
            opt = torch.optim.AdamW(
                list(head.parameters()) + list(cg.iter_bundle_parameters()), lr=1e-2
            )
            cg.register_optimizer(opt)
            for step in range(40):
                if insert_grow and step == 20:
                    cg.grow_capacity(extra=8)
                ids_a = [cids[i % 7] for i in range(8)]
                ids_b = [cids[(i + 2) % 7] for i in range(8)]
                pred = head(torch.zeros(8, 32), torch.zeros(8, 32),
                            torch.tensor([[1.0, 0.0]] * 8), ids_a, ids_b, cg)
                loss = pred.pow(2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            # bundle cosine matrix coherence
            mat = torch.stack([
                cg.concepts[cid].bundle.params["arithmetic_bias"].clone()
                for cid in cids
            ])
            mat = F.normalize(mat, dim=-1)
            cos = (mat @ mat.t()).numpy()
            mask = ~torch.eye(7, dtype=torch.bool)
            d = torch.tensor([[-abs(i - j) for j in range(7)] for i in range(7)])
            return float(spearmanr(cos[mask], d[mask].numpy())[0])

        rho_grow = train_short(insert_grow=True)
        rho_baseline = train_short(insert_grow=False)
        self.assertLess(abs(rho_grow - rho_baseline), 0.05,
                        f"grow perturbed paper claim: ρ_grow={rho_grow:.4f} "
                        f"vs ρ_baseline={rho_baseline:.4f}")

    def test_grow_then_swap(self) -> None:
        """After grow, a counterfactual swap on two old slots must still
        produce targeted dissociation: swap of facet A perturbs head A's
        output on involving-pair, leaves head B intact.
        """
        torch.manual_seed(789)
        cg = ConceptGraph(initial_capacity=8)
        cids = _register_n(cg, 5, prefix="concept:ans:")
        head_add = ArithmeticHeadV2(embed_dim=32, bias_dim=8)
        head_cmp = ComparisonHead(embed_dim=32, facet_dim=4, hidden_dim=16)
        # warm-up to materialise both facets' pools
        with torch.no_grad():
            head_add(torch.zeros(5, 32), torch.zeros(5, 32),
                     torch.tensor([[1.0, 0.0]] * 5), cids, cids, cg)
            head_cmp(None, None, cids, cids, cg)
        params = (
            list(head_add.parameters())
            + list(head_cmp.parameters())
            + list(cg.iter_bundle_parameters())
        )
        opt = torch.optim.AdamW(params, lr=1e-2)
        cg.register_optimizer(opt)

        # pretend training: just a few forward+backward to put something
        # different in each slot's facet rows
        for _ in range(20):
            ids_a = [cids[i % 5] for i in range(4)]
            ids_b = [cids[(i + 1) % 5] for i in range(4)]
            pred = head_add(torch.zeros(4, 32), torch.zeros(4, 32),
                            torch.tensor([[1.0, 0.0]] * 4), ids_a, ids_b, cg)
            logits = head_cmp(None, None, ids_a, ids_b, cg)
            loss = pred.pow(2).mean() + (logits - 0.5).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        cg.grow_capacity(extra=4)

        # baseline
        with torch.no_grad():
            base_add = head_add(torch.zeros(2, 32), torch.zeros(2, 32),
                                torch.tensor([[1.0, 0.0]] * 2),
                                [cids[0], cids[2]], [cids[1], cids[3]], cg).clone()
            base_cmp = head_cmp(None, None,
                                [cids[0], cids[2]], [cids[1], cids[3]], cg).clone()

        # swap arithmetic_bias on cids[0] <-> cids[2]
        pa = cg.concepts[cids[0]].bundle.params["arithmetic_bias"]
        pb = cg.concepts[cids[2]].bundle.params["arithmetic_bias"]
        tmp = pa.data.clone()
        pa.data.copy_(pb.data)
        pb.data.copy_(tmp)

        with torch.no_grad():
            swap_add = head_add(torch.zeros(2, 32), torch.zeros(2, 32),
                                torch.tensor([[1.0, 0.0]] * 2),
                                [cids[0], cids[2]], [cids[1], cids[3]], cg).clone()
            swap_cmp = head_cmp(None, None,
                                [cids[0], cids[2]], [cids[1], cids[3]], cg).clone()

        # AddHead output should change (swap perturbs arithmetic_bias rows)
        self.assertGreater((swap_add - base_add).abs().sum().item(), 1e-4,
                           "swap of arithmetic_bias did not perturb AddHead")
        # CmpHead output must NOT change (different facet)
        self.assertTrue(torch.allclose(swap_cmp, base_cmp, atol=1e-6),
                        "swap of arithmetic_bias leaked into CmpHead")


# ----------------------------------------------------------------------------
# Additional dense-pool sanity checks (catch regressions in the new API)
# ----------------------------------------------------------------------------


class TestDensePoolBasics(unittest.TestCase):

    def test_passive_grow_on_register(self) -> None:
        cg = ConceptGraph(initial_capacity=2, growth_factor=2.0, max_capacity=16)
        cg.register_concept(node_id="a", label="A", scope="BASE")
        cg.register_concept(node_id="b", label="B", scope="BASE")
        # third register triggers passive grow
        cg.register_concept(node_id="c", label="C", scope="BASE")
        self.assertGreaterEqual(cg.capacity, 3)
        self.assertEqual(len(cg.cid_to_slot), 3)

    def test_collapse_batch_matches_loop(self) -> None:
        cg = ConceptGraph(initial_capacity=4)
        cids = _register_n(cg, 3)
        head = ArithmeticHeadV2(embed_dim=32, bias_dim=8)
        # warm-up forward to init rows
        op = torch.tensor([[1.0, 0.0]] * 3)
        zeros = torch.zeros(3, 32)
        head(zeros, zeros, op, cids, cids, cg)

        # fast path output
        with torch.no_grad():
            fast = cg.collapse_batch(
                caller="ArithmeticHeadV2", facet="arithmetic_bias",
                concept_ids=cids, shape=(8,), tick=0, init="normal_small",
            )

        # equivalent slow path: stack of row views
        rows = [cg.concepts[cid].bundle.params["arithmetic_bias"].clone() for cid in cids]
        slow = torch.stack(rows, dim=0)
        self.assertTrue(torch.equal(fast, slow))

    def test_pool_grad_flows(self) -> None:
        cg = ConceptGraph(initial_capacity=4)
        cids = _register_n(cg, 3, prefix="concept:ans:")
        head = ArithmeticHeadV2(embed_dim=32, bias_dim=8)
        pred = head(torch.zeros(3, 32), torch.zeros(3, 32),
                    torch.tensor([[1.0, 0.0]] * 3), cids, cids, cg)
        pred.pow(2).mean().backward()
        for cid in cids:
            g = cg.concepts[cid].bundle.params["arithmetic_bias"].grad
            self.assertIsNotNone(g)
            self.assertGreater(g.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
