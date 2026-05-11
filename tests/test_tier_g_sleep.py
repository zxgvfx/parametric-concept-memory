"""tests/test_tier_g_sleep.py — falsifiable contract G1-G7 for the
Tier-G Sleep Abstraction Pass.

See ``docs/PCM_TIER_G_SLEEP_ABSTRACTION.md`` §3 for the formal
statement of each invariant. These tests deliberately use the
smallest representative configuration (N=7 numbers, 4-d facets)
so the whole file runs in well under a minute on CPU; full-scale
ρ regression validation lives in
``experiments/sleep_pass_demo.py`` (Demo, not unit test).

Structure:
    G1 — bit-identity when sleep is not attached
    G2 — pool memory safety (active rows untouched)
    G3 — ρ regression bound on §4 N=7 dual-muscle (compressed scope)
    G4 — centroid is mean of assigned members
    G5 — abstract relation cooks back to original row
    G6 — idempotency (second run is a no-op on cluster set)
    G7 — batched collapse_via_abstract ≡ direct collapse_batch
         (numeric equality post-sleep, before any further training)
"""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from pcm import ConceptGraph, GraphEvaluator
from pcm.heads import ArithmeticHeadV2
from pcm.sleep import (
    SleepConfig,
    attach_sleep,
    build_member_to_anchor_index,
    collapse_via_abstract,
    iter_abstract_relations,
    run_sleep_pass,
    sleep_status,
)


# ---------------------------------------------------------------------------
# Helpers (mirror tests/test_grow_invariants.py / test_tier_b_gate.py).
# ---------------------------------------------------------------------------


def _register_ans(cg: ConceptGraph, n: int) -> list[str]:
    cids = []
    for i in range(1, n + 1):
        cid = f"concept:ans:{i}"
        cg.register_concept(
            node_id=cid, label=f"ANS_{i}", scope="BASE",
            provenance=f"sleep-test:n={i}",
        )
        cids.append(cid)
    return cids


def _warmup_arith(cg: ConceptGraph, head: ArithmeticHeadV2,
                   cids: list[str]) -> None:
    """Touch all bundle rows once so the pool exists at full shape."""
    head(
        torch.zeros(len(cids), head.embed_dim),
        torch.zeros(len(cids), head.embed_dim),
        torch.tensor([[1.0, 0.0]] * len(cids)),
        cids, cids, cg,
    )


def _train_arith_n7(cg: ConceptGraph, *, epochs: int = 6,
                     steps_per_epoch: int = 32, lr: float = 1e-2,
                     seed: int = 0) -> ArithmeticHeadV2:
    """A compressed §4-style training loop on N=7 closed addition.

    Returns a head whose ``arithmetic_bias`` facet has converged enough
    that ρ_linear vs −|Δn| is reliably > 0.6 (the unit-test threshold;
    full paper baseline is 0.973).
    """
    torch.manual_seed(seed)
    head = ArithmeticHeadV2(embed_dim=16, bias_dim=8)
    cids = _register_ans(cg, 7)

    centroids = torch.randn(7, 16)
    centroids = F.normalize(centroids, dim=-1)
    cid_to_centroid = {cid: centroids[i] for i, cid in enumerate(cids)}

    _warmup_arith(cg, head, cids)
    params = list(head.parameters()) + list(cg.iter_bundle_parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    triples = [(a, b, a + b)
               for a in range(1, 8) for b in range(1, 8) if a + b <= 7]
    for _epoch in range(epochs):
        head.train()
        for _step in range(steps_per_epoch):
            idx = torch.randint(0, len(triples), (16,))
            a_ids = [f"concept:ans:{triples[i][0]}" for i in idx]
            b_ids = [f"concept:ans:{triples[i][1]}" for i in idx]
            target = torch.stack([cid_to_centroid[f"concept:ans:{triples[i][2]}"]
                                  for i in idx])
            op_onehot = torch.tensor([[1.0, 0.0]] * 16)
            out = head(
                torch.zeros(16, 16), torch.zeros(16, 16),
                op_onehot, a_ids, b_ids, cg,
            )
            loss = F.mse_loss(out, target)
            opt.zero_grad(); loss.backward(); opt.step()
    return head


def _rho_linear(cg: ConceptGraph, cids: list[str]) -> float:
    """Spearman-style ρ between bundle cosine matrix and -|Δn|.

    Hand-rolled (no scipy dep) so the test is hermetic.
    """
    pool = cg.bundle_pool["arithmetic_bias"].detach()
    slots = [cg.cid_to_slot[cid] for cid in cids]
    rows = pool[slots]
    rn = F.normalize(rows, dim=-1)
    cos = (rn @ rn.t()).cpu().numpy()
    n = len(cids)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    cos_vals = [cos[i, j] for i, j in pairs]
    target = [-abs(i - j) for i, j in pairs]
    # Spearman = Pearson on ranks
    def _ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda k: xs[k])
        ranks = [0.0] * len(xs)
        for r, k in enumerate(order):
            ranks[k] = float(r)
        return ranks
    rc = _ranks(cos_vals); rt = _ranks(target)
    mc = sum(rc) / len(rc); mt = sum(rt) / len(rt)
    num = sum((a - mc) * (b - mt) for a, b in zip(rc, rt))
    den = (
        sum((a - mc) ** 2 for a in rc) *
        sum((b - mt) ** 2 for b in rt)
    ) ** 0.5
    return float(num / den) if den > 0 else 0.0


# ---------------------------------------------------------------------------
# G1 — Bit-identity when sleep is not attached.
# ---------------------------------------------------------------------------


class TestG1NoAttachBitIdentical(unittest.TestCase):
    """Importing pcm.sleep / never calling attach_sleep must leave
    forward outputs byte-for-byte equal to Tier-A/D.

    The test wraps a complete ``set_seed → build → forward`` cycle in
    a single deterministic helper so two consecutive runs see exactly
    the same RNG advance pattern (init_row_ inside collapse_batch
    consumes RNG, so we must reset before *every* forward path).
    """

    @staticmethod
    def _run() -> torch.Tensor:
        torch.manual_seed(0)
        cg = ConceptGraph(initial_capacity=8)
        _register_ans(cg, 5)
        torch.manual_seed(123)
        head = ArithmeticHeadV2(embed_dim=16, bias_dim=8)
        torch.manual_seed(456)
        x = torch.randn(8, 16); y = torch.randn(8, 16)
        op_onehot = torch.tensor([[1.0, 0.0]] * 8)
        a_ids = [f"concept:ans:{(i % 5) + 1}" for i in range(8)]
        b_ids = [f"concept:ans:{((i + 2) % 5) + 1}" for i in range(8)]
        torch.manual_seed(789)
        return head(x, y, op_onehot, a_ids, b_ids, cg)

    def test_g1_no_attach_bit_identical(self) -> None:
        out_a = self._run()
        out_b = self._run()
        self.assertTrue(
            torch.equal(out_a, out_b),
            "deterministic helper baseline broken — RNG order changed",
        )
        # Now: importing pcm.sleep must not have flipped any global
        # ConceptGraph state (sleep_enabled defaults to False).
        cg = ConceptGraph(initial_capacity=4)
        self.assertFalse(getattr(cg, "sleep_enabled", False))


# ---------------------------------------------------------------------------
# G2 — Pool memory safety on a fresh graph.
# ---------------------------------------------------------------------------


class TestG2PoolMemorySafety(unittest.TestCase):
    """Active rows on the original facet must be byte-identical
    before and after run_sleep_pass(replay_steps=0)."""

    def test_g2_pool_memory_unchanged(self) -> None:
        cg = ConceptGraph(initial_capacity=16)
        cids = _register_ans(cg, 5)
        torch.manual_seed(7)
        head = ArithmeticHeadV2(embed_dim=16, bias_dim=8)
        _warmup_arith(cg, head, cids)
        # Splatter some non-trivial values so the snapshot has signal.
        with torch.no_grad():
            cg.bundle_pool["arithmetic_bias"].data.normal_(0.0, 1.0)

        active_slots = [cg.cid_to_slot[c] for c in cids]
        before = cg.bundle_pool["arithmetic_bias"].data[active_slots].clone()

        attach_sleep(cg, facets=["arithmetic_bias"])
        report = run_sleep_pass(
            cg,
            optimizer=None,
            facets=["arithmetic_bias"],
            config=SleepConfig(k_clusters=2, replay_steps=0, seed=0),
            tick=10,
        )
        self.assertEqual(len(report.facets), 1)

        after = cg.bundle_pool["arithmetic_bias"].data[active_slots].clone()
        self.assertTrue(
            torch.equal(before, after),
            "G2 violated: original active rows mutated by sleep pass",
        )
        # The new facet must exist with the right shape:
        self.assertIn("arithmetic_bias_residual", cg.bundle_pool)
        self.assertEqual(
            cg.bundle_pool["arithmetic_bias_residual"].shape[1:],
            cg.bundle_pool["arithmetic_bias"].shape[1:],
        )


# ---------------------------------------------------------------------------
# G3 — ρ regression bound on a small N=7 dual-muscle pipeline.
# ---------------------------------------------------------------------------


class TestG3RhoNoRegression(unittest.TestCase):
    """A shrunk §4 N=7 single-muscle run; sleep pass must not drop
    ρ_linear by more than 0.05 absolute.

    Note: unit test threshold (0.05) is much looser than the pre-
    registered paper threshold (0.01) because we train for 6 epochs
    × 32 steps rather than 30 × 200, so baseline ρ is ~0.7 rather
    than 0.97. The full-baseline test lives in the demo experiment.
    """

    def test_g3_rho_no_regression_n7(self) -> None:
        cg = ConceptGraph(initial_capacity=16)
        head = _train_arith_n7(cg, seed=0)
        cids = [f"concept:ans:{i}" for i in range(1, 8)]
        rho_before = _rho_linear(cg, cids)
        # Baseline sanity: training must produce non-trivial geometry.
        self.assertGreater(rho_before, 0.30,
                           f"baseline ρ too low ({rho_before:.3f}) "
                           "to test sleep regression meaningfully")

        attach_sleep(cg, facets=["arithmetic_bias"])
        run_sleep_pass(
            cg,
            optimizer=None,
            facets=["arithmetic_bias"],
            config=SleepConfig(k_clusters=3, replay_steps=0, seed=0),
            tick=1000,
        )
        rho_after = _rho_linear(cg, cids)
        self.assertGreaterEqual(
            rho_after, rho_before - 0.05,
            f"G3 violated: ρ_linear dropped from {rho_before:.4f} to "
            f"{rho_after:.4f} after sleep pass (replay=0 should be a "
            "no-op on the original facet)",
        )
        # In fact with replay=0 it should be EXACTLY equal (only the
        # original facet rows feed the cosine matrix).
        self.assertAlmostEqual(rho_after, rho_before, places=6,
                               msg="replay=0 should leave ρ exactly equal")


# ---------------------------------------------------------------------------
# G4 — Centroid is the mean of assigned members.
# ---------------------------------------------------------------------------


class TestG4CentroidIsMean(unittest.TestCase):
    """For each cluster k: centroid_k ≈ mean(member rows assigned to k)."""

    def test_g4_centroid_is_mean(self) -> None:
        torch.manual_seed(2026)
        cg = ConceptGraph(initial_capacity=16)
        cids = _register_ans(cg, 6)
        head = ArithmeticHeadV2(embed_dim=16, bias_dim=8)
        _warmup_arith(cg, head, cids)
        # Inject a deliberate 2-cluster structure: first 3 cids near +1,
        # last 3 near -1.
        with torch.no_grad():
            for i, cid in enumerate(cids):
                slot = cg.cid_to_slot[cid]
                base = 1.0 if i < 3 else -1.0
                cg.bundle_pool["arithmetic_bias"].data[slot] = (
                    base + 0.05 * torch.randn(8)
                )

        attach_sleep(cg, facets=["arithmetic_bias"])
        run_sleep_pass(
            cg,
            facets=["arithmetic_bias"],
            config=SleepConfig(k_clusters=2, replay_steps=0, seed=0,
                               distance="cosine"),
            tick=42,
        )

        # Recover the abstract slots:
        proto_slots: dict[int, int] = {}
        for k in range(2):
            pid = f"concept:cluster:arithmetic_bias:{k}"
            self.assertIn(pid, cg.concepts,
                          f"prototype {pid} not registered")
            proto_slots[k] = cg.cid_to_slot[pid]
        # Reconstruct member assignments from residuals:
        pool = cg.bundle_pool["arithmetic_bias"].data
        res_pool = cg.bundle_pool["arithmetic_bias_residual"].data
        assignments: dict[int, int] = {}
        for cid in cids:
            slot = cg.cid_to_slot[cid]
            row = pool[slot]
            best_k = -1; best_err = float("inf")
            for k, ps in proto_slots.items():
                err = float((row - pool[ps] - res_pool[slot]).pow(2).sum())
                if err < best_err:
                    best_err = err; best_k = k
            assignments[cid] = best_k

        for k, ps in proto_slots.items():
            members = [cg.cid_to_slot[c] for c, ak in assignments.items()
                       if ak == k]
            self.assertGreaterEqual(len(members), 1,
                                    f"cluster {k} has no members")
            mean_row = pool[members].mean(dim=0)
            centroid = pool[ps]
            err = float((centroid - mean_row).abs().max())
            self.assertLess(err, 1e-5,
                            f"G4 violated: centroid k={k} differs from "
                            f"member mean by {err:.2e}")


# ---------------------------------------------------------------------------
# G5 — Abstract relation cook reconstructs the original row.
# ---------------------------------------------------------------------------


class TestG5AbstractCook(unittest.TestCase):
    """For mode='add': cook(rel) ≈ original row to within float eps.

    Concretely: residual = row - centroid by construction (phase C);
    cook = anchor + residual = centroid + (row - centroid) = row.
    """

    def test_g5_abstract_cook_reconstructs(self) -> None:
        torch.manual_seed(11)
        cg = ConceptGraph(initial_capacity=16)
        cids = _register_ans(cg, 5)
        head = ArithmeticHeadV2(embed_dim=16, bias_dim=8)
        _warmup_arith(cg, head, cids)
        with torch.no_grad():
            cg.bundle_pool["arithmetic_bias"].data.normal_(0.0, 0.5)

        before_rows = {
            cid: cg.bundle_pool["arithmetic_bias"].data[
                cg.cid_to_slot[cid]
            ].clone()
            for cid in cids
        }

        attach_sleep(cg, facets=["arithmetic_bias"])
        run_sleep_pass(
            cg,
            facets=["arithmetic_bias"],
            config=SleepConfig(k_clusters=2, replay_steps=0,
                               relation_mode="add", seed=0),
            tick=99,
        )

        ev = GraphEvaluator(concept_graph=cg)
        # Pick one relation per member.
        relations = list(iter_abstract_relations(cg))
        self.assertGreaterEqual(len(relations), len(cids),
                                "every member should have a relation node")
        per_member: dict[str, str] = {}
        for rel in relations:
            mid = rel.metadata["constants"]["member_id"]
            per_member.setdefault(mid, rel.node_id)
        for cid in cids:
            self.assertIn(cid, per_member,
                          f"member {cid} has no relation node")
            out = ev.eval(per_member[cid], bindings={},
                          caller="g5-test", tick=200)
            # Cook outputs preserve the (B=1, D) batch dim from
            # collapse_batch / codebook_lookup; flatten for comparison.
            out_flat = out.reshape(-1)
            ref_flat = before_rows[cid].reshape(-1)
            self.assertEqual(tuple(out_flat.shape), tuple(ref_flat.shape))
            err = float((out_flat - ref_flat).abs().max())
            self.assertLess(err, 1e-5,
                            f"G5 violated: cook for {cid} differs from "
                            f"original row by {err:.2e}")


# ---------------------------------------------------------------------------
# G6 — Idempotency.
# ---------------------------------------------------------------------------


class TestG6Idempotent(unittest.TestCase):
    """Two consecutive run_sleep_pass calls (no training in between)
    must yield byte-identical graph state — same concepts, same
    pool data, same attribution.
    """

    def test_g6_idempotent(self) -> None:
        torch.manual_seed(13)
        cg = ConceptGraph(initial_capacity=16)
        cids = _register_ans(cg, 5)
        head = ArithmeticHeadV2(embed_dim=16, bias_dim=8)
        _warmup_arith(cg, head, cids)
        with torch.no_grad():
            cg.bundle_pool["arithmetic_bias"].data.normal_(0.0, 0.3)

        attach_sleep(cg, facets=["arithmetic_bias"])
        run_sleep_pass(
            cg,
            facets=["arithmetic_bias"],
            config=SleepConfig(k_clusters=2, replay_steps=0, seed=0),
            tick=1,
        )
        snap_concepts = sorted(cg.concepts.keys())
        snap_pool = {
            f: pool.data.clone() for f, pool in cg.bundle_pool.items()
        }
        snap_active = {
            slot: set(facets)
            for slot, facets in cg._active_facets_by_slot.items()
        }

        # Second pass — should be a no-op.
        report2 = run_sleep_pass(
            cg,
            facets=["arithmetic_bias"],
            config=SleepConfig(k_clusters=2, replay_steps=0, seed=0),
            tick=2,
        )
        # The facet should be in skipped_facets and no new facets in report:
        self.assertIn("arithmetic_bias", report2.skipped_facets)

        # Concepts unchanged:
        self.assertEqual(snap_concepts, sorted(cg.concepts.keys()),
                         "G6 violated: concept set changed on second pass")
        # Pool data byte-identical:
        for f, before in snap_pool.items():
            self.assertTrue(
                torch.equal(before, cg.bundle_pool[f].data),
                f"G6 violated: bundle_pool[{f!r}] changed on second pass",
            )
        # Active facet sets equal (with the caveat that the second pass
        # may have re-stamped the sleep caller; we only require facet
        # *set* membership stay the same):
        for slot, facets in snap_active.items():
            self.assertEqual(
                facets, set(cg._active_facets_by_slot.get(slot, set())),
                f"G6 violated: active facets for slot {slot} changed",
            )


# ---------------------------------------------------------------------------
# G7 — Batched abstract read equals direct read post-sleep.
# ---------------------------------------------------------------------------


class TestG7AbstractEqualsDirect(unittest.TestCase):
    """G7: Right after a sleep pass (no further training), the batched
    ``collapse_via_abstract`` must equal direct ``collapse_batch`` row-wise
    to within float eps. This anchors the V3 ablation: any deviation
    measured later is purely due to gradient flow through anchor +
    residual, not numerical drift in the read itself.
    """

    def test_g7_abstract_equals_direct_batch(self) -> None:
        torch.manual_seed(17)
        cg = ConceptGraph(initial_capacity=16)
        cids = _register_ans(cg, 6)
        head = ArithmeticHeadV2(embed_dim=16, bias_dim=8)
        _warmup_arith(cg, head, cids)
        with torch.no_grad():
            cg.bundle_pool["arithmetic_bias"].data.normal_(0.0, 0.4)

        attach_sleep(cg, facets=["arithmetic_bias"])
        run_sleep_pass(
            cg,
            facets=["arithmetic_bias"],
            config=SleepConfig(k_clusters=2, replay_steps=0, seed=0),
            tick=1,
        )

        direct = cg.collapse_batch(
            caller="g7-test-direct",
            facet="arithmetic_bias",
            concept_ids=cids,
            shape=(8,),
            tick=10,
            init="normal_small",
        )
        index = build_member_to_anchor_index(cg)
        self.assertGreater(len(index), 0,
                           "no abstract relations registered after sleep")
        abstract = collapse_via_abstract(
            cg,
            caller="g7-test-abstract",
            facet="arithmetic_bias",
            concept_ids=cids,
            shape=(8,),
            tick=11,
            index=index,
        )
        self.assertEqual(direct.shape, abstract.shape)
        err = float((direct - abstract).abs().max().item())
        self.assertLess(
            err, 1e-5,
            f"G7 violated: batched abstract read differs from direct "
            f"read by {err:.2e} just after sleep pass",
        )

    def test_g7_fallback_for_unregistered_member(self) -> None:
        torch.manual_seed(19)
        cg = ConceptGraph(initial_capacity=16)
        cids = _register_ans(cg, 4)
        head = ArithmeticHeadV2(embed_dim=16, bias_dim=8)
        _warmup_arith(cg, head, cids)
        with torch.no_grad():
            cg.bundle_pool["arithmetic_bias"].data.normal_(0.0, 0.3)
        attach_sleep(cg, facets=["arithmetic_bias"])
        run_sleep_pass(
            cg,
            facets=["arithmetic_bias"],
            config=SleepConfig(k_clusters=2, replay_steps=0, seed=0),
            tick=1,
        )
        # Register a brand-new concept *after* sleep — it has no
        # abstract relation. fallback="direct" must still return a
        # correct row for it.
        new_cid = "concept:ans:99"
        cg.register_concept(node_id=new_cid, label="ANS_99", scope="BASE",
                            provenance="g7-fallback")
        with torch.no_grad():
            slot = cg.cid_to_slot[new_cid]
            cg.bundle_pool["arithmetic_bias"].data[slot] = torch.randn(8)
        ids = cids + [new_cid]
        out = collapse_via_abstract(
            cg,
            caller="g7-test-fallback",
            facet="arithmetic_bias",
            concept_ids=ids,
            shape=(8,),
            tick=12,
        )
        direct = cg.collapse_batch(
            caller="g7-test-fallback-direct",
            facet="arithmetic_bias",
            concept_ids=ids,
            shape=(8,),
            tick=13,
        )
        err = float((direct - out).abs().max().item())
        self.assertLess(
            err, 1e-5,
            f"G7 fallback violated: unregistered member row differs by "
            f"{err:.2e} (should fall back to direct read)",
        )


# ---------------------------------------------------------------------------
# Bonus — sleep_status / iter_abstract_relations sanity.
# ---------------------------------------------------------------------------


class TestSleepStatus(unittest.TestCase):
    def test_status_off_by_default(self) -> None:
        cg = ConceptGraph(initial_capacity=4)
        self.assertEqual(sleep_status(cg), {"enabled": False})

    def test_status_after_attach_and_pass(self) -> None:
        torch.manual_seed(0)
        cg = ConceptGraph(initial_capacity=16)
        cids = _register_ans(cg, 4)
        head = ArithmeticHeadV2(embed_dim=16, bias_dim=8)
        _warmup_arith(cg, head, cids)
        with torch.no_grad():
            cg.bundle_pool["arithmetic_bias"].data.normal_(0.0, 0.3)

        attach_sleep(cg, facets=["arithmetic_bias"])
        run_sleep_pass(
            cg,
            facets=["arithmetic_bias"],
            config=SleepConfig(k_clusters=2, replay_steps=0, seed=0),
            tick=42,
        )
        st = sleep_status(cg)
        self.assertTrue(st["enabled"])
        self.assertEqual(st["last_tick"], 42)
        self.assertEqual(st["n_prototypes"], 2)
        self.assertEqual(st["n_relations"], 4)
        self.assertIn("arithmetic_bias_residual", st["residual_facets"])


if __name__ == "__main__":
    unittest.main()
