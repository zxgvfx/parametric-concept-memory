"""tests/test_dual_process.py — falsifiable unit tests for the
PCM v3 dual-process module ``pcm.dual_process``.

Covers:

* :class:`SuccessorHead` — forward shapes, slot-only and slot+attr
  variants, predict_step semantics, max_step validation.
* :class:`IterativeDiffCook` — convergence on a perfect-head
  oracle, non-convergence + report on stuck head, with_attr
  routing.
* :func:`route_diff` — System-1 / System-2 dispatch on a synthetic
  threshold; fallback when one path is missing.

Tests use **synthetic perfect-head oracles** (i.e. heads constructed
to behave as if E1 = 1.0) so that the cook's iteration-based error-
correction property is testable in isolation from training noise.
The full E1–E5 empirical validation lives in
``experiments/number_dual_process_poc.py``.
"""
from __future__ import annotations

import unittest

import torch

from pcm.dual_process import (
    DiffCookReport,
    DistillReport,
    IterativeDiffCook,
    SuccessorHead,
    calibrate_rpe_coverage,
    distill_cook_to_rpe,
    route_diff,
)


# ---------------------------------------------------------------------------
# Synthetic perfect-sign head — makes cook tests deterministic.
# ---------------------------------------------------------------------------


class _PerfectSignHead(SuccessorHead):
    """SuccessorHead override: ``predict_step`` returns the actual
    sign of the integer cursor encoded in the slot tensor's first
    coord. Uses no learned parameters but inherits the correct
    interface so :class:`IterativeDiffCook` can drive it."""

    def __init__(self, slot_dim: int = 4, max_step: int = 1) -> None:
        super().__init__(slot_dim=slot_dim, max_step=max_step, hidden=8)

    def predict_step(
        self,
        slot_a: torch.Tensor, slot_b: torch.Tensor,
        attr_a: torch.Tensor | None = None,
        attr_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        a = slot_a[..., 0]
        b = slot_b[..., 0]
        diff = b - a
        step = torch.sign(diff).to(torch.long)
        return step.clamp(-self.max_step, self.max_step)


# ---------------------------------------------------------------------------
# DP1 — SuccessorHead.
# ---------------------------------------------------------------------------


class TestDP1SuccessorHead(unittest.TestCase):
    def test_dp1a_slot_only_forward(self) -> None:
        torch.manual_seed(0)
        head = SuccessorHead(slot_dim=8, max_step=1, hidden=16)
        sa = torch.randn(4, 8)
        sb = torch.randn(4, 8)
        out = head(sa, sb)
        self.assertEqual(out.shape, (4, 3))  # 2*1 + 1 classes

    def test_dp1b_slot_attr_forward(self) -> None:
        torch.manual_seed(0)
        head = SuccessorHead(slot_dim=8, attr_dim=4, max_step=2)
        sa = torch.randn(3, 8); sb = torch.randn(3, 8)
        aa = torch.randn(3, 4); ab = torch.randn(3, 4)
        out = head(sa, sb, aa, ab)
        self.assertEqual(out.shape, (3, 5))  # 2*2 + 1 classes

    def test_dp1c_attr_required_when_attr_dim_set(self) -> None:
        head = SuccessorHead(slot_dim=4, attr_dim=4, max_step=1)
        sa = torch.randn(2, 4); sb = torch.randn(2, 4)
        with self.assertRaises(ValueError):
            head(sa, sb)  # missing attr_a / attr_b

    def test_dp1d_predict_step_range(self) -> None:
        torch.manual_seed(1)
        head = SuccessorHead(slot_dim=4, max_step=2, hidden=8)
        sa = torch.randn(10, 4); sb = torch.randn(10, 4)
        steps = head.predict_step(sa, sb)
        self.assertTrue((steps >= -2).all())
        self.assertTrue((steps <= 2).all())

    def test_dp1e_max_step_validation(self) -> None:
        with self.assertRaises(ValueError):
            SuccessorHead(slot_dim=4, max_step=0)
        with self.assertRaises(ValueError):
            SuccessorHead(slot_dim=4, max_step=-1)


# ---------------------------------------------------------------------------
# DP2 — IterativeDiffCook on perfect-head oracle.
# ---------------------------------------------------------------------------


def _ordinal_lookup(n_total: int):
    """Return a closure ``idx -> slot row`` where the slot row's
    first coord is the integer index. Lets the perfect head read
    cursor / target positions from the slot directly."""
    table = torch.zeros(n_total, 4)
    for i in range(n_total):
        table[i, 0] = float(i)
    return lambda idx: table[idx]


def _ordinal_dual_lookup(n_total: int):
    """Same as _ordinal_lookup but returns (slot, attr) tuples."""
    slot_table = torch.zeros(n_total, 4)
    attr_table = torch.zeros(n_total, 2)
    for i in range(n_total):
        slot_table[i, 0] = float(i)
        attr_table[i, 0] = float(i) * 0.1  # dummy monotone attr
    return lambda idx: (slot_table[idx], attr_table[idx])


class TestDP2IterativeDiffCook(unittest.TestCase):
    def test_dp2a_perfect_head_converges_short(self) -> None:
        N = 20
        head = _PerfectSignHead(slot_dim=4, max_step=1)
        cook = IterativeDiffCook(
            successor_head=head,
            identity_lookup=_ordinal_lookup(N),
            max_iters=50, cursor_min=0, cursor_max=N - 1,
        )
        diff, rep = cook(start_idx=3, target_idx=10)
        self.assertEqual(diff, 7)
        self.assertTrue(rep.converged)
        self.assertEqual(rep.n_iters, 7)

    def test_dp2b_perfect_head_converges_long_extrapolation(self) -> None:
        """The signature E3 case: head trained on |Δ|=1, cook
        bridges |Δ|=99."""
        N = 100
        head = _PerfectSignHead(slot_dim=4, max_step=1)
        cook = IterativeDiffCook(
            successor_head=head,
            identity_lookup=_ordinal_lookup(N),
            max_iters=200, cursor_min=0, cursor_max=N - 1,
        )
        diff, rep = cook(start_idx=0, target_idx=99)
        self.assertEqual(diff, 99)
        self.assertTrue(rep.converged)
        self.assertEqual(rep.n_iters, 99)

    def test_dp2c_perfect_head_converges_negative(self) -> None:
        N = 50
        head = _PerfectSignHead(slot_dim=4, max_step=1)
        cook = IterativeDiffCook(
            successor_head=head,
            identity_lookup=_ordinal_lookup(N),
            max_iters=100, cursor_min=0, cursor_max=N - 1,
        )
        diff, rep = cook(start_idx=40, target_idx=8)
        self.assertEqual(diff, -32)
        self.assertTrue(rep.converged)

    def test_dp2d_with_attr_routing(self) -> None:
        N = 20

        class _PerfectSignAttrHead(SuccessorHead):
            def __init__(self) -> None:
                super().__init__(slot_dim=4, attr_dim=2, max_step=1)

            def predict_step(self, sa, sb, aa=None, ab=None):
                # Use the slot index, just as the slot-only oracle.
                return torch.sign(sb[..., 0] - sa[..., 0]).to(torch.long)

        head = _PerfectSignAttrHead()
        cook = IterativeDiffCook(
            successor_head=head,
            identity_lookup=_ordinal_dual_lookup(N),
            max_iters=50, cursor_min=0, cursor_max=N - 1,
            with_attr=True,
        )
        diff, rep = cook(start_idx=2, target_idx=15)
        self.assertEqual(diff, 13)
        self.assertTrue(rep.converged)

    def test_dp2e_max_iters_cap(self) -> None:
        """Stuck cook (head always predicts 0) reports
        non-convergence cleanly."""
        N = 10

        class _AlwaysZero(SuccessorHead):
            def __init__(self) -> None:
                super().__init__(slot_dim=4, max_step=1)

            def predict_step(self, sa, sb, aa=None, ab=None):
                return torch.zeros(sa.shape[0], dtype=torch.long)

        cook = IterativeDiffCook(
            successor_head=_AlwaysZero(),
            identity_lookup=_ordinal_lookup(N),
            max_iters=5, cursor_min=0, cursor_max=N - 1,
        )
        diff, rep = cook(start_idx=0, target_idx=5)
        self.assertEqual(diff, 0)
        self.assertFalse(rep.converged)
        self.assertEqual(rep.final_distance, 5)
        # Should break on first zero step, not iterate to max.
        self.assertEqual(rep.n_iters, 1)

    def test_dp2f_diff_cook_report_dataclass(self) -> None:
        rep = DiffCookReport(
            n_iters=5, converged=True, final_distance=0,
            wall_seconds=0.001,
        )
        self.assertEqual(rep.n_iters, 5)
        self.assertTrue(rep.converged)
        self.assertEqual(rep.step_history, [])  # default factory


# ---------------------------------------------------------------------------
# DP3 — route_diff dispatcher.
# ---------------------------------------------------------------------------


class TestDP3RouteDiff(unittest.TestCase):
    def setUp(self) -> None:
        self.N = 30
        head = _PerfectSignHead(slot_dim=4, max_step=1)
        self.cook = IterativeDiffCook(
            successor_head=head,
            identity_lookup=_ordinal_lookup(self.N),
            max_iters=50, cursor_min=0, cursor_max=self.N - 1,
        )
        # Synthetic "perfect" RPE that just returns b - a.
        self.rpe = lambda a, b: b - a

    def test_dp3a_in_range_routes_to_rpe(self) -> None:
        diff, route = route_diff(
            5, 12, rpe_predict=self.rpe, cook=self.cook,
            train_max_abs_delta=10,
        )
        self.assertEqual(diff, 7)
        self.assertEqual(route, "rpe")

    def test_dp3b_out_of_range_routes_to_cook(self) -> None:
        diff, route = route_diff(
            2, 25, rpe_predict=self.rpe, cook=self.cook,
            train_max_abs_delta=10,
        )
        self.assertEqual(diff, 23)
        self.assertEqual(route, "cook")

    def test_dp3c_no_cook_falls_back_to_rpe(self) -> None:
        diff, route = route_diff(
            0, 25, rpe_predict=self.rpe, cook=None,
            train_max_abs_delta=10,
        )
        self.assertEqual(diff, 25)
        self.assertEqual(route, "rpe")  # falls back

    def test_dp3d_no_rpe_forces_cook(self) -> None:
        diff, route = route_diff(
            0, 8, rpe_predict=None, cook=self.cook,
            train_max_abs_delta=10,
        )
        self.assertEqual(diff, 8)
        self.assertEqual(route, "cook")  # forced

    def test_dp3e_neither_raises(self) -> None:
        with self.assertRaises(ValueError):
            route_diff(0, 1, rpe_predict=None, cook=None,
                       train_max_abs_delta=10)

    def test_dp3f_coarse_delta_supplied(self) -> None:
        # If we already know |Δ|, router doesn't need to recompute.
        diff, route = route_diff(
            5, 5, rpe_predict=self.rpe, cook=self.cook,
            train_max_abs_delta=10, coarse_delta=15,
        )
        # 5 == 5 in cook-land, but coarse_delta=15 routes to cook
        # which then sees b == a so converges with diff=0.
        self.assertEqual(route, "cook")
        self.assertEqual(diff, 0)


# ---------------------------------------------------------------------------
# DP4 — distill_cook_to_rpe (E4 sleep cache).
# ---------------------------------------------------------------------------


class TestDP4DistillCookToRPE(unittest.TestCase):
    """E4 sleep cache: cook predictions distilled into RPE table.

    Uses a tiny synthetic RPE head (single linear layer) and a
    perfect-sign cook to test the distillation primitive in
    isolation from the full v3 training pipeline. The full
    end-to-end E4 validation is in
    ``experiments/number_dual_process_sleep_poc.py``.
    """

    def _make_perfect_cook(self, n: int) -> IterativeDiffCook:
        head = _PerfectSignHead(slot_dim=4, max_step=1)
        return IterativeDiffCook(
            successor_head=head,
            identity_lookup=_ordinal_lookup(n),
            max_iters=2 * n, cursor_min=0, cursor_max=n - 1,
        )

    def test_dp4a_loss_decreases(self) -> None:
        """RPE distillation loss should decrease over training steps."""
        torch.manual_seed(0)
        N = 30
        cook = self._make_perfect_cook(N)
        n_classes = 2 * N - 1
        rpe = torch.nn.Linear(1, n_classes)

        def _step_fn(deltas: torch.Tensor) -> torch.Tensor:
            return rpe(deltas.float().unsqueeze(-1))

        sample_pairs = [(0, K) for K in range(5, N)]
        report = distill_cook_to_rpe(
            cook=cook,
            rpe_step_fn=_step_fn,
            rpe_parameters=list(rpe.parameters()),
            sample_pairs=sample_pairs,
            n_steps=200, batch_size=16,
            delta_to_idx=lambda d, _N=N: d + (_N - 1),
        )
        self.assertLess(report.final_loss, report.initial_loss)
        self.assertGreater(report.cook_oracle_acc, 0.95)

    def test_dp4b_rpe_learns_distilled_targets(self) -> None:
        """After distillation, RPE.argmax should match cook on
        the distilled pairs. Uses a proper embedding-table RPE
        (lookup by integer delta + linear classifier) — the same
        architecture pattern as the production v3 head."""
        torch.manual_seed(1)
        N = 20
        cook = self._make_perfect_cook(N)
        n_classes = 2 * N - 1
        # Embedding(2*N-1, 8) → Linear(8, n_classes); the embedding
        # is keyed on the shifted delta index so each delta can
        # learn its own classifier output.
        embed = torch.nn.Embedding(2 * N - 1, 8)
        clf = torch.nn.Linear(8, n_classes)

        def _step_fn(deltas: torch.Tensor) -> torch.Tensor:
            idx = deltas + (N - 1)
            return clf(embed(idx))

        params = list(embed.parameters()) + list(clf.parameters())
        sample_pairs = [(a, b) for a in range(N) for b in range(N)
                         if a != b][:80]
        distill_cook_to_rpe(
            cook=cook,
            rpe_step_fn=_step_fn,
            rpe_parameters=params,
            sample_pairs=sample_pairs,
            n_steps=400, batch_size=16,
            lr=1e-2,
            delta_to_idx=lambda d, _N=N: d + (_N - 1),
        )
        # Check RPE accuracy on the same distilled pairs.
        with torch.no_grad():
            hits = 0
            for a, b in sample_pairs:
                d = torch.tensor([b - a], dtype=torch.long)
                pred_class = int(_step_fn(d).argmax(-1).item())
                pred_diff = pred_class - (N - 1)
                if pred_diff == b - a:
                    hits += 1
            self.assertGreater(
                hits / len(sample_pairs), 0.7,
                "RPE failed to learn cook's predictions on distilled pairs",
            )

    def test_dp4c_empty_pairs_raises(self) -> None:
        torch.manual_seed(0)
        cook = self._make_perfect_cook(10)
        rpe = torch.nn.Linear(1, 19)

        def _step_fn(d):
            return rpe(d.float().unsqueeze(-1))

        with self.assertRaises(ValueError):
            distill_cook_to_rpe(
                cook=cook,
                rpe_step_fn=_step_fn,
                rpe_parameters=list(rpe.parameters()),
                sample_pairs=[],
                n_steps=10,
            )

    def test_dp4d_distill_report_dataclass(self) -> None:
        rep = DistillReport(
            n_steps=10, n_pairs_distilled=100,
            final_loss=0.5, initial_loss=2.0, cook_oracle_acc=0.95,
        )
        self.assertEqual(rep.n_steps, 10)
        self.assertLess(rep.final_loss, rep.initial_loss)


# ---------------------------------------------------------------------------
# DP5 — calibrate_rpe_coverage (F54 adaptive routing).
# ---------------------------------------------------------------------------


class TestDP5CalibrateRPECoverage(unittest.TestCase):
    """F54 adaptive routing: calibrate_rpe_coverage finds the
    largest K at which a synthetic RPE matches ground truth."""

    def test_dp5a_perfect_rpe_full_range(self) -> None:
        """A perfect RPE that always returns b - a should
        calibrate to the maximum possible K (n_total - 1)."""
        N = 50
        rpe = lambda a, b: b - a
        K = calibrate_rpe_coverage(
            rpe, n_total=N, threshold=0.95, sample_size=20,
        )
        self.assertEqual(K, N - 1)

    def test_dp5b_bounded_rpe_caps_at_train_max(self) -> None:
        """An RPE that only works on |Δ| ≤ 10 (returns random
        outside) calibrates to ~10."""
        train_max = 10

        def bounded_rpe(a: int, b: int) -> int:
            d = b - a
            if abs(d) <= train_max:
                return d
            return d + 1  # systematically wrong outside

        K = calibrate_rpe_coverage(
            bounded_rpe, n_total=30, threshold=0.95, sample_size=20,
        )
        self.assertLessEqual(K, train_max)
        self.assertGreaterEqual(K, train_max - 1)

    def test_dp5c_worthless_rpe_returns_zero(self) -> None:
        """An RPE that always returns the wrong answer calibrates
        to 0 (route every query elsewhere)."""
        bad_rpe = lambda a, b: b - a + 1
        K = calibrate_rpe_coverage(
            bad_rpe, n_total=20, threshold=0.95, sample_size=10,
        )
        self.assertEqual(K, 0)

    def test_dp5d_returns_largest_with_competence_islands(self) -> None:
        """If RPE is correct on K ∈ [1..5] AND [10..15] but wrong
        in between, the calibrator returns the LARGEST passing K
        (15), not the first failure point (6)."""

        def islanded_rpe(a: int, b: int) -> int:
            d = b - a
            ad = abs(d)
            if ad <= 5 or 10 <= ad <= 15:
                return d
            return d + 1  # wrong on (5, 10) and >15

        K = calibrate_rpe_coverage(
            islanded_rpe, n_total=25, threshold=0.95, sample_size=20,
        )
        self.assertGreaterEqual(K, 14)  # at least near 15
        self.assertLessEqual(K, 16)


if __name__ == "__main__":
    unittest.main()
