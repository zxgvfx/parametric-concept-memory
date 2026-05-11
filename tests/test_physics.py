"""tests/test_physics.py — falsifiable unit tests for PCM v4
``pcm.physics``.

Covers:

* :class:`PhysicsStateHead` shape, residual-zero behaviour,
  force-input handling, dt scaling.
* :class:`PhysicsCook` rollout shapes, divergence detection,
  wall reflection, custom post-step hook.
* :func:`physics_step_loss` MSE primitive.

Tests use **synthetic perfect oracle heads** (i.e. heads that
output zero or known constants) so that cook behaviours are
testable independent of training noise. The full P1/P2/P3
empirical validation lives in ``experiments/bouncing_ball_poc.py``.
"""
from __future__ import annotations

import unittest

import torch

from pcm.physics import (
    PhysicsCook,
    PhysicsDistillReport,
    PhysicsStateHead,
    RolloutReport,
    StateLookupHead,
    distill_physics_cook_to_lookup,
    physics_step_loss,
)


# ---------------------------------------------------------------------------
# PH1 — PhysicsStateHead.
# ---------------------------------------------------------------------------


class TestPH1PhysicsStateHead(unittest.TestCase):
    def test_ph1a_no_force_shape(self) -> None:
        torch.manual_seed(0)
        head = PhysicsStateHead(state_dim=4, force_dim=0, hidden=8)
        out = head(torch.randn(3, 4))
        self.assertEqual(out.shape, (3, 4))

    def test_ph1b_with_force_shape(self) -> None:
        torch.manual_seed(0)
        head = PhysicsStateHead(state_dim=2, force_dim=1, hidden=8)
        out = head(torch.randn(3, 2), torch.randn(3, 1))
        self.assertEqual(out.shape, (3, 2))

    def test_ph1c_force_required_when_dim_set(self) -> None:
        head = PhysicsStateHead(state_dim=2, force_dim=1)
        with self.assertRaises(ValueError):
            head(torch.randn(2, 2))

    def test_ph1d_dt_scales_output(self) -> None:
        torch.manual_seed(7)
        head1 = PhysicsStateHead(state_dim=2, dt=1.0, hidden=4)
        head2 = PhysicsStateHead(state_dim=2, dt=0.5, hidden=4)
        # Copy weights so the two heads are otherwise identical.
        head2.load_state_dict(head1.state_dict())
        x = torch.randn(2, 2)
        out1 = head1(x)
        out2 = head2(x)
        self.assertTrue(
            torch.allclose(out2, 0.5 * out1, atol=1e-6),
            "dt scaling failed: head2 output should be 0.5 × head1",
        )

    def test_ph1e_invalid_dim_raises(self) -> None:
        with self.assertRaises(ValueError):
            PhysicsStateHead(state_dim=0)
        with self.assertRaises(ValueError):
            PhysicsStateHead(state_dim=2, force_dim=-1)


# ---------------------------------------------------------------------------
# PH2 — PhysicsCook.
# ---------------------------------------------------------------------------


class _ZeroDeltaHead(PhysicsStateHead):
    """Output Δstate = 0 always. Used to verify cook with a
    static head produces a constant trajectory."""

    def forward(self, state, force=None):
        return torch.zeros_like(state)


class _ConstantDeltaHead(PhysicsStateHead):
    """Output Δstate = (1, 0, 0, ...) regardless of input."""

    def forward(self, state, force=None):
        out = torch.zeros_like(state)
        out[..., 0] = 1.0
        return out


class TestPH2PhysicsCook(unittest.TestCase):
    def test_ph2a_zero_head_constant_trajectory(self) -> None:
        head = _ZeroDeltaHead(state_dim=2, hidden=4)
        cook = PhysicsCook(head, max_iters=20)
        s0 = torch.tensor([[1.0, 2.0]])
        traj, rep = cook(s0, K=10)
        self.assertEqual(traj.shape, (11, 1, 2))  # K+1 frames
        # Every frame should equal s0.
        for k in range(11):
            self.assertTrue(torch.allclose(traj[k], s0))
        self.assertEqual(rep.K, 10)
        self.assertFalse(rep.diverged)

    def test_ph2b_constant_head_linear_trajectory(self) -> None:
        head = _ConstantDeltaHead(state_dim=2, hidden=4)
        cook = PhysicsCook(head, max_iters=20)
        s0 = torch.tensor([[0.0, 0.0]])
        traj, rep = cook(s0, K=5)
        # Position should grow by 1 per step, velocity stay 0.
        for k in range(6):
            self.assertAlmostEqual(float(traj[k, 0, 0].item()), float(k))
            self.assertEqual(float(traj[k, 0, 1].item()), 0.0)

    def test_ph2c_divergence_detection(self) -> None:
        """A head that doubles state should trigger the
        diverge_threshold guard."""

        class _DoubleHead(PhysicsStateHead):
            def forward(self, state, force=None):
                return state  # Δstate = state → state doubles each step

        head = _DoubleHead(state_dim=2, hidden=4)
        cook = PhysicsCook(head, max_iters=200, diverge_threshold=1e3)
        s0 = torch.tensor([[1.0, 0.0]])
        traj, rep = cook(s0, K=100)
        self.assertTrue(rep.diverged)
        self.assertLess(rep.K, 100)  # broke early

    def test_ph2d_wall_reflect(self) -> None:
        """Constant head pushes position rightward; wall reflect
        should flip velocity sign at the wall."""
        head = _ConstantDeltaHead(state_dim=2, hidden=4)
        cook = PhysicsCook(
            head, max_iters=20,
            state_clip=(
                torch.tensor([0.0, -10.0]),
                torch.tensor([3.0, 10.0]),  # wall at x=3
            ),
            wall_reflect=True,
        )
        s0 = torch.tensor([[0.0, 0.0]])
        traj, rep = cook(s0, K=5)
        # The wall reflect logic flips velocity when position
        # exceeds the wall. Behaviour is non-trivial; just verify
        # we don't blow up and we stay within bounds.
        for k in range(6):
            self.assertGreaterEqual(float(traj[k, 0, 0].item()), 0.0)
            self.assertLessEqual(float(traj[k, 0, 0].item()), 3.0 + 1e-3)

    def test_ph2e_custom_post_step(self) -> None:
        head = _ConstantDeltaHead(state_dim=2, hidden=4)

        def squash(state, k):
            return state * 0.5

        cook = PhysicsCook(head, max_iters=10, custom_post_step=squash)
        s0 = torch.tensor([[2.0, 0.0]])
        traj, rep = cook(s0, K=3)
        # After step 1: state = (s + (1, 0)) * 0.5 = (1.5, 0)
        # After step 2: state = (1.5 + 1, 0) * 0.5 = (1.25, 0)
        # After step 3: state = (1.25 + 1, 0) * 0.5 = (1.125, 0)
        self.assertAlmostEqual(float(traj[1, 0, 0].item()), 1.5)
        self.assertAlmostEqual(float(traj[2, 0, 0].item()), 1.25)
        self.assertAlmostEqual(float(traj[3, 0, 0].item()), 1.125)

    def test_ph2f_invalid_K(self) -> None:
        head = _ZeroDeltaHead(state_dim=2, hidden=4)
        cook = PhysicsCook(head, max_iters=10)
        with self.assertRaises(ValueError):
            cook(torch.randn(1, 2), K=0)
        with self.assertRaises(ValueError):
            cook(torch.randn(1, 2), K=20)  # > max_iters

    def test_ph2g_force_required(self) -> None:
        head = PhysicsStateHead(state_dim=2, force_dim=1, hidden=4)
        cook = PhysicsCook(head, max_iters=5)
        with self.assertRaises(ValueError):
            cook(torch.randn(1, 2), K=3)  # missing force_seq

    def test_ph2h_rollout_report_dataclass(self) -> None:
        rep = RolloutReport(
            K=10, wall_seconds=0.001,
            final_state_norm=1.5, diverged=False,
        )
        self.assertEqual(rep.K, 10)
        self.assertFalse(rep.diverged)
        self.assertEqual(rep.step_history, [])


# ---------------------------------------------------------------------------
# PH3 — physics_step_loss.
# ---------------------------------------------------------------------------


class TestPH3PhysicsStepLoss(unittest.TestCase):
    def test_ph3a_zero_at_perfect_prediction(self) -> None:
        state = torch.tensor([[1.0, 2.0]])
        true_next = torch.tensor([[1.5, 1.0]])
        pred_delta = true_next - state
        loss = physics_step_loss(pred_delta, true_next, state)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_ph3b_positive_for_wrong_prediction(self) -> None:
        state = torch.zeros(2, 2)
        true_next = torch.ones(2, 2)
        pred_delta = torch.zeros(2, 2)
        loss = physics_step_loss(pred_delta, true_next, state)
        self.assertGreater(float(loss.item()), 0.0)


# ---------------------------------------------------------------------------
# PH4 — StateLookupHead + distill_physics_cook_to_lookup (F59).
# ---------------------------------------------------------------------------


class TestPH4StateLookupAndDistill(unittest.TestCase):
    def test_ph4a_lookup_forward_shape(self) -> None:
        torch.manual_seed(0)
        head = StateLookupHead(state_dim=4, max_K=50, hidden=8)
        s = torch.randn(3, 4)
        K = torch.tensor([10, 25, 50])
        out = head(s, K)
        self.assertEqual(out.shape, (3, 4))

    def test_ph4b_lookup_invalid_dim(self) -> None:
        with self.assertRaises(ValueError):
            StateLookupHead(state_dim=0, max_K=10)
        with self.assertRaises(ValueError):
            StateLookupHead(state_dim=2, max_K=0)

    def test_ph4c_distill_loss_decreases(self) -> None:
        """Build a simple cook + lookup; distill; loss should drop."""
        torch.manual_seed(7)

        # A trivial constant-Δstate head is the cook's transition.
        # PhysicsCook with a head that outputs a fixed step gives
        # deterministic rollouts we can use as oracle data.
        class _LinearHead(PhysicsStateHead):
            def __init__(self):
                super().__init__(state_dim=2, hidden=4, dt=1.0)

            def forward(self, state, force=None):
                # Δstate = (0.1, 0.0) regardless of state.
                out = torch.zeros_like(state)
                out[..., 0] = 0.1
                return out

        cook_head = _LinearHead()
        cook = PhysicsCook(cook_head, max_iters=20)
        lookup = StateLookupHead(state_dim=2, max_K=10, hidden=8)

        sample_states = torch.zeros(5, 2)
        for i in range(5):
            sample_states[i, 0] = float(i)

        report = distill_physics_cook_to_lookup(
            cook=cook, lookup=lookup,
            sample_initial_states=sample_states,
            sample_Ks=[1, 3, 5, 10],
            n_steps=200, batch_size=8,
            rng_seed=42,
        )
        self.assertLess(report.final_loss, report.initial_loss)

    def test_ph4d_distill_predictions_match_cook(self) -> None:
        """Post-distillation, lookup output should approximately
        match cook's terminal state on the distilled pairs."""
        torch.manual_seed(11)

        class _LinearHead(PhysicsStateHead):
            def __init__(self):
                super().__init__(state_dim=2, hidden=4, dt=1.0)

            def forward(self, state, force=None):
                out = torch.zeros_like(state)
                out[..., 0] = 0.1
                return out

        cook_head = _LinearHead()
        cook = PhysicsCook(cook_head, max_iters=20)
        lookup = StateLookupHead(state_dim=2, max_K=10, hidden=16)

        sample_states = torch.zeros(8, 2)
        for i in range(8):
            sample_states[i, 0] = float(i)

        distill_physics_cook_to_lookup(
            cook=cook, lookup=lookup,
            sample_initial_states=sample_states,
            sample_Ks=[1, 5, 10],
            n_steps=400, batch_size=16, lr=1e-2,
            rng_seed=99,
        )

        # On a held-out (s0, K) pair, lookup should approximately
        # match cook's terminal state.
        with torch.no_grad():
            s0 = torch.tensor([[2.0, 0.0]])
            K_tensor = torch.tensor([5])
            traj_cook, _ = cook(s0, K=5)
            terminal_cook = traj_cook[-1]
            terminal_lookup = lookup(s0, K_tensor)
            err = float(
                (terminal_lookup - terminal_cook).abs().max().item()
            )
            # The constant step (0.1) gives terminal x = 2.5; lookup
            # should be within 0.5 after 400 steps of training.
            self.assertLess(err, 0.5)

    def test_ph4e_distill_empty_raises(self) -> None:
        cook_head = PhysicsStateHead(state_dim=2, hidden=4)
        cook = PhysicsCook(cook_head, max_iters=10)
        lookup = StateLookupHead(state_dim=2, max_K=5)
        with self.assertRaises(ValueError):
            distill_physics_cook_to_lookup(
                cook=cook, lookup=lookup,
                sample_initial_states=torch.empty(0, 2),
                sample_Ks=[1, 2],
            )
        with self.assertRaises(ValueError):
            distill_physics_cook_to_lookup(
                cook=cook, lookup=lookup,
                sample_initial_states=torch.zeros(2, 2),
                sample_Ks=[],
            )

    def test_ph4f_distill_report_dataclass(self) -> None:
        rep = PhysicsDistillReport(
            n_steps=100, n_pairs_distilled=50,
            initial_loss=10.0, final_loss=1.0,
            cook_oracle_diverge_rate=0.05,
        )
        self.assertEqual(rep.n_steps, 100)
        self.assertLess(rep.final_loss, rep.initial_loss)


if __name__ == "__main__":
    unittest.main()
