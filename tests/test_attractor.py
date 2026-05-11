"""Unit tests for ``pcm.attractor`` (PCM v5)."""
from __future__ import annotations

import math
import pytest
import torch

from pcm.attractor import (
    AttractorHead,
    AttractorOutput,
    AttractorTargets,
    HybridDecision,
    HybridPhysicsDispatcher,
    attractor_loss,
)


# ─────────────────────────────────────────────────────────────────
# AT1 — AttractorHead forward shape contract
# ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("state_dim,n_bodies,B", [
    (12, 3, 7),
    (16, 4, 1),
    (4, 2, 64),
])
def test_at1_attractor_head_shapes(state_dim, n_bodies, B):
    head = AttractorHead(state_dim=state_dim, n_bodies=n_bodies, hidden=32, depth=2)
    state = torch.randn(B, state_dim)
    out = head(state)
    assert isinstance(out, AttractorOutput)
    assert out.escape_logits.shape == (B, n_bodies)
    assert out.log_escape_time.shape == (B,)
    assert out.energy_logits.shape == (B, n_bodies)
    assert out.mean_final_state.shape == (B, state_dim)
    assert out.log_std_final_state.shape == (B, state_dim)


def test_at1b_attractor_head_log_std_clamped():
    """log_std clamping ensures the head can't trivially shrink
    state-NLL by predicting absurd variance."""
    head = AttractorHead(state_dim=12, n_bodies=3, hidden=32, depth=2)
    with torch.no_grad():
        for p in head.log_std.parameters():
            p.fill_(1e6)
    out = head(torch.randn(4, 12))
    assert out.log_std_final_state.max().item() <= 5.0 + 1e-6
    assert out.log_std_final_state.min().item() >= -3.0 - 1e-6


# ─────────────────────────────────────────────────────────────────
# AT2 — attractor_loss components and gradient
# ─────────────────────────────────────────────────────────────────


def _mock_targets(B: int, state_dim: int, n_bodies: int) -> AttractorTargets:
    energy = torch.softmax(torch.randn(B, n_bodies), dim=-1)
    return AttractorTargets(
        escape_label=torch.randint(0, n_bodies, (B,)),
        log_escape_time=torch.randn(B),
        energy_target=energy,
        mean_final_state=torch.randn(B, state_dim),
        std_final_state=torch.full((B, state_dim), 0.5),
    )


def test_at2_attractor_loss_returns_finite_total():
    head = AttractorHead(state_dim=12, n_bodies=3, hidden=32, depth=2)
    state = torch.randn(8, 12)
    out = head(state)
    target = _mock_targets(8, 12, 3)
    total, comps = attractor_loss(out, target)
    assert torch.isfinite(total)
    assert {"loss_escape", "loss_time", "loss_energy",
            "loss_state", "loss_total"} <= set(comps.keys())
    for v in comps.values():
        assert math.isfinite(v)


def test_at2b_attractor_loss_components_track_signal():
    """If we make energy_target match a one-hot prediction, the
    energy KL component should be smaller than for a uniform one."""
    head = AttractorHead(state_dim=12, n_bodies=3, hidden=32, depth=2)
    state = torch.randn(64, 12)
    # Force the head to output near-uniform energy probabilities
    with torch.no_grad():
        head.energy.weight.zero_()
        head.energy.bias.zero_()
    out = head(state)
    # Uniform target should give low KL
    uniform_target = _mock_targets(64, 12, 3)
    uniform_target.energy_target = torch.full((64, 3), 1.0 / 3.0)
    _, comps_uniform = attractor_loss(out, uniform_target)
    # Sharp target should give high KL
    sharp_target = _mock_targets(64, 12, 3)
    sharp_target.energy_target = torch.zeros(64, 3)
    sharp_target.energy_target[:, 0] = 1.0
    _, comps_sharp = attractor_loss(out, sharp_target)
    assert comps_sharp["loss_energy"] > comps_uniform["loss_energy"]


def test_at2c_attractor_loss_gradient_flows():
    head = AttractorHead(state_dim=12, n_bodies=3, hidden=16, depth=2)
    state = torch.randn(8, 12)
    target = _mock_targets(8, 12, 3)
    total, _ = attractor_loss(head(state), target)
    total.backward()
    grad_present = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in head.parameters()
    )
    assert grad_present


# ─────────────────────────────────────────────────────────────────
# AT3 — HybridPhysicsDispatcher routing logic
# ─────────────────────────────────────────────────────────────────


class _StubCook:
    """Minimal cook stand-in: returns deterministic pseudo-state."""

    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        self.calls: list[tuple[int, int]] = []

    def __call__(self, state: torch.Tensor, *, K: int):
        self.calls.append((state.shape[0], K))
        out = state + float(K) * 0.01
        return out, ("stub_report", K)


def test_at3_dispatcher_routes_short_horizon_to_cook():
    head = AttractorHead(state_dim=12, n_bodies=3, hidden=16, depth=2)
    cook = _StubCook(state_dim=12)
    disp = HybridPhysicsDispatcher(cook=cook, attractor=head, K_star=20)
    state = torch.randn(4, 12)
    decision = disp.predict(state, K_target=10)
    assert isinstance(decision, HybridDecision)
    assert decision.used == "cook"
    assert decision.cook_state is not None
    assert decision.attractor is None
    assert cook.calls == [(4, 10)]


def test_at3b_dispatcher_routes_long_horizon_to_attractor():
    head = AttractorHead(state_dim=12, n_bodies=3, hidden=16, depth=2)
    cook = _StubCook(state_dim=12)
    disp = HybridPhysicsDispatcher(cook=cook, attractor=head, K_star=20)
    state = torch.randn(4, 12)
    decision = disp.predict(state, K_target=50)
    assert decision.used == "attractor"
    assert decision.cook_state is None
    assert decision.attractor is not None
    assert cook.calls == []
    assert decision.attractor.escape_logits.shape == (4, 3)


def test_at3c_dispatcher_K_target_equal_K_star_uses_cook():
    head = AttractorHead(state_dim=12, n_bodies=3, hidden=16, depth=2)
    cook = _StubCook(state_dim=12)
    disp = HybridPhysicsDispatcher(cook=cook, attractor=head, K_star=20)
    state = torch.randn(2, 12)
    decision = disp.predict(state, K_target=20)
    assert decision.used == "cook"


def test_at3d_dispatcher_repr_contains_state_dim():
    head = AttractorHead(state_dim=12, n_bodies=3, hidden=16, depth=2)
    cook = _StubCook(state_dim=12)
    disp = HybridPhysicsDispatcher(cook=cook, attractor=head, K_star=11)
    s = repr(disp)
    assert "K_star=11" in s
    assert "state_dim=12" in s
    assert "n_bodies=3" in s


# ─────────────────────────────────────────────────────────────────
# AT4 — Sanity: attractor head can learn a trivial signal
# ─────────────────────────────────────────────────────────────────


def test_at4_attractor_head_learns_trivial_escape_label():
    """If the escape body is a deterministic function of the
    state's argmax, a small AttractorHead should learn it almost
    perfectly within a few steps. Smoke-level test, not full
    convergence sweep."""
    torch.manual_seed(0)
    state_dim, n_bodies, B = 6, 3, 256
    head = AttractorHead(state_dim=state_dim, n_bodies=n_bodies, hidden=32, depth=2)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)

    def _batch():
        s = torch.randn(B, state_dim)
        # Map state dim 0..2 argmax → escape label
        label = s[:, :n_bodies].argmax(dim=-1)
        targets = AttractorTargets(
            escape_label=label,
            log_escape_time=torch.zeros(B),
            energy_target=torch.full((B, n_bodies), 1.0 / n_bodies),
            mean_final_state=torch.zeros(B, state_dim),
            std_final_state=torch.full((B, state_dim), 1.0),
        )
        return s, targets

    for _ in range(80):
        s, t = _batch()
        out = head(s)
        loss, _ = attractor_loss(out, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    s, t = _batch()
    pred = head(s).escape_logits.argmax(dim=-1)
    acc = (pred == t.escape_label).float().mean().item()
    assert acc > 0.85, f"trivial escape-label task not learned, acc={acc}"
