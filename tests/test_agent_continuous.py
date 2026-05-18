"""Unit tests for ``pcm.agent`` v6.2 continuous-action layer."""
from __future__ import annotations

import math

import torch

from pcm.agent import (
    ContinuousActionRoPE,
    ContinuousPolicyHead,
    ContinuousTransitionHead,
    SlotStateEncoder,
    bc_gaussian_loss,
    continuous_rollout,
    continuous_transition_loss,
)
from pcm.agent.envs import (
    ContinuousCyclicNavEnv,
    optimal_continuous_action,
    optimal_continuous_steps,
)


# ─────────────────────────────────────────────────────────────────
# Env tests
# ─────────────────────────────────────────────────────────────────


def test_continuous_env_reset_state_property():
    env = ContinuousCyclicNavEnv(N=40, max_step=math.pi / 4)
    s = env.reset(7)
    assert s == 7
    assert env.state == 7


def test_continuous_env_step_full_action_advances_one_max_step():
    env = ContinuousCyclicNavEnv(N=40, max_step=math.pi / 4)
    env.reset(0)
    env.set_goal(99)  # arbitrary; we just check state advance
    s_next, _, _ = env.step(1.0)
    # bin_size = 2π/40 = π/20; max_step = π/4 = 5*bin_size
    # so a=1.0 advances exactly 5 bins
    assert s_next == 5


def test_continuous_env_action_clamped():
    env = ContinuousCyclicNavEnv(N=40, max_step=math.pi / 4)
    env.reset(0)
    s_a, _, _ = env.step(1.0)
    env.reset(0)
    s_b, _, _ = env.step(2.0)  # over-magnitude clamped to 1.0
    assert s_a == s_b


def test_continuous_env_modular_wrap():
    env = ContinuousCyclicNavEnv(N=40, max_step=math.pi / 4)
    env.reset(38)
    env.set_goal(99)
    # 38 + 5 bins = 43 mod 40 = 3
    s_next, _, _ = env.step(1.0)
    assert s_next == 3


def test_continuous_env_reward_at_goal():
    env = ContinuousCyclicNavEnv(N=40, max_step=math.pi / 4)
    env.reset(0)
    env.set_goal(5)  # 5 bins = max_step
    s_next, r, done = env.step(1.0)
    assert s_next == 5
    assert r == 1.0
    assert done


# ─────────────────────────────────────────────────────────────────
# Oracle tests
# ─────────────────────────────────────────────────────────────────


def test_optimal_continuous_action_within_max_step():
    # 0 -> 5 (5 bins = max_step) → optimal a = 1.0
    a = optimal_continuous_action(0, 5, 40, max_step=math.pi / 4)
    assert math.isclose(a, 1.0, abs_tol=1e-6)
    # 0 -> 2 (2 bins = 0.4 of max_step) → optimal a = 0.4
    a = optimal_continuous_action(0, 2, 40, max_step=math.pi / 4)
    assert math.isclose(a, 0.4, abs_tol=1e-6)


def test_optimal_continuous_action_beyond_max_step():
    # 0 -> 10 (10 bins = 2*max_step) → optimal a = 1.0 (cap)
    a = optimal_continuous_action(0, 10, 40, max_step=math.pi / 4)
    assert a == 1.0


def test_optimal_continuous_action_self():
    a = optimal_continuous_action(7, 7, 40, max_step=math.pi / 4)
    assert a == 0.0


def test_optimal_continuous_steps_known_cases():
    # 0 -> 5 in 1 step
    assert optimal_continuous_steps(0, 5, 40,
                                     max_step=math.pi / 4) == 1
    # 0 -> 10 in 2 steps
    assert optimal_continuous_steps(0, 10, 40,
                                     max_step=math.pi / 4) == 2
    # 0 -> 20 (antipode) in 4 steps
    assert optimal_continuous_steps(0, 20, 40,
                                     max_step=math.pi / 4) == 4
    # self loop is 0
    assert optimal_continuous_steps(5, 5, 40,
                                     max_step=math.pi / 4) == 0


# ─────────────────────────────────────────────────────────────────
# Heads tests
# ─────────────────────────────────────────────────────────────────


def test_action_rope_shape():
    rope = ContinuousActionRoPE(embed_dim=16, n_freqs=8)
    a = torch.tensor([0.0, 0.5, -0.5, 1.0])
    out = rope(a)
    assert out.shape == (4, 16)


def test_action_rope_zero_gives_consistent_output():
    rope = ContinuousActionRoPE(embed_dim=16, n_freqs=8)
    a = torch.zeros(3)
    out = rope(a)
    # All inputs are 0 → all outputs identical
    assert torch.allclose(out[0], out[1])
    assert torch.allclose(out[1], out[2])


def test_continuous_transition_head_shape():
    th = ContinuousTransitionHead(dim=16)
    slot = torch.randn(4, 16)
    a = torch.tensor([0.0, 0.5, -0.5, 1.0])
    out = th(slot, a)
    assert out.shape == (4, 16)


def test_continuous_policy_head_shape_and_clamp():
    ph = ContinuousPolicyHead(dim=16)
    slot = torch.randn(4, 16)
    goal = torch.randn(4, 16)
    mean, log_std = ph(slot, goal)
    assert mean.shape == (4,)
    assert log_std.shape == (4,)
    # log_std must be in [log_std_min, log_std_max]
    assert (log_std >= ph.log_std_min - 1e-5).all()
    assert (log_std <= ph.log_std_max + 1e-5).all()


def test_continuous_policy_head_deterministic_clamps_to_range():
    torch.manual_seed(0)
    ph = ContinuousPolicyHead(dim=16)
    slot = torch.randn(8, 16) * 5  # scale up to encourage extreme means
    goal = torch.randn(8, 16) * 5
    a = ph.deterministic(slot, goal)
    assert (a >= -1.0 - 1e-6).all()
    assert (a <= 1.0 + 1e-6).all()


# ─────────────────────────────────────────────────────────────────
# Loss tests
# ─────────────────────────────────────────────────────────────────


def test_bc_gaussian_loss_finite_grad():
    enc = SlotStateEncoder(40, 16)
    pol = ContinuousPolicyHead(16)
    s = torch.tensor([0, 5, 10])
    g = torch.tensor([7, 12, 3])
    a = torch.tensor([0.5, -0.3, 1.0])
    loss = bc_gaussian_loss(pol, enc, s, g, a)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in pol.parameters())


def test_continuous_transition_loss_converges():
    """The continuous transition head should learn the
    ``a -> displacement`` mapping in a few hundred SGD steps."""
    torch.manual_seed(7)
    N = 40
    max_step = math.pi / 4
    bin_size = 2 * math.pi / N
    enc = SlotStateEncoder(N, 16)
    th = ContinuousTransitionHead(16)
    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(th.parameters()), lr=5e-3,
    )
    for _ in range(600):
        s = torch.randint(0, N, (64,))
        a = torch.empty(64).uniform_(-1.0, 1.0)
        theta = s.float() * bin_size + a * max_step
        s_next = (theta / bin_size).round().long() % N
        loss = continuous_transition_loss(th, enc, s, a, s_next)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        s = torch.randint(0, N, (1000,))
        a = torch.empty(1000).uniform_(-1.0, 1.0)
        theta = s.float() * bin_size + a * max_step
        s_next = (theta / bin_size).round().long() % N
        slot_pred = th(enc(s), a)
        logits = slot_pred @ enc.all_slots().t()
        pred = logits.argmax(-1)
        diff = (pred - s_next)
        d_mod = diff.abs() % N
        d_circ = torch.minimum(d_mod, N - d_mod)
        within1 = (d_circ <= 1).float().mean().item()
    assert within1 >= 0.90, (
        f"continuous transition didn't converge: within1={within1}"
    )


# ─────────────────────────────────────────────────────────────────
# Rollout tests
# ─────────────────────────────────────────────────────────────────


def test_continuous_rollout_records_correct_types():
    env = ContinuousCyclicNavEnv(N=40, max_step=math.pi / 4,
                                  max_steps=4)
    env.reset(0)
    env.set_goal(5)
    enc = SlotStateEncoder(40, 16)
    pol = ContinuousPolicyHead(16)
    traj = continuous_rollout(
        env, enc, pol, goal=5, max_steps=4, reset_state=None,
    )
    assert traj.states[0] == 0
    # Continuous trajectory: actions are floats in [-1, 1]
    for a in traj.actions:
        assert isinstance(a, float)
        assert -1.0 <= a <= 1.0


def test_continuous_rollout_respects_caller_set_state():
    env = ContinuousCyclicNavEnv(N=40, max_step=math.pi / 4,
                                  max_steps=4)
    env.reset(13)
    env.set_goal(14)
    enc = SlotStateEncoder(40, 16)
    pol = ContinuousPolicyHead(16)
    traj = continuous_rollout(
        env, enc, pol, goal=14, max_steps=4, reset_state=None,
    )
    # First state must equal caller's start (13), not 0.
    assert traj.states[0] == 13


def test_continuous_rollout_with_explicit_reset_state():
    env = ContinuousCyclicNavEnv(N=40, max_step=math.pi / 4,
                                  max_steps=4)
    env.set_goal(14)
    enc = SlotStateEncoder(40, 16)
    pol = ContinuousPolicyHead(16)
    traj = continuous_rollout(
        env, enc, pol, goal=14, max_steps=4, reset_state=7,
    )
    assert traj.states[0] == 7
