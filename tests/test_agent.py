"""Unit tests for ``pcm.agent`` v6.1 layer."""
from __future__ import annotations

import torch

from pcm.agent import (
    PolicyHead,
    SlotStateEncoder,
    TransitionHead,
    ValueHead,
    bc_loss,
    rollout,
    transition_loss,
)
from pcm.agent.envs import (
    ACTION_DELTAS,
    CyclicNavEnv,
    bfs_optimal_action,
    optimal_action,
    optimal_trajectory,
    shortest_path_length,
)


# ─────────────────────────────────────────────────────────────────
# Env tests
# ─────────────────────────────────────────────────────────────────


def test_env_reset_and_step():
    env = CyclicNavEnv(N=20, max_steps=10)
    s = env.reset(7)
    assert s == 7
    assert env.state == 7
    env.set_goal(13)
    assert env.goal == 13
    s_next, r, done = env.step(0)  # +1
    assert s_next == 8
    assert r == 0.0
    assert not done


def test_env_reaches_goal_with_reward():
    env = CyclicNavEnv(N=20, max_steps=10)
    env.reset(0)
    env.set_goal(1)
    s_next, r, done = env.step(0)  # +1
    assert s_next == 1
    assert r == 1.0
    assert done


def test_env_modular_wrap():
    env = CyclicNavEnv(N=20, max_steps=10)
    env.reset(18)
    env.set_goal(1)
    s_next, r, done = env.step(0)  # +1: 18 -> 19
    assert s_next == 19
    s_next, r, done = env.step(0)  # +1: 19 -> 0
    assert s_next == 0
    s_next, r, done = env.step(0)  # +1: 0 -> 1, success
    assert s_next == 1
    assert r == 1.0
    assert done


def test_env_action_set_size():
    env = CyclicNavEnv(N=20)
    assert env.n_actions == len(ACTION_DELTAS) == 4


def test_env_invalid_action_raises():
    env = CyclicNavEnv(N=20)
    env.reset(0)
    try:
        env.step(99)
    except ValueError:
        return
    assert False, "expected ValueError"


# ─────────────────────────────────────────────────────────────────
# Oracle tests
# ─────────────────────────────────────────────────────────────────


def test_optimal_action_greedy_simple():
    # 0->5 on Z_20 — optimal greedy is +5
    a = optimal_action(0, 5, 20)
    assert ACTION_DELTAS[a] == +5
    # 0->1 on Z_20 — optimal greedy is +1 (5 would overshoot)
    a = optimal_action(0, 1, 20)
    assert ACTION_DELTAS[a] == +1


def test_optimal_trajectory_terminates():
    actions = optimal_trajectory(0, 7, 20)
    state = 0
    for a in actions:
        state = (state + ACTION_DELTAS[a]) % 20
    assert state == 7


def test_shortest_path_length_known_cases():
    # 0->10 on Z_20 — 5+5 = 2 steps
    assert shortest_path_length(0, 10, 20) == 2
    # 0->1 — 1 step
    assert shortest_path_length(0, 1, 20) == 1
    # 0->9 — BFS finds 3 (5+5-1) which is shorter than greedy 5
    assert shortest_path_length(0, 9, 20) == 3
    # Self-loop is 0
    assert shortest_path_length(5, 5, 20) == 0


def test_bfs_optimal_action_solves_hard_case():
    # 0->9 on Z_20 — greedy gives 5 steps, BFS gives 3.
    # The BFS-optimal action sequence should be +5, +5, -1.
    state = 0
    actions = []
    for _ in range(10):
        if state == 9:
            break
        a = bfs_optimal_action(state, 9, 20)
        actions.append(a)
        state = (state + ACTION_DELTAS[a]) % 20
    assert state == 9
    assert len(actions) == 3, f"BFS should solve in 3 steps, got {len(actions)}"


# ─────────────────────────────────────────────────────────────────
# Heads tests
# ─────────────────────────────────────────────────────────────────


def test_slot_state_encoder_shape():
    enc = SlotStateEncoder(20, 16)
    s = torch.tensor([0, 1, 2, 3])
    out = enc(s)
    assert out.shape == (4, 16)
    assert enc.all_slots().shape == (20, 16)


def test_transition_head_shape():
    th = TransitionHead(16, 4)
    slot = torch.randn(8, 16)
    a = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    out = th(slot, a)
    assert out.shape == (8, 16)


def test_policy_head_shape():
    ph = PolicyHead(16, 4)
    slot = torch.randn(8, 16)
    goal = torch.randn(8, 16)
    out = ph(slot, goal)
    assert out.shape == (8, 4)


def test_value_head_shape():
    vh = ValueHead(16)
    slot = torch.randn(8, 16)
    goal = torch.randn(8, 16)
    out = vh(slot, goal)
    assert out.shape == (8,)


# ─────────────────────────────────────────────────────────────────
# Loss tests
# ─────────────────────────────────────────────────────────────────


def test_bc_loss_finite_and_grad():
    enc = SlotStateEncoder(20, 16)
    pol = PolicyHead(16, 4)
    s = torch.tensor([0, 5, 10])
    g = torch.tensor([7, 12, 3])
    a = torch.tensor([0, 1, 2])
    loss = bc_loss(pol, enc, s, g, a)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in pol.parameters())


def test_transition_loss_converges_in_a_few_steps():
    """Sanity: the transition head can learn the cyclic-group
    algebra on Z_20 with 4 actions in a small number of steps."""
    torch.manual_seed(7)
    N = 20
    enc = SlotStateEncoder(N, 16)
    th = TransitionHead(16, 4)
    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(th.parameters()), lr=5e-3,
    )
    for _ in range(400):
        s = torch.randint(0, N, (64,))
        a = torch.randint(0, 4, (64,))
        deltas = torch.tensor(ACTION_DELTAS)[a]
        s_next = (s + deltas) % N
        loss = transition_loss(th, enc, s, a, s_next)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # After 400 steps acc should be near 1.0
    with torch.no_grad():
        s = torch.randint(0, N, (1000,))
        a = torch.randint(0, 4, (1000,))
        deltas = torch.tensor(ACTION_DELTAS)[a]
        s_next = (s + deltas) % N
        slot_pred = th(enc(s), a)
        logits = slot_pred @ enc.all_slots().t()
        acc = (logits.argmax(-1) == s_next).float().mean().item()
    assert acc >= 0.95, f"transition_loss didn't converge: acc={acc}"


# ─────────────────────────────────────────────────────────────────
# Rollout tests
# ─────────────────────────────────────────────────────────────────


def test_rollout_respects_caller_set_state():
    """Regression test for the rollout state-leak bug: rollout
    must NOT call env.reset() when the caller has already set
    a start state and passed reset_state=None.
    """
    env = CyclicNavEnv(N=20, max_steps=4)
    env.reset(13)
    env.set_goal(14)
    enc = SlotStateEncoder(20, 16)
    pol = PolicyHead(16, 4)
    traj = rollout(env, enc, pol, goal=14, max_steps=4, reset_state=None)
    # First state in trajectory must equal 13 (caller's start), not 0.
    assert traj.states[0] == 13


def test_rollout_with_explicit_reset_state():
    env = CyclicNavEnv(N=20, max_steps=4)
    env.set_goal(14)
    enc = SlotStateEncoder(20, 16)
    pol = PolicyHead(16, 4)
    traj = rollout(env, enc, pol, goal=14, max_steps=4, reset_state=7)
    assert traj.states[0] == 7


def test_rollout_records_done_and_success():
    """Untrained policy will rarely reach goal; check shape of
    output regardless."""
    env = CyclicNavEnv(N=20, max_steps=4)
    env.reset(0)
    env.set_goal(1)
    enc = SlotStateEncoder(20, 16)
    pol = PolicyHead(16, 4)
    traj = rollout(env, enc, pol, goal=1, max_steps=4, reset_state=None)
    assert 0 < traj.n_steps <= 4
    assert len(traj.states) == traj.n_steps + 1
    assert len(traj.actions) == traj.n_steps
