"""Unit tests for v6.2-followup (F66), v6.3 (F67), v6.4 (F68)
and v6.5 (F69) extensions of the agent base."""
from __future__ import annotations

import math

import torch

from pcm.agent import (
    Episode,
    MixedActionTransitionHead,
    MixedPolicyHead,
    PolicyHead,
    SlotStateEncoder,
    TextPerceptionHead,
    ToolEmbedding,
    TransitionHead,
    collect_episodes,
    compute_returns,
    mixed_bc_loss,
    mixed_rollout,
    mixed_transition_loss,
    on_policy_transition_step,
    reinforce_step,
    running_mean_baseline,
    text_to_slot,
)
from pcm.agent.envs import (
    ACTION_DELTAS,
    CyclicNavEnv,
    INTEGER_CALC_TOOLS,
    IntegerCalcEnv,
    apply_tool,
    ic_bfs_optimal_action,
    ic_bfs_optimal_steps,
    tool_takes_arg,
)


# ─────────────────────────────────────────────────────────────────
# F66 — mixed-arity env / oracle
# ─────────────────────────────────────────────────────────────────


def test_integer_calc_env_reset():
    env = IntegerCalcEnv(S=20)
    assert env.n_states == 41
    assert env.n_tools == 3
    s = env.reset(7)
    assert s == env.state_to_idx(7)
    assert env.state == 7


def test_integer_calc_apply_tool_neg_halve():
    assert apply_tool(10, 1, 0.0, S=20) == -10
    assert apply_tool(10, 2, 0.0, S=20) == 5
    assert apply_tool(-7, 2, 0.0, S=20) == -4  # floor div: -7//2=-4


def test_integer_calc_apply_tool_add_k_clamp():
    # ADD_K(arg=1.0) → shift +20
    assert apply_tool(5, 0, 1.0, S=20) == 20  # 5+20 capped at +S
    assert apply_tool(-15, 0, -1.0, S=20) == -20  # capped at -S


def test_tool_takes_arg():
    assert tool_takes_arg(0)
    assert not tool_takes_arg(1)
    assert not tool_takes_arg(2)


def test_bfs_optimal_action_canonical_cases():
    # 5 -> 10 in 1 step via ADD_K(0.25). Greedy BFS picks the
    # *lowest* tool index breaking ties, which here is ADD_K.
    t, a = ic_bfs_optimal_action(5, 10, 20)
    assert t == 0
    assert math.isclose(a, 0.25, abs_tol=1e-6)
    # 10 -> -10 in 1 step. Multiple 1-step solutions exist:
    # NEG (tool 1) and ADD_K(-1.0) (tool 0); both reach -10.
    # BFS picks the lowest-index tool, so ADD_K wins.
    t, _ = ic_bfs_optimal_action(10, -10, 20)
    assert t in (0, 1)
    # Verify 1-step regardless of which tool BFS chose.
    assert ic_bfs_optimal_steps(10, -10, 20) == 1
    # 10 -> 5 in 1 step via HALVE (tool 2). ADD_K(-0.25) also
    # works; BFS picks the lowest-index. Just check 1-step.
    assert ic_bfs_optimal_steps(10, 5, 20) == 1


def test_bfs_optimal_steps_reachable():
    # All states reachable in a few steps for S=20
    assert ic_bfs_optimal_steps(0, 0, 20) == 0
    assert ic_bfs_optimal_steps(0, 20, 20) == 1
    assert ic_bfs_optimal_steps(0, -20, 20) == 1


# ─────────────────────────────────────────────────────────────────
# F66 — heads
# ─────────────────────────────────────────────────────────────────


def test_tool_embedding_shape():
    te = ToolEmbedding(n_tools=3, dim=16)
    out = te(torch.tensor([0, 1, 2]))
    assert out.shape == (3, 16)


def test_mixed_transition_head_shape():
    th = MixedActionTransitionHead(dim=16, n_tools=3)
    slot = torch.randn(4, 16)
    tool = torch.tensor([0, 1, 2, 0])
    arg = torch.tensor([0.5, 0.0, 0.0, -0.5])
    out = th(slot, tool, arg)
    assert out.shape == (4, 16)


def test_mixed_policy_head_shape():
    ph = MixedPolicyHead(dim=16, n_tools=3)
    slot = torch.randn(4, 16)
    goal = torch.randn(4, 16)
    tool_logits, arg_mean, arg_log_std = ph(slot, goal)
    assert tool_logits.shape == (4, 3)
    assert arg_mean.shape == (4,)
    assert arg_log_std.shape == (4,)


def test_mixed_bc_loss_with_mask():
    enc = SlotStateEncoder(41, 16)
    pol = MixedPolicyHead(dim=16, n_tools=3)
    s = torch.tensor([25, 30, 35])
    g = torch.tensor([30, 25, 20])
    tool = torch.tensor([0, 1, 2])
    arg = torch.tensor([0.25, 0.0, 0.0])
    mask = torch.tensor([True, False, False])
    loss, diag = mixed_bc_loss(pol, enc, s, g, tool, arg, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert diag["n_arg_samples"] == 1


def test_mixed_transition_loss_converges():
    """Mixed transition head should learn the integer-calc
    algebra to ≥ 0.70 in a few hundred SGD steps. (The F66 full
    PoC reaches ≥ 0.85 with longer training; this unit test
    just checks the loss decreases meaningfully — ADD_K has a
    continuous arg that creates inherent bin-boundary slop, so
    perfect convergence requires far more than the 1200 steps
    we afford a unit test.)"""
    torch.manual_seed(7)
    S = 20
    N = 2 * S + 1
    enc = SlotStateEncoder(N, 16)
    th = MixedActionTransitionHead(16, n_tools=3)
    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(th.parameters()), lr=5e-3,
    )
    for _ in range(1200):
        s = torch.randint(-S, S + 1, (64,))
        tool = torch.randint(0, 3, (64,))
        arg = torch.empty(64).uniform_(-1.0, 1.0)
        s_next = torch.tensor([
            apply_tool(int(s[i]), int(tool[i]), float(arg[i]), S=S)
            for i in range(64)
        ])
        loss = mixed_transition_loss(
            th, enc, (s + S), tool, arg, (s_next + S),
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        s = torch.randint(-S, S + 1, (1000,))
        tool = torch.randint(0, 3, (1000,))
        arg = torch.empty(1000).uniform_(-1.0, 1.0)
        s_next = torch.tensor([
            apply_tool(int(s[i]), int(tool[i]), float(arg[i]), S=S)
            for i in range(1000)
        ])
        slot_pred = th(enc(s + S), tool, arg)
        logits = slot_pred @ enc.all_slots().t()
        acc = (logits.argmax(-1) == (s_next + S)).float().mean().item()
    assert acc >= 0.70, f"mixed transition didn't converge: acc={acc}"


def test_mixed_rollout_basic():
    env = IntegerCalcEnv(S=20, max_steps=4)
    env.reset(5)
    env.set_goal(10)
    enc = SlotStateEncoder(41, 16)
    pol = MixedPolicyHead(dim=16, n_tools=3)
    traj = mixed_rollout(
        env, enc, pol, goal=env.state_to_idx(10),
        max_steps=4, reset_state=None,
    )
    assert traj.states[0] == env.state_to_idx(5)
    for a in traj.actions:
        assert isinstance(a, tuple)
        tool, arg = a
        assert 0 <= tool < 3
        assert isinstance(arg, float)


# ─────────────────────────────────────────────────────────────────
# F67 — RL closure
# ─────────────────────────────────────────────────────────────────


def test_compute_returns_basic():
    G = compute_returns([0.0, 0.0, 1.0], gamma=0.9)
    assert math.isclose(G[2], 1.0, abs_tol=1e-6)
    assert math.isclose(G[1], 0.9, abs_tol=1e-6)
    assert math.isclose(G[0], 0.81, abs_tol=1e-6)


def test_running_mean_baseline_ema():
    state = {}
    b, state = running_mean_baseline([1.0, 1.0, 1.0], state,
                                      momentum=0.1)
    assert math.isclose(b, 0.1)
    b, state = running_mean_baseline([1.0], state, momentum=0.1)
    assert math.isclose(b, 0.19, abs_tol=1e-6)


def test_collect_episodes_shape():
    rng = torch.Generator(device="cpu").manual_seed(0)
    enc = SlotStateEncoder(20, 16)
    pol = PolicyHead(dim=16, n_actions=len(ACTION_DELTAS))

    def factory():
        return CyclicNavEnv(N=20, max_steps=8)

    def s_sampler():
        return int(torch.randint(0, 20, (1,), generator=rng).item())

    def g_sampler():
        return int(torch.randint(0, 20, (1,), generator=rng).item())

    eps = collect_episodes(
        factory, enc, pol, n_episodes=3, max_steps=8,
        goal_sampler=g_sampler, state_sampler=s_sampler,
    )
    assert len(eps) == 3
    for ep in eps:
        assert isinstance(ep, Episode)
        # state list is always actions+1 long
        assert len(ep.states) == len(ep.actions) + 1


def test_reinforce_step_finite_grad():
    enc = SlotStateEncoder(20, 16)
    pol = PolicyHead(dim=16, n_actions=4)
    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(pol.parameters()), lr=1e-2,
    )
    ep = Episode(
        states=[0, 1, 2, 3], actions=[0, 0, 0],
        rewards=[0.0, 0.0, 1.0], goal=3, success=True,
    )
    diag = reinforce_step(enc, pol, [ep], opt, baseline=0.0)
    assert math.isfinite(diag["loss"])
    assert diag["n_steps"] == 3


def test_on_policy_transition_step_runs():
    enc = SlotStateEncoder(20, 16)
    th = TransitionHead(dim=16, n_actions=4)
    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(th.parameters()), lr=1e-2,
    )
    ep = Episode(
        states=[0, 1, 2, 3], actions=[0, 0, 0],
        rewards=[0.0, 0.0, 1.0], goal=3,
    )
    diag = on_policy_transition_step(enc, th, [ep], opt)
    assert math.isfinite(diag["loss"])
    assert diag["n_steps"] == 3


# ─────────────────────────────────────────────────────────────────
# F68 — perception
# ─────────────────────────────────────────────────────────────────


def test_text_perception_head_shape():
    head = TextPerceptionHead(vocab=30, slot_dim=16, max_len=8)
    tokens = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0]])
    mask = torch.tensor([[True, True, True, True,
                          False, False, False, False]])
    out = head(tokens, mask=mask)
    assert out.shape == (1, 16)


def test_text_perception_head_no_mask():
    head = TextPerceptionHead(vocab=30, slot_dim=16, max_len=8)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    out = head(tokens)
    assert out.shape == (1, 16)


def test_text_to_slot_inference():
    head = TextPerceptionHead(vocab=30, slot_dim=16, max_len=8)
    slot = text_to_slot(head, [1, 2, 3])
    assert slot.shape == (1, 16)


def test_text_perception_alias_invariance_after_training():
    """Smoke training: after a few hundred SGD steps to map two
    distinct token sequences to the same target slot, the
    perception head should produce nearly-aligned outputs for
    them."""
    torch.manual_seed(0)
    head = TextPerceptionHead(vocab=30, slot_dim=8, d_model=16,
                              n_layers=1, n_heads=2, max_len=4)
    # Two synonyms for the same target.
    a = torch.tensor([[10, 0, 0, 0]])
    b = torch.tensor([[10, 20, 0, 0]])
    target = torch.randn(1, 8)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    for _ in range(400):
        out_a = head(a)
        out_b = head(b)
        loss = ((out_a - target) ** 2 + (out_b - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    cos = torch.nn.functional.cosine_similarity(
        head(a), head(b),
    ).item()
    assert cos >= 0.95, f"perception aliasing failed: cos={cos}"
