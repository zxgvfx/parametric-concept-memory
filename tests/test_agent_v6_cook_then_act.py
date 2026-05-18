"""Unit tests for F73 cook-then-act planner: AgentAttractorHead,
MPC planner, and hybrid dispatcher."""
from __future__ import annotations

import math

import torch

from pcm.agent import (
    AgentAttractorHead,
    AttractorTargets,
    HybridPlanResult,
    MultiArgActionTransitionHead,
    MultiArgPolicyHead,
    SlotStateEncoder,
    agent_attractor_loss,
    hybrid_multi_arg_step,
    plan_multi_arg_mpc,
)
from pcm.agent.envs import MTC_ARG_DIM


# ─────────────────────────────────────────────────────────────────
# AgentAttractorHead
# ─────────────────────────────────────────────────────────────────


def test_attractor_head_shape():
    head = AgentAttractorHead(dim=16)
    slot_s = torch.randn(4, 16)
    slot_g = torch.randn(4, 16)
    succ_logit, log_steps = head(slot_s, slot_g)
    assert succ_logit.shape == (4,)
    assert log_steps.shape == (4,)


def test_attractor_head_predict_in_range():
    head = AgentAttractorHead(dim=16)
    slot_s = torch.randn(4, 16)
    slot_g = torch.randn(4, 16)
    p_succ, exp_steps = head.predict(slot_s, slot_g)
    assert (p_succ >= 0.0).all() and (p_succ <= 1.0).all()
    assert (exp_steps > 0.0).all()


def test_attractor_loss_finite_grad():
    head = AgentAttractorHead(dim=16)
    slot_s = torch.randn(4, 16)
    slot_g = torch.randn(4, 16)
    targets = AttractorTargets(
        success=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        n_steps=torch.tensor([3, 5, 2, 7]),
    )
    loss, diag = agent_attractor_loss(head, slot_s, slot_g, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in head.parameters())
    assert math.isfinite(diag["bce"])
    assert math.isfinite(diag["mse_log_steps"])


def test_attractor_learns_simple_pattern():
    """The attractor head should learn a simple 'success = goal
    in {0, 1}' rule in a few hundred SGD steps."""
    torch.manual_seed(0)
    enc = SlotStateEncoder(20, 16)
    head = AgentAttractorHead(dim=16)
    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(head.parameters()), lr=5e-3,
    )
    for _ in range(300):
        s = torch.randint(0, 20, (64,))
        g = torch.randint(0, 20, (64,))
        # Rule: success iff goal < 10 (arbitrary learnable rule)
        succ = (g < 10).float()
        steps = torch.where(succ.bool(),
                             torch.full_like(succ, 2.0),
                             torch.full_like(succ, 8.0))
        slot_s = enc(s)
        slot_g = enc(g)
        targets = AttractorTargets(succ, steps)
        loss, _ = agent_attractor_loss(head, slot_s, slot_g, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        s = torch.randint(0, 20, (200,))
        g = torch.randint(0, 20, (200,))
        slot_s = enc(s)
        slot_g = enc(g)
        p_succ, _ = head.predict(slot_s, slot_g)
        true_succ = (g < 10).float()
        # Binarised classifier accuracy
        pred = (p_succ >= 0.5).float()
        acc = float((pred == true_succ).float().mean().item())
    assert acc >= 0.85, f"attractor didn't learn: acc={acc}"


# ─────────────────────────────────────────────────────────────────
# MPC planner
# ─────────────────────────────────────────────────────────────────


def test_plan_multi_arg_mpc_returns_valid_action():
    torch.manual_seed(0)
    enc = SlotStateEncoder(41, 16)
    th = MultiArgActionTransitionHead(16, n_tools=6, arg_dim=MTC_ARG_DIM)
    pol = MultiArgPolicyHead(16, n_tools=6, arg_dim=MTC_ARG_DIM)
    tool, args = plan_multi_arg_mpc(
        25, 15, enc, th, pol,
        n_candidates=4, rollout_steps=3,
    )
    assert 0 <= tool < 6
    assert len(args) == MTC_ARG_DIM
    for a in args:
        assert isinstance(a, float)
        assert -1.0 <= a <= 1.0


def test_plan_multi_arg_mpc_deterministic_top_candidate():
    """With ``deterministic=True``, the *first* candidate is the
    argmax — same as direct policy. The planner may pick a
    different one if its rollout has lower terminal-distance,
    but the first candidate is always the policy argmax."""
    torch.manual_seed(0)
    enc = SlotStateEncoder(41, 16)
    th = MultiArgActionTransitionHead(16, n_tools=6, arg_dim=MTC_ARG_DIM)
    pol = MultiArgPolicyHead(16, n_tools=6, arg_dim=MTC_ARG_DIM)
    # Confirm planner doesn't crash with deterministic=True
    tool, args = plan_multi_arg_mpc(
        10, 20, enc, th, pol,
        n_candidates=1, rollout_steps=2,
        deterministic=True,
    )
    assert 0 <= tool < 6


# ─────────────────────────────────────────────────────────────────
# Hybrid dispatcher
# ─────────────────────────────────────────────────────────────────


def test_hybrid_step_returns_route_S1_or_S2():
    torch.manual_seed(0)
    enc = SlotStateEncoder(41, 16)
    th = MultiArgActionTransitionHead(16, n_tools=6, arg_dim=MTC_ARG_DIM)
    pol = MultiArgPolicyHead(16, n_tools=6, arg_dim=MTC_ARG_DIM)
    attr = AgentAttractorHead(dim=16)
    res = hybrid_multi_arg_step(
        25, 15, enc, th, pol, attr,
        p_success_threshold=0.5,
    )
    assert isinstance(res, HybridPlanResult)
    assert res.route in ("S1", "S2")
    assert 0 <= res.tool_id < 6
    assert len(res.args) == MTC_ARG_DIM
    assert 0.0 <= res.p_success <= 1.0


def test_hybrid_step_threshold_routes_to_S2_below():
    """If attractor predicts low p_success, dispatcher should
    route to S2. We force this by setting threshold to 0.99."""
    torch.manual_seed(0)
    enc = SlotStateEncoder(41, 16)
    th = MultiArgActionTransitionHead(16, n_tools=6, arg_dim=MTC_ARG_DIM)
    pol = MultiArgPolicyHead(16, n_tools=6, arg_dim=MTC_ARG_DIM)
    attr = AgentAttractorHead(dim=16)
    # Random init attractor — p_success will be near 0.5 < 0.99
    res = hybrid_multi_arg_step(
        25, 15, enc, th, pol, attr,
        p_success_threshold=0.99,
    )
    assert res.route == "S2"


def test_hybrid_step_threshold_routes_to_S1_above():
    """If we set threshold to 0.0, dispatcher always routes to S1."""
    torch.manual_seed(0)
    enc = SlotStateEncoder(41, 16)
    th = MultiArgActionTransitionHead(16, n_tools=6, arg_dim=MTC_ARG_DIM)
    pol = MultiArgPolicyHead(16, n_tools=6, arg_dim=MTC_ARG_DIM)
    attr = AgentAttractorHead(dim=16)
    res = hybrid_multi_arg_step(
        25, 15, enc, th, pol, attr,
        p_success_threshold=0.0,
    )
    assert res.route == "S1"
