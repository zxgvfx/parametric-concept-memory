"""pcm.agent.planner — v6.6 cook-then-act planner.

F70 demonstrated a single-step BC policy on multi-arg tools.
F73 adds the **System 2 cook layer**: at decision time, the
agent uses its ``MultiArgActionTransitionHead`` as a world
model to roll out candidate first-actions for K steps under the
greedy policy, then scores each rollout by *terminal slot
distance to goal*. The first action of the highest-scoring
rollout is executed.

Combined with the F61-style ``AgentAttractorHead`` (which
predicts ``p_success`` and ``expected_steps`` in O(1) from
``(state, goal)``), we get the agent-side
``HybridDispatcher``: high-confidence queries (``p_success ≥
threshold``) take the cheap System 1 path; low-confidence
queries pay for System 2 planning.

This module is opt-in — F64 / F65 / F66 / F70 callers continue
to work via the direct rollout helpers in ``orchestrator.py``.

Public API::

    plan_mpc                  — single-step MPC planner
    HybridPlanResult          — dataclass for dispatcher decisions
    hybrid_step               — one decision under the hybrid policy
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .attractor_head import AgentAttractorHead
from .heads import SlotStateEncoder
from .heads_mixed import MixedActionTransitionHead, MixedPolicyHead


__all__ = [
    "plan_mpc",
    "plan_multi_arg_mpc",
    "HybridPlanResult",
    "hybrid_step",
    "hybrid_multi_arg_step",
]


# ─────────────────────────────────────────────────────────────────
# MPC planner (mixed-arity — F66 heads)
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def plan_mpc(
    state_idx: int, goal_idx: int,
    encoder: SlotStateEncoder,
    transition: MixedActionTransitionHead,
    policy: MixedPolicyHead,
    *, n_candidates: int = 4,
    rollout_steps: int = 3,
    device: str = "cpu",
    deterministic: bool = False,
    rng: torch.Generator | None = None,
) -> tuple[int, float]:
    """One-step MPC plan for the F66 mixed-arity action space.

    Generates ``n_candidates`` candidate first-actions by
    sampling from the policy's tool-categorical and arg-Gaussian.
    For each candidate, rolls out ``rollout_steps`` future steps
    under the *greedy* policy through the world-model
    ``transition``. Scores each rollout by the squared distance
    between the terminal slot and the goal slot. Returns the
    first action of the best-scoring candidate.

    Returns ``(tool_id, arg_scalar)`` — same shape as
    ``MixedPolicyHead.deterministic`` so it can swap in
    place at rollout time.
    """
    encoder.eval()
    transition.eval()
    policy.eval()

    s_t = torch.tensor([state_idx], dtype=torch.long, device=device)
    g_t = torch.tensor([goal_idx], dtype=torch.long, device=device)
    slot_s = encoder(s_t)
    slot_g = encoder(g_t)

    tool_logits, arg_mean, arg_log_std = policy(slot_s, slot_g)
    tool_probs = torch.softmax(tool_logits, dim=-1)

    # ``rng`` is optional and *only* used to derive seeds for
    # CPU-side sampling. Tool / arg sampling happens on the same
    # device as the policy outputs (no generator argument — uses
    # the global torch RNG, which is fine for stochastic
    # candidate generation).
    candidates = []
    for k in range(n_candidates):
        if deterministic and k == 0:
            tool = int(tool_logits.argmax(-1).item())
            arg = float(arg_mean.clamp(-1.0, 1.0).item())
        else:
            tool = int(torch.multinomial(tool_probs, 1).item())
            arg = float((
                arg_mean + torch.randn_like(arg_mean) * arg_log_std.exp()
            ).clamp(-1.0, 1.0).item())
        candidates.append((tool, arg))

    best_idx = 0
    best_score = float("inf")
    for ci, (tool, arg) in enumerate(candidates):
        # Simulate one step then run policy.argmax for the rest
        cur_slot = slot_s
        tool_t = torch.tensor([tool], dtype=torch.long, device=device)
        arg_t = torch.tensor([arg], dtype=torch.float32, device=device)
        cur_slot = transition(cur_slot, tool_t, arg_t)
        for _ in range(rollout_steps - 1):
            tl, am, _ = policy(cur_slot, slot_g)
            nt = int(tl.argmax(-1).item())
            na = float(am.clamp(-1.0, 1.0).item())
            nt_t = torch.tensor([nt], dtype=torch.long, device=device)
            na_t = torch.tensor([na], dtype=torch.float32, device=device)
            cur_slot = transition(cur_slot, nt_t, na_t)
        # Score: L2 distance between final slot and goal slot
        dist = ((cur_slot - slot_g) ** 2).sum().item()
        if dist < best_score:
            best_score = dist
            best_idx = ci

    return candidates[best_idx]


# ─────────────────────────────────────────────────────────────────
# MPC planner (multi-arg — F70 heads)
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def plan_multi_arg_mpc(
    state_idx: int, goal_idx: int,
    encoder: SlotStateEncoder,
    transition,                           # MultiArgActionTransitionHead
    policy,                               # MultiArgPolicyHead
    *, n_candidates: int = 4,
    rollout_steps: int = 3,
    device: str = "cpu",
    arg_clamp: tuple[float, float] = (-1.0, 1.0),
    deterministic: bool = False,
    rng: torch.Generator | None = None,
) -> tuple[int, tuple[float, ...]]:
    """Multi-arg version of :func:`plan_mpc` for the F70
    multi-arg action space. Same logic, but ``arg`` is a vector
    of length ``policy.arg_dim``."""
    encoder.eval()
    transition.eval()
    policy.eval()
    arg_dim = policy.arg_dim

    s_t = torch.tensor([state_idx], dtype=torch.long, device=device)
    g_t = torch.tensor([goal_idx], dtype=torch.long, device=device)
    slot_s = encoder(s_t)
    slot_g = encoder(g_t)

    tool_logits, arg_mean, arg_log_std = policy(slot_s, slot_g)
    tool_probs = torch.softmax(tool_logits, dim=-1)

    candidates = []
    for k in range(n_candidates):
        if deterministic and k == 0:
            tool = int(tool_logits.argmax(-1).item())
            args = tuple(
                float(arg_mean[0, j].clamp(*arg_clamp).item())
                for j in range(arg_dim)
            )
        else:
            tool = int(torch.multinomial(tool_probs, 1).item())
            args_t = (
                arg_mean[0] + torch.randn_like(arg_mean[0])
                * arg_log_std[0].exp()
            ).clamp(*arg_clamp)
            args = tuple(float(args_t[j].item()) for j in range(arg_dim))
        candidates.append((tool, args))

    best_idx = 0
    best_score = float("inf")
    for ci, (tool, args) in enumerate(candidates):
        cur_slot = slot_s
        tool_t = torch.tensor([tool], dtype=torch.long, device=device)
        args_t = torch.tensor([list(args)], dtype=torch.float32,
                              device=device)
        cur_slot = transition(cur_slot, tool_t, args_t)
        for _ in range(rollout_steps - 1):
            tl, am, _ = policy(cur_slot, slot_g)
            nt = int(tl.argmax(-1).item())
            na = tuple(
                float(am[0, j].clamp(*arg_clamp).item())
                for j in range(arg_dim)
            )
            nt_t = torch.tensor([nt], dtype=torch.long, device=device)
            na_t = torch.tensor([list(na)], dtype=torch.float32,
                                device=device)
            cur_slot = transition(cur_slot, nt_t, na_t)
        dist = ((cur_slot - slot_g) ** 2).sum().item()
        if dist < best_score:
            best_score = dist
            best_idx = ci

    return candidates[best_idx]


# ─────────────────────────────────────────────────────────────────
# Hybrid dispatcher (System 1 / System 2 routing)
# ─────────────────────────────────────────────────────────────────


@dataclass
class HybridPlanResult:
    """Trace of one hybrid-policy decision."""

    route: str                 # "S1" or "S2"
    tool_id: int
    args: tuple                # (arg,) for mixed; (a0, a1) for multi-arg
    p_success: float
    expected_steps: float


@torch.no_grad()
def hybrid_step(
    state_idx: int, goal_idx: int,
    encoder: SlotStateEncoder,
    transition: MixedActionTransitionHead,
    policy: MixedPolicyHead,
    attractor: AgentAttractorHead,
    *, p_success_threshold: float = 0.8,
    n_candidates: int = 4,
    rollout_steps: int = 3,
    device: str = "cpu",
    rng: torch.Generator | None = None,
) -> HybridPlanResult:
    """One hybrid-routed decision for the F66 mixed-arity space.

    1. Encode state + goal, run the attractor head to estimate
       ``p_success``.
    2. If ``p_success >= p_success_threshold`` — confident,
       route to **System 1** (direct policy argmax).
    3. Else — uncertain, route to **System 2** (MPC planner).
    """
    s_t = torch.tensor([state_idx], dtype=torch.long, device=device)
    g_t = torch.tensor([goal_idx], dtype=torch.long, device=device)
    slot_s = encoder(s_t)
    slot_g = encoder(g_t)
    p_succ, exp_steps = attractor.predict(slot_s, slot_g)
    p_succ_f = float(p_succ.item())
    exp_steps_f = float(exp_steps.item())

    if p_succ_f >= p_success_threshold:
        tool_logits, arg_mean, _ = policy(slot_s, slot_g)
        tool = int(tool_logits.argmax(-1).item())
        arg = float(arg_mean.clamp(-1.0, 1.0).item())
        return HybridPlanResult(
            route="S1", tool_id=tool, args=(arg,),
            p_success=p_succ_f, expected_steps=exp_steps_f,
        )
    tool, arg = plan_mpc(
        state_idx, goal_idx, encoder, transition, policy,
        n_candidates=n_candidates, rollout_steps=rollout_steps,
        device=device, rng=rng,
    )
    return HybridPlanResult(
        route="S2", tool_id=tool, args=(arg,),
        p_success=p_succ_f, expected_steps=exp_steps_f,
    )


@torch.no_grad()
def hybrid_multi_arg_step(
    state_idx: int, goal_idx: int,
    encoder: SlotStateEncoder,
    transition,
    policy,
    attractor: AgentAttractorHead,
    *, p_success_threshold: float = 0.8,
    n_candidates: int = 4,
    rollout_steps: int = 3,
    device: str = "cpu",
    rng: torch.Generator | None = None,
) -> HybridPlanResult:
    """Multi-arg version of :func:`hybrid_step` for F70 heads."""
    s_t = torch.tensor([state_idx], dtype=torch.long, device=device)
    g_t = torch.tensor([goal_idx], dtype=torch.long, device=device)
    slot_s = encoder(s_t)
    slot_g = encoder(g_t)
    p_succ, exp_steps = attractor.predict(slot_s, slot_g)
    p_succ_f = float(p_succ.item())
    exp_steps_f = float(exp_steps.item())

    if p_succ_f >= p_success_threshold:
        tool_logits, arg_mean, _ = policy(slot_s, slot_g)
        tool = int(tool_logits.argmax(-1).item())
        args = tuple(
            float(arg_mean[0, j].clamp(-1.0, 1.0).item())
            for j in range(policy.arg_dim)
        )
        return HybridPlanResult(
            route="S1", tool_id=tool, args=args,
            p_success=p_succ_f, expected_steps=exp_steps_f,
        )
    tool, args = plan_multi_arg_mpc(
        state_idx, goal_idx, encoder, transition, policy,
        n_candidates=n_candidates, rollout_steps=rollout_steps,
        device=device, rng=rng,
    )
    return HybridPlanResult(
        route="S2", tool_id=tool, args=args,
        p_success=p_succ_f, expected_steps=exp_steps_f,
    )
