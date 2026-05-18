"""pcm.agent.orchestrator — Episode rollout helper.

The agent loop in v6.1 is intentionally minimal: at each step,
the policy picks an action from the current state and goal slot,
the environment advances, and the trajectory is collected. No
search, no value-based action selection, no planner — just
``argmax policy``. The architectural claim of F64 is that this
trivial loop with a properly-trained TransitionHead and
PolicyHead is enough to navigate ``ℤ_N`` cyclic tasks; richer
loops will be added in v6.2 onwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .heads import PolicyHead, SlotStateEncoder
from .heads_continuous import ContinuousPolicyHead
from .heads_mixed import MixedPolicyHead


__all__ = ["Trajectory", "rollout", "continuous_rollout", "mixed_rollout"]


@dataclass
class Trajectory:
    """One episode's collected data.

    For discrete-action rollouts (``rollout``), ``actions`` are
    ``int`` indices. For continuous-action rollouts
    (``continuous_rollout``), ``actions`` are ``float`` scalars.
    """

    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    done: bool = False
    success: bool = False
    n_steps: int = 0


@torch.no_grad()
def rollout(
    env: Any,
    encoder: SlotStateEncoder,
    policy: PolicyHead,
    *,
    goal: int,
    max_steps: int = 32,
    device: str = "cpu",
    deterministic: bool = True,
    reset_state: int | None = None,
) -> Trajectory:
    """Roll out one episode under a greedy (argmax) policy.

    Args:
        env: environment with ``.reset()`` and ``.step(a)`` methods.
            ``.reset()`` returns an ``int`` state index;
            ``.step(a)`` returns ``(next_state, reward, done)`` —
            see :class:`pcm.agent.envs.cyclic_nav.CyclicNavEnv`.
        encoder: state-index → slot embedding.
        policy: ``(slot_state, slot_goal) → action_logits``.
        goal: target state index.
        max_steps: hard budget; episode terminates if exceeded.
        device: torch device for slot computations.
        deterministic: if True, take ``argmax``; else sample.
        reset_state: if provided, reset env to this start state;
            else use env's default reset (state 0). Pass ``None``
            if the caller has already reset the env to the
            desired start state and does NOT want this rollout
            to overwrite it.

    Returns:
        :class:`Trajectory` with collected states / actions /
        rewards and a ``success`` flag set if the episode reached
        the goal within budget.
    """
    encoder.eval()
    policy.eval()
    if reset_state is None:
        # Caller already set env state; just read it. Note: do
        # NOT use ``getattr(env, 'state', env.reset())`` — Python
        # evaluates the default eagerly, calling env.reset() as a
        # side effect and overwriting the caller-set state with 0.
        if hasattr(env, "state"):
            s = env.state
        else:
            s = env.reset()
    else:
        s = env.reset(reset_state)
    traj = Trajectory()
    traj.states.append(int(s))
    g_tensor = torch.tensor([goal], dtype=torch.long, device=device)
    slot_g = encoder(g_tensor)
    for t in range(max_steps):
        s_tensor = torch.tensor([int(s)], dtype=torch.long, device=device)
        slot_s = encoder(s_tensor)
        logits = policy(slot_s, slot_g)
        if deterministic:
            a = int(logits.argmax(dim=-1).item())
        else:
            probs = torch.softmax(logits, dim=-1)
            a = int(torch.multinomial(probs, 1).item())
        s_next, reward, done = env.step(a)
        traj.actions.append(int(a))
        traj.rewards.append(float(reward))
        traj.states.append(int(s_next))
        traj.n_steps += 1
        if done:
            traj.done = True
            traj.success = bool(s_next == goal)
            break
        s = s_next
    return traj


@torch.no_grad()
def continuous_rollout(
    env: Any,
    encoder: SlotStateEncoder,
    policy: ContinuousPolicyHead,
    *,
    goal: int,
    max_steps: int = 32,
    device: str = "cpu",
    deterministic: bool = True,
    reset_state: int | None = None,
    action_clamp: tuple[float, float] | None = (-1.0, 1.0),
) -> Trajectory:
    """Roll out one episode under a continuous-action Gaussian
    policy.

    Differs from :func:`rollout` only in:
    * ``policy`` returns ``(mean, log_std)`` instead of action
      logits, and the action is a continuous scalar;
    * ``action_clamp`` bounds the scalar before passing to
      ``env.step``.

    The env's ``step`` is expected to accept a single float and
    return ``(next_state_idx, reward, done)`` — see
    :class:`pcm.agent.envs.continuous_nav.ContinuousCyclicNavEnv`.
    """
    encoder.eval()
    policy.eval()
    if reset_state is None:
        if hasattr(env, "state"):
            s = env.state
        else:
            s = env.reset()
    else:
        s = env.reset(reset_state)
    traj = Trajectory()
    traj.states.append(int(s))
    g_tensor = torch.tensor([goal], dtype=torch.long, device=device)
    slot_g = encoder(g_tensor)
    for t in range(max_steps):
        s_tensor = torch.tensor([int(s)], dtype=torch.long, device=device)
        slot_s = encoder(s_tensor)
        mean, log_std = policy(slot_s, slot_g)
        if deterministic:
            a = mean
        else:
            a = mean + torch.randn_like(mean) * log_std.exp()
        if action_clamp is not None:
            a = a.clamp(*action_clamp)
        a_scalar = float(a.item() if a.numel() == 1 else a[0].item())
        s_next, reward, done = env.step(a_scalar)
        traj.actions.append(a_scalar)
        traj.rewards.append(float(reward))
        traj.states.append(int(s_next))
        traj.n_steps += 1
        if done:
            traj.done = True
            traj.success = bool(s_next == goal)
            break
        s = s_next
    return traj


@torch.no_grad()
def mixed_rollout(
    env: Any,
    encoder: SlotStateEncoder,
    policy: MixedPolicyHead,
    *,
    goal: int,
    max_steps: int = 16,
    device: str = "cpu",
    deterministic: bool = True,
    reset_state: int | None = None,
    arg_clamp: tuple[float, float] | None = (-1.0, 1.0),
) -> Trajectory:
    """Roll out one episode under a mixed-arity tool-call policy.

    The env's ``step`` is expected to accept ``(tool_id, arg)``
    and return ``(next_state_idx, reward, done)`` — see
    :class:`pcm.agent.envs.integer_calc.IntegerCalcEnv`.

    Actions stored in ``traj.actions`` are tuples
    ``(tool_id: int, arg: float)``.
    """
    encoder.eval()
    policy.eval()
    if reset_state is None:
        if hasattr(env, "state"):
            s = env.state_to_idx(env.state) if hasattr(env, "state_to_idx") \
                else int(env.state)
        else:
            s = env.reset()
    else:
        s = env.reset(reset_state)
    traj = Trajectory()
    traj.states.append(int(s))
    g_tensor = torch.tensor([goal], dtype=torch.long, device=device)
    slot_g = encoder(g_tensor)
    for t in range(max_steps):
        s_tensor = torch.tensor([int(s)], dtype=torch.long, device=device)
        slot_s = encoder(s_tensor)
        tool_logits, arg_mean, arg_log_std = policy(slot_s, slot_g)
        if deterministic:
            tool = int(tool_logits.argmax(dim=-1).item())
            arg_val = arg_mean
        else:
            tool_probs = torch.softmax(tool_logits, dim=-1)
            tool = int(torch.multinomial(tool_probs, 1).item())
            arg_val = arg_mean + torch.randn_like(arg_mean) * arg_log_std.exp()
        if arg_clamp is not None:
            arg_val = arg_val.clamp(*arg_clamp)
        arg_scalar = float(arg_val.item() if arg_val.numel() == 1
                           else arg_val[0].item())
        s_next, reward, done = env.step(tool, arg_scalar)
        traj.actions.append((tool, arg_scalar))
        traj.rewards.append(float(reward))
        traj.states.append(int(s_next))
        traj.n_steps += 1
        if done:
            traj.done = True
            traj.success = bool(s_next == goal)
            break
        s = s_next
    return traj
