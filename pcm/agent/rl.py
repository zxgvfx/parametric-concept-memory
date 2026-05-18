"""pcm.agent.rl — v6.3 RL closure: REINFORCE + on-policy transition.

F64 (v6.1) and F65 (v6.2) trained the agent via behavioural
cloning on oracle labels. v6.3 closes the RL loop: the policy is
trained from **sparse reward** (+1 at goal, 0 elsewhere) without
any oracle action labels. The transition head is trained on the
*on-policy* data the agent itself collects.

Architectural claim of v6.3: the F62 ``UniversalCombiner``
operator that worked under BC also works under REINFORCE — the
learning signal switches from CE-on-labels to advantage-weighted
log-prob, but the architecture is unchanged. This mirrors the
F54 "calibrate_rpe_coverage" framework where System-1 retrieval
(cached policy) and System-2 cook (planning rollout) coexist
under one operator.

Minimum-viable implementation:

* :func:`collect_episodes` — Monte-Carlo rollouts with a
  stochastic (sampled-action) policy.
* :func:`compute_returns` — discounted per-step returns
  ``G_t = sum_{k≥t} γ^{k−t} r_k``.
* :func:`reinforce_step` — one batched policy-gradient update
  with a running-mean baseline (lighter than a learned value
  head — sufficient for F67's small env).
* :func:`on_policy_transition_step` — one transition-head SGD
  step on the agent's own (s, a, s_next) data.

The public API stays opt-in: v6.1 and v6.2 callers do not see
any change. v6.3 simply adds an alternate training mode that
reuses the same heads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .heads import PolicyHead, SlotStateEncoder, TransitionHead


__all__ = [
    "Episode",
    "collect_episodes",
    "compute_returns",
    "reinforce_step",
    "on_policy_transition_step",
    "running_mean_baseline",
]


@dataclass
class Episode:
    """One collected episode for RL training."""

    states: list[int] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    goal: int = 0
    success: bool = False


# ─────────────────────────────────────────────────────────────────
# Rollout / episode collection
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def collect_episodes(
    env_factory,
    encoder: SlotStateEncoder,
    policy: PolicyHead,
    *,
    n_episodes: int,
    max_steps: int,
    goal_sampler,
    state_sampler,
    device: str = "cpu",
    deterministic: bool = False,
) -> list[Episode]:
    """Collect a batch of episodes under the current stochastic
    policy.

    Args:
        env_factory: zero-arg callable returning a fresh env.
        encoder: state encoder.
        policy: discrete-action policy head.
        n_episodes: how many episodes to collect.
        max_steps: per-episode budget.
        goal_sampler: ``() -> goal_idx`` callable.
        state_sampler: ``() -> start_state_idx`` callable.
        device: torch device for forward passes.
        deterministic: if True, take argmax (eval mode); else
            sample from the action distribution (training).

    Returns:
        List of :class:`Episode`. ``goal`` is the integer index.
    """
    encoder.eval()
    policy.eval()
    out: list[Episode] = []
    for _ in range(n_episodes):
        env = env_factory()
        goal = int(goal_sampler())
        s0 = int(state_sampler())
        env.reset(s0)
        env.set_goal(goal)
        ep = Episode(goal=goal)
        s = s0
        ep.states.append(s)
        for _ in range(max_steps):
            s_t = torch.tensor([s], dtype=torch.long, device=device)
            g_t = torch.tensor([goal], dtype=torch.long, device=device)
            logits = policy(encoder(s_t), encoder(g_t))
            if deterministic:
                a = int(logits.argmax(dim=-1).item())
            else:
                probs = torch.softmax(logits, dim=-1)
                a = int(torch.multinomial(probs, 1).item())
            s_next, r, done = env.step(a)
            ep.actions.append(a)
            ep.rewards.append(float(r))
            ep.states.append(int(s_next))
            if done:
                ep.success = bool(s_next == goal)
                break
            s = int(s_next)
        out.append(ep)
    return out


# ─────────────────────────────────────────────────────────────────
# Return / advantage
# ─────────────────────────────────────────────────────────────────


def compute_returns(
    rewards: list[float], *, gamma: float = 0.95,
) -> list[float]:
    """Compute discounted per-step returns ``G_t = sum_{k≥t}
    γ^{k−t} r_k``. Pure Python — these lists are short."""
    G: list[float] = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        G.append(running)
    G.reverse()
    return G


def running_mean_baseline(
    new_returns: list[float], state: dict,
    *, momentum: float = 0.05,
) -> tuple[float, dict]:
    """Simple EMA baseline over past returns. Returns the
    *current* baseline and the updated state dict."""
    mean = state.get("mean", 0.0)
    if new_returns:
        batch_mean = sum(new_returns) / len(new_returns)
        mean = (1 - momentum) * mean + momentum * batch_mean
    state["mean"] = mean
    return mean, state


# ─────────────────────────────────────────────────────────────────
# REINFORCE step (with running-mean baseline)
# ─────────────────────────────────────────────────────────────────


def reinforce_step(
    encoder: SlotStateEncoder,
    policy: PolicyHead,
    episodes: list[Episode],
    optimizer: torch.optim.Optimizer,
    *,
    baseline: float = 0.0,
    gamma: float = 0.95,
    entropy_bonus: float = 0.0,
    device: str = "cpu",
) -> dict:
    """One REINFORCE update over a batch of episodes.

    Loss = - mean_t [(G_t − baseline) · log π(a_t | s_t, g)]
           - entropy_bonus · H[π]

    The baseline is a *scalar* (e.g. EMA of recent returns) —
    sufficient for small envs. For richer envs, plug a learned
    value head in via :class:`~pcm.agent.heads.ValueHead` (out
    of scope for v6.3 MVP).
    """
    encoder.train()
    policy.train()
    all_states: list[int] = []
    all_actions: list[int] = []
    all_goals: list[int] = []
    all_advantages: list[float] = []
    all_returns: list[float] = []
    for ep in episodes:
        G = compute_returns(ep.rewards, gamma=gamma)
        for i, a in enumerate(ep.actions):
            all_states.append(ep.states[i])
            all_actions.append(a)
            all_goals.append(ep.goal)
            all_advantages.append(G[i] - baseline)
            all_returns.append(G[i])
    if not all_states:
        return {"loss": 0.0, "n_steps": 0, "mean_return": 0.0,
                "entropy": 0.0}

    s_t = torch.tensor(all_states, dtype=torch.long, device=device)
    a_t = torch.tensor(all_actions, dtype=torch.long, device=device)
    g_t = torch.tensor(all_goals, dtype=torch.long, device=device)
    adv_t = torch.tensor(all_advantages, dtype=torch.float32, device=device)

    logits = policy(encoder(s_t), encoder(g_t))
    log_probs = F.log_softmax(logits, dim=-1)
    log_pi = log_probs.gather(1, a_t.unsqueeze(-1)).squeeze(-1)
    pg_loss = -(log_pi * adv_t).mean()

    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    loss = pg_loss - entropy_bonus * entropy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "pg_loss": float(pg_loss.item()),
        "entropy": float(entropy.item()),
        "n_steps": len(all_states),
        "mean_return": float(sum(all_returns) / max(len(all_returns), 1)),
    }


# ─────────────────────────────────────────────────────────────────
# On-policy transition step (learn world model from agent's own data)
# ─────────────────────────────────────────────────────────────────


def on_policy_transition_step(
    encoder: SlotStateEncoder,
    transition: TransitionHead,
    episodes: list[Episode],
    optimizer: torch.optim.Optimizer,
    *,
    device: str = "cpu",
) -> dict:
    """One transition-head SGD step on (s, a, s_next) triples
    collected by the current policy.

    This is the v6.3 analogue of F64's BC transition loss: the
    agent learns its own world model from its own experience,
    without privileged access to ground-truth transitions.
    """
    encoder.train()
    transition.train()
    s_list: list[int] = []
    a_list: list[int] = []
    sn_list: list[int] = []
    for ep in episodes:
        for i, a in enumerate(ep.actions):
            s_list.append(ep.states[i])
            a_list.append(a)
            sn_list.append(ep.states[i + 1])
    if not s_list:
        return {"loss": 0.0, "n_steps": 0}
    s_t = torch.tensor(s_list, dtype=torch.long, device=device)
    a_t = torch.tensor(a_list, dtype=torch.long, device=device)
    sn_t = torch.tensor(sn_list, dtype=torch.long, device=device)
    slot_pred = transition(encoder(s_t), a_t)
    logits = slot_pred @ encoder.all_slots().t()
    loss = F.cross_entropy(logits, sn_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item()), "n_steps": len(s_list)}
