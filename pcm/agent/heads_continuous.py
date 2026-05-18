"""pcm.agent.heads_continuous — v6.2 continuous-action layer.

F64 (v6.1) showed the F62 universal-operator architecture
extends from passive group action ``(slot, Δ) → slot'`` to
discrete-action MDP transition ``(state, action_idx) →
next_state``. F65 (v6.2) takes the next step: replace the
*discrete* action embedding (``nn.Embedding`` keyed by 4 action
indices) with a **RoPE-style continuous action encoder** —
mirroring F62c's extension of the universal operator from
discrete ℤ_N to continuous Lie group S¹, but on the action side
of the (state, action) → next_state operator instead of the
displacement side of the (slot, Δ) → slot' operator.

The agent is now a stochastic continuous-action policy with
``(mean, log_std)`` outputs over a scalar action ``a ∈ [-1, 1]``.
The transition head consumes the continuous action through a
RoPE encoder, exactly the F62c construction.

Architecture
============

::

    a (scalar, continuous in [-1, 1])
        ↓
    ContinuousActionRoPE
        a → [cos(f_k a), sin(f_k a) for k=1..K] → Linear → action_emb (D-dim)
        ↓
    UniversalCombiner(slot_state, action_emb) → predicted slot_next
        (identical to F62 / F62c's combiner — F65's only architectural
         change vs F64 is swapping discrete embedding lookup for
         RoPE on a real-valued action)

Public API::

    ContinuousActionRoPE        — analytic continuous-action encoder
    ContinuousTransitionHead    — UniversalCombiner over continuous a
    ContinuousPolicyHead        — Gaussian goal-conditioned policy
    bc_gaussian_loss            — Gaussian NLL behavioural cloning
    continuous_transition_loss  — next-state classifier loss
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import SlotStateEncoder


__all__ = [
    "ContinuousActionRoPE",
    "ContinuousTransitionHead",
    "ContinuousPolicyHead",
    "bc_gaussian_loss",
    "continuous_transition_loss",
]


# ─────────────────────────────────────────────────────────────────
# Continuous action encoder — RoPE on a single scalar
# ─────────────────────────────────────────────────────────────────


class ContinuousActionRoPE(nn.Module):
    """Analytic encoder for a *scalar* continuous action.

    ``a ∈ ℝ`` (typically clamped to ``[-1, 1]`` by the env) is
    mapped to a vector via the same RoPE construction used in
    F62c, but applied to a 1-D action coordinate rather than a
    1-D angular displacement::

        rpe(a) = Linear([cos(f_k · a · π), sin(f_k · a · π)
                         for k = 1..K])

    The factor of ``π`` puts the input range ``[-1, 1]`` into
    the natural ``[-π, π]`` range RoPE was designed for; the
    log-frequencies ``log f_k`` are learnable, initialised to the
    standard RoPE schedule.

    Output dim equals the slot dim so the encoder drops into the
    F62 ``UniversalCombiner`` unchanged.
    """

    def __init__(
        self, embed_dim: int, n_freqs: int = 8,
        base: float = 100.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_freqs = n_freqs
        log_init = torch.linspace(
            0.0, -math.log(base), n_freqs,
        )
        self.log_freq = nn.Parameter(log_init)
        self.proj = nn.Linear(2 * n_freqs, embed_dim)
        with torch.no_grad():
            self.proj.weight.mul_(0.3)
            self.proj.bias.zero_()

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """``a`` of shape ``(B,)`` (any real-valued scalars) →
        ``(B, embed_dim)``."""
        if a.dim() == 0:
            a = a.unsqueeze(0)
        freqs = torch.exp(self.log_freq)               # (n_freqs,)
        phase = (a * math.pi).unsqueeze(-1) * freqs    # (B, n_freqs)
        x = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────
# Continuous transition head — UniversalCombiner with RoPE encoder
# ─────────────────────────────────────────────────────────────────


class ContinuousTransitionHead(nn.Module):
    """``(slot_state, a_scalar) → predicted slot_next`` via
    F62c-style RoPE action encoder + F62 UniversalCombiner.

    The combiner is the F62 ``UniversalCombiner`` (residual MLP on
    the concatenation of slot and action embedding); only the
    action encoder side changes vs F64.
    """

    def __init__(
        self, dim: int, n_freqs: int = 8, hidden: int = 128,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.action_rope = ContinuousActionRoPE(
            embed_dim=dim, n_freqs=n_freqs,
        )
        self.combiner = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self, slot_state: torch.Tensor, a: torch.Tensor,
    ) -> torch.Tensor:
        a_emb = self.action_rope(a)
        delta = self.combiner(torch.cat([slot_state, a_emb], dim=-1))
        return slot_state + delta


# ─────────────────────────────────────────────────────────────────
# Continuous policy head — Gaussian goal-conditioned policy
# ─────────────────────────────────────────────────────────────────


class ContinuousPolicyHead(nn.Module):
    """``(slot_state, slot_goal) → (a_mean, a_log_std)``.

    Implements a simple Gaussian goal-conditioned policy. The
    action distribution is ``a ~ N(mean, exp(log_std)²)``. At
    inference time the deterministic action is just ``mean``,
    optionally clamped into the env's valid range.

    log_std is bounded into ``[log_std_min, log_std_max]`` to
    avoid the policy collapsing to a delta or exploding to
    pure noise.
    """

    def __init__(
        self, dim: int, hidden: int = 128,
        log_std_min: float = -3.0, log_std_max: float = 0.5,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.trunk = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.log_std_head = nn.Linear(hidden, 1)

    def forward(
        self, slot_state: torch.Tensor, slot_goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(torch.cat([slot_state, slot_goal], dim=-1))
        mean = self.mean_head(h).squeeze(-1)
        log_std = self.log_std_head(h).squeeze(-1)
        log_std = torch.clamp(
            log_std, min=self.log_std_min, max=self.log_std_max,
        )
        return mean, log_std

    def deterministic(
        self, slot_state: torch.Tensor, slot_goal: torch.Tensor,
        *, clamp: tuple[float, float] | None = (-1.0, 1.0),
    ) -> torch.Tensor:
        mean, _ = self(slot_state, slot_goal)
        if clamp is not None:
            mean = mean.clamp(*clamp)
        return mean


# ─────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────


def bc_gaussian_loss(
    policy: ContinuousPolicyHead, encoder: SlotStateEncoder,
    s: torch.Tensor, g: torch.Tensor, a_star: torch.Tensor,
) -> torch.Tensor:
    """Gaussian NLL on ``a_star`` under the goal-conditioned policy.

    ``a_star`` is the oracle continuous action. Returns mean NLL
    over the batch. Equivalent to the F64 ``bc_loss`` (cross-
    entropy on discrete actions) but for continuous-action
    policies with a Gaussian likelihood.
    """
    slot_s = encoder(s)
    slot_g = encoder(g)
    mean, log_std = policy(slot_s, slot_g)
    var = (2 * log_std).exp()
    return ((a_star - mean) ** 2 / (2 * var) + log_std).mean()


def continuous_transition_loss(
    transition: ContinuousTransitionHead, encoder: SlotStateEncoder,
    s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy on next-state index given (state, continuous
    action). Predicted next-slot is dot-producted with the full
    slot table; argmax must equal the true next-state index.

    Identical to F64's ``transition_loss`` except the action is a
    real-valued scalar passed through ``ContinuousActionRoPE``
    inside the transition head, instead of a discrete index
    looked up in an embedding table.
    """
    slot_s = encoder(s)
    slot_next_pred = transition(slot_s, a)
    all_slots = encoder.all_slots()
    logits = slot_next_pred @ all_slots.t()
    return F.cross_entropy(logits, s_next)
