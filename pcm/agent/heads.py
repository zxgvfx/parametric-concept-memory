"""pcm.agent.heads — Agent base heads (transition / policy / value).

The architectural claim of v6.1 is that the F62 universal-
operator combiner ``(slot, Δ) → next_slot`` is the right
primitive for an MDP transition function ``(state, action) →
next_state``: action plays the role of displacement, the
combiner plays the role of the transition. The policy then
inverts the transition (given current and goal slots, pick the
action that bridges them).

We *literally* reuse ``UniversalCombiner`` from
``experiments.cross_discipline_operator`` for the transition
head. The policy and value heads are small MLPs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "SlotStateEncoder",
    "TransitionHead",
    "PolicyHead",
    "ValueHead",
    "bc_loss",
    "transition_loss",
]


# ─────────────────────────────────────────────────────────────────
# State encoder
# ─────────────────────────────────────────────────────────────────


class SlotStateEncoder(nn.Module):
    """Map a discrete environment state index ``s ∈ {0..N-1}`` to
    a slot vector ``ℝ^D``.

    For the F64 PoC this is a plain ``nn.Embedding`` because the
    environment state is a single integer; later v6 extensions
    (continuous state, image observations, …) will swap this for
    a richer encoder while keeping the rest of the agent
    architecture unchanged.
    """

    def __init__(self, n_states: int, dim: int) -> None:
        super().__init__()
        self.n_states = n_states
        self.dim = dim
        self.slot = nn.Embedding(n_states, dim)
        nn.init.normal_(self.slot.weight, std=1.0)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.slot(s)

    def all_slots(self) -> torch.Tensor:
        return self.slot.weight


# ─────────────────────────────────────────────────────────────────
# Transition head — F62 UniversalCombiner with action embedding
# ─────────────────────────────────────────────────────────────────


class TransitionHead(nn.Module):
    """``(slot_state, action_idx) → slot_next``.

    Architecture: action embedding ``E[a] ∈ ℝ^D`` plays the role
    of F62's RPE displacement; a 2-layer MLP residual combiner
    (identical to F62 ``UniversalCombiner``) maps
    ``(slot, E[a])`` to the predicted next-slot vector.

    Output is the *predicted next-state slot*; combined with a
    classifier over all slots this gives the next-state index
    distribution (see :func:`transition_loss`).
    """

    def __init__(
        self, dim: int, n_actions: int, hidden: int = 128,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_actions = n_actions
        self.action_emb = nn.Embedding(n_actions, dim)
        # Smaller-scale init so identity dominates initially —
        # mirrors F62's ``_smaller_rpe_init``.
        nn.init.normal_(self.action_emb.weight, std=0.3)
        self.combiner = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self, slot_state: torch.Tensor, action_idx: torch.Tensor,
    ) -> torch.Tensor:
        a_emb = self.action_emb(action_idx)
        delta = self.combiner(torch.cat([slot_state, a_emb], dim=-1))
        return slot_state + delta


# ─────────────────────────────────────────────────────────────────
# Policy head — goal-conditioned action selector
# ─────────────────────────────────────────────────────────────────


class PolicyHead(nn.Module):
    """``(slot_state, slot_goal) → action_logits``.

    Implementation: a 2-layer MLP on the concatenation of the
    current and goal slot vectors. We do not assume any specific
    relation between (state, goal) and the optimal action; the
    policy must learn it from oracle behaviour.

    For multi-step planning, the policy is invoked iteratively at
    each env step with ``slot_state`` updated to the new
    observation; see ``orchestrator.rollout``.
    """

    def __init__(
        self, dim: int, n_actions: int, hidden: int = 128,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(
        self, slot_state: torch.Tensor, slot_goal: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([slot_state, slot_goal], dim=-1))


# ─────────────────────────────────────────────────────────────────
# Value head — optional, for v6.3 RL closure
# ─────────────────────────────────────────────────────────────────


class ValueHead(nn.Module):
    """``(slot_state, slot_goal) → scalar expected return``.

    Not used by the F64 BC PoC (BC has no reward signal), but
    provided for v6.3 onwards where we close the RL loop with
    trial-based reward learning. Placed in the public API so the
    v6.1 → v6.3 upgrade is purely additive.
    """

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, slot_state: torch.Tensor, slot_goal: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([slot_state, slot_goal], dim=-1)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────


def bc_loss(
    policy: PolicyHead, encoder: SlotStateEncoder,
    s: torch.Tensor, g: torch.Tensor, a_star: torch.Tensor,
) -> torch.Tensor:
    """Behavioural-cloning cross-entropy on
    ``(state, goal, oracle_action)`` triples."""
    slot_s = encoder(s)
    slot_g = encoder(g)
    logits = policy(slot_s, slot_g)
    return F.cross_entropy(logits, a_star)


def transition_loss(
    transition: TransitionHead, encoder: SlotStateEncoder,
    s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor,
) -> torch.Tensor:
    """Transition cross-entropy.

    Predicted next-slot is dot-producted with the full slot table
    to give logits over next-state indices, then cross-entropy
    against the true next-state index. Identical to F62's
    operator-classifier loss with action playing the role of Δ.
    """
    slot_s = encoder(s)
    slot_next_pred = transition(slot_s, a)
    all_slots = encoder.all_slots()
    logits = slot_next_pred @ all_slots.t()
    return F.cross_entropy(logits, s_next)
