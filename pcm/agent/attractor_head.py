"""pcm.agent.attractor_head — v6.6 outcome-distribution head.

F61 ``AttractorHead`` predicted *distributional* outcomes
(escape body, energy partition, mean final state, log-std) for
chaotic 3-body systems where pointwise trajectory prediction
breaks down past the Lyapunov horizon. F73 brings the same
architectural pattern to the agent stack: instead of rolling out
a trajectory step-by-step (cook / System 2), predict the
*outcome distribution* of the trajectory in one shot (System 1
attractor lookup).

Specifically, given a ``(state_slot, goal_slot)`` pair, the
``AgentAttractorHead`` outputs:

* ``p_success`` ∈ [0, 1] — predicted probability of reaching
  the goal within a fixed budget under the current policy.
* ``log_expected_steps`` ∈ ℝ — log of the expected number of
  steps to reach the goal (Gaussian-style regression head).

The head is trained on supervised tuples
``(state, goal, did_reach, n_steps)`` collected by running the
agent's current policy on a held-out distribution of episodes.
At inference time the head provides O(1) outcome estimates that
the hybrid dispatcher (System-1 retrieval vs System-2 cook)
uses to route queries: high-``p_success`` episodes go to the
direct policy (cheap), low-``p_success`` episodes invoke MPC
planning (expensive).

This is the cleanest agent-side analogue of F61's
``HybridPhysicsDispatcher`` — the same operator pattern
(attractor head for one-shot outcome + cook for step-by-step
state) applied to MDP transitions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "AgentAttractorHead",
    "AttractorTargets",
    "agent_attractor_loss",
]


class AgentAttractorHead(nn.Module):
    """``(slot_state, slot_goal) → (p_success_logit, log_expected_steps)``.

    Two-output one-shot outcome predictor mirroring F61's
    ``AttractorHead`` design. Trunk is a 2-layer MLP on the
    concatenation of state and goal slots; two heads on top:

    * **success head** outputs a scalar logit; ``sigmoid`` gives
      ``p_success``.
    * **steps head** outputs ``log_expected_steps`` — Gaussian
      regression on ``log(n_steps + 1)`` so the prediction is
      strictly positive and handles the long-tail of step counts
      gracefully.
    """

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.trunk = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.success_head = nn.Linear(hidden, 1)
        self.steps_head = nn.Linear(hidden, 1)

    def forward(
        self, slot_state: torch.Tensor, slot_goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(torch.cat([slot_state, slot_goal], dim=-1))
        return self.success_head(h).squeeze(-1), self.steps_head(h).squeeze(-1)

    def predict(
        self, slot_state: torch.Tensor, slot_goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(p_success, expected_steps)`` (both already
        post-activated)."""
        success_logit, log_steps = self(slot_state, slot_goal)
        return torch.sigmoid(success_logit), torch.exp(log_steps)


class AttractorTargets:
    """Holder for one supervision pair on the attractor head."""

    def __init__(
        self, success: torch.Tensor, n_steps: torch.Tensor,
    ) -> None:
        self.success = success      # (B,) bool/float
        self.n_steps = n_steps      # (B,) long, count of steps


def agent_attractor_loss(
    head: AgentAttractorHead,
    slot_state: torch.Tensor,
    slot_goal: torch.Tensor,
    targets: AttractorTargets,
    *, steps_weight: float = 0.3,
) -> tuple[torch.Tensor, dict]:
    """Combined loss: BCE on success + MSE on log-expected steps.

    Args:
        slot_state / slot_goal: encoded states.
        targets: ``(success, n_steps)`` — success in ``{0, 1}``,
            ``n_steps`` is the actual episode length (≥ 1).
        steps_weight: weight on the steps-regression term.
    """
    success_logit, log_steps_pred = head(slot_state, slot_goal)
    bce = F.binary_cross_entropy_with_logits(
        success_logit, targets.success.float(),
    )
    log_steps_target = torch.log(
        targets.n_steps.float().clamp(min=1.0)
    )
    mse = F.mse_loss(log_steps_pred, log_steps_target)
    total = bce + steps_weight * mse
    return total, {
        "bce": float(bce.item()),
        "mse_log_steps": float(mse.item()),
    }
