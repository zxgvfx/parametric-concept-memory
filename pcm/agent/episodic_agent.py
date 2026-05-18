"""pcm.agent.episodic_agent — F78 episodic-grounded agent.

Integrates F75 :class:`EpisodicBuffer` into F73 hybrid agent
(F70 multi-arg transition + policy + F61-style ``AgentAttractorHead``).

The result is a **three-tier dispatcher**:

* **RECALL** (System 1.5) — episodic buffer lookup. If we've
  succeeded on a similar ``(state, goal)`` query before, copy
  the stored action. Cost: O(buffer_size · slot_dim) cosine
  similarity scan. Much cheaper than MPC.
* **S1** — direct policy argmax (when attractor predicts high
  ``p_success``). The F73 baseline.
* **S2** — MPC planner over the world model (when attractor
  predicts low ``p_success``). The F73 fallback.

Cognitive parallel: humans don't re-plan from scratch every
time. They first **recall** ("have I done this before? what
worked?"). Only if nothing matches do they engage System-1
intuition (S1) or System-2 deliberate planning (S2).

The architecture is **additive over F73** — disable the
buffer via ``recall_threshold=1.01`` (impossible match) and
the agent behaves *exactly* like F73's hybrid dispatcher
(M5 ablation invariant).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ..episodic import EpisodicBuffer
from .attractor_head import AgentAttractorHead
from .heads import SlotStateEncoder
from .heads_mixed import MultiArgActionTransitionHead, MultiArgPolicyHead
from .planner import plan_multi_arg_mpc


__all__ = [
    "EpisodicAgentDecision",
    "EpisodicAgent",
]


# ─────────────────────────────────────────────────────────────────
# Decision record
# ─────────────────────────────────────────────────────────────────


@dataclass
class EpisodicAgentDecision:
    """Trace of one decision by :class:`EpisodicAgent`."""

    route: str  # "RECALL", "S1", or "S2"
    tool_id: int
    args: tuple[float, ...]
    p_success: float | None = None
    recall_sim: float | None = None
    recall_n_steps: int | None = None


# ─────────────────────────────────────────────────────────────────
# EpisodicAgent
# ─────────────────────────────────────────────────────────────────


class EpisodicAgent:
    """F75 + F73 integrated agent.

    Stores successful past ``(state, goal, tool, args, n_steps)``
    tuples in an :class:`EpisodicBuffer` keyed by
    ``concat(slot_state, slot_goal)``. At decision time, tries
    a similarity match first; falls through to F73 hybrid
    dispatcher otherwise.

    Args:
        encoder, transition, policy, attractor: the F73 agent
            stack (same modules as ``hybrid_multi_arg_step``).
        buffer_capacity: FIFO ring size for episodic memory.
        recall_threshold: cosine threshold for accepting a
            buffer match as the answer. Default 0.95.
        p_success_threshold: when no recall, the F73
            attractor-routed S1/S2 split threshold.
        mpc_n_candidates, mpc_rollout_steps: MPC planner config.
        salience_mode: how to score buffer writes.
            ``"inverse_steps"`` (default): salience = 1/n_steps;
            ``"none"``: salience = 0 (all equal).
    """

    def __init__(
        self,
        encoder: SlotStateEncoder,
        transition: MultiArgActionTransitionHead,
        policy: MultiArgPolicyHead,
        attractor: AgentAttractorHead,
        *,
        buffer_capacity: int = 200,
        recall_threshold: float = 0.95,
        p_success_threshold: float = 0.8,
        mpc_n_candidates: int = 4,
        mpc_rollout_steps: int = 3,
        salience_mode: str = "inverse_steps",
        device: str = "cpu",
    ) -> None:
        self.encoder = encoder
        self.transition = transition
        self.policy = policy
        self.attractor = attractor
        self.buffer = EpisodicBuffer(
            capacity=buffer_capacity,
            slot_dim=2 * encoder.dim,
            device=device,
        )
        self.recall_threshold = recall_threshold
        self.p_success_threshold = p_success_threshold
        self.mpc_n_candidates = mpc_n_candidates
        self.mpc_rollout_steps = mpc_rollout_steps
        self.salience_mode = salience_mode
        self.device = device
        self._episode_counter = 0
        # Counters for diagnostics
        self.n_recalls = 0
        self.n_s1 = 0
        self.n_s2 = 0

    # ─── Joint slot encoding ──────────────────────────────────────

    @torch.no_grad()
    def _joint_slot(
        self, state_idx: int, goal_idx: int,
    ) -> torch.Tensor:
        s = torch.tensor([state_idx], dtype=torch.long, device=self.device)
        g = torch.tensor([goal_idx], dtype=torch.long, device=self.device)
        slot_s = self.encoder(s)[0].detach()
        slot_g = self.encoder(g)[0].detach()
        return torch.cat([slot_s, slot_g], dim=-1)

    # ─── Buffer write (after episode) ─────────────────────────────

    def remember(
        self, state_idx: int, goal_idx: int,
        tool_id: int, args: tuple[float, ...],
        n_steps: int, success: bool,
    ) -> bool:
        """Record a finished episode if it succeeded. Returns
        whether the episode was actually written to the buffer."""
        if not success:
            return False
        slot = self._joint_slot(state_idx, goal_idx)
        if self.salience_mode == "inverse_steps":
            salience = 1.0 / max(int(n_steps), 1)
        elif self.salience_mode == "none":
            salience = 0.0
        else:
            raise ValueError(
                f"unknown salience_mode {self.salience_mode!r}"
            )
        self._episode_counter += 1
        self.buffer.append(
            slot, timestamp=self._episode_counter,
            salience=salience,
            metadata={
                "state_idx": int(state_idx),
                "goal_idx": int(goal_idx),
                "tool_id": int(tool_id),
                "args": tuple(float(a) for a in args),
                "n_steps": int(n_steps),
            },
        )
        return True

    # ─── Decision (RECALL / S1 / S2 dispatcher) ───────────────────

    @torch.no_grad()
    def decide(
        self, state_idx: int, goal_idx: int,
    ) -> EpisodicAgentDecision:
        """Pick the next action under the three-tier dispatcher."""
        slot = self._joint_slot(state_idx, goal_idx)

        # ─── Tier 1: try recall ─────────────────────────────────
        if len(self.buffer) > 0:
            matches = self.buffer.recall_by_similarity(slot, k=1)
            if matches:
                r = matches[0]
                qn = F.normalize(slot.view(1, -1), dim=-1)
                kn = F.normalize(r.slot.view(1, -1), dim=-1)
                cos = float((qn * kn).sum().item())
                if cos >= self.recall_threshold:
                    self.n_recalls += 1
                    return EpisodicAgentDecision(
                        route="RECALL",
                        tool_id=int(r.metadata["tool_id"]),
                        args=tuple(r.metadata["args"]),
                        p_success=None,
                        recall_sim=cos,
                        recall_n_steps=int(r.metadata["n_steps"]),
                    )

        # ─── Tier 2/3: F73 hybrid (attractor-routed S1 vs S2) ───
        s = torch.tensor([state_idx], dtype=torch.long, device=self.device)
        g = torch.tensor([goal_idx], dtype=torch.long, device=self.device)
        slot_s = self.encoder(s)
        slot_g = self.encoder(g)
        p_succ, _ = self.attractor.predict(slot_s, slot_g)
        p_succ_f = float(p_succ.item())

        if p_succ_f >= self.p_success_threshold:
            self.n_s1 += 1
            tool_logits, arg_mean, _ = self.policy(slot_s, slot_g)
            tool = int(tool_logits.argmax(-1).item())
            args = tuple(
                float(arg_mean[0, j].clamp(-1.0, 1.0).item())
                for j in range(self.policy.arg_dim)
            )
            return EpisodicAgentDecision(
                route="S1", tool_id=tool, args=args,
                p_success=p_succ_f,
            )

        self.n_s2 += 1
        tool, args = plan_multi_arg_mpc(
            state_idx, goal_idx, self.encoder, self.transition,
            self.policy,
            n_candidates=self.mpc_n_candidates,
            rollout_steps=self.mpc_rollout_steps,
            device=self.device,
        )
        return EpisodicAgentDecision(
            route="S2", tool_id=tool, args=args,
            p_success=p_succ_f,
        )

    def reset_counters(self) -> None:
        self.n_recalls = 0
        self.n_s1 = 0
        self.n_s2 = 0

    def diagnostics(self) -> dict[str, Any]:
        return {
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer.capacity,
            "n_recalls": self.n_recalls,
            "n_s1": self.n_s1,
            "n_s2": self.n_s2,
        }
