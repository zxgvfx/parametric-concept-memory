"""pcm.agent.heads_mixed — v6.2-followup mixed-arity tool calls (F66).

v6.1 (F64) shipped discrete actions; v6.2 (F65) shipped continuous
actions. Real agent tool calls are *heterogeneous*: some tools take
typed args (e.g. ``PUSH(value)``), some are nullary
(``ADD``, ``POP``). F66 demonstrates that the F62
``UniversalCombiner`` handles this case too, by *summing* a discrete
tool embedding (F64-style) with a RoPE-encoded continuous argument
(F65-style) and feeding the sum through the unchanged combiner.

Architectural primitive: **action = tool_emb(tool_id) +
arg_rope(arg)**. For nullary tools, the caller supplies arg=0
which encodes to a fixed RoPE vector; the combiner can ignore it
because the tool_emb is informative enough. For typed-arg tools,
the arg-RoPE provides the continuous parameter.

This is the smallest possible test of the v6 thesis: a single
``UniversalCombiner`` can serve *any* action interface (discrete,
continuous, heterogeneous) by composing F64 and F65 encoders in the
input.

Public API::

    ToolEmbedding                 — F64-style discrete tool encoder
    MixedActionTransitionHead     — (slot, tool_id, arg) → next slot
    MixedPolicyHead               — (slot, goal_slot) → (tool_logits,
                                                          arg_mean,
                                                          arg_log_std)
    mixed_bc_loss                 — BC for typed-arg actions
    mixed_transition_loss         — transition CE under mixed actions
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import SlotStateEncoder
from .heads_continuous import ContinuousActionRoPE


__all__ = [
    "ToolEmbedding",
    "MixedActionTransitionHead",
    "MixedPolicyHead",
    "MultiArgActionTransitionHead",
    "MultiArgPolicyHead",
    "mixed_bc_loss",
    "mixed_transition_loss",
    "multi_arg_bc_loss",
    "multi_arg_transition_loss",
]


# ─────────────────────────────────────────────────────────────────
# Discrete tool encoder (mirrors F64 action embedding)
# ─────────────────────────────────────────────────────────────────


class ToolEmbedding(nn.Module):
    """Discrete tool-ID → embedding vector.

    Identical in spirit to the action-embedding lookup in F64's
    ``TransitionHead``, but factored out so it can be summed with
    the F65 ``ContinuousActionRoPE`` for typed-arg actions.
    """

    def __init__(self, n_tools: int, dim: int) -> None:
        super().__init__()
        self.n_tools = n_tools
        self.dim = dim
        self.emb = nn.Embedding(n_tools, dim)
        nn.init.normal_(self.emb.weight, std=0.3)

    def forward(self, tool_id: torch.Tensor) -> torch.Tensor:
        return self.emb(tool_id)


# ─────────────────────────────────────────────────────────────────
# Mixed-action transition head
# ─────────────────────────────────────────────────────────────────


class MixedActionTransitionHead(nn.Module):
    """``(slot_state, tool_id, arg) → predicted slot_next``.

    The action embedding is the **sum** of a discrete tool
    embedding and a RoPE-encoded continuous argument. For nullary
    tools, the caller passes ``arg=0`` and the arg-RoPE produces a
    fixed vector that the combiner can route through with the tool
    embedding doing the discriminative work.

    The combiner is the F62 ``UniversalCombiner`` — exactly the
    same module as F64 and F65 used. v6.1 + v6.2 (this commit)
    introduces no new combiner architecture.
    """

    def __init__(
        self, dim: int, n_tools: int,
        n_freqs: int = 8, hidden: int = 128,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_tools = n_tools
        self.tool_emb = ToolEmbedding(n_tools, dim)
        self.arg_rope = ContinuousActionRoPE(
            embed_dim=dim, n_freqs=n_freqs,
        )
        self.combiner = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self, slot_state: torch.Tensor,
        tool_id: torch.Tensor, arg: torch.Tensor,
    ) -> torch.Tensor:
        action_emb = self.tool_emb(tool_id) + self.arg_rope(arg)
        delta = self.combiner(torch.cat([slot_state, action_emb], dim=-1))
        return slot_state + delta


# ─────────────────────────────────────────────────────────────────
# Mixed-action policy head
# ─────────────────────────────────────────────────────────────────


class MixedPolicyHead(nn.Module):
    """``(slot_state, slot_goal) → (tool_logits, arg_mean, arg_log_std)``.

    Outputs three things per state-goal pair:

    * ``tool_logits`` of shape ``(B, n_tools)`` — categorical over
      which tool to call.
    * ``arg_mean`` of shape ``(B,)`` — mean of the continuous arg
      distribution (used only if the selected tool takes an arg).
    * ``arg_log_std`` of shape ``(B,)`` — log-std of the arg
      distribution, clamped to a safe range.

    The arg head outputs unconditionally; nullary tools' arg
    predictions are simply ignored downstream. This keeps the
    head shape-stable across tool types.
    """

    def __init__(
        self, dim: int, n_tools: int, hidden: int = 128,
        log_std_min: float = -3.0, log_std_max: float = 0.5,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_tools = n_tools
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.trunk = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.tool_head = nn.Linear(hidden, n_tools)
        self.arg_mean = nn.Linear(hidden, 1)
        self.arg_log_std = nn.Linear(hidden, 1)

    def forward(
        self, slot_state: torch.Tensor, slot_goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(torch.cat([slot_state, slot_goal], dim=-1))
        tool_logits = self.tool_head(h)
        arg_mean = self.arg_mean(h).squeeze(-1)
        arg_log_std = self.arg_log_std(h).squeeze(-1)
        arg_log_std = torch.clamp(
            arg_log_std, min=self.log_std_min, max=self.log_std_max,
        )
        return tool_logits, arg_mean, arg_log_std

    def deterministic(
        self, slot_state: torch.Tensor, slot_goal: torch.Tensor,
        *, arg_clamp: tuple[float, float] | None = (-1.0, 1.0),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tool_logits, arg_mean, _ = self(slot_state, slot_goal)
        tool = tool_logits.argmax(dim=-1)
        if arg_clamp is not None:
            arg_mean = arg_mean.clamp(*arg_clamp)
        return tool, arg_mean


# ─────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────


def mixed_bc_loss(
    policy: MixedPolicyHead, encoder: SlotStateEncoder,
    s: torch.Tensor, g: torch.Tensor,
    tool_star: torch.Tensor, arg_star: torch.Tensor,
    arg_mask: torch.Tensor | None = None,
    arg_weight: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Combined BC loss for mixed-arity action policies.

    * ``tool_star``: oracle tool index, ``(B,) long``.
    * ``arg_star``: oracle continuous arg, ``(B,) float`` (set to
      any value for nullary tools — masked out below).
    * ``arg_mask``: ``(B,) bool``, True where the tool consumes an
      arg. Nullary tools should have ``arg_mask=False`` and their
      arg loss is dropped from the gradient.
    * ``arg_weight``: scalar weight on the arg-NLL term.

    Returns ``(total_loss, diagnostics)`` where diagnostics holds
    per-term losses for logging.
    """
    slot_s = encoder(s)
    slot_g = encoder(g)
    tool_logits, arg_mean, arg_log_std = policy(slot_s, slot_g)

    tool_loss = F.cross_entropy(tool_logits, tool_star)

    if arg_mask is None:
        arg_mask = torch.ones_like(tool_star, dtype=torch.bool)
    if arg_mask.any():
        var = (2 * arg_log_std[arg_mask]).exp()
        arg_nll = (
            (arg_star[arg_mask] - arg_mean[arg_mask]) ** 2
            / (2 * var)
            + arg_log_std[arg_mask]
        ).mean()
    else:
        arg_nll = arg_mean.new_zeros(())

    total = tool_loss + arg_weight * arg_nll
    diag = {
        "tool_loss": float(tool_loss.item()),
        "arg_nll": float(arg_nll.item()),
        "n_arg_samples": int(arg_mask.sum().item()),
    }
    return total, diag


def mixed_transition_loss(
    transition: MixedActionTransitionHead, encoder: SlotStateEncoder,
    s: torch.Tensor, tool_id: torch.Tensor, arg: torch.Tensor,
    s_next: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy on next-state index for mixed-arity action.

    Predicted next-slot is dot-producted with the full slot table;
    argmax must equal the true next-state index. Identical
    construction to F64's ``transition_loss`` and F65's
    ``continuous_transition_loss`` — only the action encoding
    changes (sum of discrete tool emb + continuous arg RoPE).
    """
    slot_s = encoder(s)
    slot_next_pred = transition(slot_s, tool_id, arg)
    all_slots = encoder.all_slots()
    logits = slot_next_pred @ all_slots.t()
    return F.cross_entropy(logits, s_next)


# ─────────────────────────────────────────────────────────────────
# v6.2-followup² (F70) — multi-arg tools
# ─────────────────────────────────────────────────────────────────


class MultiArgActionTransitionHead(nn.Module):
    """``(slot_state, tool_id, args[arg_dim]) → predicted slot_next``.

    Generalises :class:`MixedActionTransitionHead` to support tools
    with ``arg_dim`` continuous arguments. The action embedding is
    the *sum* of the discrete tool embedding and one
    :class:`ContinuousActionRoPE` per arg slot::

        action_emb = tool_emb(tool_id) + Σ_k arg_rope_k(args[..., k])

    Per-slot RoPE encoders have independent learnable frequencies
    so the model can learn distinct encodings per argument
    position (no positional collapse). The combiner remains the
    F62 ``UniversalCombiner`` — no architectural change beyond the
    input-side sum.
    """

    def __init__(
        self, dim: int, n_tools: int, arg_dim: int,
        n_freqs: int = 8, hidden: int = 128,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_tools = n_tools
        self.arg_dim = arg_dim
        self.tool_emb = ToolEmbedding(n_tools, dim)
        self.arg_ropes = nn.ModuleList([
            ContinuousActionRoPE(embed_dim=dim, n_freqs=n_freqs)
            for _ in range(arg_dim)
        ])
        self.combiner = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self, slot_state: torch.Tensor,
        tool_id: torch.Tensor, args: torch.Tensor,
    ) -> torch.Tensor:
        """``args`` of shape ``(B, arg_dim)``."""
        action_emb = self.tool_emb(tool_id)
        for k, rope in enumerate(self.arg_ropes):
            action_emb = action_emb + rope(args[..., k])
        delta = self.combiner(torch.cat([slot_state, action_emb], dim=-1))
        return slot_state + delta


class MultiArgPolicyHead(nn.Module):
    """``(slot_state, slot_goal) → (tool_logits, arg_means, arg_log_stds)``.

    ``arg_means`` and ``arg_log_stds`` have shape ``(B, arg_dim)``;
    nullary tools' arg predictions are masked out at the loss /
    rollout level by the caller.
    """

    def __init__(
        self, dim: int, n_tools: int, arg_dim: int, hidden: int = 128,
        log_std_min: float = -3.0, log_std_max: float = 0.5,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_tools = n_tools
        self.arg_dim = arg_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.trunk = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.tool_head = nn.Linear(hidden, n_tools)
        self.arg_mean = nn.Linear(hidden, arg_dim)
        self.arg_log_std = nn.Linear(hidden, arg_dim)

    def forward(
        self, slot_state: torch.Tensor, slot_goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(torch.cat([slot_state, slot_goal], dim=-1))
        tool_logits = self.tool_head(h)
        arg_mean = self.arg_mean(h)
        arg_log_std = torch.clamp(
            self.arg_log_std(h),
            min=self.log_std_min, max=self.log_std_max,
        )
        return tool_logits, arg_mean, arg_log_std

    def deterministic(
        self, slot_state: torch.Tensor, slot_goal: torch.Tensor,
        *, arg_clamp: tuple[float, float] | None = (-1.0, 1.0),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tool_logits, arg_mean, _ = self(slot_state, slot_goal)
        tool = tool_logits.argmax(dim=-1)
        if arg_clamp is not None:
            arg_mean = arg_mean.clamp(*arg_clamp)
        return tool, arg_mean


def multi_arg_bc_loss(
    policy: MultiArgPolicyHead, encoder: SlotStateEncoder,
    s: torch.Tensor, g: torch.Tensor,
    tool_star: torch.Tensor, args_star: torch.Tensor,
    arg_mask: torch.Tensor,
    arg_weight: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """BC loss for multi-arg policies.

    Args:
        tool_star: ``(B,)`` long — oracle tool indices.
        args_star: ``(B, arg_dim)`` float — oracle arg vectors.
        arg_mask: ``(B, arg_dim)`` bool — True where the tool
            actually consumes that arg slot. Per-slot mask
            (not just per-tool) lets us train tools with
            different arities under a single loss.
    """
    slot_s = encoder(s)
    slot_g = encoder(g)
    tool_logits, arg_mean, arg_log_std = policy(slot_s, slot_g)
    tool_loss = F.cross_entropy(tool_logits, tool_star)
    if arg_mask.any():
        m = arg_mask.float()
        var = (2 * arg_log_std).exp()
        elem = (
            (args_star - arg_mean) ** 2 / (2 * var) + arg_log_std
        )
        denom = m.sum().clamp(min=1.0)
        arg_nll = (elem * m).sum() / denom
    else:
        arg_nll = tool_loss.new_zeros(())
    total = tool_loss + arg_weight * arg_nll
    diag = {
        "tool_loss": float(tool_loss.item()),
        "arg_nll": float(arg_nll.item()),
        "n_arg_slots": int(arg_mask.sum().item()),
    }
    return total, diag


def multi_arg_transition_loss(
    transition: MultiArgActionTransitionHead, encoder: SlotStateEncoder,
    s: torch.Tensor, tool_id: torch.Tensor, args: torch.Tensor,
    s_next: torch.Tensor,
) -> torch.Tensor:
    slot_s = encoder(s)
    slot_next_pred = transition(slot_s, tool_id, args)
    all_slots = encoder.all_slots()
    logits = slot_next_pred @ all_slots.t()
    return F.cross_entropy(logits, s_next)
