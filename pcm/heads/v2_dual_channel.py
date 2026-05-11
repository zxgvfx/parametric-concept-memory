"""pcm.heads.v2_dual_channel — public PCM v2 muscle heads.

Three public heads built on top of :mod:`pcm.dual_channel`:

* :class:`DualChannelPairHead` — the v2 archetype for any
  pair-input task. Combines:
    1. translation-invariant slot-attention path on
       ``slot_b - slot_a`` (avoids the v1 fc1 pair-fingerprint
       memorisation failure documented in F40 / S6);
    2. attribute-difference path with a 2-layer ReLU MLP (the
       linear single-layer that v3-MVP started with cannot
       resolve direction classes that are non-separable in
       ``attr_a - attr_b`` space; see PCM_V2 §6 D4 ablation);
    3. (optional) :class:`pcm.dual_channel.RelativePositionEmbedding`
       lookup keyed on the caller-supplied displacement
       (the V3-RPE lever from F44/F48 that saturates
       OOD on space, colour, phoneme, and bounded number);
    4. a gate (`fixed` / `schedule` / `learned`) controlling
       how much of the slot-attention path's logits are added
       on top of the RPE / attribute logits — see F45 / F46 for
       the empirical justification.

* :class:`SlotIdentityAuxHead` — the v2 LastDigit/RowIndex
  analogue. Takes a slot row only, predicts slot identity. Used
  to drive the slot facet to learn cell identity in parallel
  with whatever pair task the main head is running.

Both heads accept either pre-collapsed tensors or
``(cg, concept_ids)`` and call
:func:`pcm.dual_channel.collapse_dual_channel` themselves; the
former is more flexible for manual mini-batching, the latter
matches the v1 head ergonomics.

The :class:`DualChannelPairHead` API is deliberately
domain-agnostic — the same class is used for spatial direction
(5-class), numeric difference (2N-1 class), cyclic colour shift
(N class), or phonetic feature delta (joint discrete class) by
varying the ``rpe_ranges`` argument.

See ``docs/PCM_V2_DUAL_CHANNEL_DESIGN.md`` and
``docs/SHORT_REPORT_2026_S1_S6.md §V3-RPE`` for the full
empirical justification.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dual_channel import (
    RelativePositionEmbedding,
    collapse_dual_channel,
)


__all__ = [
    "DualChannelPairHead",
    "SlotIdentityAuxHead",
]


class DualChannelPairHead(nn.Module):
    """v2 pair-input head with translation-invariant slot path,
    attribute-difference MLP path, and optional RPE path.

    Args:
        n_classes: number of output direction / difference classes.
        slot_dim, attr_dim: per-facet dim of the dual-channel pair.
        hidden: MLP hidden width.
        rpe_ranges: when not ``None``, allocates an internal RPE
            table over these displacement axes. Caller supplies
            integer delta tensors at forward time. When ``None``,
            the head ignores RPE entirely (slot+attr only).
        rpe_embed_dim: per-displacement RPE embedding dim.
        gate_mode: ``"fixed"``, ``"schedule"``, or ``"learned"``.
            See PCM_V2 §6.A1/A2 for the empirical decision tree.
        attn_logit_scale: only used with ``gate_mode="fixed"``;
            scales the slot-attention logits before sum.
        gate_init_logit: only used with ``gate_mode="learned"``;
            initial gate logit (default 4.0 → sigmoid≈0.98).
        use_attr_path: include the attr-difference MLP path. Set
            to False to test RPE-only or slot-only ablations.
        use_slot_path: include the slot-attention path.

    Notes:
        At least one of (slot, attr, RPE) must be enabled. Gating
        only affects the slot path; the attr and RPE paths are
        always used at full weight if their flag is on.
    """

    def __init__(
        self,
        n_classes: int,
        *,
        slot_dim: int,
        attr_dim: int,
        hidden: int = 64,
        rpe_ranges: list[tuple[int, int]] | None = None,
        rpe_embed_dim: int = 32,
        gate_mode: str = "fixed",
        attn_logit_scale: float = 1.0,
        gate_init_logit: float = 4.0,
        use_attr_path: bool = True,
        use_slot_path: bool = True,
    ) -> None:
        super().__init__()
        if not (use_attr_path or use_slot_path or rpe_ranges is not None):
            raise ValueError(
                "DualChannelPairHead must enable at least one of "
                "{attr, slot, RPE} paths"
            )
        if gate_mode not in {"fixed", "schedule", "learned"}:
            raise ValueError(f"unknown gate_mode {gate_mode!r}")
        self.n_classes = int(n_classes)
        self.slot_dim = int(slot_dim)
        self.attr_dim = int(attr_dim)
        self.hidden = int(hidden)
        self.use_attr_path = use_attr_path
        self.use_slot_path = use_slot_path
        self.gate_mode = gate_mode
        self.attn_logit_scale = float(attn_logit_scale)
        self._schedule_lambda = 1.0
        if gate_mode == "learned":
            self.attn_gate_logit = nn.Parameter(
                torch.tensor(float(gate_init_logit))
            )

        if use_attr_path:
            self.attr_diff = nn.Sequential(
                nn.Linear(attr_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_classes),
            )

        if use_slot_path:
            self.q_proj = nn.Linear(slot_dim, hidden, bias=False)
            self.k_proj = nn.Linear(slot_dim, hidden, bias=False)
            self.v_proj = nn.Linear(slot_dim, hidden, bias=False)
            self.slot_out = nn.Linear(hidden, n_classes)

        if rpe_ranges is not None:
            self.rpe = RelativePositionEmbedding(rpe_ranges, rpe_embed_dim)
            self.rpe_classifier = nn.Sequential(
                nn.Linear(rpe_embed_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_classes),
            )
        else:
            self.rpe = None

    # -- gate helpers -------------------------------------------------

    def set_progress(self, progress: float) -> None:
        """Schedule mode only: ``progress ∈ [0, 1]`` decays λ
        linearly 1.0 → 0.0 over the first half of training."""
        self._schedule_lambda = max(0.0, 1.0 - 2.0 * float(progress))

    def current_attn_lambda(self) -> torch.Tensor | float:
        if self.gate_mode == "learned":
            return torch.sigmoid(self.attn_gate_logit)
        if self.gate_mode == "schedule":
            return self._schedule_lambda
        return self.attn_logit_scale

    def gate_l1(self) -> torch.Tensor:
        """Return L1-style penalty on the gate. Caller adds with
        weight β > 0 to its training loss. Zero in non-learned
        modes (no parameter to penalise)."""
        if self.gate_mode == "learned":
            return torch.sigmoid(self.attn_gate_logit)
        if self.gate_mode == "schedule":
            return torch.tensor(self._schedule_lambda)
        return torch.tensor(self.attn_logit_scale)

    # -- forward ------------------------------------------------------

    def forward(
        self,
        slot_a: torch.Tensor | None = None,
        slot_b: torch.Tensor | None = None,
        attr_a: torch.Tensor | None = None,
        attr_b: torch.Tensor | None = None,
        deltas: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """All paths additive in logit space. Caller passes
        whichever arguments correspond to enabled paths."""
        logits: torch.Tensor | None = None

        if self.rpe is not None:
            if deltas is None:
                raise ValueError(
                    "rpe_ranges set but no deltas passed to forward()"
                )
            rpe_emb = self.rpe(*deltas)
            rpe_logits = self.rpe_classifier(rpe_emb)
            logits = rpe_logits if logits is None else logits + rpe_logits

        if self.use_attr_path:
            if attr_a is None or attr_b is None:
                raise ValueError("attr path enabled but attr_a/attr_b missing")
            attr_logits = self.attr_diff(attr_a - attr_b)
            logits = attr_logits if logits is None else logits + attr_logits

        if self.use_slot_path:
            if slot_a is None or slot_b is None:
                raise ValueError("slot path enabled but slot_a/slot_b missing")
            slot_diff = slot_b - slot_a
            q = self.q_proj(slot_diff)
            k = self.k_proj(slot_diff)
            v = self.v_proj(slot_diff)
            scale = 1.0 / float(q.shape[-1]) ** 0.5
            score = (q * k).sum(dim=-1, keepdim=True) * scale
            attn = torch.tanh(score)
            slot_logits = self.slot_out(attn * v)
            lam = self.current_attn_lambda()
            if isinstance(lam, torch.Tensor):
                slot_contrib = lam * slot_logits
            else:
                slot_contrib = float(lam) * slot_logits
            logits = (
                slot_contrib if logits is None else logits + slot_contrib
            )

        assert logits is not None
        return logits


class SlotIdentityAuxHead(nn.Module):
    """v2 single-input identity head used as a Tier-D auxiliary
    that drives the slot facet to learn unique concept identity.

    Equivalent in spirit to v1's ``LastDigitHead`` (number) or
    ``RowIndexHead`` (space) — exposes a supervised signal on
    the slot facet at every concept in the inventory, regardless
    of whether that concept appears in the main pair task.
    """

    def __init__(
        self,
        slot_dim: int,
        n_concepts: int,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(slot_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_concepts)

    def forward(self, slot_rows: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(slot_rows)))


# ---------------------------------------------------------------------------
# Convenience: collapse + forward in one shot.
# ---------------------------------------------------------------------------


def pair_collapse_and_forward(
    head: "DualChannelPairHead",
    cg,
    *,
    base_facet: str,
    ids_a: list[str],
    ids_b: list[str],
    slot_shape: tuple[int, ...],
    attr_shape: tuple[int, ...],
    deltas: tuple[torch.Tensor, ...] | None = None,
    tick: int = 0,
    device=None,
) -> torch.Tensor:
    """Collapse both ids on the dual-channel facet then forward
    ``head``. Drop-in for the per-domain ``head(ids_a, ids_b, cg, ...)``
    pattern from v1 muscles."""
    slot_a, attr_a = collapse_dual_channel(
        cg, caller=f"{base_facet}-pair-a", base_facet=base_facet,
        concept_ids=ids_a,
        slot_shape=slot_shape, attr_shape=attr_shape,
        tick=tick, device=device,
    )
    slot_b, attr_b = collapse_dual_channel(
        cg, caller=f"{base_facet}-pair-b", base_facet=base_facet,
        concept_ids=ids_b,
        slot_shape=slot_shape, attr_shape=attr_shape,
        tick=tick + 1, device=device,
    )
    return head(slot_a, slot_b, attr_a, attr_b, deltas=deltas)
