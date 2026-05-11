"""pcm.gate — Tier B: differentiable slot gates for synaptogenesis +
pruning, per the bio-inspired upgrade plan §3.B.

This module adds an **opt-in** gate ``g[facet, slot] in (0, 1)`` to the
dense pool. With gates enabled, the muscle output becomes::

    bias = sigmoid(gate[facet, slot]) * pool[facet, slot]

This makes the lottery-ticket / lazy-init story explicit: every slot is
allocated up front, but its function is *learnt* via the gate. An L0/L1
regulariser pushes unused slots to ``gate -> 0`` (pruning) and used
slots to ``gate -> 1`` (synaptogenesis), reproducing the developmental
biology curve from Frontiers 2025 / bioRxiv 2025.

Key design properties:

- **Default off** (``ConceptGraph.gate_enabled = False``): forward output
  is bit-identical to Tier A, so all paper claims (§4-§6 + §B) hold
  unchanged. This is enforced by ``collapse_batch`` skipping the gate
  multiplication when no gate Parameter exists.
- **G1-G6 invariants**: ``grow_capacity`` extends gate Parameters along
  with the pool; new gate rows get the calibrated init logit so they
  start "undifferentiated" (sigmoid(-2.197) = 0.1).
- **Causal interventions**: ``set_slot_gate`` lets ablation studies set
  ``g[f, slot] -> 0`` reversibly, complementing the Tier-A
  ``ablate(facet)`` zero-fill.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from . import config as _cfg
from .param_bundle import migrate_param_in_optimizer

if TYPE_CHECKING:
    from .concept_graph import ConceptGraph


def _logit_for_prob(p: float) -> float:
    p = max(min(p, 1.0 - 1e-9), 1e-9)
    return float(torch.logit(torch.tensor(p)).item())


def attach_gates(
    cg: "ConceptGraph",
    *,
    init_logit: float | None = None,
) -> None:
    """Enable Tier-B slot gates on ``cg``.

    Idempotent: subsequent calls are no-ops once gates are enabled. New
    facets created via ``collapse`` or ``collapse_batch`` will lazily
    grow a gate Parameter the first time they appear.

    Important ordering: call **before** the first forward (or before
    optimizer construction) so the gates appear in
    ``cg.iter_gate_parameters()``.
    """
    if cg.gate_enabled:
        return
    cg.gate_enabled = True
    cg._gate_init_logit = float(
        init_logit if init_logit is not None else _cfg.GATE_INIT_LOGIT
    )
    # Materialise gates for any facet that already has a pool (e.g. when
    # gates are enabled mid-run after warm-up). We start every existing
    # row at the calibrated low logit so unused slots are pre-pruned.
    for facet, pool in cg.bundle_pool.items():
        if facet in cg.bundle_gates:
            continue
        gate_data = torch.full(
            (cg.capacity,), cg._gate_init_logit,
            device=pool.device, dtype=pool.dtype,
        )
        # Active rows start "open" so the first forward is bit-identical
        # to Tier A *modulo* the sigmoid(0) factor; we use logit=10 (~1.0)
        # to keep migration smooth for already-trained models.
        for slot in cg._active_facets_by_slot:
            if facet in cg._active_facets_by_slot[slot]:
                gate_data[slot] = 10.0
        cg.bundle_gates[facet] = nn.Parameter(gate_data)


def grow_gates(cg: "ConceptGraph", old_capacity: int, new_capacity: int) -> None:
    """Internal helper: extend every gate Parameter to ``new_capacity``,
    preserving old logits exactly and seeding new rows at ``init_logit``.

    Called by ``ConceptGraph.grow_capacity`` after the pool has been
    extended; satisfies G1-G6 for gates.
    """
    if not cg.gate_enabled or not cg.bundle_gates:
        return
    opt = cg._registered_optimizer
    init_logit = cg._gate_init_logit
    for facet, old_g in list(cg.bundle_gates.items()):
        new_data = torch.full(
            (new_capacity,), init_logit,
            device=old_g.device, dtype=old_g.dtype,
        )
        new_data[:old_capacity] = old_g.data
        new_g = nn.Parameter(new_data, requires_grad=old_g.requires_grad)
        if opt is not None:
            migrate_param_in_optimizer(opt, old_g, new_g, old_capacity)
        cg.bundle_gates[facet] = new_g


def ensure_facet_gate(cg: "ConceptGraph", facet: str) -> None:
    """Create the gate Parameter for ``facet`` if it doesn't exist yet.

    Called from ``ConceptGraph._ensure_facet`` whenever a new facet pool
    is created with gates enabled.
    """
    if not cg.gate_enabled or facet in cg.bundle_gates:
        return
    pool = cg.bundle_pool[facet]
    gate_data = torch.full(
        (cg.capacity,), cg._gate_init_logit,
        device=pool.device, dtype=pool.dtype,
    )
    cg.bundle_gates[facet] = nn.Parameter(gate_data)


def open_slot_gate(cg: "ConceptGraph", facet: str, slot_idx: int) -> None:
    """Snap ``gate[facet, slot] -> 1.0`` (logit = +10).

    Called when a slot is first observed for a facet so that the gate
    immediately admits the row (matching Tier A's "first collapse opens
    the synapse" behaviour). The L0 regulariser will later relax this
    back if the slot ends up unused.
    """
    if not cg.gate_enabled:
        return
    g = cg.bundle_gates.get(facet)
    if g is None:
        return
    with torch.no_grad():
        g.data[slot_idx] = 10.0


def apply_gates(cg: "ConceptGraph", facet: str, rows: torch.Tensor,
                slot_indices: torch.Tensor) -> torch.Tensor:
    """Multiply ``rows`` (shape (B, D)) by ``sigmoid(gate[facet, slots])``.

    Returns ``rows`` unchanged if gates are disabled or the facet has
    no gate (preserving Tier A bit-identity).
    """
    if not cg.gate_enabled:
        return rows
    g = cg.bundle_gates.get(facet)
    if g is None:
        return rows
    gate_vals = torch.sigmoid(g[slot_indices])  # (B,)
    return rows * gate_vals.unsqueeze(-1)


def gate_l0_loss(cg: "ConceptGraph", facet: str | None = None) -> torch.Tensor:
    """L0 surrogate (sum of sigmoids) for one facet or the whole graph.

    With ``facet=None``, returns the sum across every active gate; this
    is the standard "synaptogenesis pressure" term to add to the
    training loss.
    """
    if not cg.gate_enabled or not cg.bundle_gates:
        return torch.tensor(0.0)
    if facet is not None:
        g = cg.bundle_gates.get(facet)
        if g is None:
            return torch.tensor(0.0)
        return torch.sigmoid(g).sum()
    total = None
    for g in cg.bundle_gates.values():
        s = torch.sigmoid(g).sum()
        total = s if total is None else total + s
    return total if total is not None else torch.tensor(0.0)


def gate_l1_loss(cg: "ConceptGraph") -> torch.Tensor:
    """Equivalent L1 form (``|sigmoid(g)|`` is identical to ``sigmoid(g)``
    since gates are non-negative). Provided for API symmetry."""
    return gate_l0_loss(cg)


def set_slot_gate(
    cg: "ConceptGraph", facet: str, slot_idx: int, value: float
) -> None:
    """Hard-set the **probability** ``sigmoid(g[facet, slot])`` to ``value``.

    Used by Tier-B causal interventions (``value=0`` ablates a slot's
    contribution; ``value=1`` re-opens it). Reversible.
    """
    if not cg.gate_enabled:
        raise RuntimeError("gates are not enabled on this ConceptGraph")
    g = cg.bundle_gates.get(facet)
    if g is None:
        raise KeyError(f"no gate Parameter for facet {facet!r}")
    with torch.no_grad():
        g.data[slot_idx] = _logit_for_prob(value)


def gate_status(cg: "ConceptGraph") -> dict[str, dict]:
    """Diagnostic snapshot: per-facet gate distribution.

    Returns ``{facet: {"mean": float, "p_open": float, "p_pruned": float,
    "n": int}}`` where ``p_open = mean(g > GATE_GROW_THRESHOLD)`` and
    ``p_pruned = mean(g < GATE_PRUNE_THRESHOLD)``. The latter two
    correspond to plan §3.B's "synaptogenesis filled" vs "pruned"
    populations.
    """
    out: dict[str, dict] = {}
    if not cg.gate_enabled:
        return out
    for facet, g in cg.bundle_gates.items():
        probs = torch.sigmoid(g.detach())
        out[facet] = {
            "mean": float(probs.mean().item()),
            "p_open": float((probs > _cfg.GATE_GROW_THRESHOLD).float().mean().item()),
            "p_pruned": float((probs < _cfg.GATE_PRUNE_THRESHOLD).float().mean().item()),
            "n": int(probs.numel()),
        }
    return out


def prune_pruned_slots(cg: "ConceptGraph", threshold: float | None = None) -> int:
    """Recycle slots whose gates have collapsed below ``threshold`` for
    every facet they were active in. Returns count of recycled slots.

    The slot row is zero-filled (Tier-A ``ablate`` semantics); the slot
    index goes back into ``_free_slots`` for the next ``register_concept``.
    The associated concept_id is **not** removed automatically; callers
    that want full pruning should follow up with
    ``cg._remove_concept(cid)`` for each recycled slot's concept.
    """
    if not cg.gate_enabled:
        return 0
    threshold = threshold if threshold is not None else _cfg.GATE_PRUNE_THRESHOLD
    recycled: list[int] = []
    for slot, facets in list(cg._active_facets_by_slot.items()):
        if not facets:
            continue
        all_pruned = True
        for f in facets:
            g = cg.bundle_gates.get(f)
            if g is None or torch.sigmoid(g[slot]).item() >= threshold:
                all_pruned = False
                break
        if all_pruned:
            recycled.append(slot)
    # Drop concept_ids whose slots are recycled (so registration can reuse them).
    for slot in recycled:
        cid = cg.slot_to_cid.get(slot)
        if cid is not None and cid in cg.concepts:
            cg._remove_concept(cid)
    return len(recycled)


__all__ = [
    "attach_gates",
    "grow_gates",
    "ensure_facet_gate",
    "open_slot_gate",
    "apply_gates",
    "gate_l0_loss",
    "gate_l1_loss",
    "set_slot_gate",
    "gate_status",
    "prune_pruned_slots",
]
