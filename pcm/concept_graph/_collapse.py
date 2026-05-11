r"""Collapse + cook mixin for ConceptGraph.

Owns everything in the dense-pool collapse + Tier-D cook-subgraph
registration flow:

- ``register_muscle_subgraph`` — D94 helper that registers a Tier-D
  ``parametric_muscle_subgraph`` ConceptNode.
- ``_ensure_facet`` — lazy-allocates the dense pool ``(capacity, *shape)``
  for a facet on first observation (also lazy-creates the optional
  Tier-B gate Parameter).
- ``_init_slot_if_unset`` — first-observation per-row init + Tier-B gate
  open-on-first-touch.
- ``_record_attribution`` — write ``consumed_by`` / ``collapse_history``
  / ``_active_facets_by_slot`` for one (slot, facet, caller, tick) tuple.
- ``_collapse_one`` — single-slot collapse called by
  ``ParamBundle.request`` (legacy per-row API).
- ``collapse_batch`` — GPU-friendly ``F.embedding`` batched lookup that
  is the new fast path for muscle heads.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..param_bundle import ContextualizedConcept, init_row_
from .nodes import ConceptNode


__all__ = ["_CollapseMixin"]


class _CollapseMixin:
    """Collapse + cook subgraph registration."""

    # ── D94 Tier-D cook subgraph registration ─────────────────────────────

    def register_muscle_subgraph(
        self,
        node_id: str,
        *,
        inputs: list[str],
        nodes: list[dict],
        output: str,
        constants: dict | None = None,
        facet_specs: dict | None = None,
        label: str | None = None,
        scope: str = "BASE",
        provenance: str | None = None,
        tick: int = 0,
    ) -> ConceptNode:
        """Register a Tier-D ``parametric_muscle_subgraph`` ConceptNode.

        Convenience wrapper around :meth:`register_concept` that fills in
        ``kind`` and ``metadata`` for the cook interpreter. The resulting
        node is *not* given a slot in any ``bundle_pool`` (cook nodes
        carry no bundle row of their own); ``cid_to_slot`` still maps the
        node id to a fresh slot for uniformity, but downstream cook
        interpretation never reads its row.

        Args:
            node_id: e.g. ``"muscle.arith_v2"``. Must be unique in the graph.
            inputs: ordered list of input binding names (matched against the
                ``bindings`` dict at cook time).
            nodes: ordered list of op invocations; see schema in
                ``docs/PCM_NODE_AS_FUNCTION_DESIGN.md`` §1.1.
            output: id of the node whose result is the subgraph output.
            constants: literal name -> value map; readable as ``@name``.
            facet_specs: declarative facet shape table; reserved for future
                Tier-E / Tier-F use. Currently passed through as-is.
        """
        node = self.register_concept(
            node_id=node_id,
            label=label or node_id,
            scope=scope,
            provenance=provenance,
            tick=tick,
        )
        node.kind = "parametric_muscle_subgraph"
        node.metadata = {
            "inputs": list(inputs),
            "constants": dict(constants or {}),
            "facet_specs": dict(facet_specs or {}),
            "nodes": list(nodes),
            "output": str(output),
        }
        return node

    # ── D93 facet pool lifecycle + per-row init ───────────────────────────

    def _ensure_facet(
        self,
        facet: str,
        shape: tuple[int, ...],
        device: torch.device | str | None,
        init: str,
    ) -> nn.Parameter:
        """Ensure ``bundle_pool[facet]`` exists with shape (capacity, *shape).

        Pool is zero-initialised globally; per-row init happens on first
        observation via ``_init_slot_if_unset``. Validates shape if facet
        already exists.
        """
        if not facet or not facet.replace("_", "").isalnum():
            raise ValueError(
                f"facet name must be alnum+underscore (got {facet!r}); "
                "ParameterDict-style key restriction."
            )
        if facet in self.bundle_pool:
            existing_shape = self._facet_shapes[facet]
            if existing_shape != shape:
                raise ValueError(
                    f"facet {facet!r} already initialised with shape {existing_shape}, "
                    f"requested {shape}"
                )
            pool = self.bundle_pool[facet]
            if device is not None:
                target = torch.device(device)
                # Compare device *type* + *index* but treat
                # ``device('cuda')`` (no index) as matching
                # ``device('cuda', 0)`` so callers passing the
                # string "cuda" don't trigger a Parameter rebuild
                # on every collapse — that silently broke v2 MVP
                # training (see PCM_V2_DUAL_CHANNEL_DESIGN §V2
                # debug log) because optimizer references became
                # stale after each rebuild.
                same_type = pool.device.type == target.type
                idx_match = (
                    target.index is None
                    or pool.device.index == target.index
                )
                if not (same_type and idx_match):
                    self.bundle_pool[facet] = nn.Parameter(
                        pool.data.to(device),
                        requires_grad=pool.requires_grad,
                    )
            return self.bundle_pool[facet]

        pool_device = torch.device(device) if device is not None else torch.device("cpu")
        pool_data = torch.zeros(self.capacity, *shape, device=pool_device, dtype=torch.float32)
        pool = nn.Parameter(pool_data, requires_grad=True)
        self.bundle_pool[facet] = pool
        self._facet_shapes[facet] = tuple(int(s) for s in shape)
        self._facet_default_init[facet] = init
        # Tier-B: lazy-create the gate Parameter for this facet (no-op if disabled).
        from .. import gate as _gate_mod
        _gate_mod.ensure_facet_gate(self, facet)
        return pool

    def _init_slot_if_unset(self, facet: str, slot_idx: int, init: str) -> None:
        """Apply per-row init the first time (facet, slot_idx) is observed.

        Other slots in the same pool stay at zero ("undifferentiated
        neurons" in the bionic story). Idempotent on repeat calls.

        On first observation, also snaps the Tier-B gate open
        (logit=10 → sigmoid≈1) so the slot starts contributing immediately.
        Subsequent calls do *not* reset the gate, so user-driven
        ``set_slot_gate`` interventions are durable.
        """
        key = (facet, slot_idx)
        if key in self._initialized_rows:
            return
        pool = self.bundle_pool[facet]
        with torch.no_grad():
            init_row_(pool.data[slot_idx], init)  # type: ignore[arg-type]
        self._initialized_rows.add(key)
        # Tier-B: open the gate once (guarded by _initialized_rows).
        if self.gate_enabled:
            from .. import gate as _gate_mod
            _gate_mod.open_slot_gate(self, facet, slot_idx)

    def _record_attribution(
        self,
        slot_idx: int,
        facet: str,
        caller: str,
        tick: int,
    ) -> None:
        per_slot_cb = self._consumed_by_by_slot.setdefault(slot_idx, {})
        per_slot_cb.setdefault(facet, set()).add(caller)
        per_slot_h = self._collapse_history_by_slot.setdefault(slot_idx, {})
        per_slot_h.setdefault(facet, []).append((caller, int(tick)))
        self._active_facets_by_slot.setdefault(slot_idx, set()).add(facet)

    def _collapse_one(
        self,
        slot_idx: int,
        facet: str,
        shape: tuple[int, ...],
        caller: str,
        concept_id: str,
        tick: int,
        init: str,
        device: torch.device | str | None,
    ) -> ContextualizedConcept:
        """Single-slot collapse; backs ``ParamBundle.request`` legacy API."""
        self._ensure_facet(facet, shape, device, init)
        self._init_slot_if_unset(facet, slot_idx, init)  # also opens gate on first obs
        self._record_attribution(slot_idx, facet, caller, tick)
        row = self.bundle_pool[facet][slot_idx]
        if self.gate_enabled and facet in self.bundle_gates:
            row = row * torch.sigmoid(self.bundle_gates[facet][slot_idx])
        return ContextualizedConcept(
            concept_id=concept_id,
            caller=caller,
            facet=facet,
            facet_params=row,
            tick=int(tick),
        )

    def collapse_batch(
        self,
        caller: str,
        facet: str,
        concept_ids: list[str],
        shape: Iterable[int],
        tick: int = 0,
        init: str = "normal_small",
        device: torch.device | str | None = None,
    ) -> Tensor:
        """GPU-friendly batched collapse: returns ``F.embedding`` lookup over
        the dense pool (B, *shape), with the **full** D91 attribution side
        effects performed in one pass (deduplicated over unique slots).

        This is the new fast path that supersedes the per-call
        ``for cid in concept_ids: node.collapse(...).as_tensor()`` loop in
        muscle heads.
        """
        shape_t = tuple(int(s) for s in shape)
        self._ensure_facet(facet, shape_t, device, init)

        slot_indices: list[int] = []
        unique_slots: set[int] = set()
        for cid in concept_ids:
            if cid not in self.cid_to_slot:
                raise KeyError(f"concept {cid!r} not in ConceptGraph")
            slot = self.cid_to_slot[cid]
            slot_indices.append(slot)
            if slot not in unique_slots:
                unique_slots.add(slot)
                # _init_slot_if_unset itself opens the gate on first
                # observation; we do NOT re-open per-batch (otherwise a
                # user-driven set_slot_gate(0) would not survive).
                self._init_slot_if_unset(facet, slot, init)
        # Per-slot attribution write (deduplicated; single set.add per unique slot).
        for slot in unique_slots:
            self._record_attribution(slot, facet, caller, tick)

        pool = self.bundle_pool[facet]
        slots_t = torch.as_tensor(slot_indices, dtype=torch.long, device=pool.device)
        rows = F.embedding(slots_t, pool)
        # Tier-B: multiplicative gate (no-op + bit-identical when disabled).
        if self.gate_enabled and facet in self.bundle_gates:
            from .. import gate as _gate_mod
            rows = _gate_mod.apply_gates(self, facet, rows, slots_t)
        return rows
