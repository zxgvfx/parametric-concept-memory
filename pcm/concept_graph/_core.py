r"""Core mixin for ConceptGraph.

Owns the lifecycle primitives:

- ``__init__`` — config validation + state buckets (concepts, surfaces,
  edges, procedures, dense pool, attribution tables).
- ``register_concept`` — slot allocation + bundle binding.
- ``_allocate_slot`` / ``_reset_slot_metadata`` — D93 slot bookkeeping.
- ``register_optimizer`` — register the optimizer for grow-time moment migration.
- ``grow_capacity`` — D93 G1-G6 capacity-grow protocol.
- ``_evict_oldest_concept`` / ``_remove_concept`` / ``_append_provenance``
  — LRU eviction with D86 protected-scope guard.

The mixin holds no abstract slots; concrete attributes are populated by
``__init__`` and consumed by sibling mixins via ``self.*`` attribute
access. The :class:`pcm.concept_graph.graph.ConceptGraph` final class
combines all mixins.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable

import torch
import torch.nn as nn

from .. import config as _cfg
from ..param_bundle import ParamBundle, migrate_param_in_optimizer
from .nodes import (
    ConceptEdge,
    ConceptNode,
    DEFAULT_FEAT_DIM,
    DEFAULT_MAX_EDGES,
    DEFAULT_MAX_NODES,
    DEFAULT_MAX_SURFACES,
    HasSurfaceFormEdge,
    MAX_PROVENANCE_PER_NODE,
    PROTECTED_SCOPES,
    ProcedureNode,
    SurfaceFormNode,
)


__all__ = ["_CoreMixin"]


log = logging.getLogger(__name__)


class _CoreMixin:
    """Core lifecycle: __init__, register_concept, slot allocation, grow."""

    def __init__(
        self,
        feat_dim: int = DEFAULT_FEAT_DIM,
        max_nodes: int = DEFAULT_MAX_NODES,
        max_surfaces: int = DEFAULT_MAX_SURFACES,
        max_edges: int = DEFAULT_MAX_EDGES,
        *,
        initial_capacity: int | None = None,
        growth_factor: float | None = None,
        max_capacity: int | None = None,
    ) -> None:
        self.feat_dim = feat_dim
        self.max_nodes = max_nodes
        self.max_surfaces = max_surfaces
        self.max_edges = max_edges

        # D93 dense-pool config — bionic pre-allocation knobs.
        self._initial_capacity = int(
            initial_capacity if initial_capacity is not None else _cfg.INITIAL_CAPACITY
        )
        self._growth_factor = float(
            growth_factor if growth_factor is not None else _cfg.GROWTH_FACTOR
        )
        self._max_capacity = int(
            max_capacity if max_capacity is not None else _cfg.MAX_CAPACITY
        )
        if self._initial_capacity < 1:
            raise ValueError("initial_capacity must be >= 1")
        if self._growth_factor < 1.0:
            raise ValueError("growth_factor must be >= 1.0 (1.0 disables grow)")
        if self._max_capacity < self._initial_capacity:
            raise ValueError("max_capacity must be >= initial_capacity")

        self.concepts: dict[str, ConceptNode] = {}
        self.surfaces: dict[str, SurfaceFormNode] = {}
        self.has_surface_form: dict[tuple[str, str], HasSurfaceFormEdge] = {}
        self.edges: dict[tuple[str, str, str], ConceptEdge] = {}
        self.procedures: dict[str, ProcedureNode] = {}

        # adjacency 索引 (查询加速; PHILOSOPHY §4 同一职责一种实现)
        self._concept_to_surfaces: dict[str, list[str]] = defaultdict(list)
        self._surface_to_concept: dict[str, str] = {}
        self._adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._concept_to_procedures: dict[str, list[str]] = defaultdict(list)

        # ── D93 dense-pool storage (bionic pre-allocation) ────────────────
        self.bundle_pool: dict[str, nn.Parameter] = {}
        self._facet_shapes: dict[str, tuple[int, ...]] = {}
        self._facet_default_init: dict[str, str] = {}
        # ── Tier-B optional slot gates ────────────────────────────────────
        self.bundle_gates: dict[str, nn.Parameter] = {}
        self.gate_enabled: bool = bool(_cfg.ENABLE_SLOT_GATE)
        self._gate_init_logit: float = float(_cfg.GATE_INIT_LOGIT)
        # ── Concept identity ↔ row index ──────────────────────────────────
        self.cid_to_slot: dict[str, int] = {}
        self.slot_to_cid: dict[int, str] = {}
        self._free_slots: list[int] = []
        self._next_slot_idx: int = 0
        self.capacity: int = self._initial_capacity
        # ── Per-slot attribution / history ────────────────────────────────
        self._consumed_by_by_slot: dict[int, dict[str, set[str]]] = {}
        self._collapse_history_by_slot: dict[int, dict[str, list[tuple[str, int]]]] = {}
        self._active_facets_by_slot: dict[int, set[str]] = {}
        self._initialized_rows: set[tuple[str, int]] = set()
        # ── Bookkeeping ───────────────────────────────────────────────────
        self._grow_events: int = 0
        self._registered_optimizer: torch.optim.Optimizer | None = None

    # ── ConceptNode 操作 ─────────────────────────────────────────

    def register_concept(
        self,
        node_id: str,
        label: str,
        scope: str = "level",
        connected_networks: Iterable[str] | None = None,
        provenance: str | None = None,
        tick: int = 0,
    ) -> ConceptNode:
        """添加或更新 ConceptNode (hub). 已存在则 EMA-style 更新元数据.

        Side-effects (D93 dense-pool):
            - Allocates a stable ``slot_idx`` for the concept (永不变更).
            - Binds a ``ParamBundle(graph, slot_idx)`` proxy to ``node.bundle``.
            - May trigger a ``grow_capacity`` if the slot pool is exhausted
              (passive grow per plan §2.5.4).
        """
        if node_id in self.concepts:
            node = self.concepts[node_id]
            node.last_tick = tick
            if provenance:
                self._append_provenance(node, provenance)
            if connected_networks:
                node.connected_networks.update(connected_networks)
            return node

        if len(self.concepts) >= self.max_nodes:
            self._evict_oldest_concept()

        slot_idx = self._allocate_slot()

        node = ConceptNode(
            node_id=node_id,
            label=label,
            scope=scope,
            connected_networks=set(connected_networks) if connected_networks else set(),
            grounding_provenance=[provenance] if provenance else [],
            last_tick=tick,
            valid_from=tick,
        )
        node.bundle = ParamBundle(self, slot_idx)
        self.concepts[node_id] = node
        self.cid_to_slot[node_id] = slot_idx
        self.slot_to_cid[slot_idx] = node_id
        return node

    # ── D93 slot allocation + grow protocol ───────────────────────────────

    def _allocate_slot(self) -> int:
        """Return the next free slot, growing the pool if exhausted (G* safe).

        Recycles ``_free_slots`` first (LIFO) so prune+register stays
        determinstic; otherwise hands out monotonically increasing indices
        from ``_next_slot_idx``.
        """
        if self._free_slots:
            slot = self._free_slots.pop()
            self._reset_slot_metadata(slot)
            return slot
        if self._next_slot_idx >= self.capacity:
            self.grow_capacity(extra=1)
        slot = self._next_slot_idx
        self._next_slot_idx += 1
        return slot

    def _reset_slot_metadata(self, slot: int) -> None:
        """Drop attribution / history for a recycled slot (Tier-B prune helper)."""
        self._consumed_by_by_slot.pop(slot, None)
        self._collapse_history_by_slot.pop(slot, None)
        self._active_facets_by_slot.pop(slot, None)
        for facet in self.bundle_pool:
            self._initialized_rows.discard((facet, slot))
            with torch.no_grad():
                self.bundle_pool[facet].data[slot].zero_()

    def register_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Tell the graph which optimizer owns ``bundle_pool`` parameters.

        Once registered, ``grow_capacity`` will automatically migrate Adam
        moments / SGD momentum buffers (G4 invariant). If you choose not
        to register, you must pass the optimizer to ``grow_capacity`` at
        each call site.
        """
        self._registered_optimizer = optimizer

    def grow_capacity(
        self,
        extra: int = 1,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> int:
        """Grow the dense pool capacity by at least ``extra`` rows (G1-G6).

        Strategy: doubling growth (``growth_factor=2.0``) so register-driven
        passive grows are amortised O(1). Returns the new capacity.

        Invariants enforced (see plan §2.5.3):
          G1 row data identical for old slots
          G2 cid_to_slot unchanged
          G3 attribution tables unchanged
          G4 optimizer moments preserved (new rows zero)
          G5 forward bit-identical for any batch over old slots
          G6 grad path unchanged for old slots
        """
        if extra < 1:
            return self.capacity
        opt = optimizer if optimizer is not None else self._registered_optimizer

        new_cap = max(
            self.capacity + extra,
            int(self.capacity * self._growth_factor),
        )
        if new_cap > self._max_capacity:
            raise RuntimeError(
                f"ConceptGraph.grow_capacity: requested {new_cap} > max_capacity "
                f"{self._max_capacity}. Increase max_capacity or prune unused slots."
            )

        old_cap = self.capacity
        for facet, old_p in list(self.bundle_pool.items()):
            new_data = torch.zeros(
                new_cap, *old_p.shape[1:],
                dtype=old_p.dtype, device=old_p.device,
            )
            new_data[:old_cap] = old_p.data  # G1 bit-copy
            new_p = nn.Parameter(new_data, requires_grad=old_p.requires_grad)
            if opt is not None:
                migrate_param_in_optimizer(opt, old_p, new_p, old_cap)  # G4
            self.bundle_pool[facet] = new_p

        # Tier-B: extend gate Parameters in lockstep (no-op when disabled).
        from .. import gate as _gate_mod
        _gate_mod.grow_gates(self, old_cap, new_cap)

        self.capacity = new_cap
        self._grow_events += 1
        return new_cap

    # ── Eviction + provenance bookkeeping ─────────────────────────────────

    def _append_provenance(self, node: ConceptNode, provenance: str) -> None:
        """LRU-cap provenance list 防 unbounded 增长 (PHILOSOPHY §7)."""
        node.grounding_provenance.append(provenance)
        if len(node.grounding_provenance) > MAX_PROVENANCE_PER_NODE:
            node.grounding_provenance = (
                node.grounding_provenance[:1]
                + node.grounding_provenance[-(MAX_PROVENANCE_PER_NODE - 1):]
            )

    def _evict_oldest_concept(self) -> None:
        """LRU evict, 但 BASE/CORE scope 节点不淘汰 (D86)."""
        candidates = [
            (nid, node)
            for nid, node in self.concepts.items()
            if node.scope not in PROTECTED_SCOPES
        ]
        if not candidates:
            log.warning(
                "ConceptGraph at capacity (%d) but only protected nodes exist; "
                "skipping evict (D86)",
                len(self.concepts),
            )
            return
        oldest_id = min(candidates, key=lambda x: x[1].last_tick)[0]
        self._remove_concept(oldest_id)

    def _remove_concept(self, node_id: str) -> None:
        """移除 ConceptNode + 相关 has_surface_form / ConceptEdge.

        D93: also recycles the row index back to ``_free_slots`` and
        zero-fills the row so a future ``register_concept`` reusing the
        slot starts from a clean state. Pool capacity is **not** shrunk
        (G1 invariant — other slots' rows must stay bit-identical).
        """
        if node_id not in self.concepts:
            return
        node = self.concepts.pop(node_id)
        for sf_id in list(node.surface_forms):
            sf = self.surfaces.get(sf_id)
            if sf and sf.grounded_to == node_id:
                sf.grounded_to = None
            self.has_surface_form.pop((node_id, sf_id), None)
            self._surface_to_concept.pop(sf_id, None)
        self._concept_to_surfaces.pop(node_id, None)
        edge_keys = [k for k in self.edges if k[0] == node_id or k[1] == node_id]
        for k in edge_keys:
            self.edges.pop(k, None)
        self._adjacency.pop(node_id, None)
        for adj_list in self._adjacency.values():
            adj_list[:] = [(t, et) for t, et in adj_list if t != node_id]
        # D93 slot recycling
        slot = self.cid_to_slot.pop(node_id, None)
        if slot is not None:
            self.slot_to_cid.pop(slot, None)
            self._reset_slot_metadata(slot)
            self._free_slots.append(slot)
