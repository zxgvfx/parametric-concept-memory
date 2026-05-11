"""Per-concept ``ParamBundle`` proxy (D91 + D93 dense pool).

Holds **no** tensor storage of its own. All parameters live on
``ConceptGraph.bundle_pool[facet]`` indexed by ``slot_idx``. The
legacy ``params`` / ``consumed_by`` / ``collapse_history`` dicts are
re-exposed here as proxies so downstream code (heads, swap utilities,
tests) keeps working unmodified.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Iterator

import torch
import torch.nn as nn

from ._proxy import _ParamProxyDict
from ._types import ContextualizedConcept, InitStrategy


if TYPE_CHECKING:
    from ..concept_graph import ConceptGraph


__all__ = ["ParamBundle"]


class ParamBundle:
    """Per-concept multi-facet bundle proxy (D91 + D93 dense-pool).

    Lifecycle::

        register_concept(cid)  -> graph.cid_to_slot[cid] = slot_idx
                                  graph.concepts[cid].bundle = ParamBundle(graph, slot_idx)
        node.collapse(...)     -> ensures pool[facet] exists (lazy);
                                  initialises this row if first observation;
                                  records consumer + history;
                                  returns ContextualizedConcept(row view).
    """

    def __init__(self, graph: "ConceptGraph", slot_idx: int) -> None:
        self._graph = graph
        self._slot_idx = int(slot_idx)

    # ── identity ────────────────────────────────────────────────────────────

    @property
    def slot_idx(self) -> int:
        """Row index assigned to this concept in the dense pool."""
        return self._slot_idx

    @property
    def graph(self) -> "ConceptGraph":
        return self._graph

    # ── legacy dict proxies ─────────────────────────────────────────────────

    @property
    def params(self) -> _ParamProxyDict:
        return _ParamProxyDict(self._graph, self._slot_idx)

    @property
    def consumed_by(self) -> dict[str, set[str]]:
        # Returns the *actual* dict (mutable) so legacy code that
        # iterates and inspects sets continues to work.
        return self._graph._consumed_by_by_slot.setdefault(self._slot_idx, {})

    @property
    def collapse_history(self) -> dict[str, list[tuple[str, int]]]:
        return self._graph._collapse_history_by_slot.setdefault(self._slot_idx, {})

    # ── core API ────────────────────────────────────────────────────────────

    def request(
        self,
        facet: str,
        shape: Iterable[int],
        caller: str,
        concept_id: str = "",
        tick: int = 0,
        init: InitStrategy = "normal_small",
        device: torch.device | str | None = None,
    ) -> ContextualizedConcept:
        """Collapse this slot under (caller, facet); see ConceptGraph._collapse_one."""
        return self._graph._collapse_one(
            slot_idx=self._slot_idx,
            facet=facet,
            shape=tuple(int(s) for s in shape),
            caller=caller,
            concept_id=concept_id,
            tick=int(tick),
            init=init,
            device=device,
        )

    def ablate(self, facet: str) -> None:
        """Zero this slot's row of the given facet (G6-safe; other slots untouched)."""
        if facet in self._graph.bundle_pool:
            with torch.no_grad():
                self._graph.bundle_pool[facet].data[self._slot_idx].zero_()

    def remove(self, facet: str) -> None:
        """Drop this slot's link to facet (debug / prune)."""
        if facet in self._graph.bundle_pool:
            with torch.no_grad():
                self._graph.bundle_pool[facet].data[self._slot_idx].zero_()
        per_slot_cb = self._graph._consumed_by_by_slot.get(self._slot_idx)
        if per_slot_cb is not None:
            per_slot_cb.pop(facet, None)
        per_slot_h = self._graph._collapse_history_by_slot.get(self._slot_idx)
        if per_slot_h is not None:
            per_slot_h.pop(facet, None)
        active = self._graph._active_facets_by_slot.get(self._slot_idx)
        if active is not None:
            active.discard(facet)

    # ── queries ─────────────────────────────────────────────────────────────

    def facets(self) -> list[str]:
        return sorted(self._graph._active_facets_by_slot.get(self._slot_idx, set()))

    def consumers(self) -> set[str]:
        out: set[str] = set()
        for s in self.consumed_by.values():
            out |= s
        return out

    def liveness(self) -> int:
        return len(self.consumers())

    def n_collapses(self) -> int:
        return sum(len(v) for v in self.collapse_history.values())

    def n_parameters(self) -> int:
        total = 0
        for f in self.facets():
            pool = self._graph.bundle_pool.get(f)
            if pool is None:
                continue
            row_numel = 1
            for s in pool.shape[1:]:
                row_numel *= int(s)
            total += row_numel
        return total

    # ── compatibility shims ─────────────────────────────────────────────────

    def parameters(self) -> Iterator[nn.Parameter]:
        """Per-slot bundles do not own ``nn.Parameter`` — storage lives on the
        graph. Returns an empty iterator. Use ``ConceptGraph.iter_bundle_parameters``
        to feed an optimizer.
        """
        return iter(())

    def to(self, device: torch.device | str) -> "ParamBundle":
        """No-op: device is owned by ``ConceptGraph.bundle_pool``."""
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Backward-compat shim: returns ``{f"params.{facet}": row_clone}``.

        Used by analysis code (e.g. ``scale_study.train_scale_one``) that
        wants per-slot detached snapshots. Always returns a fresh CPU
        clone so downstream callers can safely mutate.
        """
        out: dict[str, torch.Tensor] = {}
        for f in self.facets():
            pool = self._graph.bundle_pool.get(f)
            if pool is None:
                continue
            out[f"params.{f}"] = pool.data[self._slot_idx].detach().clone()
        return out

    def describe(self, recent_events: int = 20) -> dict:
        """Metadata summary (no tensor values)."""
        facets: dict[str, dict] = {}
        for name in self.facets():
            pool = self._graph.bundle_pool.get(name)
            if pool is None:
                continue
            shape = tuple(int(s) for s in pool.shape[1:])
            row_numel = 1
            for s in shape:
                row_numel *= s
            hist = self.collapse_history.get(name, [])
            tail = list(hist[-recent_events:]) if recent_events > 0 else []
            facets[name] = {
                "shape": shape,
                "n_params": row_numel,
                "consumers": sorted(self.consumed_by.get(name, set())),
                "n_collapses": len(hist),
                "recent_collapses": [
                    {"caller": c, "tick": int(t)} for (c, t) in tail
                ],
            }
        return {
            "facets": facets,
            "n_facets": len(facets),
            "n_params_total": self.n_parameters(),
            "liveness": self.liveness(),
            "n_collapses_total": self.n_collapses(),
        }

    # ── pickling support (so ConceptNode dataclass still pickles) ───────────

    def __getstate__(self) -> dict:
        # Bundles are graph-bound proxies; pickling them in isolation is
        # meaningless (the graph drives storage). Return an empty dict and
        # let ConceptGraph re-bind on load.
        return {"_slot_idx": self._slot_idx}

    def __setstate__(self, state: dict) -> None:
        self._slot_idx = int(state.get("_slot_idx", 0))
        self._graph = None  # type: ignore[assignment]
