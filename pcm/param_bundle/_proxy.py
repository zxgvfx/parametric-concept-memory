"""Read-only dict view: ``bundle.params`` → BundleRowView per facet.

Backs the legacy iteration / membership / ``.get`` / ``.items`` surface
on ``ParamBundle.params``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from ._row_view import BundleRowView


if TYPE_CHECKING:
    from ..concept_graph import ConceptGraph


__all__ = ["_ParamProxyDict"]


class _ParamProxyDict:
    """Read-only dict view: facet -> BundleRowView for one slot.

    Backs the legacy ``bundle.params`` attribute. Iteration / membership
    only sees facets that have actually been collapsed for this slot
    (matches the old lazy-init semantics).
    """

    __slots__ = ("_graph", "_slot_idx")

    def __init__(self, graph: "ConceptGraph", slot_idx: int) -> None:
        self._graph = graph
        self._slot_idx = int(slot_idx)

    def _active_facets(self) -> set[str]:
        return self._graph._active_facets_by_slot.get(self._slot_idx, set())

    def __getitem__(self, facet: str) -> BundleRowView:
        if facet not in self._active_facets():
            raise KeyError(facet)
        pool = self._graph.bundle_pool[facet]
        return BundleRowView(pool, self._slot_idx)

    def __contains__(self, facet: object) -> bool:
        return isinstance(facet, str) and facet in self._active_facets()

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._active_facets()))

    def __len__(self) -> int:
        return len(self._active_facets())

    def keys(self) -> list[str]:
        return sorted(self._active_facets())

    def values(self) -> list[BundleRowView]:
        return [
            BundleRowView(self._graph.bundle_pool[f], self._slot_idx)
            for f in self.keys()
        ]

    def items(self) -> list[tuple[str, BundleRowView]]:
        return [
            (f, BundleRowView(self._graph.bundle_pool[f], self._slot_idx))
            for f in self.keys()
        ]

    def get(self, facet: str, default=None):
        if facet in self._active_facets():
            return BundleRowView(self._graph.bundle_pool[facet], self._slot_idx)
        return default
