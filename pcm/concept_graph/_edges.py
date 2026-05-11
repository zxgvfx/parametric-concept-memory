r"""ConceptNode → ConceptNode edge mixin (v1 borrow).

Owns ``add_edge`` and ``_evict_weakest_edge``. Edge types:
``co_occurrence`` / ``temporal`` / ``spatial_adjacency`` / ``character_pair``.
"""
from __future__ import annotations

from .nodes import ConceptEdge, MAX_EDGE_WEIGHT


__all__ = ["_EdgeMixin"]


class _EdgeMixin:
    """ConceptNode edge management."""

    def add_edge(
        self,
        src_id: str,
        tgt_id: str,
        edge_type: str,
        tick: int = 0,
        weight_delta: float = 0.1,
    ) -> ConceptEdge:
        """ConceptNode → ConceptNode 边 (4 类: co_occurrence/temporal/
        spatial_adjacency/character_pair).
        """
        if src_id not in self.concepts or tgt_id not in self.concepts:
            raise KeyError(f"Both concept nodes must exist: {src_id}, {tgt_id}")

        key = (src_id, tgt_id, edge_type)
        if key in self.edges:
            edge = self.edges[key]
            edge.weight = min(edge.weight + weight_delta, MAX_EDGE_WEIGHT)
            edge.count += 1
            edge.last_tick = tick
            return edge

        if len(self.edges) >= self.max_edges:
            self._evict_weakest_edge()

        edge = ConceptEdge(
            source_id=src_id,
            target_id=tgt_id,
            weight=weight_delta,
            edge_type=edge_type,
            count=1,
            last_tick=tick,
        )
        self.edges[key] = edge
        self._adjacency[src_id].append((tgt_id, edge_type))
        return edge

    def _evict_weakest_edge(self) -> None:
        if not self.edges:
            return
        weakest = min(self.edges, key=lambda k: self.edges[k].weight)
        edge = self.edges.pop(weakest)
        adj = self._adjacency.get(edge.source_id, [])
        adj[:] = [
            (t, et) for t, et in adj
            if not (t == edge.target_id and et == edge.edge_type)
        ]
