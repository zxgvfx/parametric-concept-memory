"""Custom-id graph builder (A4 assay)."""
from __future__ import annotations

from typing import Callable

from pcm.concept_graph import ConceptGraph


__all__ = ["build_graph_with_id_fn"]


def build_graph_with_id_fn(
    n_min: int,
    n_max: int,
    id_fn: Callable[[int], str],
) -> tuple[ConceptGraph, dict[int, str]]:
    """按 ``id_fn(n)`` 生成 concept_id. 返回 ``(cg, {n -> concept_id})``."""
    cg = ConceptGraph(feat_dim=128)
    id_map: dict[int, str] = {}
    for n in range(n_min, n_max + 1):
        cid = id_fn(n)
        cg.register_concept(
            node_id=cid,
            label=f"ANS_{n}",
            scope="BASE",
            provenance=f"purity_audit:n={n}",
        )
        id_map[n] = cid
    return cg, id_map
