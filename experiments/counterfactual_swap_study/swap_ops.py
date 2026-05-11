"""In-place bundle-row swap utilities (paper §B core intervention).

These two functions are the *causal* mechanism behind the entire study:
swapping ``cg.bundle_pool[facet][slot_a]`` with ``[slot_b]`` and seeing
the downstream task accuracy follow the swap proves that the bundle row
**is** the concept identity (not merely correlated with it).

The swap is done in-place on ``Parameter.data`` so the optimizer state
and autograd graph remain valid. Pre-D93 the same logic lived in the
single-file ``counterfactual_swap_study.py``.
"""
from __future__ import annotations

from pcm.concept_graph import ConceptGraph


__all__ = ["swap_bundle_facet", "swap_all_facets"]


def swap_bundle_facet(
    cg: ConceptGraph,
    cid_a: str,
    cid_b: str,
    facet: str,
) -> None:
    """Swap ``cid_a`` / ``cid_b`` in place on the given facet's bundle row.

    Implementation note: ``bundle.params[facet]`` returns a row-view
    proxy (D93 BundleRowView) whose ``.data`` aliases the dense pool
    row; ``copy_`` therefore writes through to the pool tensor without
    creating new ``Parameter`` objects.
    """
    pa = cg.concepts[cid_a].bundle.params[facet]
    pb = cg.concepts[cid_b].bundle.params[facet]
    tmp = pa.data.clone()
    pa.data.copy_(pb.data)
    pb.data.copy_(tmp)


def swap_all_facets(cg: ConceptGraph, cid_a: str, cid_b: str) -> None:
    """Swap every facet that both concepts have in common.

    Used by the "swap_both" condition in the per-seed run loops to
    demonstrate the additive nature of facet-targeted swaps.
    """
    facets = set(cg.concepts[cid_a].bundle.params.keys())
    facets &= set(cg.concepts[cid_b].bundle.params.keys())
    for f in facets:
        swap_bundle_facet(cg, cid_a, cid_b, f)
