"""pcm.migrate_ckpt — One-shot migrator: legacy per-bundle ckpt -> dense pool.

Pre-D93 PCM stored each concept's parameter bundle as its own
``nn.ParameterDict`` ("params.<facet>") — i.e. the ckpt produced by
``torch.save(node.bundle.state_dict(), ...)`` had keys like
``"params.arithmetic_bias"``. After D93 the storage moved to a single
dense pool ``ConceptGraph.bundle_pool[facet]`` with shape (capacity, D).

This module provides a functional migrator that reads either:

* a per-concept dict ``{cid: {"params.<facet>": tensor}}``, or
* an explicit list of ``(cid, facet, tensor)`` tuples,

and writes the corresponding rows into the dense pool of an existing
``ConceptGraph``. The graph is mutated in place; the migrator does not
load any new optimizer state (callers should rebuild the optimizer on
top of ``cg.iter_bundle_parameters()`` and call ``cg.register_optimizer``).

Usage::

    from pcm import ConceptGraph
    from pcm.migrate_ckpt import migrate_legacy_bundle_dict

    cg = ConceptGraph(...)
    for cid in concept_ids:
        cg.register_concept(cid, label=...)
    legacy = torch.load("outputs/ans_encoder/legacy_bundles.pt")
    migrate_legacy_bundle_dict(cg, legacy)
"""
from __future__ import annotations

import logging
from typing import Iterable

import torch

from .concept_graph import ConceptGraph

log = logging.getLogger(__name__)


def _coerce_legacy_key(key: str) -> str:
    """Strip the legacy ``"params."`` prefix on facet keys."""
    return key[len("params."):] if key.startswith("params.") else key


def migrate_legacy_bundle_dict(
    cg: ConceptGraph,
    legacy: dict[str, dict[str, torch.Tensor]],
    *,
    init: str = "normal_small",
    device: torch.device | str | None = None,
    strict: bool = False,
) -> dict[str, list[str]]:
    """Migrate a ``{cid: {"params.<facet>": row_tensor}}`` legacy ckpt into ``cg``.

    Concepts already present in ``cg.cid_to_slot`` are migrated; concepts
    in ``legacy`` not yet registered are silently skipped (or raised when
    ``strict=True``). Facet pools are lazily created with shape from the
    first observed row.

    Returns a report ``{"migrated": [cids...], "skipped_unknown_cid": [...],
    "skipped_unknown_facet": [...]}`` for diagnostics.
    """
    migrated: list[str] = []
    skipped_unknown_cid: list[str] = []
    skipped_unknown_facet: list[str] = []

    for cid, fdict in legacy.items():
        if cid not in cg.cid_to_slot:
            if strict:
                raise KeyError(
                    f"legacy ckpt has cid {cid!r} but it is not registered "
                    f"in the target ConceptGraph"
                )
            skipped_unknown_cid.append(cid)
            continue

        slot = cg.cid_to_slot[cid]
        for raw_key, row_tensor in fdict.items():
            facet = _coerce_legacy_key(raw_key)
            shape = tuple(int(s) for s in row_tensor.shape)
            cg._ensure_facet(facet, shape, device, init)
            pool = cg.bundle_pool[facet]
            with torch.no_grad():
                pool.data[slot].copy_(row_tensor.to(pool.device, dtype=pool.dtype))
            cg._initialized_rows.add((facet, slot))
            cg._active_facets_by_slot.setdefault(slot, set()).add(facet)
        migrated.append(cid)

    log.info(
        "migrate_legacy_bundle_dict: migrated=%d, skipped_unknown_cid=%d",
        len(migrated), len(skipped_unknown_cid),
    )
    return {
        "migrated": migrated,
        "skipped_unknown_cid": skipped_unknown_cid,
        "skipped_unknown_facet": skipped_unknown_facet,
    }


def migrate_legacy_bundles_iter(
    cg: ConceptGraph,
    rows: Iterable[tuple[str, str, torch.Tensor]],
    *,
    device: torch.device | str | None = None,
    init: str = "normal_small",
    strict: bool = False,
) -> dict[str, list[str]]:
    """Variant accepting an iterator of ``(cid, facet, row_tensor)`` tuples.

    Useful when the legacy ckpt stores rows in a different layout (e.g.
    one ``.pt`` per concept) and the caller wants streaming migration.
    """
    legacy: dict[str, dict[str, torch.Tensor]] = {}
    for cid, facet, row in rows:
        legacy.setdefault(cid, {})[f"params.{facet}"] = row
    return migrate_legacy_bundle_dict(
        cg, legacy, init=init, device=device, strict=strict
    )


def export_dense_bundle_dict(cg: ConceptGraph) -> dict[str, dict[str, torch.Tensor]]:
    """Inverse: serialise the dense pool back into the legacy
    ``{cid: {"params.<facet>": row_tensor}}`` shape so old viewers /
    diagnostics keep working unmodified.
    """
    out: dict[str, dict[str, torch.Tensor]] = {}
    for cid, slot in cg.cid_to_slot.items():
        per_cid: dict[str, torch.Tensor] = {}
        for facet, pool in cg.bundle_pool.items():
            if (facet, slot) in cg._initialized_rows:
                per_cid[f"params.{facet}"] = pool.data[slot].detach().cpu().clone()
        if per_cid:
            out[cid] = per_cid
    return out


__all__ = [
    "migrate_legacy_bundle_dict",
    "migrate_legacy_bundles_iter",
    "export_dense_bundle_dict",
]
