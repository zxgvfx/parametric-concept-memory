"""ParamBundle package — D91/D92 + bio-inspired pre-allocation core.

This package implements three layered concepts that together realise the
bionic dense-pool architecture described in
``docs/PCM_BIO_PREALLOC_UPGRADE.md``:

- **D91 Parametric Concept Memory** — every ``ConceptNode`` owns a
  multi-facet parameter bundle, consumed by muscles via ``collapse``.
- **D92 Contextual Concept Collapse** — concepts have no resting state;
  semantic content materialises only as ``ContextualizedConcept`` handles
  produced by collapse.
- **D93 Pre-allocated Dense Pool** — the per-concept bundles are
  *views* into a single pre-allocated dense tensor pool living on the
  ``ConceptGraph``. This unlocks GPU-friendly ``F.embedding`` lookups,
  bit-identical capacity grow, and slot-level pruning, while preserving
  every D91/D92 invariant.

Pre-2026-05 this lived in a single ``pcm/param_bundle.py`` (~558 lines).
It has been split across:

- :mod:`._types`    — ``InitStrategy`` + ``ContextualizedConcept`` dataclass.
- :mod:`._row_view` — ``BundleRowView`` (legacy ``Parameter``-shaped facet view).
- :mod:`._proxy`    — ``_ParamProxyDict`` (legacy ``bundle.params`` dict).
- :mod:`._bundle`   — ``ParamBundle`` (the per-concept proxy itself).
- :mod:`._init`     — ``init_row_`` strategy dispatcher.
- :mod:`._opt`      — optimiser-state migration + iterator helpers.

Backward compatibility: every name that used to be importable from
``pcm.param_bundle`` is re-exported here. ``bundle.params[facet].data``,
``bundle.consumed_by[facet]``, ``bundle.state_dict()['params.<facet>']``
all behave identically to the pre-split code.
"""
from __future__ import annotations

from ._bundle import ParamBundle
from ._init import init_row_
from ._opt import (
    aggregate_consumed_by,
    iter_bundle_parameters,
    migrate_param_in_optimizer,
)
from ._proxy import _ParamProxyDict
from ._row_view import BundleRowView
from ._types import ContextualizedConcept, InitStrategy


__all__ = [
    # public-API classes
    "ContextualizedConcept",
    "BundleRowView",
    "ParamBundle",
    "InitStrategy",
    # functions
    "init_row_",
    "migrate_param_in_optimizer",
    "iter_bundle_parameters",
    "aggregate_consumed_by",
]
