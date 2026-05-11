"""Optimiser-state migration + bundle aggregation helpers.

Houses the G4 capacity-grow invariant (preserve per-row Adam moments)
and two small convenience functions that earlier callers imported
from ``pcm.param_bundle`` directly.
"""
from __future__ import annotations

from typing import Iterable, Iterator

import torch
import torch.nn as nn

from ._bundle import ParamBundle


__all__ = [
    "migrate_param_in_optimizer",
    "iter_bundle_parameters",
    "aggregate_consumed_by",
]


def migrate_param_in_optimizer(
    optimizer: torch.optim.Optimizer,
    old_p: nn.Parameter,
    new_p: nn.Parameter,
    old_capacity: int,
) -> None:
    """Replace ``old_p`` with ``new_p`` inside ``optimizer``, copying per-row
    moment buffers (``exp_avg``, ``exp_avg_sq`` etc.) for the first
    ``old_capacity`` rows. New rows get zero moments (G4 invariant).

    Works for AdamW / Adam / SGD with momentum / RMSprop. Unknown
    optimizer state buffers are zero-initialised on the new shape so the
    next ``step`` does not blow up.
    """
    state_old = optimizer.state.pop(old_p, None)
    if state_old is None:
        # Param was never stepped; just rebind in param_groups.
        for group in optimizer.param_groups:
            group["params"] = [new_p if p is old_p else p for p in group["params"]]
        return

    state_new: dict = {}
    for key, val in state_old.items():
        if isinstance(val, torch.Tensor):
            if val.shape == old_p.shape:
                # Per-element buffer (exp_avg, exp_avg_sq, momentum_buffer, ...)
                buf = torch.zeros_like(new_p.data)
                buf[:old_capacity] = val
                state_new[key] = buf
            else:
                # Scalar tensor (step in some optimizers); copy verbatim.
                state_new[key] = val.clone()
        else:
            state_new[key] = val

    optimizer.state[new_p] = state_new

    for group in optimizer.param_groups:
        group["params"] = [new_p if p is old_p else p for p in group["params"]]


def iter_bundle_parameters(bundles: Iterable[ParamBundle]) -> Iterator[nn.Parameter]:
    """Compat shim. With the dense-pool architecture this is degenerate
    (per-bundle parameters live on the graph) — callers should use
    ``ConceptGraph.iter_bundle_parameters`` directly.
    """
    seen: set[int] = set()
    for b in bundles:
        if b._graph is None:
            continue
        for p in b._graph.bundle_pool.values():
            if id(p) in seen:
                continue
            seen.add(id(p))
            yield p


def aggregate_consumed_by(
    bundles: dict[str, ParamBundle],
) -> dict[str, dict[str, set[str]]]:
    """Cross-bundle attribution merge (concept_id × facet × {callers})."""
    return {cid: {f: set(c) for f, c in b.consumed_by.items()} for cid, b in bundles.items()}
