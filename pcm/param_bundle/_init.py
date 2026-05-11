"""Per-row tensor initialisation strategies.

Pure functions (no graph dependencies); used by
:meth:`ConceptGraph._init_slot_if_unset` on first observation of a
``(facet, slot)`` pair so paper §4.4 small-init feature-learning
behaviour stays identical to the pre-D93 ``_init_parameter``.
"""
from __future__ import annotations

import torch

from ._types import InitStrategy


__all__ = ["init_row_"]


def init_row_(
    row: torch.Tensor,
    strategy: InitStrategy,
    generator: torch.Generator | None = None,
) -> None:
    """Initialise a single row in-place. Matches the old ``_init_parameter`` semantics
    so D91 paper-claim regressions stay bit-identical (within RNG order).

    ``generator`` is optional; when ``None``, ``torch.randn_like`` (default RNG)
    is used. Pass an explicit generator if a specific deterministic schedule is
    required.
    """
    if strategy == "zero":
        row.zero_()
    elif strategy == "normal_small":
        if generator is None:
            row.normal_(0.0, 1.0).mul_(0.01)
        else:
            tmp = torch.randn(row.shape, device=row.device, dtype=row.dtype,
                              generator=generator)
            row.copy_(tmp.mul_(0.01))
    elif strategy == "normal":
        if generator is None:
            row.normal_(0.0, 1.0)
        else:
            tmp = torch.randn(row.shape, device=row.device, dtype=row.dtype,
                              generator=generator)
            row.copy_(tmp)
    elif strategy == "identity":
        row.fill_(1.0)
    else:
        raise ValueError(f"unknown init strategy: {strategy}")
