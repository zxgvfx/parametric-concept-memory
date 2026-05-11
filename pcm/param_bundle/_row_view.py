"""Tensor-like row-view proxy backing legacy ``bundle.params[facet]``."""
from __future__ import annotations

import torch
import torch.nn as nn


__all__ = ["BundleRowView"]


class BundleRowView:
    """Tensor-like view of a single row in ``bundle_pool[facet]``.

    Provides the legacy ``nn.Parameter`` surface that downstream code
    relies on:

    - ``.data``      — read/write row view (writeable; ``copy_`` works)
    - ``.grad``      — row of ``pool.grad`` (or ``None`` if no backward yet)
    - ``.requires_grad`` — mirrors the pool's flag
    - ``.shape`` / ``.numel()`` / ``.detach()`` / ``.clone()`` / ``.cpu()`` / ``.zero_()``
    - ``.device`` / ``.dtype``

    This object is **not** a real ``nn.Parameter`` — it cannot be
    appended to ``optimizer.param_groups`` directly. The optimizer must
    receive the underlying pool via ``ConceptGraph.iter_bundle_parameters``.
    """

    __slots__ = ("_pool", "_slot_idx")

    def __init__(self, pool: nn.Parameter, slot_idx: int) -> None:
        self._pool = pool
        self._slot_idx = int(slot_idx)

    # ── identity-ish accessors ──────────────────────────────────────────────

    @property
    def data(self) -> torch.Tensor:
        # Returns a *view* into pool.data. Mutating in place affects pool.
        return self._pool.data[self._slot_idx]

    @data.setter
    def data(self, value: torch.Tensor) -> None:
        self._pool.data[self._slot_idx] = value

    @property
    def grad(self) -> torch.Tensor | None:
        if self._pool.grad is None:
            return None
        return self._pool.grad[self._slot_idx]

    @property
    def requires_grad(self) -> bool:
        return bool(self._pool.requires_grad)

    @property
    def shape(self) -> torch.Size:
        return self._pool.data[self._slot_idx].shape

    @property
    def device(self) -> torch.device:
        return self._pool.device

    @property
    def dtype(self) -> torch.dtype:
        return self._pool.dtype

    # ── tensor-like ops ─────────────────────────────────────────────────────

    def numel(self) -> int:
        return int(self._pool.data[self._slot_idx].numel())

    def detach(self) -> torch.Tensor:
        return self._pool.data[self._slot_idx].detach()

    def cpu(self) -> torch.Tensor:
        return self._pool.data[self._slot_idx].detach().cpu()

    def clone(self) -> torch.Tensor:
        return self._pool.data[self._slot_idx].clone()

    def zero_(self) -> "BundleRowView":
        with torch.no_grad():
            self._pool.data[self._slot_idx].zero_()
        return self

    def __repr__(self) -> str:
        return (
            f"BundleRowView(slot={self._slot_idx}, "
            f"shape={tuple(self.shape)}, device={self.device})"
        )
