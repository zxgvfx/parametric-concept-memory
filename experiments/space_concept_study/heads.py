"""Muscle heads for the 5×5 grid study.

- :class:`MoveHead`     consumes ``motion_bias`` (64-d), 5-class direction.
- :class:`DistanceHead` consumes ``distance_offset`` (8-d), 9-class L1.

Both use the dense-pool ``cg.collapse_batch`` fast path. With
``use_abstract=True`` the collapse is routed via Tier-G's
:func:`pcm.sleep.collapse_with_optional_abstract` (anchor + residual).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcm.sleep import collapse_with_optional_abstract

from ._config import (
    CALLER_DIST,
    CALLER_MOVE,
    DIST_DIM,
    FACET_DIST,
    FACET_MOVE,
    MOTION_DIM,
)


__all__ = ["MoveHead", "DistanceHead"]


class MoveHead(nn.Module):
    """(cell_a, cell_b) → 5-class direction logits, 消费 motion_bias."""

    def __init__(
        self,
        facet_dim: int = MOTION_DIM,
        hidden: int = 128,
        n_classes: int = 5,
        *,
        use_abstract: bool = False,
        assignment: str = "hard",
        soft_tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.facet_dim = facet_dim
        self.use_abstract = use_abstract
        self.assignment = assignment
        self.soft_tau = soft_tau
        self.fc1 = nn.Linear(2 * facet_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, ids_a, ids_b, cg, tick=0):
        ba = self._collapse(ids_a, cg, tick)
        bb = self._collapse(ids_b, cg, tick)
        x = torch.cat([ba, bb], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def _collapse(self, ids, cg, tick):
        device = next(self.parameters()).device
        return collapse_with_optional_abstract(
            cg, caller=CALLER_MOVE, facet=FACET_MOVE, concept_ids=ids,
            shape=(self.facet_dim,), tick=tick, init="normal_small", device=device,
            use_abstract=self.use_abstract,
            assignment=self.assignment, soft_tau=self.soft_tau,
        )


class DistanceHead(nn.Module):
    """(cell_a, cell_b) → 9-class L1 distance logits, 消费 distance_offset."""

    def __init__(
        self,
        facet_dim: int = DIST_DIM,
        hidden: int = 64,
        n_classes: int = 9,
        *,
        use_abstract: bool = False,
        assignment: str = "hard",
        soft_tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.facet_dim = facet_dim
        self.use_abstract = use_abstract
        self.assignment = assignment
        self.soft_tau = soft_tau
        self.fc1 = nn.Linear(2 * facet_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, ids_a, ids_b, cg, tick=0):
        oa = self._collapse(ids_a, cg, tick)
        ob = self._collapse(ids_b, cg, tick)
        x = torch.cat([oa, ob], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def _collapse(self, ids, cg, tick):
        device = next(self.parameters()).device
        return collapse_with_optional_abstract(
            cg, caller=CALLER_DIST, facet=FACET_DIST, concept_ids=ids,
            shape=(self.facet_dim,), tick=tick, init="normal_small", device=device,
            use_abstract=self.use_abstract,
            assignment=self.assignment, soft_tau=self.soft_tau,
        )
