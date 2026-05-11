"""Single-input phoneme classification heads (one per attribute axis)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcm.concept_graph import ConceptGraph
from pcm.sleep import collapse_with_optional_abstract

from ._config import (
    CALLER_M,
    CALLER_P,
    CALLER_V,
    FACET_M,
    FACET_P,
    FACET_V,
    MANNER_DIM,
    N_MANNER,
    N_PLACE,
    N_VOICE,
    PLACE_DIM,
    VOICE_DIM,
)


__all__ = [
    "_SingleInputHead",
    "build_voicing_head",
    "build_manner_head",
    "build_place_head",
]


class _SingleInputHead(nn.Module):
    """(phoneme) → n_classes logits, consuming a single facet.

    Tier-G: ``use_abstract=True`` 时 collapse 走 anchor + residual 路径.
    """

    def __init__(
        self,
        caller: str,
        facet: str,
        facet_dim: int,
        n_classes: int,
        hidden: int = 64,
        *,
        use_abstract: bool = False,
        assignment: str = "hard",
        soft_tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.caller = caller
        self.facet = facet
        self.facet_dim = facet_dim
        self.use_abstract = use_abstract
        self.assignment = assignment
        self.soft_tau = soft_tau
        self.fc1 = nn.Linear(facet_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, ids: list[str], cg: ConceptGraph, tick: int = 0) -> torch.Tensor:
        x = self._collapse(ids, cg, tick)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def _collapse(self, ids: list[str], cg: ConceptGraph, tick: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return collapse_with_optional_abstract(
            cg, caller=self.caller, facet=self.facet, concept_ids=ids,
            shape=(self.facet_dim,), tick=tick, init="normal_small", device=device,
            use_abstract=self.use_abstract,
            assignment=self.assignment, soft_tau=self.soft_tau,
        )


def _make(caller: str, facet: str, facet_dim: int, n_classes: int,
          *, use_abstract: bool, assignment: str,
          soft_tau: float) -> _SingleInputHead:
    return _SingleInputHead(
        caller, facet, facet_dim, n_classes,
        use_abstract=use_abstract, assignment=assignment, soft_tau=soft_tau,
    )


def build_voicing_head(*, use_abstract: bool = False, assignment: str = "hard",
                       soft_tau: float = 0.5) -> _SingleInputHead:
    return _make(CALLER_V, FACET_V, VOICE_DIM, N_VOICE,
                 use_abstract=use_abstract, assignment=assignment,
                 soft_tau=soft_tau)


def build_manner_head(*, use_abstract: bool = False, assignment: str = "hard",
                      soft_tau: float = 0.5) -> _SingleInputHead:
    return _make(CALLER_M, FACET_M, MANNER_DIM, N_MANNER,
                 use_abstract=use_abstract, assignment=assignment,
                 soft_tau=soft_tau)


def build_place_head(*, use_abstract: bool = False, assignment: str = "hard",
                     soft_tau: float = 0.5) -> _SingleInputHead:
    return _make(CALLER_P, FACET_P, PLACE_DIM, N_PLACE,
                 use_abstract=use_abstract, assignment=assignment,
                 soft_tau=soft_tau)
