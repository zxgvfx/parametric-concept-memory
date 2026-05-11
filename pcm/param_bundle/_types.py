"""Type aliases + ephemeral D92 collapse handle.

``ContextualizedConcept`` is the single most-passed-around datatype in
the muscle-head hot path; defined here in its own module so other
parts of the package can ``from ._types import ContextualizedConcept``
without dragging in tensor-view machinery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


__all__ = ["InitStrategy", "ContextualizedConcept"]


InitStrategy = Literal["zero", "normal_small", "normal", "identity"]


@dataclass
class ContextualizedConcept:
    """Concept materialised under a (caller, facet) observation (D92).

    Carries all of the concept's **current semantic content** towards the
    caller. Ephemeral: one is produced per ``ConceptNode.collapse`` call
    and discarded once the surrounding forward returns.
    """

    concept_id: str
    caller: str
    facet: str
    facet_params: torch.Tensor   # row view into the dense pool
    tick: int = 0

    def as_tensor(self) -> torch.Tensor:
        return self.facet_params

    def __repr__(self) -> str:
        return (
            f"CC(id={self.concept_id}, caller={self.caller}, facet={self.facet}, "
            f"shape={tuple(self.facet_params.shape)}, tick={self.tick})"
        )
