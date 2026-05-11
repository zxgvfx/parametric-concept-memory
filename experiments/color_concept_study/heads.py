"""Color domain muscle heads (mix + adjacency)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcm.concept_graph import ConceptGraph
from pcm.sleep import collapse_with_optional_abstract

from ._config import (
    ADJ_DIM,
    BIAS_DIM,
    CALLER_ADJ,
    CALLER_MIX,
    EMBED_DIM,
    FACET_ADJ,
    FACET_MIX,
)


__all__ = ["ColorMixingHead", "ColorAdjacencyHead", "RipeFruitHead"]


class ColorMixingHead(nn.Module):
    """2 bundle → predicted bundle (会被 cosine-argmax 映射到 color class).

    类似 ArithmeticHeadV2, 但无 op_onehot (mixing 是对称的单一 op).
    Tier-G: ``use_abstract=True`` 时 collapse 走 anchor + residual 路径.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        bias_dim: int = BIAS_DIM,
        *,
        use_abstract: bool = False,
        assignment: str = "hard",
        soft_tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.bias_dim = bias_dim
        self.use_abstract = use_abstract
        self.assignment = assignment
        self.soft_tau = soft_tau
        self.fc1 = nn.Linear(2 * bias_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        concept_ids_a: list[str],
        concept_ids_b: list[str],
        cg: ConceptGraph,
        tick: int = 0,
    ) -> torch.Tensor:
        ba = self._collapse(concept_ids_a, cg, tick)
        bb = self._collapse(concept_ids_b, cg, tick)
        x = torch.cat([ba, bb], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def _collapse(self, ids: list[str], cg: ConceptGraph, tick: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return collapse_with_optional_abstract(
            cg, caller=CALLER_MIX, facet=FACET_MIX, concept_ids=ids,
            shape=(self.bias_dim,), tick=tick, init="normal_small", device=device,
            use_abstract=self.use_abstract,
            assignment=self.assignment, soft_tau=self.soft_tau,
        )


class ColorAdjacencyHead(nn.Module):
    """2 bundle → 3-class logits."""

    def __init__(
        self,
        facet_dim: int = ADJ_DIM,
        hidden: int = 64,
        n_classes: int = 3,
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

    def forward(
        self,
        concept_ids_a: list[str],
        concept_ids_b: list[str],
        cg: ConceptGraph,
        tick: int = 0,
    ) -> torch.Tensor:
        oa = self._collapse(concept_ids_a, cg, tick)
        ob = self._collapse(concept_ids_b, cg, tick)
        x = torch.cat([oa, ob], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def _collapse(self, ids: list[str], cg: ConceptGraph, tick: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return collapse_with_optional_abstract(
            cg, caller=CALLER_ADJ, facet=FACET_ADJ, concept_ids=ids,
            shape=(self.facet_dim,), tick=tick, init="normal_small", device=device,
            use_abstract=self.use_abstract,
            assignment=self.assignment, soft_tau=self.soft_tau,
        )


class RipeFruitHead(nn.Module):
    """PAPER §6.8 — single-input "is this hue a ripe fruit?" binary head.

    Mirrors the foraging-pressure leg of the trichromacy evolution
    hypothesis (Jacobs 2009 *Curr Biol*): primates with red-vs-green
    discrimination locate ripe fruit / young leaves more reliably,
    creating an asymmetric task signal that singles out a small
    subset of hues as "behaviourally salient".

    The head consumes ``mixing_bias`` (same facet as ``ColorMixingHead``
    so gradient lands on the same bundle rows) and outputs binary
    logits ``ripe / not-ripe``. ``ripe_set`` defaults to {0, 1, 11}
    on a 12-hue ring (a 3-hue red wedge), modelling "the red region
    of fruit ripeness".
    """

    def __init__(
        self,
        bias_dim: int = BIAS_DIM,
        hidden: int = 64,
        *,
        use_abstract: bool = False,
        assignment: str = "hard",
        soft_tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.bias_dim = bias_dim
        self.use_abstract = use_abstract
        self.assignment = assignment
        self.soft_tau = soft_tau
        self.fc1 = nn.Linear(bias_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 2)

    def forward(
        self,
        concept_ids: list[str],
        cg: ConceptGraph,
        tick: int = 0,
    ) -> torch.Tensor:
        x = self._collapse(concept_ids, cg, tick)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def _collapse(self, ids: list[str], cg: ConceptGraph, tick: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return collapse_with_optional_abstract(
            cg, caller="RipeFruitHead", facet=FACET_MIX, concept_ids=ids,
            shape=(self.bias_dim,), tick=tick, init="normal_small", device=device,
            use_abstract=self.use_abstract,
            assignment=self.assignment, soft_tau=self.soft_tau,
        )
