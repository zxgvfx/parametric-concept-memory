"""Color concept study (paper §5 hue wheel) — split package.

Pre-2026-05 this was a single ``color_concept_study.py`` (~568 lines).
Layout:

- :mod:`._config`        — module constants (DEVICE, dims, facet names).
- :mod:`.topology`       — circular_dist + mix_pair + triple enumeration.
- :mod:`.heads`          — ``ColorMixingHead`` + ``ColorAdjacencyHead``.
- :mod:`.graph_builder`  — ``build_color_graph`` + centroid helper.
- :mod:`.train`          — ``train_one``.
- :mod:`.metrics`        — ``_cos_matrix`` + ρ helpers.
- :mod:`.runners`        — ``run_e1_multi_seed`` + ``run_e2_shuffled`` +
                           ``run_e4_permutation``.
- :mod:`.__main__`       — CLI entry.

Public API is preserved: every name that used to be importable from
``experiments.color_concept_study`` is re-exported here. Sister modules
(``counterfactual_swap_study``, ``render_paper_figures``) import these
names directly from this package.
"""
from __future__ import annotations

from ._config import (
    ADJ_DIM,
    BATCH_SIZE,
    BIAS_DIM,
    CALLER_ADJ,
    CALLER_MIX,
    DEVICE,
    EMBED_DIM,
    EPOCHS,
    FACET_ADJ,
    FACET_MIX,
    LR,
    N_COLORS,
    STEPS_PER_EPOCH,
)
from .graph_builder import (
    _apply_shuffle,
    build_color_graph,
    make_random_orthogonal_centroids,
)
from .heads import ColorAdjacencyHead, ColorMixingHead
from .metrics import (
    _cos_matrix,
    _cross_facet_align,
    _rho_circular,
    _rho_linear,
)
from .runners import run_e1_multi_seed, run_e2_shuffled, run_e4_permutation
from .topology import (
    circular_dist,
    enumerate_adjacency_triples,
    enumerate_mixing_triples,
    mix_pair,
)
from .train import train_one


__all__ = [
    # config
    "DEVICE",
    "N_COLORS", "EMBED_DIM", "BIAS_DIM", "ADJ_DIM",
    "BATCH_SIZE", "LR", "EPOCHS", "STEPS_PER_EPOCH",
    "CALLER_MIX", "FACET_MIX", "CALLER_ADJ", "FACET_ADJ",
    # topology
    "circular_dist", "mix_pair",
    "enumerate_mixing_triples", "enumerate_adjacency_triples",
    # heads
    "ColorMixingHead", "ColorAdjacencyHead",
    # graph_builder
    "build_color_graph", "make_random_orthogonal_centroids",
    # train
    "train_one",
    # metrics (private leading underscore preserved)
    "_cos_matrix", "_rho_circular", "_rho_linear", "_cross_facet_align",
    # runners
    "run_e1_multi_seed", "run_e2_shuffled", "run_e4_permutation",
]
