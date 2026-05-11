"""Purity audit (paper §4.4 information-leakage ablations) — split package.

Pre-2026-05 this lived in a single ``purity_audit.py`` (~545 lines).
Layout:

- :mod:`.centroids`     — random-orthogonal / random-gaussian centroid factories.
- :mod:`.graph_builder` — ``build_graph_with_id_fn`` (custom-id graph).
- :mod:`.train`         — ``purity_train_one`` universal training loop.
- :mod:`.metrics`       — ``rho_by_n`` + ``rho_with_inverse_remap`` + helpers.
- :mod:`.assays`        — A1 / A2 / A3 / A4 ablation runners.
- :mod:`.reporting`     — markdown report renderer.
- :mod:`.__main__`      — CLI entry.

Public API preserved for sibling experiments
(``scale_study`` / ``quad_study`` / ``emergent_base10_study``) that
import ``build_graph_with_id_fn`` and ``make_random_orthogonal_centroids``
directly from this package.
"""
from __future__ import annotations

from .assays import (
    assay_a1_random_centroids,
    assay_a2_shuffle_inverse,
    assay_a3_init_scale,
    assay_a4_random_id,
)
from .centroids import (
    make_random_gaussian_centroids,
    make_random_orthogonal_centroids,
)
from .graph_builder import build_graph_with_id_fn
from .metrics import (
    _cos_matrix_by_n,
    _stats,
    rho_by_n,
    rho_with_inverse_remap,
)
from .reporting import render_report
from .train import purity_train_one


__all__ = [
    # centroids
    "make_random_orthogonal_centroids",
    "make_random_gaussian_centroids",
    # graph builder
    "build_graph_with_id_fn",
    # train
    "purity_train_one",
    # metrics
    "_cos_matrix_by_n", "rho_by_n", "rho_with_inverse_remap", "_stats",
    # assays
    "assay_a1_random_centroids",
    "assay_a2_shuffle_inverse",
    "assay_a3_init_scale",
    "assay_a4_random_id",
    # reporting
    "render_report",
]
