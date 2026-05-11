"""Space concept study (paper §6.2 5×5 lattice) — split package.

Pre-2026-05 this was a single ``space_concept_study.py`` (~565 lines).
Split layout:

- :mod:`._config`   — module constants (DEVICE, dims, facet names).
- :mod:`.topology`  — grid coord / triple enumeration helpers.
- :mod:`.heads`     — ``MoveHead`` + ``DistanceHead``.
- :mod:`.train`     — ``build_space_graph`` + ``train_one``.
- :mod:`.metrics`   — ``_cos_matrix`` + ρ helpers + ``_mds_grid_fit``.
- :mod:`.runners`   — ``run_e1_multi_seed`` + ``run_e2_shuffled`` +
                      ``run_e4_permutation``.
- :mod:`.__main__`  — CLI entry.

External callers (e.g. :mod:`experiments.render_paper_figures._bundles`)
import names like ``cid_of`` / ``train_one`` / ``N_ROWS`` / ``FACET_MOVE``
directly from the top-level package; those are re-exported here.
"""
from __future__ import annotations

from ._config import (
    BATCH_SIZE,
    CALLER_DIST,
    CALLER_MOVE,
    CLS_DOWN,
    CLS_LEFT,
    CLS_RIGHT,
    CLS_SAME,
    CLS_UP,
    DEVICE,
    DIST_DIM,
    EMBED_DIM,
    EPOCHS,
    FACET_DIST,
    FACET_MOVE,
    LR,
    MOTION_DIM,
    MOVE_CLASSES,
    N_CELLS,
    N_COLS,
    N_ROWS,
    STEPS_PER_EPOCH,
)
from .heads import DistanceHead, MoveHead
from .metrics import (
    _cos_matrix,
    _cross_facet_align,
    _mds_grid_fit,
    _rho_col_within,
    _rho_L1,
    _rho_linear_flat,
    _rho_row_within,
)
from .runners import run_e1_multi_seed, run_e2_shuffled, run_e4_permutation
from .topology import (
    cid_of,
    enumerate_distance_triples,
    enumerate_move_triples,
    idx_of_rc,
    l1_dist,
    move_class,
    rc_of_idx,
)
from .train import build_space_graph, train_one


__all__ = [
    # config
    "DEVICE", "N_ROWS", "N_COLS", "N_CELLS",
    "EMBED_DIM", "MOTION_DIM", "DIST_DIM",
    "BATCH_SIZE", "LR", "EPOCHS", "STEPS_PER_EPOCH",
    "MOVE_CLASSES",
    "CLS_UP", "CLS_DOWN", "CLS_LEFT", "CLS_RIGHT", "CLS_SAME",
    "CALLER_MOVE", "FACET_MOVE",
    "CALLER_DIST", "FACET_DIST",
    # topology
    "cid_of", "rc_of_idx", "idx_of_rc", "l1_dist", "move_class",
    "enumerate_move_triples", "enumerate_distance_triples",
    # heads
    "MoveHead", "DistanceHead",
    # train
    "build_space_graph", "train_one",
    # metrics (private leading underscore preserved for back-compat)
    "_cos_matrix", "_rho_L1", "_rho_linear_flat",
    "_rho_row_within", "_rho_col_within",
    "_cross_facet_align", "_mds_grid_fit",
    # runners
    "run_e1_multi_seed", "run_e2_shuffled", "run_e4_permutation",
]
