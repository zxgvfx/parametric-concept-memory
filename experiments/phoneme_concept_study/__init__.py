"""Phoneme concept study (paper §6.3) — split package.

Pre-2026-05 this was a single ``phoneme_concept_study.py`` (~512 lines).
Layout:

- :mod:`._config`   — module constants (DEVICE, dims, facet names).
- :mod:`.inventory` — PHONEMES table + cid_of / feat_of / hamming /
                      build_phoneme_graph / _apply_shuffle.
- :mod:`.heads`     — ``_SingleInputHead`` + 3 attribute-axis builders.
- :mod:`.train`     — ``train_one`` (single_v / single_m / single_p / triple).
- :mod:`.metrics`   — ρ helpers + intra/inter gap + permutation test.
- :mod:`.runners`   — ``run_e1_multi_seed`` + ``run_e2_shuffled``.
- :mod:`.__main__`  — CLI entry.

External callers (e.g. :mod:`experiments.render_paper_figures._bundles`)
import names like ``cid_of`` / ``train_one`` / ``PHONEMES`` /
``FACET_V`` directly from the top-level package.
"""
from __future__ import annotations

from ._config import (
    BATCH_SIZE,
    CALLER_M,
    CALLER_P,
    CALLER_V,
    DEVICE,
    EMBED_DIM,
    EPOCHS,
    FACET_M,
    FACET_P,
    FACET_V,
    LR,
    MANNER_DIM,
    N_MANNER,
    N_PLACE,
    N_VOICE,
    PLACE_DIM,
    STEPS_PER_EPOCH,
    VOICE_DIM,
)
from .heads import (
    _SingleInputHead,
    build_manner_head,
    build_place_head,
    build_voicing_head,
)
from .inventory import (
    N_PH,
    PHONEMES,
    _apply_shuffle,
    build_phoneme_graph,
    cid_of,
    feat_of,
    hamming,
)
from .metrics import (
    _cos_matrix,
    _cross_facet_align,
    _intra_vs_inter_gap,
    _perm_test_align,
    _rho_hamming_total,
    _rho_same_axis,
)
from .runners import run_e1_multi_seed, run_e2_shuffled
from .train import train_one


__all__ = [
    # config
    "DEVICE",
    "EMBED_DIM", "VOICE_DIM", "MANNER_DIM", "PLACE_DIM",
    "BATCH_SIZE", "LR", "EPOCHS", "STEPS_PER_EPOCH",
    "N_VOICE", "N_MANNER", "N_PLACE",
    "CALLER_V", "FACET_V",
    "CALLER_M", "FACET_M",
    "CALLER_P", "FACET_P",
    # inventory
    "PHONEMES", "N_PH",
    "cid_of", "feat_of", "hamming",
    "build_phoneme_graph",
    # heads
    "_SingleInputHead",
    "build_voicing_head", "build_manner_head", "build_place_head",
    # train
    "train_one",
    # metrics (private leading underscore preserved)
    "_cos_matrix", "_rho_hamming_total", "_rho_same_axis",
    "_intra_vs_inter_gap", "_cross_facet_align", "_perm_test_align",
    # runners
    "run_e1_multi_seed", "run_e2_shuffled",
]
