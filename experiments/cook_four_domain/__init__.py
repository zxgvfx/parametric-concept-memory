"""Cook vs direct end-to-end training comparison across paper §4-§6.3.

Pre-2026-05 the four-domain comparison lived in a single
``cook_four_domain.py`` (~650 lines). It has been split per domain so
each file stays under the 500-line cap.

Layout:

- :mod:`._common`   — DEVICE constant + ρ helpers + diff_metrics.
- :mod:`.number`    — §4 dual (ArithmeticHeadV2 + ComparisonHead, N=7).
- :mod:`.color`     — §5 dual (ColorMixingHead + ColorAdjacencyHead, 12 hues).
- :mod:`.space`     — §6.2 dual (MoveHead + DistanceHead, 5×5 grid).
- :mod:`.phoneme`   — §6.3 triple (Voicing + Manner + Place, 20 phonemes).
- :mod:`.__main__`  — CLI dispatcher.

Each ``train_<domain>(use_cook, seed, epochs, steps)`` runs one full
training pass using either the legacy ``head.forward(...)`` muscle path
or the Tier-D ``parametric_muscle_subgraph`` cook path. Backbones are
shared with the head via :class:`pcm.heads.HeadAsBackbone`, so the cook
trajectory is end-to-end bit-identical to the direct trajectory.
See ``docs/COOK_FOUR_DOMAIN_COMPARISON.md`` for the result table.
"""
from __future__ import annotations

from ._common import diff_metrics, rho_linear, stack_rows
from .color import train_color
from .number import train_number
from .phoneme import train_phoneme
from .space import train_space


__all__ = [
    "stack_rows", "rho_linear", "diff_metrics",
    "train_number", "train_color", "train_space", "train_phoneme",
]
