"""Paper-figure renderer (split package).

Pre-2026-05 every figure lived in a single ``render_paper_figures.py``
(~620 lines). It has been split into one file per figure plus the
shared ``_style`` and ``_bundles`` helpers. The CLI invocation
``python -m experiments.render_paper_figures`` continues to work
unchanged.

Per-figure modules:

- :mod:`.F2_number`        — render_F2_number_cos_heatmaps
- :mod:`.F4_four_domain`   — render_F4_four_domain_panel
- :mod:`.F5_base10`        — render_F5_base10_spike_null
- :mod:`.F6_swap`          — render_F6_swap_dissociation
- :mod:`.F7_alignment`     — render_F7_h5pp_alignment_schema
- :mod:`.F8_space_mds`     — render_F8_space_mds_trained_vs_shuffle
"""
from __future__ import annotations

from .F2_number import render_F2_number_cos_heatmaps
from .F4_four_domain import render_F4_four_domain_panel
from .F5_base10 import render_F5_base10_spike_null
from .F6_swap import render_F6_swap_dissociation
from .F7_alignment import render_F7_h5pp_alignment_schema
from .F8_space_mds import render_F8_space_mds_trained_vs_shuffle


FIGURES = {
    "F2": render_F2_number_cos_heatmaps,
    "F4": render_F4_four_domain_panel,
    "F5": render_F5_base10_spike_null,
    "F6": render_F6_swap_dissociation,
    "F7": render_F7_h5pp_alignment_schema,
    "F8": render_F8_space_mds_trained_vs_shuffle,
}


__all__ = [
    "FIGURES",
    "render_F2_number_cos_heatmaps",
    "render_F4_four_domain_panel",
    "render_F5_base10_spike_null",
    "render_F6_swap_dissociation",
    "render_F7_h5pp_alignment_schema",
    "render_F8_space_mds_trained_vs_shuffle",
]
