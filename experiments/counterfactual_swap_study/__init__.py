"""Appendix B: bundle = concept identity 的因果证据 (split package).

Pre-2026-05 this lived in a single ``counterfactual_swap_study.py``
(~580 lines). It has been split across:

- :mod:`._config`         — module-level constants.
- :mod:`.swap_ops`        — the in-place swap primitive.
- :mod:`.number_domain`   — number-domain training + per-seed loop.
- :mod:`.color_domain`    — color-domain training + per-seed loop.
- :mod:`.reporting`       — aggregation + console summary.
- :mod:`.__main__`        — CLI entry point.

Public API is preserved: every name that used to be importable from
``experiments.counterfactual_swap_study`` is re-exported here. ``python
-m experiments.counterfactual_swap_study`` still works.

以往的证据 (shuffle / ρ / cross-facet alignment) 都是**相关性**层面;
本实验做**因果介入**: 训练完后把两个 concept 的 bundle 直接互换, 看
下游推理是否跟着 "语义互换", 以此证明 bundle params ≡ concept identity
itself (not just a correlate). 同时用 facet-specific swap 做 double-
dissociation.
"""
from __future__ import annotations

from ._config import (
    COLOR_SWAP_A,
    COLOR_SWAP_B,
    DEVICE,
    NUM_BATCH_SIZE,
    NUM_BIAS_DIM,
    NUM_EMBED_DIM,
    NUM_EPOCHS,
    NUM_LR,
    NUM_N_MAX,
    NUM_N_MIN,
    NUM_ORD_DIM,
    NUM_STEPS_PER_EPOCH,
    NUM_SWAP_A,
    NUM_SWAP_B,
)
from .color_domain import (
    eval_color_adj,
    eval_color_mix,
    run_color_seed,
    train_color_dual,
)
from .number_domain import (
    build_number_graph,
    eval_number_add,
    eval_number_cmp,
    run_number_seed,
    train_number_dual,
)
from .reporting import agg_stats, print_color_summary, print_number_summary
from .swap_ops import swap_all_facets, swap_bundle_facet


__all__ = [
    # config
    "DEVICE",
    "NUM_EMBED_DIM", "NUM_BIAS_DIM", "NUM_ORD_DIM",
    "NUM_N_MIN", "NUM_N_MAX",
    "NUM_BATCH_SIZE", "NUM_LR",
    "NUM_EPOCHS", "NUM_STEPS_PER_EPOCH",
    "NUM_SWAP_A", "NUM_SWAP_B",
    "COLOR_SWAP_A", "COLOR_SWAP_B",
    # swap_ops
    "swap_bundle_facet", "swap_all_facets",
    # number domain
    "build_number_graph",
    "train_number_dual",
    "eval_number_add", "eval_number_cmp",
    "run_number_seed",
    # color domain
    "train_color_dual",
    "eval_color_mix", "eval_color_adj",
    "run_color_seed",
    # reporting
    "agg_stats",
    "print_number_summary", "print_color_summary",
]
