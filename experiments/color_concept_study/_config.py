"""Module-level constants for the color concept study (paper §5)."""
from __future__ import annotations

import torch


__all__ = [
    "DEVICE",
    "N_COLORS",
    "EMBED_DIM",
    "BIAS_DIM",
    "ADJ_DIM",
    "BATCH_SIZE",
    "LR",
    "EPOCHS",
    "STEPS_PER_EPOCH",
    "CALLER_MIX",
    "FACET_MIX",
    "CALLER_ADJ",
    "FACET_ADJ",
]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_COLORS = 12
EMBED_DIM = 128
BIAS_DIM = 64
ADJ_DIM = 8
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 30
STEPS_PER_EPOCH = 200

CALLER_MIX = "ColorMixingHead"
FACET_MIX = "mixing_bias"
CALLER_ADJ = "ColorAdjacencyHead"
FACET_ADJ = "adjacency_offset"
