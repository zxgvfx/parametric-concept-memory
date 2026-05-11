"""Module-level constants for the phoneme study (paper §6.3)."""
from __future__ import annotations

import torch


__all__ = [
    "DEVICE",
    "EMBED_DIM", "VOICE_DIM", "MANNER_DIM", "PLACE_DIM",
    "BATCH_SIZE", "LR", "EPOCHS", "STEPS_PER_EPOCH",
    "N_VOICE", "N_MANNER", "N_PLACE",
    "CALLER_V", "FACET_V",
    "CALLER_M", "FACET_M",
    "CALLER_P", "FACET_P",
]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_DIM = 128
VOICE_DIM = 16
MANNER_DIM = 16
PLACE_DIM = 16
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 60
STEPS_PER_EPOCH = 120

N_VOICE, N_MANNER, N_PLACE = 2, 4, 4

CALLER_V, FACET_V = "VoicingHead", "voice_bias"
CALLER_M, FACET_M = "MannerHead", "manner_bias"
CALLER_P, FACET_P = "PlaceHead", "place_bias"
