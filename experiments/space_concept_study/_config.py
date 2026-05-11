"""Module-level constants for the space concept study (5×5 grid, paper §6.2).

Imported by every sibling module so caller / facet names and grid
dimensions are defined exactly once.
"""
from __future__ import annotations

import torch


__all__ = [
    "DEVICE",
    "N_ROWS", "N_COLS", "N_CELLS",
    "EMBED_DIM", "MOTION_DIM", "DIST_DIM",
    "BATCH_SIZE", "LR", "EPOCHS", "STEPS_PER_EPOCH",
    "MOVE_CLASSES",
    "CLS_UP", "CLS_DOWN", "CLS_LEFT", "CLS_RIGHT", "CLS_SAME",
    "CALLER_MOVE", "FACET_MOVE",
    "CALLER_DIST", "FACET_DIST",
]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Grid + dim config
N_ROWS = 5
N_COLS = 5
N_CELLS = N_ROWS * N_COLS          # 25
EMBED_DIM = 128
MOTION_DIM = 64
DIST_DIM = 8
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 30
STEPS_PER_EPOCH = 200

# Move classes (5-class)
MOVE_CLASSES = ["up", "down", "left", "right", "same"]
CLS_UP, CLS_DOWN, CLS_LEFT, CLS_RIGHT, CLS_SAME = 0, 1, 2, 3, 4

# Muscle registry
CALLER_MOVE = "MoveHead"
FACET_MOVE = "motion_bias"
CALLER_DIST = "DistanceHead"
FACET_DIST = "distance_offset"
