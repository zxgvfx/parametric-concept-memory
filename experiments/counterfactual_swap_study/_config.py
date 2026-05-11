"""Module-level constants for the counterfactual swap study (Appendix B).

Exposed as a private module so the per-domain modules and ``__main__``
share the exact same numbers without circular imports.
"""
from __future__ import annotations

import torch


__all__ = [
    "DEVICE",
    "NUM_EMBED_DIM",
    "NUM_BIAS_DIM",
    "NUM_ORD_DIM",
    "NUM_N_MIN",
    "NUM_N_MAX",
    "NUM_BATCH_SIZE",
    "NUM_LR",
    "NUM_EPOCHS",
    "NUM_STEPS_PER_EPOCH",
    "NUM_SWAP_A",
    "NUM_SWAP_B",
    "COLOR_SWAP_A",
    "COLOR_SWAP_B",
]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────────────────────────────
# Number-domain config (paper §B.1)
# ──────────────────────────────────────────────────────────────────────
NUM_EMBED_DIM = 128
NUM_BIAS_DIM = 64
NUM_ORD_DIM = 8
NUM_N_MIN = 1
NUM_N_MAX = 7
NUM_BATCH_SIZE = 32
NUM_LR = 1e-3
NUM_EPOCHS = 12
NUM_STEPS_PER_EPOCH = 120

# Swap pair: pick two concepts with max inter-distance that still allow
# plenty of (a, b) pairs on each side (避开 boundary 1 和 7 以便 add/sub
# 都能采样).
NUM_SWAP_A = 3
NUM_SWAP_B = 5

# ──────────────────────────────────────────────────────────────────────
# Color-domain config (paper §B.2)
# ──────────────────────────────────────────────────────────────────────
# circular_dist=3, not antipodal — mixing triples contain (2, 5).
COLOR_SWAP_A = 2
COLOR_SWAP_B = 5
