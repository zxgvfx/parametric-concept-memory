"""pcm.config — Bio-inspired pre-allocation defaults (8GB friendly).

This module centralises the capacity, growth and gate hyperparameters
used by the new dense ``bundle_pool`` storage. The defaults are tuned
for an 8GB consumer GPU (RTX 30/40-series 8GB, RTX 4060, etc.) and
cover the full paper-scale experiment matrix (N up to 100, 10 facets,
batch size 128) with ample safety margin.

The configuration follows the bionic narrative:

- ``INITIAL_CAPACITY`` — "neonatal cortical sheet" size; over-provisioned
  so a few thousand small experiments never trigger a grow.
- ``GROWTH_FACTOR`` — adult-neurogenesis-style amortised expansion;
  ``2.0`` gives O(log N) total reallocations.
- ``MAX_CAPACITY`` — VRAM safety stop; exceeding this raises rather
  than silently OOMing.

All values can be overridden via ``ConceptGraph(initial_capacity=...,
growth_factor=..., max_capacity=...)`` arguments.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Capacity / grow defaults (Tier A + Tier B).
# ---------------------------------------------------------------------------
# 8GB VRAM budget breakdown (see plan §2.6.1):
#   PyTorch / CUDA baseline            ~600 MB
#   bundle_pool (10 facets x 64-d)     ~80 MB at MAX_CAPACITY
#   AdamW moments (x2)                 ~160 MB
#   activations + cache                ~1 GB
#   ── total                            ~1.8 GB at full N_max
# Leaves ~6 GB safety margin on an 8GB card.
INITIAL_CAPACITY: int = 128
GROWTH_FACTOR: float = 2.0
MAX_CAPACITY: int = 32_000

# Default numerical batch size used by the paper experiments.
DEFAULT_BATCH_SIZE: int = 128

# ---------------------------------------------------------------------------
# Gate (Tier B): off by default to preserve Tier A bit-identical behaviour.
# ---------------------------------------------------------------------------
ENABLE_SLOT_GATE: bool = bool(int(os.environ.get("PCM_ENABLE_GATE", "0")))
GATE_INIT_LOGIT: float = -2.197  # sigmoid^{-1}(0.1); matches plan §3.B "0.1 mean"
GATE_DTYPE: torch.dtype = torch.float32
GATE_L0_LAMBDA: float = 1e-4
GATE_PRUNE_THRESHOLD: float = 0.05  # gate < 0.05 -> slot eligible for prune
GATE_GROW_THRESHOLD: float = 0.80   # mean(gate) > 0.80 -> auto-grow trigger

# ---------------------------------------------------------------------------
# PEER (Tier C): off by default; in 8GB micro-mode we cap at 16K experts.
# ---------------------------------------------------------------------------
ENABLE_PEER: bool = bool(int(os.environ.get("PCM_ENABLE_PEER", "0")))
PEER_NUM_EXPERTS_8GB: int = 16_384
PEER_NUM_EXPERTS_24GB: int = 1_048_576
PEER_TOP_K: int = 16
PEER_NUM_HEADS: int = 8


@dataclass(frozen=True)
class CapacityConfig:
    """Frozen snapshot of capacity / grow knobs (passed into ConceptGraph)."""

    initial_capacity: int = INITIAL_CAPACITY
    growth_factor: float = GROWTH_FACTOR
    max_capacity: int = MAX_CAPACITY

    def validate(self) -> None:
        if self.initial_capacity < 1:
            raise ValueError("initial_capacity must be >= 1")
        if self.growth_factor < 1.0:
            raise ValueError("growth_factor must be >= 1.0 (1.0 = no grow)")
        if self.max_capacity < self.initial_capacity:
            raise ValueError("max_capacity must be >= initial_capacity")


def default_config() -> CapacityConfig:
    """Return the 8GB-safe default ``CapacityConfig``."""
    cfg = CapacityConfig()
    cfg.validate()
    return cfg


__all__ = [
    "INITIAL_CAPACITY",
    "GROWTH_FACTOR",
    "MAX_CAPACITY",
    "DEFAULT_BATCH_SIZE",
    "ENABLE_SLOT_GATE",
    "GATE_INIT_LOGIT",
    "GATE_DTYPE",
    "GATE_L0_LAMBDA",
    "GATE_PRUNE_THRESHOLD",
    "GATE_GROW_THRESHOLD",
    "ENABLE_PEER",
    "PEER_NUM_EXPERTS_8GB",
    "PEER_NUM_EXPERTS_24GB",
    "PEER_TOP_K",
    "PEER_NUM_HEADS",
    "CapacityConfig",
    "default_config",
]
