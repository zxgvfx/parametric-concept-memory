"""PAPER §7.5-space — three-causal-layer priors for the spatial domain.

Mirror of ``experiments.number_decimal_priors`` /
``experiments.phoneme_transfer_priors`` for a 2-D lattice.
Provides:

* **B layer (biological / hardware prior)** —
  :func:`make_cardinal_axis_centroids` builds row + col axis
  cones (analogous to LMS for colour, decimal cones for
  numbers, articulator cones for phonemes); each cell's
  centroid is the sum of its row-axis and col-axis cone
  projections, mirroring the place-cell × grid-cell
  factorisation in mammalian spatial cognition (Hafting et al.
  2005 *Nature*; O'Keefe & Burgess 1996).
* **C layer (ecological statistics)** — :func:`center_bias_weights`
  per-cell sampling that boosts central cells, modelling the
  natural distribution of movement frequency in a bounded
  environment.
* **D layer (task-driven asymmetry)** — :class:`RowIndexHead` is
  a single-input head that classifies each cell's *row index*,
  exposing a row identity signal across the entire grid
  (including held-out outer-ring cells when the head is
  trained on the full inventory).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcm.sleep import collapse_with_optional_abstract


__all__ = [
    "make_cardinal_axis_centroids",
    "center_bias_weights",
    "RowIndexHead",
]


# ────────────────────────────────────────────────────────────────────────
# B layer — cardinal-axis centroids (row × col cone factorisation)
# ────────────────────────────────────────────────────────────────────────


def make_cardinal_axis_centroids(
    n_rows: int, n_cols: int, dim: int, seed: int,
    *, row_weight: float = 1.0, col_weight: float = 1.0,
) -> torch.Tensor:
    """Build cell centroids spanning a (n_rows + n_cols)-d row/col cone
    subspace; cell (r, c)'s centroid is::

        c_{r,c} = row_weight * cones[r] + col_weight * cones[n_rows + c]

    L2-normalised. Returns ``(n_rows * n_cols, dim)`` tensor in
    flattened row-major order matching ``idx_of_rc(r, c) = r * n_cols + c``.

    Falsifiability: under this centroid set, the supervised
    geometry is row-and-column factorised but the move task itself
    (a 5-class direction classifier on cell pairs) still has full
    translational symmetry, so any post-training emergence of
    cardinal axes can be cleanly attributed to the centroid layer.
    """
    n_total = n_rows + n_cols
    if dim < n_total:
        raise ValueError(
            f"dim ({dim}) must be >= n_rows+n_cols ({n_total})"
        )
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, n_total, generator=g)
    Q, _ = torch.linalg.qr(A)
    cones = F.normalize(Q.t(), dim=-1)  # (n_total, dim)

    n_cells = n_rows * n_cols
    centroids = torch.zeros(n_cells, dim)
    for r in range(n_rows):
        for c in range(n_cols):
            cell_idx = r * n_cols + c
            cell = (row_weight * cones[r]
                    + col_weight * cones[n_rows + c])
            centroids[cell_idx] = cell
    centroids = F.normalize(centroids, dim=-1)
    return centroids


# ────────────────────────────────────────────────────────────────────────
# C layer — center-bias sampling weights
# ────────────────────────────────────────────────────────────────────────


def center_bias_weights(
    n_rows: int, n_cols: int, *, sigma: float = 1.5,
) -> list[float]:
    """Per-cell weight = exp(-d^2 / (2 sigma^2)) where d = chebyshev
    distance to grid centre. Boosts central cells, models natural
    movement-frequency distribution in bounded rooms (people stay
    near the middle more than at the corners).

    Returns weights in flattened row-major order. Magnitude ratio
    centre:corner ≈ 5–10× under default sigma."""
    out: list[float] = []
    cr = (n_rows - 1) / 2
    cc = (n_cols - 1) / 2
    for r in range(n_rows):
        for c in range(n_cols):
            d = max(abs(r - cr), abs(c - cc))
            out.append(math.exp(-(d ** 2) / (2 * sigma ** 2)))
    return out


# ────────────────────────────────────────────────────────────────────────
# D layer — row-index classification head (single input)
# ────────────────────────────────────────────────────────────────────────


class RowIndexHead(nn.Module):
    """Single-input head: predict ``r`` for cell at (r, c).

    Trained on the **full inventory** (training + outer-ring cells),
    so target-language cells' ``motion_bias`` rows still receive
    axis-relevant gradient via the row-classification task even
    though they never participate in any movement triple. This is
    the spatial analogue of ``LastDigitHead`` for numbers and
    ``MinimalPairHead`` for phonemes.
    """

    def __init__(
        self,
        n_rows: int,
        facet_dim: int,
        hidden: int = 64,
        *,
        use_abstract: bool = False,
        assignment: str = "hard",
        soft_tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_rows = n_rows
        self.facet_dim = facet_dim
        self.use_abstract = use_abstract
        self.assignment = assignment
        self.soft_tau = soft_tau
        self.fc1 = nn.Linear(facet_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_rows)

    def forward(self, ids: list[str], cg, tick: int = 0) -> torch.Tensor:
        x = self._collapse(ids, cg, tick)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def _collapse(self, ids: list[str], cg, tick: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return collapse_with_optional_abstract(
            cg, caller="RowIndexHead", facet="motion_bias",
            concept_ids=ids,
            shape=(self.facet_dim,), tick=tick, init="normal_small",
            device=device,
            use_abstract=self.use_abstract,
            assignment=self.assignment, soft_tau=self.soft_tau,
        )
