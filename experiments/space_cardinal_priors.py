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
    "MaskInferHead",
    "enumerate_mask_infer_samples",
    "InverseMoveHead",
    "enumerate_inverse_move_samples",
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


# ────────────────────────────────────────────────────────────────────────
# D layer alt — Causal-JEPA-style mask-infer head
# ────────────────────────────────────────────────────────────────────────


def enumerate_mask_infer_samples(
    n_rows: int, n_cols: int, *, neighbors: str = "cardinal4",
) -> list[tuple[int, list[int]]]:
    """Enumerate (center_idx, [neighbor_idxs]) pairs over the full grid.

    The mask-infer task is: given the bundle rows of the neighbours,
    predict the centre's flattened index. Boundary cells whose
    cardinal neighbour falls off-grid pad with the centre itself
    (self-loop), keeping a fixed neighbour count so the head can
    use a fixed-shape input.

    ``neighbors`` choices:

    * ``"cardinal4"`` — N/S/E/W four neighbours (the default,
      mirrors the place-cell × grid-cell adjacency assumption).
    """
    if neighbors != "cardinal4":
        raise ValueError(f"unknown neighbour scheme {neighbors!r}")
    out: list[tuple[int, list[int]]] = []
    for r in range(n_rows):
        for c in range(n_cols):
            cidx = r * n_cols + c
            ns: list[int] = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    ns.append(nr * n_cols + nc)
                else:
                    ns.append(cidx)  # pad with self
            out.append((cidx, ns))
    return out


class MaskInferHead(nn.Module):
    """Causal-JEPA-style mask-infer head (paper §15 / S3).

    Given the bundle rows of a centre cell's cardinal neighbours,
    predict the centre cell's flattened index (``r * n_cols + c``).

    Why this matters for §7.5-space ``mixed_OOD``: outer-ring cells
    that never appear as the *target* of a movement triple still
    get fed into ``MaskInferHead.fc1`` as **neighbours** of inner
    cells, so their bundle rows are exposed to a head whose
    weights live in the same input-distribution coverage problem
    that gave rise to the 0.000 ``mixed_OOD`` ceiling. By rotating
    every cell through the *neighbour* slot the head's input
    distribution becomes joint-symmetric, plausibly breaking the
    ceiling without per-task data augmentation.

    Architecture:

    * Concatenate the four neighbour bundle rows along the feature
      dim → ``fc1: (n_neighbours * facet_dim) → hidden``.
    * Two ReLU MLP layers + ``fc3: hidden → n_cells`` logits.
    """

    def __init__(
        self,
        n_cells: int,
        facet_dim: int,
        hidden: int = 128,
        *,
        n_neighbors: int = 4,
        use_abstract: bool = False,
        assignment: str = "hard",
        soft_tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_cells = n_cells
        self.facet_dim = facet_dim
        self.n_neighbors = n_neighbors
        self.use_abstract = use_abstract
        self.assignment = assignment
        self.soft_tau = soft_tau
        self.fc1 = nn.Linear(n_neighbors * facet_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_cells)

    def forward(
        self, neighbor_ids_per_sample: list[list[str]], cg, tick: int = 0,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        B = len(neighbor_ids_per_sample)
        flat = [cid for sample in neighbor_ids_per_sample for cid in sample]
        emb = collapse_with_optional_abstract(
            cg, caller="MaskInferHead", facet="motion_bias",
            concept_ids=flat,
            shape=(self.facet_dim,), tick=tick, init="normal_small",
            device=device,
            use_abstract=self.use_abstract,
            assignment=self.assignment, soft_tau=self.soft_tau,
        )
        x = emb.view(B, self.n_neighbors * self.facet_dim)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


# ────────────────────────────────────────────────────────────────────────
# D layer alt-2 — inverse-move head (pair-input mask-infer)
# ────────────────────────────────────────────────────────────────────────


def enumerate_inverse_move_samples(
    n_rows: int, n_cols: int,
) -> list[tuple[int, int, int]]:
    """Enumerate ``(cell_a_idx, direction, cell_b_idx)`` triples
    over the full grid for the inverse-move task.

    Direction encoding matches :func:`space_concept_study.topology.move_class`
    (CLS_SAME=0, CLS_UP=1, CLS_DOWN=2, CLS_LEFT=3, CLS_RIGHT=4).
    Boundary cells emit only the same-cell case for off-grid
    directions to keep targets unique.
    """
    out: list[tuple[int, int, int]] = []
    deltas = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }
    for r in range(n_rows):
        for c in range(n_cols):
            cell_a = r * n_cols + c
            for direction, (dr, dc) in deltas.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    cell_b = nr * n_cols + nc
                    out.append((cell_a, direction, cell_b))
    return out


class InverseMoveHead(nn.Module):
    """Pair-input mask-infer head: ``(cell_a, direction) → cell_b id``.

    Unlike :class:`MaskInferHead` (which wraps four neighbour bundles
    into a single feature vector), this head's input is a single
    bundle row plus a one-hot direction code, mirroring the pair
    shape of :class:`MoveHead`. The point is that during training,
    ``cell_a`` is sampled uniformly across **all 7×7 = 49 cells**
    (including outer-ring), so ``fc1`` directly observes outer-ring
    bundle rows as one half of a pair input.

    This addresses the §7.5-space ``mixed_OOD = 0.000`` ceiling
    diagnosed in F36/F37: the original ``MoveHead.fc1`` only saw
    (inner, inner) joint pairs during training, so its
    input-distribution coverage was strictly inner-x-inner. Adding
    InverseMoveHead trains a parallel pair-shape MLP whose joint
    distribution is full grid × direction, plausibly producing
    bundle-row representations that, when re-used by the (still
    inner-only) MoveHead, fall closer to a region MoveHead.fc1
    can interpolate over.

    Hypothesis (S3-v2): InverseMoveHead pushes the ``test_outer_OOD``
    accuracy substantially above BCD_RowIndex/BCD_MaskInfer and may
    crack the ``test_mixed_OOD`` ceiling.
    """

    def __init__(
        self,
        n_cells: int,
        facet_dim: int,
        hidden: int = 128,
        *,
        n_directions: int = 5,
        use_abstract: bool = False,
        assignment: str = "hard",
        soft_tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_cells = n_cells
        self.facet_dim = facet_dim
        self.n_directions = n_directions
        self.use_abstract = use_abstract
        self.assignment = assignment
        self.soft_tau = soft_tau
        self.fc1 = nn.Linear(facet_dim + n_directions, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_cells)

    def forward(
        self,
        cell_a_ids: list[str],
        direction_idx: torch.Tensor,
        cg,
        tick: int = 0,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        emb = collapse_with_optional_abstract(
            cg, caller="InverseMoveHead", facet="motion_bias",
            concept_ids=cell_a_ids,
            shape=(self.facet_dim,), tick=tick, init="normal_small",
            device=device,
            use_abstract=self.use_abstract,
            assignment=self.assignment, soft_tau=self.soft_tau,
        )
        d_oh = F.one_hot(direction_idx, num_classes=self.n_directions).float()
        x = torch.cat([emb, d_oh], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)
