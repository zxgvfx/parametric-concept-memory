"""PAPER §7.4 — 1, 2, 3 layer decimal priors for the number domain.

Mirror of ``experiments.color_concept_study.graph_builder /
heads`` but for the number domain (PAPER §7 negative). Provides:

* **B layer (biological prior)** —
  :func:`make_decimal_cone_centroids` builds centroids from 10
  orthogonal "digit cones" (units 0–9) plus 10 "tens cones",
  so number ``n`` projects mainly onto ``cone[n % 10]`` and
  partially onto ``cone[10 + (n // 10)]``. This is the §6.8 LMS
  analogue: one orthogonal axis per digit.
* **D layer (task-driven asymmetry)** —
  :class:`LastDigitHead` and :class:`CarryHead` consume the same
  ``arithmetic_bias`` facet as :class:`QuadArithHead`, so gradient
  lands on the same bundle rows. They expose digit identity as a
  single-input or pair classification task.
* **C layer (ecological statistics)** — helper builders for
  per-number sampling weights (round-number bias, small-number
  bias, Zipf-like bias).

These are toggled from
``experiments/sleep_number_decimal.py`` for the 5-condition
ablation; this module exposes pure constructors only.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcm.sleep import collapse_with_optional_abstract


__all__ = [
    "make_decimal_cone_centroids",
    "round_number_weights",
    "zipf_weights",
    "small_number_weights",
    "LastDigitHead",
    "CarryHead",
]


# ────────────────────────────────────────────────────────────────────────
# B layer — decimal-cone centroids (LMS analogue for digits)
# ────────────────────────────────────────────────────────────────────────


def make_decimal_cone_centroids(
    N: int, dim: int, seed: int,
    *, units_weight: float = 1.0, tens_weight: float = 0.5,
    use_tens: bool = True,
) -> torch.Tensor:
    """Build centroids for integers ``1..N`` along 10 (or 20) orthogonal
    digit cones, mirroring :func:`make_lms_like_centroids` for colour.

    Construction:

    1. Sample ``n_cones`` orthogonal direction vectors via QR — 10 unit
       cones for digits 0-9, plus 10 tens cones (digits 0-9 of the
       tens place) when ``use_tens=True`` (so ``n_cones = 20``).
    2. For each integer ``n ∈ [1, N]``, build
       ``c_n = units_weight * cone[n % 10]
            + tens_weight  * cone[10 + (n // 10)]``  (or just the
       first term when ``use_tens=False``), then L2-normalise.

    The result has shape ``(N, dim)``, drop-in compatible with
    :func:`experiments.purity_audit.make_random_orthogonal_centroids`.

    Falsifiability: under this centroid set, the **task** itself
    (closed four-arithmetic on a linear ``[1, N]`` grid) is still
    additive-symmetric, i.e. the gradient signal alone has no
    base-10 preference. The asymmetry lives entirely in the
    supervised geometry. PAPER §7 reported a clean null on
    ``spike_10`` under :func:`make_random_orthogonal_centroids`.
    The §7.4 reverse claim is whether ``spike_10`` becomes
    significantly positive once decimal-cone centroids are
    supplied.
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    n_cones = 20 if use_tens else 10
    if dim < n_cones:
        raise ValueError(
            f"dim ({dim}) must be >= number of cones ({n_cones}); "
            "rerun with a larger embedding dimension."
        )

    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, n_cones, generator=g)
    Q, _ = torch.linalg.qr(A)
    cones = F.normalize(Q.t(), dim=-1)  # (n_cones, dim)

    centroids = torch.zeros(N, dim)
    for idx in range(N):
        n = idx + 1  # natural number 1..N
        units = n % 10
        c = units_weight * cones[units]
        if use_tens:
            tens = (n // 10) % 10
            c = c + tens_weight * cones[10 + tens]
        centroids[idx] = c
    centroids = F.normalize(centroids, dim=-1)
    return centroids


# ────────────────────────────────────────────────────────────────────────
# C layer — ecological sampling weights
# ────────────────────────────────────────────────────────────────────────


def round_number_weights(N: int, *, boost: float = 5.0) -> list[float]:
    """Per-number weight that boosts multiples of 10 by ``boost``×.

    Models the human discourse bias toward "round" numbers
    (Zipf-Mandelbrot style); see Dehaene's Number Sense literature
    for empirical frequency tables of integer use in natural
    languages."""
    out = []
    for n in range(1, N + 1):
        out.append(boost if n % 10 == 0 else 1.0)
    return out


def zipf_weights(N: int, *, alpha: float = 1.0) -> list[float]:
    """Zipf 1/n^alpha weights — small numbers exponentially more
    frequent (matches natural-language number frequency)."""
    return [1.0 / (n ** alpha) for n in range(1, N + 1)]


def small_number_weights(N: int, *, decay: float = 0.05) -> list[float]:
    """Exponential decay favouring small numbers."""
    return [math.exp(-decay * (n - 1)) for n in range(1, N + 1)]


# ────────────────────────────────────────────────────────────────────────
# D layer — task-driven asymmetry: heads that expose digit identity
# ────────────────────────────────────────────────────────────────────────


class LastDigitHead(nn.Module):
    """Single-input head: predict ``n % 10`` (10-class).

    Consumes the same ``arithmetic_bias`` facet as
    :class:`QuadArithHead`, so gradient lands on the same bundle
    rows. This is the strongest base-10 task asymmetry: it forces
    bundle rows to share representation along the digit axis,
    independent of additive linear position.

    Falsifiability: ``LastDigitHead`` makes ``cos(n, n+10)`` a
    direct training target via shared loss — if the bundle still
    fails to develop ``spike_10``, the head is computationally
    bypassed, falsifying the §7 explanation that "task signal is
    the missing ingredient".
    """

    def __init__(
        self,
        bias_dim: int,
        hidden: int = 64,
        *,
        use_abstract: bool = False,
        assignment: str = "hard",
        soft_tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.bias_dim = bias_dim
        self.use_abstract = use_abstract
        self.assignment = assignment
        self.soft_tau = soft_tau
        self.fc1 = nn.Linear(bias_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)

    def forward(self, ids: list[str], cg, tick: int = 0) -> torch.Tensor:
        x = self._collapse(ids, cg, tick)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def _collapse(self, ids: list[str], cg, tick: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return collapse_with_optional_abstract(
            cg, caller="LastDigitHead", facet="arithmetic_bias",
            concept_ids=ids,
            shape=(self.bias_dim,), tick=tick, init="normal_small",
            device=device,
            use_abstract=self.use_abstract,
            assignment=self.assignment, soft_tau=self.soft_tau,
        )


class CarryHead(nn.Module):
    """Pair-input head: predict ``a % 10 + b % 10 ≥ 10`` (binary).

    A weaker but more "ecologically realistic" alternative to
    :class:`LastDigitHead`: human learners encounter carry as a
    derived signal during arithmetic practice, not as a
    standalone classification task. Provided as an alternative D
    layer for ablations that want to test whether the strength of
    base-10 emergence depends on the directness of the task signal.
    """

    def __init__(
        self,
        bias_dim: int,
        hidden: int = 64,
        *,
        use_abstract: bool = False,
        assignment: str = "hard",
        soft_tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.bias_dim = bias_dim
        self.use_abstract = use_abstract
        self.assignment = assignment
        self.soft_tau = soft_tau
        self.fc1 = nn.Linear(2 * bias_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 2)

    def forward(
        self,
        ids_a: list[str], ids_b: list[str],
        cg, tick: int = 0,
    ) -> torch.Tensor:
        ba = self._collapse(ids_a, cg, tick)
        bb = self._collapse(ids_b, cg, tick)
        x = torch.cat([ba, bb], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def _collapse(self, ids: list[str], cg, tick: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return collapse_with_optional_abstract(
            cg, caller="CarryHead", facet="arithmetic_bias",
            concept_ids=ids,
            shape=(self.bias_dim,), tick=tick, init="normal_small",
            device=device,
            use_abstract=self.use_abstract,
            assignment=self.assignment, soft_tau=self.soft_tau,
        )
