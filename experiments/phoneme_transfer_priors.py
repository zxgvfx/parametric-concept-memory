"""PAPER §6.9 — three-causal-layer ablation for phoneme cross-language transfer.

Mirror of ``experiments.sleep_color_primaries`` /
``experiments.number_decimal_priors`` for the phoneme domain.
Provides:

* **B layer (biological / hardware prior)** —
  :func:`make_articulator_centroids` builds a per-axis
  voice / manner / place cone basis (mirroring the LMS / decimal
  cone constructions). Each phoneme's centroid is a convex
  combination of its (voice, manner, place) cone projections.
* **C layer (ecological / phonotactic statistics)** —
  :func:`zipf_phonotactic_weights` and
  :func:`source_natural_weights` give per-phoneme sampling
  weights that mimic natural frequency distributions in source
  languages (e.g. /t/ /n/ /s/ are universally over-represented;
  cf. PHOIBLE database, Maddieson 1984).
* **D layer (task-driven asymmetry)** —
  :class:`MinimalPairHead` is a pair-input binary head that
  predicts whether two phonemes differ in *exactly one* axis
  (i.e. they are a minimal pair like /p/ vs /b/, /t/ vs /d/).
  Trained on the full inventory, this provides
  axis-relevant gradient to held-out (target-language)
  phonemes without revealing the per-axis ground-truth label.

These map exactly onto the §6.8 / §7.4 5-condition protocol:
* A baseline = random centroid + uniform sampling + V/M/P heads
  trained only on the source-language subset.
* B / C / D inject the corresponding layer.
* B+C+D combines all three.

Cognitive-science rationale (Sun et al. 2023 *Nat Neurosci* on
generalisation-conditional consolidation; Werker & Tees 1984
*Infant Behav Dev* on infant cross-language phonetic
discrimination): the transfer-relevant axes (voice / manner /
place) are *features* the source-language learner already
practises, but the held-out target-language phonemes are novel
combinations of those features. Whether the system can
recognise those combinations on first exposure depends on
whether the three causal layers individually or jointly suffice
to seed the held-out bundle rows with correct axis identity.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcm.sleep import collapse_with_optional_abstract


__all__ = [
    "make_articulator_centroids",
    "zipf_phonotactic_weights",
    "source_natural_weights",
    "MinimalPairHead",
]


# ────────────────────────────────────────────────────────────────────────
# B layer — articulator-grouped centroids (LMS analogue for phonemes)
# ────────────────────────────────────────────────────────────────────────


def make_articulator_centroids(
    phoneme_features: list[tuple[int, int, int]], dim: int, seed: int,
    *, n_voice: int = 2, n_manner: int = 4, n_place: int = 4,
    voice_weight: float = 1.0, manner_weight: float = 1.0,
    place_weight: float = 1.0,
) -> torch.Tensor:
    """Build per-phoneme centroids spanning a (n_voice + n_manner +
    n_place)-dimensional articulator subspace.

    Construction:

    1. Sample ``n_total = n_voice + n_manner + n_place`` orthogonal
       cone vectors via QR. Layout:
       ``cones[0..n_voice-1]`` = voice cones,
       ``cones[n_voice..n_voice+n_manner-1]`` = manner cones,
       ``cones[n_voice+n_manner..]`` = place cones.
    2. For phoneme ``i`` with features ``(v, m, p)``:
       ``c_i = voice_weight * cones[v]
            + manner_weight * cones[n_voice + m]
            + place_weight  * cones[n_voice + n_manner + p]``,
       L2-normalised.

    Returns a ``(N_PH, dim)`` tensor on CPU; caller is responsible
    for ``.to(DEVICE)`` if needed.
    """
    n_total = int(n_voice + n_manner + n_place)
    if dim < n_total:
        raise ValueError(
            f"dim ({dim}) must be >= n_voice+n_manner+n_place ({n_total})"
        )
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, n_total, generator=g)
    Q, _ = torch.linalg.qr(A)
    cones = F.normalize(Q.t(), dim=-1)  # (n_total, dim)

    centroids = torch.zeros(len(phoneme_features), dim)
    for i, (v, m, p) in enumerate(phoneme_features):
        if not (0 <= v < n_voice and 0 <= m < n_manner and 0 <= p < n_place):
            raise ValueError(
                f"phoneme {i} feat ({v}, {m}, {p}) out of range "
                f"({n_voice}, {n_manner}, {n_place})"
            )
        c = (voice_weight * cones[v]
             + manner_weight * cones[n_voice + m]
             + place_weight * cones[n_voice + n_manner + p])
        centroids[i] = c
    centroids = F.normalize(centroids, dim=-1)
    return centroids


# ────────────────────────────────────────────────────────────────────────
# C layer — phonotactic frequency weights
# ────────────────────────────────────────────────────────────────────────


def zipf_phonotactic_weights(N: int, *, alpha: float = 1.0) -> list[float]:
    """Zipf 1/rank^alpha weights with a shuffle in rank order.

    Returns weights in *position* order ``[w_0, ..., w_{N-1}]`` where
    ``w_i = 1 / rank(i)^alpha`` and rank is determined by phoneme
    index (smaller index = more frequent, mimicking the typical
    convention that earlier inventory positions are unmarked /
    frequent stops and nasals).
    """
    return [1.0 / ((i + 1) ** alpha) for i in range(N)]


def source_natural_weights(
    phoneme_features: list[tuple[int, int, int]],
    *, frequent_manners: tuple[int, ...] = (0, 1),
    frequent_boost: float = 4.0,
) -> list[float]:
    """Boost phonemes whose manner is in ``frequent_manners`` (default
    stops + fricatives, the cross-linguistically frequent classes
    per Maddieson 1984's PHOIBLE survey)."""
    out: list[float] = []
    for (_v, m, _p) in phoneme_features:
        out.append(frequent_boost if m in frequent_manners else 1.0)
    return out


# ────────────────────────────────────────────────────────────────────────
# D layer — minimal pair classification head
# ────────────────────────────────────────────────────────────────────────


class MinimalPairHead(nn.Module):
    """Pair-input head: predict whether two phonemes are a minimal pair.

    Output is 4-class:
      0 = identical (a == b),
      1 = differ on voice axis only,
      2 = differ on manner axis only,
      3 = differ on place axis only,
      ... or "other" (multiple-axis differ) lumped into class 0
      bucket since it should be predicted as "no minimal pair".

    Following Werker & Tees 1984, infants up to 6 months can
    discriminate any phonetic contrast; native-language tuning
    later collapses non-functional contrasts. Our minimal-pair
    head models the universal-discrimination phase. Crucially,
    the head is trained on the **full inventory** including
    target-language phonemes, providing axis-relevant gradient
    to held-out bundle rows without ever predicting their per-
    axis labels directly.

    Consumes a single configurable facet (default voice_bias)
    per call; for full coverage the host trains three instances
    (one per axis) or rotates the consumed facet across batches.
    Here we keep the head simple and let the caller decide.
    """

    def __init__(
        self,
        facet: str,
        facet_dim: int,
        hidden: int = 64,
        n_axes: int = 3,
        *,
        use_abstract: bool = False,
        assignment: str = "hard",
        soft_tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.facet = facet
        self.facet_dim = facet_dim
        self.use_abstract = use_abstract
        self.assignment = assignment
        self.soft_tau = soft_tau
        self.fc1 = nn.Linear(2 * facet_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        # 4 classes: same / voice-diff / manner-diff / place-diff
        self.fc3 = nn.Linear(hidden, n_axes + 1)

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
            cg, caller="MinimalPairHead", facet=self.facet,
            concept_ids=ids,
            shape=(self.facet_dim,), tick=tick, init="normal_small",
            device=device,
            use_abstract=self.use_abstract,
            assignment=self.assignment, soft_tau=self.soft_tau,
        )


def minimal_pair_label(
    feat_a: tuple[int, int, int], feat_b: tuple[int, int, int],
) -> int:
    """Return minimal-pair class:
       0 = identical or differ on >1 axis;
       1 = differ on voice only;
       2 = differ on manner only;
       3 = differ on place only.
    """
    diffs = sum(1 for x, y in zip(feat_a, feat_b) if x != y)
    if diffs == 0:
        return 0
    if diffs > 1:
        return 0  # not a minimal pair
    if feat_a[0] != feat_b[0]:
        return 1
    if feat_a[1] != feat_b[1]:
        return 2
    if feat_a[2] != feat_b[2]:
        return 3
    return 0  # unreachable
