"""Color graph builder + centroid factory + shuffle helper."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from pcm.concept_graph import ConceptGraph

from ._config import DEVICE, EMBED_DIM, N_COLORS


__all__ = [
    "build_color_graph",
    "make_random_orthogonal_centroids",
    "make_lms_like_centroids",
    "make_cone_opponent_centroids",
    "_apply_shuffle",
]


def build_color_graph(n_colors: int = N_COLORS) -> ConceptGraph:
    cg = ConceptGraph(feat_dim=EMBED_DIM)
    for i in range(n_colors):
        cg.register_concept(
            node_id=f"concept:color:{i}",
            label=f"COLOR_{i}",
            scope="BASE",
            provenance=f"color_study:hue={i}/{n_colors}",
        )
    return cg


def make_random_orthogonal_centroids(K: int, dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, K, generator=g)
    Q, _ = torch.linalg.qr(A)
    return F.normalize(Q.t(), dim=-1).to(DEVICE)


def make_lms_like_centroids(
    K: int, dim: int, seed: int,
    *, peaks: tuple[int, int, int] = (0, 4, 8),
) -> torch.Tensor:
    """PAPER §6.8 — LMS-like supervised centroids ("biological prior").

    Build ``K`` centroids that span only a 3-dimensional subspace of the
    ``dim``-d embedding space, mirroring the trichromatic sampling of
    human cone receptors L/M/S (Stockman & Sharpe 2000). Concretely:

    1. Draw three orthogonal "cone" basis vectors ``c_L, c_M, c_S`` in
       ``dim`` dims from the seeded RNG (so cones rotate per seed but
       remain mutually orthogonal — what stays seed-invariant is the
       *3-d subspace structure*, not its embedding rotation).
    2. For each hue ``i ∈ [0, K)``, compute three half-cosine
       responses peaked at ``peaks[0/1/2]`` on the ``K``-cycle (default
       0/4/8 for the three primaries on a 12-hue ring).
    3. Centroid ``c_i = L(i)·c_L + M(i)·c_M + S(i)·c_S``, then L2-norm.

    The result is bit-shape compatible with
    :func:`make_random_orthogonal_centroids` (returns ``(K, dim)`` on
    ``DEVICE``) so any caller can switch between "random" and "LMS"
    centroids by a single one-line change.

    Falsifiability: under this centroid set, the **task** itself
    (cyclic mixing) still has full rotational symmetry, so there is
    no asymmetry in the gradient signal alone. The asymmetry lives
    entirely in the supervised geometry. If sleep-pass anchors then
    align with the (0, 4, 8) peaks (RGB-like), it proves PCM faithfully
    preserves an injected biological prior; if they keep rotating with
    seed, it proves even centroid-level supervision can't override
    cyclic task symmetry.
    """
    if K < 3:
        raise ValueError(f"LMS centroids require K >= 3, got {K}")
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, 3, generator=g)
    Q, _ = torch.linalg.qr(A)
    cones = F.normalize(Q.t(), dim=-1)  # (3, dim) orthogonal cone basis

    p_L, p_M, p_S = peaks
    centroids = torch.zeros(K, dim)
    for i in range(K):
        angle = 2 * math.pi * i / K
        a_L = 2 * math.pi * (p_L % K) / K
        a_M = 2 * math.pi * (p_M % K) / K
        a_S = 2 * math.pi * (p_S % K) / K
        L_resp = max(0.0, math.cos(angle - a_L))
        M_resp = max(0.0, math.cos(angle - a_M))
        S_resp = max(0.0, math.cos(angle - a_S))
        c = L_resp * cones[0] + M_resp * cones[1] + S_resp * cones[2]
        centroids[i] = c
    centroids = F.normalize(centroids, dim=-1)
    return centroids.to(DEVICE)


def make_cone_opponent_centroids(
    K: int, dim: int, seed: int,
    *, peaks: tuple[int, int, int] = (0, 4, 8),
) -> torch.Tensor:
    """PAPER §16 (S5) — cone-opponent centroids ("infant prior").

    Variant of :func:`make_lms_like_centroids` that projects the
    three cone responses onto **two opponent axes** plus a
    luminance axis, rather than three independent cone axes:

    * **R-G axis** ← (L − M) / 2
    * **Y-B axis** ← ((L + M)/2 − S) / 2
    * **Lum axis** ← (L + M + S) / 3

    This more faithfully reflects the **prelinguistic categorical
    perception** observed in 5–7-month-old infants by NIRS (Yang
    et al. 2016 *PNAS*; Skelton et al. 2017) and the cone-opponent
    mechanism that underlies it (Solomon & Lennie 2007 *Nat Rev
    Neurosci*; Conway et al. 2018), which carves the hue ring into
    **four primary regions** (red / green / yellow / blue) rather
    than three additive primaries (R / G / B).

    Layout: same ``(K, dim)`` shape on ``DEVICE`` as
    :func:`make_lms_like_centroids`, drop-in replaceable. Three
    cone responses share the same ``peaks`` argument so this
    function is one-line swap with the LMS variant.

    Falsifiability: under opponent centroids the **task** is still
    fully cyclic, so any post-training emergence of *four* anchor
    clusters (vs three under LMS) is attributable solely to the
    centroid prior, not to gradient asymmetry.
    """
    if K < 3:
        raise ValueError(
            f"cone-opponent centroids require K >= 3, got {K}"
        )
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, 3, generator=g)
    Q, _ = torch.linalg.qr(A)
    rg_axis = Q.t()[0]
    yb_axis = Q.t()[1]
    lum_axis = Q.t()[2]

    p_L, p_M, p_S = peaks
    centroids = torch.zeros(K, dim)
    for i in range(K):
        angle = 2 * math.pi * i / K
        a_L = 2 * math.pi * (p_L % K) / K
        a_M = 2 * math.pi * (p_M % K) / K
        a_S = 2 * math.pi * (p_S % K) / K
        # Linear (signed) cone responses; opponent processing
        # uses signed differences so we keep the sign here too.
        L_resp = math.cos(angle - a_L)
        M_resp = math.cos(angle - a_M)
        S_resp = math.cos(angle - a_S)
        rg = (L_resp - M_resp) / 2.0
        yb = ((L_resp + M_resp) / 2.0 - S_resp) / 2.0
        lum = max(0.0, (L_resp + M_resp + S_resp) / 3.0)
        centroids[i] = rg * rg_axis + yb * yb_axis + lum * lum_axis
    centroids = F.normalize(centroids, dim=-1)
    return centroids.to(DEVICE)


def _apply_shuffle(ids: list[str], sm: dict[int, int] | None) -> list[str]:
    if sm is None:
        return ids
    out = []
    for cid in ids:
        k = int(cid.split(":")[-1])
        out.append(f"concept:color:{sm[k]}")
    return out
