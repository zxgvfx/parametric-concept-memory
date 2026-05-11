"""Geometry metrics for the spatial study (paper §6.2).

Each helper consumes the per-cell ``bundle_state`` produced by
:func:`.train.train_one` and returns a Spearman ρ or a Procrustes-MDS
fit. Names are intentionally underscore-prefixed so the package
boundary mirrors the original single-file layout.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from ._config import N_CELLS, N_COLS, N_ROWS
from .topology import cid_of, idx_of_rc, l1_dist, rc_of_idx


__all__ = [
    "_cos_matrix",
    "_rho_L1",
    "_rho_linear_flat",
    "_rho_row_within",
    "_rho_col_within",
    "_cross_facet_align",
    "_mds_grid_fit",
]


def _cos_matrix(bs: dict, facet: str) -> torch.Tensor:
    rows = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            rows.append(bs[cid_of(r, c)][f"params.{facet}"])
    M = F.normalize(torch.stack(rows), dim=-1)
    return M @ M.t()


def _rho_L1(cos: torch.Tensor) -> float:
    mask = ~torch.eye(N_CELLS, dtype=torch.bool)
    off = cos[mask].numpy()
    d = torch.zeros(N_CELLS, N_CELLS)
    for i in range(N_CELLS):
        for j in range(N_CELLS):
            d[i, j] = -l1_dist(rc_of_idx(i), rc_of_idx(j))
    return float(spearmanr(off, d[mask].numpy())[0])


def _rho_linear_flat(cos: torch.Tensor) -> float:
    """1D-flattened control: ρ(cos, −|i−j|). If > ρ_L1, PCM 退化到了 1D bias."""
    mask = ~torch.eye(N_CELLS, dtype=torch.bool)
    off = cos[mask].numpy()
    d = torch.tensor(
        [[-abs(i - j) for j in range(N_CELLS)] for i in range(N_CELLS)],
        dtype=torch.float,
    )
    return float(spearmanr(off, d[mask].numpy())[0])


def _rho_row_within(cos: torch.Tensor) -> float:
    """同行 pair 内: ρ(cos_ab, −|c_a − c_b|)."""
    xs, ys = [], []
    for r in range(N_ROWS):
        for c1 in range(N_COLS):
            for c2 in range(N_COLS):
                if c1 == c2:
                    continue
                i = idx_of_rc(r, c1); j = idx_of_rc(r, c2)
                xs.append(cos[i, j].item())
                ys.append(-abs(c1 - c2))
    if len(xs) < 3:
        return float("nan")
    return float(spearmanr(xs, ys)[0])


def _rho_col_within(cos: torch.Tensor) -> float:
    xs, ys = [], []
    for c in range(N_COLS):
        for r1 in range(N_ROWS):
            for r2 in range(N_ROWS):
                if r1 == r2:
                    continue
                i = idx_of_rc(r1, c); j = idx_of_rc(r2, c)
                xs.append(cos[i, j].item())
                ys.append(-abs(r1 - r2))
    return float(spearmanr(xs, ys)[0])


def _cross_facet_align(bs: dict, f1: str, f2: str) -> float:
    ca = _cos_matrix(bs, f1)
    cb = _cos_matrix(bs, f2)
    mask = ~torch.eye(N_CELLS, dtype=torch.bool)
    return float(spearmanr(ca[mask].numpy(), cb[mask].numpy())[0])


def _mds_grid_fit(bs: dict, facet: str, seed: int = 0) -> dict:
    """把 bundle cos 矩阵用 MDS 投到 2D, 和 GT grid 坐标做 Procrustes align.

    返回 disparity (∈ [0, 1]; 0=完美 grid, 1=最差) + 每点残差.
    """
    import numpy as np
    from scipy.spatial import procrustes
    from sklearn.manifold import MDS

    cos = _cos_matrix(bs, facet)
    cos_np = cos.numpy()
    diss = np.clip(1.0 - cos_np, 0.0, 2.0)
    np.fill_diagonal(diss, 0.0)
    mds = MDS(
        n_components=2, dissimilarity="precomputed",
        random_state=seed, normalized_stress="auto",
        n_init=4, max_iter=500,
    )
    coords = mds.fit_transform(diss)

    gt = np.array(
        [[r, c] for r in range(N_ROWS) for c in range(N_COLS)],
        dtype=float,
    )
    gt_n, coords_n, disparity = procrustes(gt, coords)
    residuals = np.linalg.norm(gt_n - coords_n, axis=1)
    return {
        "disparity": float(disparity),
        "mean_residual": float(residuals.mean()),
        "max_residual": float(residuals.max()),
        "mds_stress": float(mds.stress_),
    }
