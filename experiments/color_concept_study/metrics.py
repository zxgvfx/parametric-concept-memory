"""Geometry metrics for the color study (paper §5)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from ._config import N_COLORS
from .topology import circular_dist


__all__ = [
    "_cos_matrix",
    "_rho_circular",
    "_rho_linear",
    "_cross_facet_align",
]


def _cos_matrix(bs: dict, facet: str) -> torch.Tensor:
    rows = []
    for i in range(N_COLORS):
        cid = f"concept:color:{i}"
        rows.append(bs[cid][f"params.{facet}"])
    M = F.normalize(torch.stack(rows), dim=-1)
    return M @ M.t()


def _rho_circular(cos: torch.Tensor) -> float:
    mask = ~torch.eye(N_COLORS, dtype=torch.bool)
    off = cos[mask].numpy()
    d = torch.tensor(
        [[-circular_dist(i, j) for j in range(N_COLORS)] for i in range(N_COLORS)],
        dtype=torch.float,
    )
    return float(spearmanr(off, d[mask].numpy())[0])


def _rho_linear(cos: torch.Tensor) -> float:
    mask = ~torch.eye(N_COLORS, dtype=torch.bool)
    off = cos[mask].numpy()
    d = torch.tensor(
        [[-abs(i - j) for j in range(N_COLORS)] for i in range(N_COLORS)],
        dtype=torch.float,
    )
    return float(spearmanr(off, d[mask].numpy())[0])


def _cross_facet_align(bs: dict, f1: str, f2: str) -> float:
    ca = _cos_matrix(bs, f1)
    cb = _cos_matrix(bs, f2)
    mask = ~torch.eye(N_COLORS, dtype=torch.bool)
    return float(spearmanr(ca[mask].numpy(), cb[mask].numpy())[0])
