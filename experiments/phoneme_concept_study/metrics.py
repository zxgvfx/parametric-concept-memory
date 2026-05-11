"""Metrics for the phoneme study (paper §6.3)."""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .inventory import N_PH, cid_of, feat_of, hamming


__all__ = [
    "_cos_matrix",
    "_rho_hamming_total",
    "_rho_same_axis",
    "_intra_vs_inter_gap",
    "_cross_facet_align",
    "_perm_test_align",
]


def _cos_matrix(bs: dict, facet: str) -> torch.Tensor:
    rows = []
    for i in range(N_PH):
        rows.append(bs[cid_of(i)][f"params.{facet}"])
    M = F.normalize(torch.stack(rows), dim=-1)
    return M @ M.t()


def _rho_hamming_total(cos: torch.Tensor) -> float:
    """Spearman(off-diag cos, −hamming): 总体 categorical metric 贴合度."""
    mask = ~torch.eye(N_PH, dtype=torch.bool)
    off = cos[mask].numpy()
    d = torch.tensor(
        [[-hamming(i, j) for j in range(N_PH)] for i in range(N_PH)],
        dtype=torch.float,
    )
    return float(spearmanr(off, d[mask].numpy())[0])


def _rho_same_axis(cos: torch.Tensor, axis: int) -> float:
    """Spearman(off-diag cos, indicator[same axis value]).

    axis ∈ {0, 1, 2} for (voice, manner, place).
    高 ρ ⇒ facet 几何把同 class 集合塞得更近.
    """
    mask = ~torch.eye(N_PH, dtype=torch.bool)
    off = cos[mask].numpy()
    ind = torch.tensor(
        [[1.0 if feat_of(i)[axis] == feat_of(j)[axis] else 0.0
          for j in range(N_PH)] for i in range(N_PH)],
        dtype=torch.float,
    )
    return float(spearmanr(off, ind[mask].numpy())[0])


def _intra_vs_inter_gap(cos: torch.Tensor, axis: int) -> dict:
    intra_vals, inter_vals = [], []
    for i in range(N_PH):
        for j in range(N_PH):
            if i == j:
                continue
            c = cos[i, j].item()
            if feat_of(i)[axis] == feat_of(j)[axis]:
                intra_vals.append(c)
            else:
                inter_vals.append(c)

    def _m(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    return {
        "intra_mean": _m(intra_vals),
        "inter_mean": _m(inter_vals),
        "gap": _m(intra_vals) - _m(inter_vals),
        "n_intra": len(intra_vals),
        "n_inter": len(inter_vals),
    }


def _cross_facet_align(bs: dict, f1: str, f2: str) -> float:
    ca = _cos_matrix(bs, f1)
    cb = _cos_matrix(bs, f2)
    mask = ~torch.eye(N_PH, dtype=torch.bool)
    return float(spearmanr(ca[mask].numpy(), cb[mask].numpy())[0])


def _perm_test_align(bs: dict, f1: str, f2: str, n_perm: int = 1000) -> dict:
    ca = _cos_matrix(bs, f1)
    cb = _cos_matrix(bs, f2)
    mask = ~torch.eye(N_PH, dtype=torch.bool)
    oa = ca[mask].numpy()
    ob = cb[mask].numpy()
    observed = float(spearmanr(oa, ob)[0])

    rng = random.Random(0)
    ge = 0
    null = []
    for _ in range(n_perm):
        perm = list(range(N_PH))
        rng.shuffle(perm)
        cb_perm = cb[perm][:, perm]
        obp = cb_perm[mask].numpy()
        r = float(spearmanr(oa, obp)[0])
        null.append(r)
        if abs(r) >= abs(observed):
            ge += 1
    p = (ge + 1) / (n_perm + 1)
    return {
        "observed": observed,
        "p_value": p,
        "null_mean": sum(null) / len(null),
        "conclusion": (
            "significant (p<0.01)" if p < 0.01
            else "significant (p<0.05)" if p < 0.05
            else "not significant"
        ),
    }
