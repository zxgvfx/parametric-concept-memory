"""Helpers shared across the four cook-vs-direct training comparisons."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


__all__ = ["DEVICE", "stack_rows", "rho_linear", "diff_metrics"]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def stack_rows(bs: dict, cids: list[str], facet: str) -> torch.Tensor:
    rows = [bs[cid][f"params.{facet}"] for cid in cids]
    return F.normalize(torch.stack(rows), dim=-1)


def rho_linear(bs: dict, cids: list[str], facet: str, target_idx) -> float:
    """ρ(cos, target_distance) — used for §4 / §5 / §6.2 / §6.3."""
    M = stack_rows(bs, cids, facet)
    n = len(cids)
    cos = (M @ M.t()).cpu().numpy()
    iu = [(i, j) for i in range(n) for j in range(n) if i != j]
    cos_vals = [float(cos[i, j]) for i, j in iu]
    d = [target_idx(i, j) for i, j in iu]
    return float(spearmanr(cos_vals, d)[0])


def diff_metrics(d: dict, c: dict) -> dict:
    """Compute |direct - cook| diffs for the metrics they share."""
    out = {}
    for k, v_d in d.items():
        if k not in c:
            continue
        v_c = c[k]
        if isinstance(v_d, (int, float)) and isinstance(v_c, (int, float)):
            out[k] = abs(float(v_d) - float(v_c))
        elif isinstance(v_d, dict) and isinstance(v_c, dict):
            out[k] = {sub: abs(float(v_d[sub]) - float(v_c[sub]))
                      for sub in v_d if sub in v_c
                      and isinstance(v_d[sub], (int, float))}
    return out
