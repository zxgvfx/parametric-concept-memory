"""Per-n bundle metrics shared by the four assays."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from experiments.robustness_study import _rho_vs_order


__all__ = [
    "_cos_matrix_by_n",
    "rho_by_n",
    "rho_with_inverse_remap",
    "_stats",
]


def _cos_matrix_by_n(bundle_by_n: dict[int, dict], facet: str,
                     ns: list[int]) -> torch.Tensor:
    rows = [bundle_by_n[n][facet] for n in ns]
    M = F.normalize(torch.stack(rows), dim=-1)
    return M @ M.t()


def rho_by_n(bundle_by_n: dict[int, dict], facet: str, ns: list[int]) -> float:
    cos = _cos_matrix_by_n(bundle_by_n, facet, ns)
    return _rho_vs_order(cos, ns)


def rho_with_inverse_remap(
    bundle_by_n: dict[int, dict],
    facet: str,
    ns: list[int],
    shuffle_map: dict[int, int],
) -> float:
    """A2 核心: 用 shuffle_map 把 bundle 还原到正确的数量坐标后再测 ρ.

    shuffle_map 是 "自然数 n -> 训练中替代的 bundle_id". 所以训练后
    ``bundle[n]`` 实际承载的是 "数量 ``shuffle_map^-1(n)`` 的语义".
    要测它是否真的学到了数量序, 我们按 shuffle_map 重排: 对自然序 i,
    取 ``bundle[sm[i]]``.
    """
    remapped: dict[int, dict] = {i: bundle_by_n[shuffle_map[i]] for i in ns}
    return rho_by_n(remapped, facet, ns)


def _stats(xs: list[float]) -> dict:
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
    return {"mean": m, "std": sd, "min": min(xs), "max": max(xs), "n": len(xs)}
