"""Random centroid generators (A1 assay)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.robustness_study import DEVICE


__all__ = [
    "make_random_orthogonal_centroids",
    "make_random_gaussian_centroids",
]


def make_random_orthogonal_centroids(n_classes: int, dim: int, seed: int) -> torch.Tensor:
    """生成 n_classes 个**随机正交**单位向量, 作为 arithmetic 分类 target.

    这故意剥离任何"数量序"信息 (cos(c_i, c_j) ≈ 0 for i ≠ j), 模拟
    一个没有内禀 ordinal 的 supervision target.
    """
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, n_classes, generator=g)
    Q, _ = torch.linalg.qr(A)   # (dim, n_classes)
    return F.normalize(Q.t(), dim=-1).to(DEVICE)


def make_random_gaussian_centroids(n_classes: int, dim: int, seed: int) -> torch.Tensor:
    """生成 n_classes 个独立 L2-normalized 高斯向量 (近似正交, 对照)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_classes, dim, generator=g)
    return F.normalize(X, dim=-1).to(DEVICE)
