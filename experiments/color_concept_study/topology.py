"""Circular hue-wheel topology helpers for the color study."""
from __future__ import annotations

import cmath
import math
from typing import Optional

from ._config import N_COLORS


__all__ = [
    "circular_dist",
    "mix_pair",
    "enumerate_mixing_triples",
    "enumerate_adjacency_triples",
]


def circular_dist(i: int, j: int, N: int = N_COLORS) -> int:
    d = abs(i - j)
    return min(d, N - d)


def mix_pair(i: int, j: int, N: int = N_COLORS) -> Optional[int]:
    """Circular midpoint. 对 opposite pair (d = N/2) 返回 None (跳过, 歧义)."""
    if circular_dist(i, j, N) == N // 2:
        return None
    angle_i = 2 * math.pi * i / N
    angle_j = 2 * math.pi * j / N
    z = cmath.exp(1j * angle_i) + cmath.exp(1j * angle_j)
    if abs(z) < 1e-8:
        return None
    mid_angle = (cmath.phase(z)) % (2 * math.pi)
    return int(round(mid_angle * N / (2 * math.pi))) % N


def enumerate_mixing_triples(N: int = N_COLORS) -> list[tuple[int, int, int]]:
    triples: list[tuple[int, int, int]] = []
    for a in range(N):
        for b in range(N):
            if a == b:
                continue
            c = mix_pair(a, b, N)
            if c is None:
                continue
            triples.append((a, b, c))
    return triples


def enumerate_adjacency_triples(N: int = N_COLORS) -> list[tuple[int, int, int]]:
    """3-class: 0=adjacent (d=1), 1=near (d=2-3), 2=far (d≥4)."""
    triples: list[tuple[int, int, int]] = []
    for a in range(N):
        for b in range(N):
            if a == b:
                continue
            d = circular_dist(a, b, N)
            if d == 1:
                lab = 0
            elif d <= 3:
                lab = 1
            else:
                lab = 2
            triples.append((a, b, lab))
    return triples
