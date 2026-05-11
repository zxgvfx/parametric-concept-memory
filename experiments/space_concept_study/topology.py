"""5×5 grid topology helpers (paper §6.2).

Pure functions that map between grid coordinates, flattened indices,
concept ids, and the labelled triples enumerated for the
``MoveHead`` (5-class direction) and ``DistanceHead`` (9-class L1)
training tasks.
"""
from __future__ import annotations

from typing import Optional

from ._config import (
    CLS_DOWN,
    CLS_LEFT,
    CLS_RIGHT,
    CLS_SAME,
    CLS_UP,
    N_COLS,
    N_ROWS,
)


__all__ = [
    "cid_of",
    "rc_of_idx",
    "idx_of_rc",
    "l1_dist",
    "move_class",
    "enumerate_move_triples",
    "enumerate_distance_triples",
]


def cid_of(r: int, c: int) -> str:
    return f"concept:space:{r}_{c}"


def rc_of_idx(idx: int) -> tuple[int, int]:
    return divmod(idx, N_COLS)


def idx_of_rc(r: int, c: int) -> int:
    return r * N_COLS + c


def l1_dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def move_class(a: tuple[int, int], b: tuple[int, int]) -> Optional[int]:
    """a → b 的方向. 只对 L1=1 相邻 + same-cell 返回 class, 否则 None."""
    dr, dc = b[0] - a[0], b[1] - a[1]
    if (dr, dc) == (0, 0):
        return CLS_SAME
    if (dr, dc) == (-1, 0):
        return CLS_UP
    if (dr, dc) == (1, 0):
        return CLS_DOWN
    if (dr, dc) == (0, -1):
        return CLS_LEFT
    if (dr, dc) == (0, 1):
        return CLS_RIGHT
    return None


def enumerate_move_triples() -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    out = []
    for r1 in range(N_ROWS):
        for c1 in range(N_COLS):
            for r2 in range(N_ROWS):
                for c2 in range(N_COLS):
                    cls = move_class((r1, c1), (r2, c2))
                    if cls is not None:
                        out.append(((r1, c1), (r2, c2), cls))
    return out


def enumerate_distance_triples() -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    out = []
    for r1 in range(N_ROWS):
        for c1 in range(N_COLS):
            for r2 in range(N_ROWS):
                for c2 in range(N_COLS):
                    out.append(((r1, c1), (r2, c2),
                                l1_dist((r1, c1), (r2, c2))))
    return out
