"""Aggregation + console-printer helpers for the swap study.

Pure presentation: take per-seed result dicts and dump human-readable
tables to stdout. Computation belongs in the per-domain modules.
"""
from __future__ import annotations

import math


__all__ = [
    "agg_stats",
    "print_number_summary",
    "print_color_summary",
]


def agg_stats(values: list[float]) -> dict:
    """Mean / std / min / max / n over a list of floats.

    Returns a dict suitable for embedding in ``summary.json``. Uses an
    n-1 corrected sample standard deviation; falls back to ``nan`` when
    the input is empty so callers don't have to special-case.
    """
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    m = sum(values) / len(values)
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / max(len(values) - 1, 1))
    return {"mean": m, "std": sd, "min": min(values), "max": max(values),
            "n": len(values)}


def _mean_path(rows: list[dict], path: tuple) -> float:
    vs = []
    for r in rows:
        node = r
        for k in path:
            node = node[k]
        vs.append(node["acc"])
    return sum(vs) / len(vs) if vs else float("nan")


def print_number_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("NUMBER DOMAIN — per-condition accuracy, averaged over seeds")
    print("=" * 70)
    conds = ("baseline", "swap_arith_only", "swap_ord_only", "swap_both")
    print(f"{'cond':<20} | AddHead inv | AddHead not | CmpHead inv | CmpHead not")
    print("-" * 78)
    for c in conds:
        add_inv = _mean_path(rows, (c, "add", "involving_swap"))
        add_not = _mean_path(rows, (c, "add", "not_involving"))
        cmp_inv = _mean_path(rows, (c, "cmp", "involving_swap"))
        cmp_not = _mean_path(rows, (c, "cmp", "not_involving"))
        print(f"{c:<20} |  {add_inv*100:6.1f}%   |  {add_not*100:6.1f}%   "
              f"|  {cmp_inv*100:6.1f}%   |  {cmp_not*100:6.1f}%")


def print_color_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("COLOR DOMAIN — per-condition accuracy, averaged over seeds")
    print("=" * 70)
    conds = ("baseline", "swap_mix_only", "swap_adj_only", "swap_both")
    print(f"{'cond':<20} | MixHead inv | MixHead not | AdjHead inv | AdjHead not")
    print("-" * 78)
    for c in conds:
        mix_inv = _mean_path(rows, (c, "mix", "involving_swap"))
        mix_not = _mean_path(rows, (c, "mix", "not_involving"))
        adj_inv = _mean_path(rows, (c, "adj", "involving_swap"))
        adj_not = _mean_path(rows, (c, "adj", "not_involving"))
        print(f"{c:<20} |  {mix_inv*100:6.1f}%   |  {mix_not*100:6.1f}%   "
              f"|  {adj_inv*100:6.1f}%   |  {adj_not*100:6.1f}%")
