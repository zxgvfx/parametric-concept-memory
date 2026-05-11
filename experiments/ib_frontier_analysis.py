"""PAPER §18 (S4) — Information-Bottleneck (IB) frontier evaluation.

Post-processing analysis: given a PCM run that emits a
``hue_to_proto`` (or ``num_to_proto``) mapping per seed, compute
each run's location in the (Complexity, Accuracy) plane defined by
the IB framework (Tishby, Pereira & Bialek 1999; Zaslavsky et al.
2018 *PNAS* "Efficient compression in colour naming"; Nat Hum
Behav 2025 s41562-025-02336-w).

For a hard assignment ``q(c|x) ∈ {0,1}`` of items ``x ∈ X`` to
codebook entries ``c ∈ C`` and a reference target ``y ∈ Y``:

* **Complexity** ≡ ``I(X; C) = H(C) − H(C|X) = H(C)``
  (since the assignment is deterministic).
* **Informativeness** ≡ ``I(C; Y)``.

The IB frontier traces the curve ``min_q I(X;C) − β·I(C;Y)`` over
``β ∈ [0, ∞)``. Human languages cluster along the predicted
frontier; we ask whether PCM does too — and how each causal
condition (A / B / C / D / BCD) and second-level partition
(warm-cool / Hering4 / parity) compares against it.

This module is **pure analysis** — no training, no PCM imports
beyond reading existing ``summary.json`` files. It is designed to
be re-run cheaply over different experiment outputs.

Usage::

    python -m experiments.ib_frontier_analysis \\
        --hsae outputs/hsae_color_6to2_8x30/summary.json \\
        --hsae-domain color \\
        --out outputs/ib_color
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


__all__ = [
    "entropy",
    "joint_entropy",
    "mutual_info",
    "ib_point",
    "main",
]


# ─────────────────────────────────────────────────────────────────
# Information-theoretic primitives.
# ─────────────────────────────────────────────────────────────────


def entropy(counter: Counter, total: int) -> float:
    """Shannon entropy in bits over ``Counter.values()``."""
    if total <= 0:
        return 0.0
    h = 0.0
    for v in counter.values():
        if v <= 0:
            continue
        p = v / total
        h -= p * math.log2(p)
    return h


def joint_entropy(pairs: list[tuple[int, int]]) -> float:
    n = len(pairs)
    if n == 0:
        return 0.0
    return entropy(Counter(pairs), n)


def mutual_info(xs: list[int], ys: list[int]) -> float:
    """``I(X;Y) = H(X) + H(Y) − H(X,Y)``, ε-free."""
    if len(xs) != len(ys) or not xs:
        return 0.0
    n = len(xs)
    return (
        entropy(Counter(xs), n)
        + entropy(Counter(ys), n)
        - joint_entropy(list(zip(xs, ys)))
    )


def ib_point(
    items: list[int],
    item_to_codebook: list[int],
    item_to_target: list[int],
) -> dict[str, float]:
    """One point on the IB plane.

    Args:
        items: ``[0, 1, ..., N-1]`` (used only for length).
        item_to_codebook: per-item codebook id (anchor / cluster).
        item_to_target: per-item reference label.

    Returns:
        dict with ``complexity`` (``I(X;C)``) and ``accuracy``
        (``I(C;Y)``). Both in bits.
    """
    if not (len(items) == len(item_to_codebook) == len(item_to_target)):
        raise ValueError("items, codebook, target lists must have same length")
    return {
        "complexity": mutual_info(items, item_to_codebook),
        "accuracy": mutual_info(item_to_codebook, item_to_target),
    }


# ─────────────────────────────────────────────────────────────────
# Reference targets for color (12 hues) and number (variable N).
# ─────────────────────────────────────────────────────────────────


def color_targets(n_colors: int = 12) -> dict[str, list[int]]:
    """Reference labellings for the color domain."""
    return {
        "RGB": [0 if h in {0, 1, 11} else 1 if h in {3, 4, 5} else 2
                for h in range(n_colors)],
        "Hering4": [{0: 0, 1: 0, 11: 0,
                     2: 1, 3: 1,
                     4: 2, 5: 2, 6: 2,
                     7: 3, 8: 3, 9: 3, 10: 3}.get(h, 3)
                    for h in range(n_colors)],
        "WarmCool6": [h % 2 for h in range(n_colors)],
        "warm_cool": [0 if h <= 5 else 1 for h in range(n_colors)],
        "red_wedge": [1 if h in {0, 1, 11} else 0 for h in range(n_colors)],
    }


def number_targets(n_total: int) -> dict[str, list[int]]:
    """Reference labellings for the number domain."""
    return {
        "parity": [n % 2 for n in range(n_total)],
        "small_large": [0 if n < n_total // 2 else 1 for n in range(n_total)],
        "last_digit": [n % 10 for n in range(n_total)],
        "decade": [n // 10 for n in range(n_total)],
    }


# ─────────────────────────────────────────────────────────────────
# Per-seed analysis from a ``hue_to_proto`` / ``num_to_proto`` map.
# ─────────────────────────────────────────────────────────────────


def analyse_seed(
    item_to_codebook: list[int],
    targets: dict[str, list[int]],
) -> dict[str, dict[str, float]]:
    items = list(range(len(item_to_codebook)))
    out = {}
    complexity = mutual_info(items, item_to_codebook)
    for tgt_name, tgt_labels in targets.items():
        if len(tgt_labels) != len(item_to_codebook):
            continue
        accuracy = mutual_info(item_to_codebook, tgt_labels)
        out[tgt_name] = {
            "complexity": complexity,
            "accuracy": accuracy,
            "efficiency": (
                accuracy / complexity if complexity > 1e-9 else 0.0
            ),
        }
    return out


def aggregate_per_target(
    seed_results: list[dict[str, dict[str, float]]],
    target_name: str,
) -> dict[str, float]:
    points: list[tuple[float, float]] = []
    for r in seed_results:
        if target_name not in r:
            continue
        p = r[target_name]
        points.append((p["complexity"], p["accuracy"]))
    if not points:
        return {"n": 0}
    cs, as_ = zip(*points)
    n = len(points)
    cm = sum(cs) / n
    am = sum(as_) / n
    cs_v = sum((c - cm) ** 2 for c in cs) / max(n - 1, 1)
    as_v = sum((a - am) ** 2 for a in as_) / max(n - 1, 1)
    eff = [a / c if c > 1e-9 else 0.0 for c, a in points]
    em = sum(eff) / n
    return {
        "n": n,
        "complexity_mean": cm,
        "complexity_std": math.sqrt(cs_v),
        "accuracy_mean": am,
        "accuracy_std": math.sqrt(as_v),
        "efficiency_mean": em,
    }


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────


def _load_hsae_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hsae", type=Path, required=True,
                    help="path to sleep_hierarchical_anchors summary.json")
    ap.add_argument("--hsae-domain", choices=["color", "number"],
                    default="color")
    ap.add_argument("--n-total", type=int, default=30,
                    help="number domain only: |items|")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/ib_frontier"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = _load_hsae_summary(args.hsae)
    seed_rows = summary.get("per_seed", [])

    if args.hsae_domain == "color":
        n_items = 12
        targets = color_targets(n_items)
        proto_key = "hue_to_proto"
    else:
        n_items = args.n_total
        targets = number_targets(n_items)
        proto_key = "num_to_proto"

    seed_results: list[dict] = []
    for r in seed_rows:
        mapping = r.get(proto_key)
        if not mapping or len(mapping) != n_items:
            continue
        seed_results.append({
            "seed": r.get("seed"),
            **analyse_seed(mapping, targets),
        })

    if not seed_results:
        print(
            f"No usable {proto_key} entries found in {args.hsae}; "
            "did you run sleep_hierarchical_anchors with --domain "
            f"{args.hsae_domain}?"
        )
        return

    # Per-target aggregation.
    target_names = sorted({
        k for r in seed_results
        for k in r if k != "seed" and isinstance(r[k], dict)
    })
    aggregates = {
        t: aggregate_per_target(seed_results, t)
        for t in target_names
    }

    out = {
        "config": vars(args) | {"out": str(args.out)},
        "per_seed": seed_results,
        "aggregates": aggregates,
    }
    (args.out / "summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str)
    )

    print("=" * 76)
    print(
        f"  PAPER §18 (S4) IB frontier — domain={args.hsae_domain}, "
        f"n_seeds={len(seed_results)}"
    )
    print("=" * 76)
    print(f"  {'target':<14s} {'complexity':>14s} {'accuracy':>14s} "
          f"{'efficiency':>12s}  n")
    for t, a in aggregates.items():
        if a["n"] == 0:
            continue
        print(
            f"  {t:<14s} "
            f"{a['complexity_mean']:>+8.3f}±{a['complexity_std']:.3f}   "
            f"{a['accuracy_mean']:>+8.3f}±{a['accuracy_std']:.3f}    "
            f"{a['efficiency_mean']:>+8.3f}    {a['n']}"
        )
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
