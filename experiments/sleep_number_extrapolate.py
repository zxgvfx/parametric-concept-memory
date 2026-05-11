"""PAPER §7.5 — number length extrapolation.

Tests whether the §7.4 layered priors (decimal cones + round-number
sampling + last-digit head) let PCM **extrapolate** to numbers
beyond the QuadArithHead training range.

Setup:

* Concept registry: 1..N_total (e.g. 200) — every integer up to
  the test range has a bundle row; rows for n > N_train are
  initialised either randomly (A baseline) or via decimal-cone
  prior (B / BCD).
* :class:`QuadArithHead` trains on triples within ``a, b, c ∈
  [1, N_train]``. This is the "easy" range the system explicitly
  practises arithmetic on.
* :class:`LastDigitHead` (when enabled by D / BCD) sees uniform
  samples from ``[1, N_total]``, so the bundle rows for length-
  OOD numbers receive last-digit gradient even though they never
  participate in any arithmetic triple.

Three test splits per condition:

* **T1 random** — random hold-out from in-range triples; replicates
  §7.4's "A=0.82, BCD=0.74" trade-off.
* **T2 length-OOD-100** — triples with ``a > N_train OR b > N_train``
  and ``a, b, c ≤ 100``; tests whether bundle rows for 31–100 hold
  enough structure to support arithmetic.
* **T3 length-OOD-200** — same, with ``a, b, c ≤ 200``.

Predicted pattern (parallels human "I can do 28+15 because I know
what 8+5 is, applied to the tens place" reasoning):

* T1: A ≥ BCD (BCD's geometric compression slightly hurts random
  interpolation).
* T2 / T3: A ≪ BCD (A has no structure for n > N_train, BCD's
  decimal cone + last-digit head provide a learnable column
  representation).

Usage::

    python -m experiments.sleep_number_extrapolate \\
        --N-train 30 --N-total 100 --n-seeds 5 \\
        --epochs 30 --steps 240 \\
        --out outputs/extrapolate_30_100
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from experiments.quad_study import (
    DEVICE,
    OPS,
    enumerate_triples,
    eval_on_triples,
    train_quad,
)
from experiments.number_decimal_priors import round_number_weights


# ────────────────────────────────────────────────────────────────────────
# Conditions (mirror §7.4)
# ────────────────────────────────────────────────────────────────────────


CONDITIONS = {
    "A_baseline":   {"centroid": "random",        "sample": None,           "ld": False},
    "B_decimal":    {"centroid": "decimal_cones", "sample": None,           "ld": False},
    "C_roundbias":  {"centroid": "random",        "sample": "round_number", "ld": False},
    "D_lastdigit":  {"centroid": "random",        "sample": None,           "ld": True},
    "BCD_combined": {"centroid": "decimal_cones", "sample": "round_number", "ld": True},
}


# ────────────────────────────────────────────────────────────────────────
# Triple split helpers
# ────────────────────────────────────────────────────────────────────────


def _build_splits(
    N_train: int, N_total: int, ood_ratio: float, seed: int,
) -> dict[str, list[tuple[float, float, str, float]]]:
    """Return train + 3 test splits.

    * train: a, b, c ∈ [1, N_train], minus ``ood_ratio`` random hold-out.
    * test_random_in_range: the held-out portion of in-range triples.
    * test_length_100: triples with ``max(a,b) > N_train`` and
      ``a,b,c ≤ 100``, capped at ``min(N_total, 100)``.
    * test_length_200: as above, capped at ``min(N_total, 200)``.
    """
    rng = random.Random(seed)
    in_range = enumerate_triples(N_train, 1.0)

    train: list = []
    test_random_in_range: list = []
    for op, trips in in_range.items():
        trips = list(trips)
        rng.shuffle(trips)
        n_test = max(1, int(len(trips) * ood_ratio))
        for (a, b, c) in trips[:n_test]:
            test_random_in_range.append((a, b, op, c))
        for (a, b, c) in trips[n_test:]:
            train.append((a, b, op, c))

    def _length_test(N_cap: int) -> list:
        if N_cap <= N_train:
            return []
        full = enumerate_triples(min(N_cap, N_total), 1.0)
        out: list = []
        for op, trips in full.items():
            for (a, b, c) in trips:
                if max(a, b) > N_train:
                    out.append((a, b, op, c))
        return out

    return {
        "train": train,
        "test_random_in_range": test_random_in_range,
        "test_length_100": _length_test(100) if N_total >= 100 else [],
        "test_length_200": _length_test(200) if N_total >= 200 else [],
    }


# ────────────────────────────────────────────────────────────────────────
# Per-seed runner
# ────────────────────────────────────────────────────────────────────────


def _run_one(
    seed: int, cond_name: str, cond: dict, *,
    N_train: int, N_total: int,
    epochs: int, steps_per_epoch: int,
    ood_ratio: float,
) -> dict:
    splits = _build_splits(N_train, N_total, ood_ratio, seed)
    weights = (round_number_weights(N_train, boost=5.0)
               if cond["sample"] == "round_number" else None)

    t0 = time.time()
    r = train_quad(
        N_train, 1.0, seed,
        train_triples=splits["train"],
        epochs=epochs, steps_per_epoch=steps_per_epoch,
        n_total=N_total,
        centroid_mode=cond["centroid"],
        digit_sample_weight=weights,
        enable_last_digit_head=bool(cond["ld"]),
        sleep_every=None,
        use_abstract=False,
    )

    out = {
        "seed": seed, "condition": cond_name,
        "centroid_mode": cond["centroid"],
        "sample_mode": cond["sample"],
        "last_digit_head": cond["ld"],
        "wall_s": time.time() - t0,
    }

    train_acc = eval_on_triples(
        r["head"], r["cg"], r["centroids"], splits["train"], 1.0,
    )
    out["train_acc_overall"] = sum(train_acc.values()) / max(len(train_acc), 1)

    for split_name in ("test_random_in_range", "test_length_100",
                       "test_length_200"):
        triples = splits[split_name]
        if not triples:
            out[split_name] = float("nan")
            continue
        acc = eval_on_triples(
            r["head"], r["cg"], r["centroids"], triples, 1.0,
        )
        out[split_name] = sum(acc.values()) / max(len(acc), 1)
        out[f"{split_name}__per_op"] = {op: acc[op] for op in OPS}

    return out


def _stats(xs):
    xs = [x for x in xs
          if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
    return {"mean": m, "std": sd, "min": min(xs), "max": max(xs), "n": len(xs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N-train", type=int, default=30)
    ap.add_argument("--N-total", type=int, default=100)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=86000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=240)
    ap.add_argument("--ood-ratio", type=float, default=0.15)
    ap.add_argument("--conditions", nargs="+",
                    default=list(CONDITIONS.keys()))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/extrapolate"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"  PAPER §7.5 length extrapolation: N_train={args.N_train}, "
          f"N_total={args.N_total}, n_seeds={args.n_seeds}")
    print(f"  device={DEVICE}; conditions={args.conditions}")
    print("=" * 80)

    summary: dict = {
        "config": vars(args) | {"out": str(args.out)},
        "by_condition": {},
    }
    for cond_name in args.conditions:
        if cond_name not in CONDITIONS:
            print(f"  ! unknown condition {cond_name!r}, skipping")
            continue
        cond = CONDITIONS[cond_name]
        print(f"\n── {cond_name}: centroid={cond['centroid']}, "
              f"sample={cond['sample']}, last-digit-head={cond['ld']} ──")
        rows: list[dict] = []
        for si in range(args.n_seeds):
            seed = args.seed_base + si
            r = _run_one(
                seed, cond_name, cond,
                N_train=args.N_train, N_total=args.N_total,
                epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
                ood_ratio=args.ood_ratio,
            )
            rows.append(r)
            print(
                f"  [seed={seed}] "
                f"train={r['train_acc_overall']:.3f} "
                f"in_range={r['test_random_in_range']:.3f}  "
                f"len100={r['test_length_100']:.3f}  "
                f"len200={r['test_length_200']:.3f}  "
                f"({r['wall_s']:.1f}s)"
            )

        cs = {
            "config": cond,
            "per_seed": rows,
            "train_acc": _stats([r["train_acc_overall"] for r in rows]),
            "test_random_in_range": _stats(
                [r["test_random_in_range"] for r in rows]),
            "test_length_100": _stats(
                [r["test_length_100"] for r in rows]),
            "test_length_200": _stats(
                [r["test_length_200"] for r in rows]),
        }
        summary["by_condition"][cond_name] = cs
        rir = cs["test_random_in_range"]
        l100 = cs["test_length_100"]
        l200 = cs["test_length_200"]
        print(
            f"  → in_range = {rir['mean']:.3f} ± {rir['std']:.3f}  "
            f"len100 = {l100['mean']:.3f} ± {l100['std']:.3f}  "
            f"len200 = {l200['mean']:.3f} ± {l200['std']:.3f}"
        )

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print("\n" + "═" * 80)
    print("  cross-condition summary (3 test splits, mean ± std):")
    print(f"  {'condition':<16s} {'in_range':>16s} {'len_100':>16s} "
          f"{'len_200':>16s}")
    for cond_name in args.conditions:
        if cond_name not in summary["by_condition"]:
            continue
        cs = summary["by_condition"][cond_name]
        rir = cs["test_random_in_range"]
        l100 = cs["test_length_100"]
        l200 = cs["test_length_200"]
        print(
            f"  {cond_name:<16s} "
            f"{rir['mean']:>8.3f}±{rir['std']:.3f}   "
            f"{l100['mean']:>8.3f}±{l100['std']:.3f}   "
            f"{l200['mean']:>8.3f}±{l200['std']:.3f}"
        )
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
