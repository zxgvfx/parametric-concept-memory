"""PAPER §7.5-color — colour hue holdout (mirror of §7.5 number length OOD).

Tests whether the §6.8 layered priors (LMS-like centroid + green-peak
sampling + ripe-fruit head) let PCM **extrapolate** mixing to a target
hue that was never in any training triple as the *output* (``c``).

Setup:

* All 12 hues are registered (PCM concept inventory unchanged).
* Mixing triples whose target hue ``c ∈ holdout_target_hues`` are
  held out from training and become the test set; remaining
  triples go into training.
* The held-out hue still receives gradient via the ripe-fruit head
  (D condition) when in the ripe set, and via centroid prior
  (B condition) regardless. This is the closest colour-domain
  analogue of §7.5's number length OOD: the held-out concept is
  registered and prior-influenced, but never the output of the
  primary task.

Conditions are the standard §6.8 5-cell A / B / C / D / B+C+D set.

The §7.5-number experiment found that A/B/C flat-line at chance
while D/BCD lift OOD by +1.1 pp (statistically robust but small).
We expect a parallel pattern here: layered priors give a small
cross-symmetry-class shift on the held-out target hue, but the
absolute OOD accuracy may stay low because the colour mixing
function is fully cyclic and the held-out hue is just one out of
12 candidate outputs.

Usage::

    python -m experiments.sleep_color_holdout \\
        --holdout-hue 5 --n-seeds 5 --epochs 30 \\
        --out outputs/color_holdout_h5
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from experiments.color_concept_study._config import (
    DEVICE, EMBED_DIM, EPOCHS, N_COLORS, STEPS_PER_EPOCH,
)
from experiments.color_concept_study.graph_builder import (
    make_lms_like_centroids,
    make_random_orthogonal_centroids,
)
from experiments.color_concept_study.train import train_one


def _green_peak_weights() -> list[float]:
    out = []
    for i in range(N_COLORS):
        d = min(abs(i - 4), N_COLORS - abs(i - 4))
        out.append(math.exp(-d / 3.0))
    return out


CONDITIONS = {
    "A_baseline":    {"centroid": "random", "sample": None,         "ripe": False},
    "B_lms":         {"centroid": "lms",    "sample": None,         "ripe": False},
    "C_greenpeak":   {"centroid": "random", "sample": "green_peak", "ripe": False},
    "D_ripehead":    {"centroid": "random", "sample": None,         "ripe": True},
    "BCD_combined":  {"centroid": "lms",    "sample": "green_peak", "ripe": True},
}


def _build_centroids(mode: str, seed: int):
    if mode == "random":
        return make_random_orthogonal_centroids(N_COLORS, EMBED_DIM, seed)
    if mode == "lms":
        return make_lms_like_centroids(N_COLORS, EMBED_DIM, seed)
    raise ValueError(f"unknown centroid mode {mode!r}")


def _build_weights(mode):
    if mode is None:
        return None
    if mode == "green_peak":
        return _green_peak_weights()
    raise ValueError(f"unknown sample mode {mode!r}")


def _run_one(seed: int, cond_name: str, cond: dict, *,
             holdout_hue: int, epochs: int, steps: int) -> dict:
    centroids = _build_centroids(cond["centroid"], seed)
    weights = _build_weights(cond["sample"])
    t0 = time.time()
    r = train_one(
        "single", seed, centroids,
        epochs=epochs, steps_per_epoch=steps,
        mix_sample_weight=weights,
        enable_ripe_head=bool(cond["ripe"]),
        holdout_target_hues=(holdout_hue,),
        sleep_every=None,
        use_abstract=False,
    )
    return {
        "seed": seed,
        "condition": cond_name,
        "centroid_mode": cond["centroid"],
        "sample_mode": cond["sample"],
        "ripe_head": cond["ripe"],
        "holdout_hue": holdout_hue,
        "wall_s": time.time() - t0,
        "train_acc": r["mix_acc"],
        "ood_acc_holdout_hue": r["mix_holdout_acc"],
    }


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
    ap.add_argument("--holdout-hue", type=int, default=5)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=88000)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    ap.add_argument("--conditions", nargs="+",
                    default=list(CONDITIONS.keys()))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/color_holdout"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  PAPER §7.5-color hue-holdout ablation: holdout_hue={args.holdout_hue}, "
          f"n_seeds={args.n_seeds}")
    print(f"  device={DEVICE}; conditions={args.conditions}")
    print("=" * 78)

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
              f"sample={cond['sample']}, ripe={cond['ripe']} ──")
        rows = []
        for si in range(args.n_seeds):
            seed = args.seed_base + si
            r = _run_one(
                seed, cond_name, cond,
                holdout_hue=args.holdout_hue,
                epochs=args.epochs, steps=args.steps_per_epoch,
            )
            rows.append(r)
            print(
                f"  [seed={seed}] train={r['train_acc']:.3f}  "
                f"ood_holdout_hue={r['ood_acc_holdout_hue']:.3f}  "
                f"({r['wall_s']:.1f}s)"
            )

        cs = {
            "config": cond,
            "per_seed": rows,
            "train_acc": _stats([r["train_acc"] for r in rows]),
            "ood_acc_holdout_hue": _stats([r["ood_acc_holdout_hue"] for r in rows]),
        }
        summary["by_condition"][cond_name] = cs
        print(
            f"  → train={cs['train_acc']['mean']:.3f}±{cs['train_acc']['std']:.3f}  "
            f"ood={cs['ood_acc_holdout_hue']['mean']:.3f}±"
            f"{cs['ood_acc_holdout_hue']['std']:.3f}"
        )

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print("\n" + "═" * 78)
    print(f"  cross-condition summary, target hue={args.holdout_hue}, "
          f"chance ≈ 1/{N_COLORS} = {1/N_COLORS:.3f}:")
    print(f"  {'condition':<16s} {'train_acc':>16s} {'ood_holdout_hue':>20s}")
    for cond_name in args.conditions:
        if cond_name not in summary["by_condition"]:
            continue
        cs = summary["by_condition"][cond_name]
        ta = cs["train_acc"]
        oa = cs["ood_acc_holdout_hue"]
        print(
            f"  {cond_name:<16s} "
            f"{ta['mean']:>8.3f}±{ta['std']:.3f}   "
            f"{oa['mean']:>10.3f}±{oa['std']:.3f}"
        )
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
