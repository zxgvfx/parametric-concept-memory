"""PAPER §7.5-color — colour hue holdout (mirror of §7.5 number length OOD).

This is the simplest application of the canonical PCM
:func:`pcm.diagnostics.run_causal_ablation` protocol: tests
whether the §6.8 layered priors (LMS-like centroid + green-peak
sampling + ripe-fruit head) let PCM **extrapolate** mixing to a
target hue that was never in any training triple as the *output*
``c``.

Setup:

* All 12 hues are registered (PCM concept inventory unchanged).
* Mixing triples whose target hue ``c ∈ holdout_target_hues`` are
  held out from training and become the test set; remaining
  triples go into training.
* The held-out hue still receives gradient via the ripe-fruit
  head (D condition) when in the ripe set, and via centroid
  prior (B condition) regardless. This is the closest colour-
  domain analogue of §7.5's number length OOD: the held-out
  concept is registered and prior-influenced, but never the
  output of the primary task.

The script is a *dogfood* of the F33 protocol: rather than
re-implementing 5-condition × N-seed orchestration locally, it
declares its B/C/D layer activations as
:class:`pcm.diagnostics.AblationCondition` instances and lets
:func:`pcm.diagnostics.run_causal_ablation` drive the run loop,
per-metric aggregation, summary-dict layout, and verbose
console output.

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

from pcm.diagnostics import (
    AblationCondition, AblationLayers, run_causal_ablation,
)
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


# Condition catalog. Layer-flag values chosen for direct dispatch
# in `_run_one` below: B = centroid mode, C = sample mode,
# D = ripe-head bool.
CONDITIONS = (
    AblationCondition("A_baseline", AblationLayers()),
    AblationCondition("B_lms", AblationLayers(B="lms")),
    AblationCondition("C_greenpeak", AblationLayers(C="green_peak")),
    AblationCondition("D_ripehead", AblationLayers(D=True)),
    AblationCondition("BCD_combined",
                      AblationLayers(B="lms", C="green_peak", D=True)),
)


def _run_one(seed: int, layers: AblationLayers, *,
             holdout_hue: int, epochs: int, steps: int) -> dict:
    centroids = (
        make_lms_like_centroids(N_COLORS, EMBED_DIM, seed)
        if layers.B == "lms"
        else make_random_orthogonal_centroids(N_COLORS, EMBED_DIM, seed)
    )
    weights = _green_peak_weights() if layers.C == "green_peak" else None
    t0 = time.time()
    r = train_one(
        "single", seed, centroids,
        epochs=epochs, steps_per_epoch=steps,
        mix_sample_weight=weights,
        enable_ripe_head=bool(layers.is_active("D")),
        holdout_target_hues=(holdout_hue,),
        sleep_every=None,
        use_abstract=False,
    )
    return {
        "wall_s": time.time() - t0,
        "holdout_hue": holdout_hue,
        "centroid_mode": layers.B,
        "sample_mode": layers.C,
        "ripe_head": bool(layers.is_active("D")),
        "train_acc": r["mix_acc"],
        "ood_acc_holdout_hue": r["mix_holdout_acc"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-hue", type=int, default=5)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=88000)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    ap.add_argument("--out", type=Path, default=Path("outputs/color_holdout"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  PAPER §7.5-color hue-holdout ablation: holdout_hue={args.holdout_hue}, "
          f"n_seeds={args.n_seeds}")
    print(f"  device={DEVICE}; protocol=pcm.diagnostics.run_causal_ablation")
    print("=" * 78)

    summary = run_causal_ablation(
        _run_one,
        n_seeds=args.n_seeds,
        seed_base=args.seed_base,
        conditions=CONDITIONS,
        primary_metrics=("train_acc", "ood_acc_holdout_hue"),
        holdout_hue=args.holdout_hue,
        epochs=args.epochs,
        steps=args.steps_per_epoch,
    )
    summary["config"] |= vars(args) | {"out": str(args.out)}

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "═" * 78)
    print(f"  cross-condition summary, target hue={args.holdout_hue}, "
          f"chance ≈ 1/{N_COLORS} = {1/N_COLORS:.3f}:")
    print(f"  {'condition':<16s} {'train_acc':>16s} {'ood_holdout_hue':>20s}")
    for cond in CONDITIONS:
        cs = summary["by_condition"][cond.name]
        ta = cs["train_acc"]
        oa = cs["ood_acc_holdout_hue"]
        print(
            f"  {cond.name:<16s} "
            f"{ta['mean']:>8.3f}±{ta['std']:.3f}   "
            f"{oa['mean']:>10.3f}±{oa['std']:.3f}"
        )
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
