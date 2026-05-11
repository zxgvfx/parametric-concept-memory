"""PAPER §6.9 — phoneme cross-language transfer.

Mirror of ``experiments.sleep_color_primaries`` and
``experiments.sleep_number_decimal`` for the phoneme domain.
Splits the 20-phoneme inventory into a *source language*
(seen by the V/M/P heads during training) and a *target
language* (held out from V/M/P training; receives gradient
only via prior layers).

Conditions (paralleling §6.8 / §7.4):

* **A** baseline — random orthogonal centroids, uniform sampling
  over source phonemes, no minimal-pair head.
* **B** + articulator centroid — voice / manner / place cones
  pre-shape every phoneme's bundle row (target-language rows
  get their per-axis cone projection at init).
* **C** + Zipf source sampling — source-language phonemes
  sampled with Zipf 1/rank weights; target untouched.
* **D** + minimal-pair head — pair-input classifier on full
  inventory pairs; target-language bundle rows receive
  axis-relevant gradient via "do these two phonemes differ on
  exactly one axis" without ever predicting their per-axis
  ground-truth labels directly.
* **B+C+D** — all three layers stacked.

Cognitive-science rationale (Werker & Tees 1984; Sun et al. 2023
*Nat Neurosci*; Saffran et al. 1996 *Science* on infant
statistical learning of phoneme sequences): infants learning a
first language acquire feature axes (voice/manner/place)
through massive exposure to a subset; they retain the ability
to discriminate non-native contrasts up to ~6 months. We model
the universal-discrimination phase: target-language phonemes
were never task-classified, but their bundle representations
should still encode the right axis identity if the feature
geometry transferred.

Usage::

    python -m experiments.sleep_phoneme_transfer --n-seeds 5 \\
        --n-target 7 --epochs 60 \\
        --out outputs/phoneme_transfer
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from experiments.phoneme_concept_study._config import (
    DEVICE, EMBED_DIM, EPOCHS, FACET_M, FACET_P, FACET_V,
    MANNER_DIM, PLACE_DIM, STEPS_PER_EPOCH, VOICE_DIM,
)
from experiments.phoneme_concept_study.inventory import (
    N_PH, PHONEMES, feat_of,
)
from experiments.phoneme_concept_study.train import train_one
from experiments.phoneme_transfer_priors import (
    make_articulator_centroids,
    source_natural_weights,
    zipf_phonotactic_weights,
)


# Each phoneme features list (voice, manner, place).
PHONEME_FEATS = [(p[1], p[2], p[3]) for p in PHONEMES]


from pcm.diagnostics import (
    AblationCondition, AblationLayers, run_causal_ablation,
)


CONDITIONS = (
    AblationCondition("A_baseline", AblationLayers()),
    AblationCondition("B_articulator", AblationLayers(B="articulator")),
    AblationCondition("C_zipfsource", AblationLayers(C="zipf")),
    AblationCondition("D_minpair", AblationLayers(D=True)),
    AblationCondition("BCD_combined",
                      AblationLayers(B="articulator", C="zipf", D=True)),
)


def _split_source_target(seed: int, n_target: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed + 5555)
    perm = list(range(N_PH))
    rng.shuffle(perm)
    target = sorted(perm[:n_target])
    source = sorted(perm[n_target:])
    return source, target


def _build_centroid_init(
    mode: str, seed: int,
) -> dict[str, torch.Tensor] | None:
    if mode == "random":
        return None
    if mode == "articulator":
        # 128-d cones; per-axis pool will read the first {VOICE_DIM /
        # MANNER_DIM / PLACE_DIM} dims of the same articulator-grouped
        # vectors. Different seed per axis to avoid identity overlap.
        cents_v = make_articulator_centroids(
            PHONEME_FEATS, EMBED_DIM, seed,
            voice_weight=1.5, manner_weight=0.7, place_weight=0.7,
        )
        cents_m = make_articulator_centroids(
            PHONEME_FEATS, EMBED_DIM, seed + 1,
            voice_weight=0.7, manner_weight=1.5, place_weight=0.7,
        )
        cents_p = make_articulator_centroids(
            PHONEME_FEATS, EMBED_DIM, seed + 2,
            voice_weight=0.7, manner_weight=0.7, place_weight=1.5,
        )
        return {FACET_V: cents_v, FACET_M: cents_m, FACET_P: cents_p}
    raise ValueError(f"unknown centroid mode {mode!r}")


def _build_sample_weight(mode, src_idx) -> list[float] | None:
    if mode is None:
        return None
    if mode == "zipf":
        return zipf_phonotactic_weights(len(src_idx), alpha=1.0)
    if mode == "manner_natural":
        feats_src = [PHONEME_FEATS[i] for i in src_idx]
        return source_natural_weights(feats_src)
    raise ValueError(f"unknown sample mode {mode!r}")


def _run_one(
    seed: int, layers: AblationLayers, *,
    n_target: int, epochs: int, steps_per_epoch: int,
) -> dict:
    src_idx, tgt_idx = _split_source_target(seed, n_target)
    centroid_init = _build_centroid_init(
        layers.B if layers.B else "random", seed,
    )
    sample_w = _build_sample_weight(layers.C, src_idx)

    t0 = time.time()
    r = train_one(
        "triple", seed,
        epochs=epochs, steps_per_epoch=steps_per_epoch,
        source_indices=src_idx,
        sample_weight=sample_w,
        centroid_init=centroid_init,
        enable_minimal_pair_head=bool(layers.is_active("D")),
    )

    out = {
        "centroid_mode": layers.B,
        "sample_mode": layers.C,
        "minimal_pair_head": bool(layers.is_active("D")),
        "n_source": r["n_source"], "n_target": r["n_target"],
        "source_indices": src_idx, "target_indices": tgt_idx,
        "wall_s": time.time() - t0,
        "accs_full": r["accs"],
        "accs_source": r["accs_source"],
        "accs_target": r["accs_target"],
    }
    # Flatten per-axis source/target accuracies into top-level keys so
    # they can be aggregated by run_causal_ablation as primary metrics.
    for axis in ("v", "m", "p"):
        out[f"source_{axis}"] = r["accs_source"].get(axis)
        out[f"target_{axis}"] = r["accs_target"].get(axis)
        sa = r["accs_source"].get(axis, 0) or 0
        ta = r["accs_target"].get(axis, 0) or 0
        out[f"delta_{axis}"] = ta - sa
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=90000)
    ap.add_argument("--n-target", type=int, default=7,
                    help="number of phonemes held out as target language "
                         f"(out of {N_PH}); rest become source")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/phoneme_transfer"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"  PAPER §6.9 phoneme cross-language transfer: "
          f"N_PH={N_PH}, n_target={args.n_target}, "
          f"n_source={N_PH - args.n_target}, n_seeds={args.n_seeds}")
    print(f"  device={DEVICE}; protocol=pcm.diagnostics.run_causal_ablation")
    print("=" * 80)

    chance = {"v": 1 / 2, "m": 1 / 4, "p": 1 / 4}

    summary = run_causal_ablation(
        _run_one,
        n_seeds=args.n_seeds,
        seed_base=args.seed_base,
        conditions=CONDITIONS,
        primary_metrics=(
            "source_v", "source_m", "source_p",
            "target_v", "target_m", "target_p",
            "delta_v", "delta_m", "delta_p",
        ),
        n_target=args.n_target,
        epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
    )
    summary["config"] |= vars(args) | {"out": str(args.out)}
    summary["chance_levels"] = chance

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "═" * 80)
    print(f"  cross-condition target-language accuracy (n_target={args.n_target}):")
    print(f"  chance: V=0.500, M=0.250, P=0.250")
    print(f"  {'condition':<16s} "
          f"{'tgt_V':>14s} {'tgt_M':>14s} {'tgt_P':>14s}")
    for cond in CONDITIONS:
        cs = summary["by_condition"][cond.name]
        print(
            f"  {cond.name:<16s} "
            f"{cs['target_v']['mean']:>6.3f}±{cs['target_v']['std']:.3f}  "
            f"{cs['target_m']['mean']:>6.3f}±{cs['target_m']['std']:.3f}  "
            f"{cs['target_p']['mean']:>6.3f}±{cs['target_p']['std']:.3f}"
        )
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
