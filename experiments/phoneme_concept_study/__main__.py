"""CLI entry: ``python -m experiments.phoneme_concept_study --n-seeds 3``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._config import DEVICE, EPOCHS, FACET_M, FACET_P, FACET_V, STEPS_PER_EPOCH
from .inventory import N_PH, PHONEMES
from .metrics import _perm_test_align
from .runners import run_e1_multi_seed, run_e2_shuffled
from .train import train_one


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    ap.add_argument("--out", type=Path, default=Path("outputs/phoneme_concept"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], help="e1 / e2 / e4")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.n_seeds = 1
        args.epochs = 20
        args.steps_per_epoch = 60

    summary: dict = {
        "n_seeds": args.n_seeds,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "device": DEVICE,
        "n_phonemes": N_PH,
        "phoneme_set": [
            {"label": p[0], "voice": p[1], "manner": p[2], "place": p[3]}
            for p in PHONEMES
        ],
    }

    if "e1" not in args.skip:
        print("=" * 60); print("E1: Multi-seed (triple orthogonal muscles)"); print("=" * 60)
        summary["E1_multi_seed"] = run_e1_multi_seed(
            args.n_seeds, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch
        )

    if "e2" not in args.skip:
        print("=" * 60); print("E2: Shuffled concept counterfactual"); print("=" * 60)
        summary["E2_shuffled"] = run_e2_shuffled(
            args.n_seeds, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch
        )

    if "e4" not in args.skip:
        print("=" * 60); print("E4: Cross-facet permutation tests"); print("=" * 60)
        t = train_one("triple", 1000, epochs=args.epochs,
                      steps_per_epoch=args.steps_per_epoch)
        bs = t["bundle_state"]
        summary["E4_permutation"] = {
            "v_m": _perm_test_align(bs, FACET_V, FACET_M),
            "v_p": _perm_test_align(bs, FACET_V, FACET_P),
            "m_p": _perm_test_align(bs, FACET_M, FACET_P),
        }

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60); print("SUMMARY"); print("=" * 60)
    if "E1_multi_seed" in summary:
        e1 = summary["E1_multi_seed"]
        print("E1 — per-facet geometry on own axis (ρ_same):")
        for ax in ("v", "m", "p"):
            s = e1[f"rho_same_axis_{ax}"]
            print(f"    {ax}: {s['mean']:+.3f} ± {s['std']:.3f}")
        print("E1 — cross-facet alignment (expect ≈ 0):")
        for pair in ("vm", "vp", "mp"):
            s = e1[f"cross_facet_align_{pair}"]
            print(f"    {pair}: {s['mean']:+.3f} ± {s['std']:.3f}")
        print("E1 — intra/inter class cos gap (expect positive):")
        for ax in ("v", "m", "p"):
            s = e1[f"gap_{ax}"]
            print(f"    {ax}: {s['mean']:+.3f} ± {s['std']:.3f}")
    if "E2_shuffled" in summary:
        e2 = summary["E2_shuffled"]
        print("E2 — |ρ_same| shuffled (expect ≈ 0):")
        for ax in ("v", "m", "p"):
            s = e2[f"abs_rho_same_{ax}_stats"]
            print(f"    {ax}: {s['mean']:.3f} ± {s['std']:.3f}")
    if "E4_permutation" in summary:
        e4 = summary["E4_permutation"]
        print("E4 — cross-facet permutation tests:")
        for pair, v in e4.items():
            print(f"    {pair}: ρ = {v['observed']:+.3f}  "
                  f"null_mean = {v['null_mean']:+.3f}  "
                  f"p = {v['p_value']:.3f}  → {v['conclusion']}")


if __name__ == "__main__":
    main()
