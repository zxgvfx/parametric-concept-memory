"""CLI entry: ``python -m experiments.space_concept_study --n-seeds 5``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._config import DEVICE, EPOCHS, N_CELLS, N_COLS, N_ROWS, STEPS_PER_EPOCH
from .runners import run_e1_multi_seed, run_e2_shuffled, run_e4_permutation
from .train import train_one


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    ap.add_argument("--out", type=Path, default=Path("outputs/space_concept"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], help="e1 / e2 / e4")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.n_seeds = 1
        args.epochs = 10
        args.steps_per_epoch = 80

    summary: dict = {
        "n_seeds": args.n_seeds,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "device": DEVICE,
        "n_cells": N_CELLS,
        "grid": [N_ROWS, N_COLS],
    }

    if "e1" not in args.skip:
        print("=" * 60); print("E1: Multi-seed (single vs dual)"); print("=" * 60)
        summary["E1_multi_seed"] = run_e1_multi_seed(
            args.n_seeds, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch
        )

    if "e2" not in args.skip:
        print("=" * 60); print("E2: Shuffled concept counterfactual"); print("=" * 60)
        summary["E2_shuffled"] = run_e2_shuffled(
            args.n_seeds, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch
        )

    if "e4" not in args.skip:
        print("=" * 60); print("E4: Cross-facet permutation test"); print("=" * 60)
        d = train_one("dual", 1000,
                      epochs=args.epochs, steps_per_epoch=args.steps_per_epoch)
        summary["E4_permutation"] = run_e4_permutation(d["bundle_state"])

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60); print("SUMMARY"); print("=" * 60)
    if "E1_multi_seed" in summary:
        e1 = summary["E1_multi_seed"]
        sL = e1["single_rho_L1"]; sl = e1["single_rho_linear_flat"]
        sr = e1["single_rho_row_within"]; sc = e1["single_rho_col_within"]
        md = e1["single_mds_disparity"]
        al = e1["dual_cross_facet_align"]
        print(f"E1  single ρ_L1         = {sL['mean']:+.3f} ± {sL['std']:.3f}")
        print(f"E1  single ρ_linear_flat= {sl['mean']:+.3f} ± {sl['std']:.3f}"
              f"  (should be << ρ_L1)")
        print(f"E1  single ρ_row_within = {sr['mean']:+.3f} ± {sr['std']:.3f}")
        print(f"E1  single ρ_col_within = {sc['mean']:+.3f} ± {sc['std']:.3f}")
        print(f"E1  single MDS disp     = {md['mean']:.3f} ± {md['std']:.3f}"
              f"  (0=perfect grid)")
        print(f"E1  dual cross-facet align = {al['mean']:+.3f} ± {al['std']:.3f}")
    if "E2_shuffled" in summary:
        e2 = summary["E2_shuffled"]
        print(f"E2  |ρ_L1| shuffled = {e2['abs_rho_L1_stats']['mean']:.3f} "
              f"± {e2['abs_rho_L1_stats']['std']:.3f}")
    if "E4_permutation" in summary:
        e4 = summary["E4_permutation"]
        print(f"E4  observed = {e4['observed_cross_facet_rho']:+.3f}  "
              f"null_mean = {e4['null_mean']:+.3f}  p = {e4['p_value']:.3f}  "
              f"→ {e4['conclusion']}")


if __name__ == "__main__":
    main()
