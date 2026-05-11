"""CLI entry: ``python -m experiments.color_concept_study --n-seeds 5``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._config import DEVICE, EMBED_DIM, EPOCHS, N_COLORS, STEPS_PER_EPOCH
from .graph_builder import make_random_orthogonal_centroids
from .runners import run_e1_multi_seed, run_e2_shuffled, run_e4_permutation
from .train import train_one


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    ap.add_argument("--out", type=Path, default=Path("outputs/color_concept"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], help="e1 / e2 / e4")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.n_seeds = 1

    summary: dict = {
        "n_seeds": args.n_seeds,
        "device": DEVICE,
        "n_colors": N_COLORS,
    }

    if "e1" not in args.skip:
        print("=" * 60); print("E1: Multi-seed (single vs dual)"); print("=" * 60)
        summary["E1_multi_seed"] = run_e1_multi_seed(args.n_seeds)

    if "e2" not in args.skip:
        print("=" * 60); print("E2: Shuffled concept counterfactual"); print("=" * 60)
        summary["E2_shuffled"] = run_e2_shuffled(args.n_seeds)

    if "e4" not in args.skip:
        print("=" * 60); print("E4: Cross-facet permutation test"); print("=" * 60)
        centroids = make_random_orthogonal_centroids(N_COLORS, EMBED_DIM, 1000)
        d = train_one("dual", 1000, centroids)
        summary["E4_permutation"] = run_e4_permutation(d["bundle_state"])

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60); print("SUMMARY"); print("=" * 60)
    if "E1_multi_seed" in summary:
        e1 = summary["E1_multi_seed"]
        sc = e1["single_rho_circular"]; sl = e1["single_rho_linear"]
        dc = e1["dual_rho_mix_circular"]; al = e1["dual_cross_facet_align"]
        print(f"E1  single ρ_circ = {sc['mean']:+.3f} ± {sc['std']:.3f}  "
              f"ρ_lin = {sl['mean']:+.3f}")
        print(f"E1  dual   ρ_mix_circ = {dc['mean']:+.3f} ± {dc['std']:.3f}")
        print(f"E1  dual   cross-facet align = {al['mean']:+.3f} ± {al['std']:.3f}")
    if "E2_shuffled" in summary:
        e2 = summary["E2_shuffled"]
        s = e2["abs_rho_circular_stats"]
        print(f"E2  |ρ_circ| shuffled = {s['mean']:.3f} ± {s['std']:.3f}")
    if "E4_permutation" in summary:
        e4 = summary["E4_permutation"]
        print(f"E4  observed = {e4['observed_cross_facet_rho']:+.3f}  "
              f"null_mean = {e4['null_mean']:+.3f}  p = {e4['p_value']:.3f}  "
              f"→ {e4['conclusion']}")


if __name__ == "__main__":
    main()
