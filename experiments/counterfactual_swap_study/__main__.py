"""Entry point: ``python -m experiments.counterfactual_swap_study [--smoke]``.

Sequences the number-domain run, the color-domain run, and the JSON
summary write. Backwards-compatible CLI: every flag the pre-split
single-file ``counterfactual_swap_study.py`` accepted is preserved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._config import (
    COLOR_SWAP_A,
    COLOR_SWAP_B,
    DEVICE,
    NUM_EPOCHS,
    NUM_STEPS_PER_EPOCH,
    NUM_SWAP_A,
    NUM_SWAP_B,
)
from .color_domain import run_color_seed
from .number_domain import run_number_seed
from .reporting import print_color_summary, print_number_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--num-steps", type=int, default=NUM_STEPS_PER_EPOCH)
    ap.add_argument("--color-epochs", type=int, default=30)
    ap.add_argument("--color-steps", type=int, default=200)
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed, reduced epochs, sanity check only")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="domains to skip: 'number' / 'color'")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/counterfactual_swap"))
    args = ap.parse_args()

    if args.smoke:
        args.n_seeds = 1
        args.num_epochs = 6
        args.num_steps = 60
        args.color_epochs = 10
        args.color_steps = 80

    args.out.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "config": {
            "device": DEVICE,
            "n_seeds": args.n_seeds,
            "num_epochs": args.num_epochs,
            "num_steps": args.num_steps,
            "color_epochs": args.color_epochs,
            "color_steps": args.color_steps,
            "num_swap_pair": [NUM_SWAP_A, NUM_SWAP_B],
            "color_swap_pair": [COLOR_SWAP_A, COLOR_SWAP_B],
        }
    }

    if "number" not in args.skip:
        print("=" * 70)
        print(f"NUMBER DOMAIN — swap concept:ans:{NUM_SWAP_A} ↔ "
              f"concept:ans:{NUM_SWAP_B}")
        print("=" * 70)
        rows = []
        for s in range(args.n_seeds):
            seed = 1000 + s
            r = run_number_seed(seed, args.num_epochs, args.num_steps)
            rows.append(r)
            b = r["baseline"]; a = r["swap_arith_only"]; o = r["swap_ord_only"]
            print(f"[seed={seed}] baseline add_all={b['add']['all']['acc']*100:.1f}% "
                  f"cmp_all={b['cmp']['all']['acc']*100:.1f}% | "
                  f"swap-arith: add_inv={a['add']['involving_swap']['acc']*100:.1f}% "
                  f"cmp_inv={a['cmp']['involving_swap']['acc']*100:.1f}% | "
                  f"swap-ord: add_inv={o['add']['involving_swap']['acc']*100:.1f}% "
                  f"cmp_inv={o['cmp']['involving_swap']['acc']*100:.1f}% "
                  f"({r['wall_s']:.1f}s)")
        summary["number_domain"] = {"per_seed": rows}
        print_number_summary(rows)

    if "color" not in args.skip:
        print("=" * 70)
        print(f"COLOR DOMAIN — swap concept:color:{COLOR_SWAP_A} ↔ "
              f"concept:color:{COLOR_SWAP_B}")
        print("=" * 70)
        rows = []
        for s in range(args.n_seeds):
            seed = 2000 + s
            r = run_color_seed(seed, args.color_epochs, args.color_steps)
            rows.append(r)
            b = r["baseline"]; a = r["swap_mix_only"]; o = r["swap_adj_only"]
            print(f"[seed={seed}] baseline mix_all={b['mix']['all']['acc']*100:.1f}% "
                  f"adj_all={b['adj']['all']['acc']*100:.1f}% | "
                  f"swap-mix: mix_inv={a['mix']['involving_swap']['acc']*100:.1f}% "
                  f"adj_inv={a['adj']['involving_swap']['acc']*100:.1f}% | "
                  f"swap-adj: mix_inv={o['mix']['involving_swap']['acc']*100:.1f}% "
                  f"adj_inv={o['adj']['involving_swap']['acc']*100:.1f}% "
                  f"({r['wall_s']:.1f}s)")
        summary["color_domain"] = {"per_seed": rows}
        print_color_summary(rows)

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nsummary written: {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
