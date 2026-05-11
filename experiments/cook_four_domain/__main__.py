"""CLI entry: ``python -m experiments.cook_four_domain``."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ._common import DEVICE, diff_metrics
from .color import train_color
from .number import train_number
from .phoneme import train_phoneme
from .space import train_space


DOMAINS = {
    "number":  (train_number,  {"epochs": 6, "steps": 60}),
    "color":   (train_color,   {"epochs": 8, "steps": 80}),
    "space":   (train_space,   {"epochs": 6, "steps": 80}),
    "phoneme": (train_phoneme, {"epochs": 6, "steps": 60}),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--smoke", action="store_true",
                    help="reduced epochs/steps")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/cook_four_domain"))
    ap.add_argument("--only", nargs="*", default=None,
                    help=f"subset of domains: {list(DOMAINS)}")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    keys = args.only or list(DOMAINS)
    summary: dict = {"config": {"seed": args.seed, "device": DEVICE,
                                 "domains": keys}}
    print(f"[cook_four_domain] device={DEVICE} seed={args.seed} domains={keys}")

    overall_pass = True
    for domain in keys:
        if domain not in DOMAINS:
            print(f"  skip unknown domain: {domain}")
            continue
        train_fn, hp = DOMAINS[domain]
        if args.smoke:
            hp = {"epochs": max(1, hp["epochs"] // 2),
                   "steps": max(20, hp["steps"] // 2)}
        print()
        print("=" * 72)
        print(f"  DOMAIN: {domain}   epochs={hp['epochs']}  steps={hp['steps']}")
        print("=" * 72)

        t0 = time.time()
        direct = train_fn(False, args.seed, **hp)
        t_direct = time.time() - t0
        print(f"  [direct] {direct}  ({t_direct:.1f}s)")

        t1 = time.time()
        cook = train_fn(True, args.seed, **hp)
        t_cook = time.time() - t1
        print(f"  [cook]   {cook}  ({t_cook:.1f}s)")

        delta = diff_metrics(direct, cook)
        leaves: list[float] = []
        for v in delta.values():
            if isinstance(v, dict):
                leaves.extend(v.values())
            else:
                leaves.append(v)
        max_delta = max(leaves) if leaves else 0.0
        bit_identical = max_delta < 1e-6
        if not bit_identical:
            overall_pass = False
        print(f"  [delta]  {delta}  max={max_delta:.3e}  "
              f"{'PASS bit-identical' if bit_identical else 'NUMERICAL DRIFT'}")

        summary[domain] = {
            "epochs": hp["epochs"], "steps": hp["steps"],
            "direct": direct, "cook": cook, "delta": delta,
            "max_delta": max_delta,
            "bit_identical_within_1e6": bit_identical,
            "wall_direct_s": t_direct, "wall_cook_s": t_cook,
        }

    summary["overall_bit_identical"] = overall_pass
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print("=" * 72)
    print("  OVERALL: " + ("PASS — every domain bit-identical (<1e-6)"
                            if overall_pass
                            else "DRIFT — see per-domain delta"))
    print("=" * 72)
    print(f"summary written: {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
