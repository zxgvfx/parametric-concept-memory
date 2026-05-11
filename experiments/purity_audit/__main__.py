"""CLI entry: ``python -m experiments.purity_audit --encoder-ckpt ...``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.robustness_study import DEVICE, EPOCHS, STEPS_PER_EPOCH

from .assays import (
    assay_a1_random_centroids,
    assay_a2_shuffle_inverse,
    assay_a3_init_scale,
    assay_a4_random_id,
)
from .reporting import render_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-ckpt", type=Path, required=True)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("outputs/purity_audit"))
    ap.add_argument("--skip", nargs="*", default=[],
                    help="跳过 (a1/a1b/a2/a3/a4)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    enc_ckpt = torch.load(args.encoder_ckpt, map_location="cpu", weights_only=False)

    summary: dict = {
        "n_seeds": args.n_seeds,
        "device": DEVICE,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "encoder_ckpt": str(args.encoder_ckpt),
    }

    if "a1" not in args.skip:
        print("=" * 60); print("A1: Random Orthogonal Centroids"); print("=" * 60)
        summary["A1_random_orthogonal"] = assay_a1_random_centroids(
            enc_ckpt, args.n_seeds, "orthogonal"
        )

    if "a1b" not in args.skip:
        print("=" * 60); print("A1b: Random Gaussian Centroids"); print("=" * 60)
        summary["A1_random_gaussian"] = assay_a1_random_centroids(
            enc_ckpt, args.n_seeds, "gaussian"
        )

    if "a2" not in args.skip:
        print("=" * 60); print("A2: Shuffle-Inverse"); print("=" * 60)
        summary["A2_shuffle_inverse"] = assay_a2_shuffle_inverse(enc_ckpt, args.n_seeds)

    if "a3" not in args.skip:
        print("=" * 60); print("A3: Init-Scale"); print("=" * 60)
        summary["A3_init_scale"] = assay_a3_init_scale(enc_ckpt, args.n_seeds)

    if "a4" not in args.skip:
        print("=" * 60); print("A4: Random Concept-ID"); print("=" * 60)
        summary["A4_random_id"] = assay_a4_random_id(enc_ckpt, args.n_seeds)

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    (args.out / "report.md").write_text(render_report(summary))
    print("\nWrote:")
    print(f"  {args.out/'summary.json'}")
    print(f"  {args.out/'report.md'}")


if __name__ == "__main__":
    main()
