"""PAPER §6.8 — recovering perceptual primitives from biological priors,
ecological pressure, and task asymmetry.

Five-condition ablation on the 12-hue color domain. After training,
runs a sleep pass with ``k_clusters=3`` and inspects which hues the
three anchors land on, exactly as in
``experiments.sleep_inspect_color_anchors`` but now sweeping
*ablation conditions* rather than ``k`` values.

Conditions correspond to the three causal layers of biological
trichromacy (Stockman & Sharpe 2000; Jacobs 2009; Conway et al. 2007;
Berlin & Kay 1969):

* **A** — *baseline*: random orthogonal centroids, uniform mixing
  sampling, no auxiliary head. PCM has nothing to break the cyclic
  symmetry; matches §6.7's negative result.
* **B** — *biological prior*: ``make_lms_like_centroids`` injects three
  cone-like axes peaked at hue 0/4/8 in the supervised geometry. Tests
  whether sleep faithfully recovers a centroid-level RGB-aligned
  topology when one is supplied.
* **C** — *ecological statistics*: a non-uniform per-hue sampling
  weight (default: green peak at hue 4, simulating a "lots of green
  leaves" environment). Tests whether sleep biases anchor placement
  toward statistically dense regions.
* **D** — *task asymmetry*: an auxiliary ``RipeFruitHead`` that
  classifies hues in {0, 1, 11} as ripe. Tests whether a single
  binary task that picks out a 3-hue red wedge lets sleep break
  cyclic symmetry on the *task* side rather than the supervision side.
* **B+C+D** — all three layers stacked.

For each condition, runs ``--n-seeds`` trainings, registers sleep
prototypes with ``k=3``, and reports the same diagnostics used in
``sleep_inspect_color_anchors`` (anchor nearest hues, equidistance,
RGB / CMY / RYB / CMYK / WarmCool6 hit rates).

Usage::

    python -m experiments.sleep_color_primaries \
        --n-seeds 8 --epochs 30 --steps-per-epoch 200 \
        --out outputs/color_primaries_5cond
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from experiments.color_concept_study._config import (
    DEVICE, EMBED_DIM, EPOCHS, FACET_MIX, N_COLORS, STEPS_PER_EPOCH,
)
from experiments.color_concept_study.graph_builder import (
    make_lms_like_centroids,
    make_random_orthogonal_centroids,
)
from experiments.color_concept_study.train import train_one
from experiments.sleep_inspect_color_anchors import (
    PERCEPTUAL_PRIORS,
    _equidistant_score,
    _prior_match,
    _rotation_class,
    _nearest_hue_for_anchor,
)
from pcm.sleep import PROTO_CID_TEMPLATE
import torch
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────
# Ablation condition definitions
# ────────────────────────────────────────────────────────────────────────


def _green_peak_weights() -> list[float]:
    """Non-uniform sampling: peak at hue 4 ("green"), exponential falloff
    around the cycle. ratio of max:min ≈ 4:1, so the bias is real but
    not pathological."""
    import math
    out = []
    for i in range(N_COLORS):
        d = min(abs(i - 4), N_COLORS - abs(i - 4))
        out.append(math.exp(-d / 3.0))
    return out


def _red_peak_weights() -> list[float]:
    """Alternative: peak at hue 0 (red region)."""
    import math
    out = []
    for i in range(N_COLORS):
        d = min(abs(i - 0), N_COLORS - abs(i - 0))
        out.append(math.exp(-d / 3.0))
    return out


CONDITIONS = {
    "A_baseline":    {"centroid": "random",  "sample": None,         "ripe": False},
    "B_lms":         {"centroid": "lms",     "sample": None,         "ripe": False},
    "C_greenpeak":   {"centroid": "random",  "sample": "green_peak", "ripe": False},
    "D_ripehead":    {"centroid": "random",  "sample": None,         "ripe": True},
    "BCD_combined":  {"centroid": "lms",     "sample": "green_peak", "ripe": True},
}


def _build_centroids(mode: str, seed: int) -> torch.Tensor:
    if mode == "random":
        return make_random_orthogonal_centroids(N_COLORS, EMBED_DIM, seed)
    if mode == "lms":
        return make_lms_like_centroids(N_COLORS, EMBED_DIM, seed)
    raise ValueError(f"unknown centroid mode {mode!r}")


def _build_weights(mode: str | None) -> list[float] | None:
    if mode is None:
        return None
    if mode == "green_peak":
        return _green_peak_weights()
    if mode == "red_peak":
        return _red_peak_weights()
    raise ValueError(f"unknown sample mode {mode!r}")


# ────────────────────────────────────────────────────────────────────────
# Diagnostic per seed
# ────────────────────────────────────────────────────────────────────────


def _run_one_seed(seed: int, cond_name: str, cond: dict, *,
                  k: int, epochs: int, steps: int,
                  sleep_warmup: int, sleep_every: int) -> dict:
    centroids = _build_centroids(cond["centroid"], seed)
    weights = _build_weights(cond["sample"])
    t0 = time.time()
    r = train_one(
        "single", seed, centroids,
        epochs=epochs, steps_per_epoch=steps,
        mix_sample_weight=weights,
        enable_ripe_head=bool(cond["ripe"]),
        sleep_every=sleep_every, sleep_warmup=sleep_warmup,
        sleep_k_clusters=k,
        sleep_assignment="hard",
        use_abstract=False,
    )
    bs = r["bundle_state"]
    proto_rows: list[torch.Tensor] = []
    for c in range(k):
        proto_id = PROTO_CID_TEMPLATE.format(facet=FACET_MIX, k=c)
        if proto_id not in bs:
            continue
        key = f"params.{FACET_MIX}"
        if key not in bs[proto_id]:
            continue
        proto_rows.append(bs[proto_id][key])

    hue_rows = torch.stack([
        bs[f"concept:color:{i}"][f"params.{FACET_MIX}"] for i in range(N_COLORS)
    ])
    nearest: list[int] = []
    cluster_sizes: Counter[int] = Counter()
    if proto_rows:
        anchor_stack = torch.stack(proto_rows)
        for a in range(len(proto_rows)):
            nh, _ = _nearest_hue_for_anchor(anchor_stack[a], hue_rows)
            nearest.append(nh)

        an = F.normalize(anchor_stack.flatten(start_dim=1), dim=-1)
        hn = F.normalize(hue_rows.flatten(start_dim=1), dim=-1)
        sims = hn @ an.t()
        for hue_idx in range(N_COLORS):
            cluster_sizes[int(sims[hue_idx].argmax().item())] += 1

    eq = _equidistant_score(nearest)
    rot = _rotation_class(nearest) if eq["is_equidistant"] else None
    priors = _prior_match(nearest)

    return {
        "seed": seed,
        "condition": cond_name,
        "centroid_mode": cond["centroid"],
        "sample_mode": cond["sample"],
        "ripe_head": cond["ripe"],
        "anchor_nearest_hues": nearest,
        "spacings": eq["spacings"],
        "is_equidistant": eq["is_equidistant"],
        "max_dev": eq["max_dev"],
        "rotation_class": rot,
        "cluster_sizes": [cluster_sizes[c] for c in range(len(nearest))],
        "perceptual_prior_match": priors,
        "wall_s": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=82000)
    ap.add_argument("--k", type=int, default=3,
                    help="cluster count for the sleep pass (default 3 = "
                         "the 'three primaries' question)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    ap.add_argument("--sleep-warmup", type=int, default=15)
    ap.add_argument("--sleep-every", type=int, default=5)
    ap.add_argument("--conditions", nargs="+",
                    default=list(CONDITIONS.keys()))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/color_primaries"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  PAPER §6.8 color primaries ablation: n_seeds={args.n_seeds}, "
          f"k={args.k}, epochs={args.epochs}, steps={args.steps_per_epoch}")
    print(f"  device={DEVICE}; conditions={args.conditions}")
    print("=" * 72)

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
        rows: list[dict] = []
        for si in range(args.n_seeds):
            seed = args.seed_base + si
            r = _run_one_seed(
                seed, cond_name, cond,
                k=args.k, epochs=args.epochs, steps=args.steps_per_epoch,
                sleep_warmup=args.sleep_warmup, sleep_every=args.sleep_every,
            )
            rows.append(r)
            eq_str = "EQUI" if r["is_equidistant"] else f"dev={r['max_dev']}"
            priors_hit = ",".join(
                name for name, m in r["perceptual_prior_match"].items() if m
            ) or "(none)"
            print(
                f"  [seed={seed}] hues={r['anchor_nearest_hues']}  "
                f"spacings={r['spacings']}  {eq_str}  "
                f"sizes={r['cluster_sizes']}  prior={priors_hit}  "
                f"({r['wall_s']:.1f}s)"
            )

        n_equi = sum(1 for r in rows if r["is_equidistant"])
        rot_counter: Counter[int] = Counter()
        for r in rows:
            if r["rotation_class"] is not None:
                rot_counter[r["rotation_class"]] += 1
        prior_counter: Counter[str] = Counter()
        for r in rows:
            for name, m in r["perceptual_prior_match"].items():
                if m:
                    prior_counter[name] += 1
        red_anchor_rate = sum(
            1 for r in rows
            if any(h in {0, 1, 11} for h in r["anchor_nearest_hues"])
        ) / max(args.n_seeds, 1)

        cond_summary = {
            "config": cond,
            "per_seed": rows,
            "n_equidistant": n_equi,
            "fraction_equidistant": n_equi / max(args.n_seeds, 1),
            "rotation_class_histogram": dict(rot_counter),
            "perceptual_prior_match_counts": dict(prior_counter),
            "fraction_anchor_in_red_wedge": red_anchor_rate,
        }
        summary["by_condition"][cond_name] = cond_summary
        print(
            f"  → equidistant: {n_equi}/{args.n_seeds}, "
            f"rotation hist: {dict(rot_counter)}, "
            f"red-wedge anchor: {red_anchor_rate:.2f}"
        )
        for name in PERCEPTUAL_PRIORS.keys():
            cnt = prior_counter.get(name, 0)
            if cnt:
                print(f"     {name}: {cnt}/{args.n_seeds} match")

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    # Cross-condition summary table
    print("\n" + "═" * 72)
    print("  cross-condition summary (RGB hits, red-wedge fraction):")
    print(f"  {'condition':<16s} {'equi':>5s} {'RGB':>5s} {'CMY':>5s} "
          f"{'RYB':>5s} {'CMYK':>5s} {'WC6':>5s} {'red_wedge':>10s}")
    for cond_name in args.conditions:
        if cond_name not in summary["by_condition"]:
            continue
        cs = summary["by_condition"][cond_name]
        priors = cs["perceptual_prior_match_counts"]
        print(
            f"  {cond_name:<16s} {cs['n_equidistant']:>3d}/{args.n_seeds:<2d} "
            f"{priors.get('RGB', 0):>3d}/{args.n_seeds:<2d} "
            f"{priors.get('CMY', 0):>3d}/{args.n_seeds:<2d} "
            f"{priors.get('RYB', 0):>3d}/{args.n_seeds:<2d} "
            f"{priors.get('CMYK_aligned', 0):>3d}/{args.n_seeds:<2d} "
            f"{priors.get('WarmCool6', 0):>3d}/{args.n_seeds:<2d} "
            f"{cs['fraction_anchor_in_red_wedge']:>10.2f}"
        )
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
