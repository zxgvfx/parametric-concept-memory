"""PAPER §16 (S5) — cone-opponent vs LMS biological prior.

Five-condition ablation that asks whether replacing PCM's §6.8
LMS-like trichromatic centroids with **cone-opponent (R-G, Y-B,
Lum)** centroids causes the sleep-pass anchors to migrate from a
3-anchor RGB layout (additive primaries) to a 4-anchor Hering
layout (red / yellow / green / blue), more faithfully reflecting:

* Solomon & Lennie 2007 ``Nat Rev Neurosci`` — the cone-opponent
  pathway in primate retina.
* Yang et al. 2016 ``PNAS`` (NIRS infants) — 5–7-month-old
  prelinguistic boundaries cluster around four cardinal regions.
* Skelton et al. 2017 ``PNAS`` — categorical perception predicts
  later linguistic boundaries; the four-region split is robust
  across cultures (Berlin & Kay 1969 stage I–III).

Conditions:

* ``A_baseline``     — random orthogonal centroids; cyclic.
* ``B_lms``          — §6.8's :func:`make_lms_like_centroids` with
                       k=3 sleep clusters. Replicates §6.8's
                       RGB anchor result.
* ``B_opponent``     — :func:`make_cone_opponent_centroids` with
                       k=4 sleep clusters. Asks if opponent
                       carves the ring into four cardinal anchors.
* ``BCD_lms``        — §6.8's full B+C+D stack with LMS, k=3.
* ``BCD_opponent``   — full stack with cone-opponent, k=4.

Each condition runs ``--n-seeds`` trainings; we report:

* ``Hering4_match`` — fraction of seeds whose anchors map to a
  rotation of {R=0, Y=2, G=4, B=8}. Predicted ↑ for ``opponent``.
* ``RGB_match``     — fraction whose anchors map to {R=0, G=4, B=8}.
                      Predicted ↑ for ``lms``.
* ``red_wedge``     — fraction whose anchors include hue 0/1/11.

Usage::

    python -m experiments.sleep_color_cone_opponent \\
        --n-seeds 8 --epochs 30 --steps-per-epoch 200 \\
        --out outputs/color_cone_opponent
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.color_concept_study._config import (
    DEVICE, EMBED_DIM, EPOCHS, FACET_MIX, N_COLORS, STEPS_PER_EPOCH,
)
from experiments.color_concept_study.graph_builder import (
    make_cone_opponent_centroids, make_lms_like_centroids,
    make_random_orthogonal_centroids,
)
from experiments.color_concept_study.train import train_one
from experiments.sleep_color_primaries import (
    _green_peak_weights, _red_peak_weights,
)
from experiments.sleep_inspect_color_anchors import (
    PERCEPTUAL_PRIORS, _equidistant_score, _nearest_hue_for_anchor,
    _prior_match, _rotation_class,
)
from pcm.diagnostics import (
    AblationCondition, AblationLayers, run_causal_ablation,
)
from pcm.sleep import PROTO_CID_TEMPLATE


# ────────────────────────────────────────────────────────────────────────
# Conditions
# ────────────────────────────────────────────────────────────────────────


CONDITIONS = (
    AblationCondition("A_baseline", AblationLayers()),
    AblationCondition("B_lms", AblationLayers(B="lms")),
    AblationCondition("B_opponent", AblationLayers(B="opponent")),
    AblationCondition(
        "BCD_lms",
        AblationLayers(B="lms", C="green_peak", D=True),
    ),
    AblationCondition(
        "BCD_opponent",
        AblationLayers(B="opponent", C="green_peak", D=True),
    ),
)


# Per-condition k (LMS predicts 3 anchors, opponent predicts 4).
_K_BY_CONDITION = {
    "A_baseline": 3,
    "B_lms": 3,
    "B_opponent": 4,
    "BCD_lms": 3,
    "BCD_opponent": 4,
}


def _build_centroids(mode: str | None, seed: int) -> torch.Tensor:
    if mode is None or mode == "random":
        return make_random_orthogonal_centroids(N_COLORS, EMBED_DIM, seed)
    if mode == "lms":
        return make_lms_like_centroids(N_COLORS, EMBED_DIM, seed)
    if mode == "opponent":
        return make_cone_opponent_centroids(N_COLORS, EMBED_DIM, seed)
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
# Per-seed runner
# ────────────────────────────────────────────────────────────────────────


def _condition_name_from_layers(layers: AblationLayers) -> str:
    """Recover condition name from layer flags so we can pick k."""
    for cond in CONDITIONS:
        if cond.layers == layers:
            return cond.name
    return "A_baseline"


def _run_one_seed(
    seed: int, layers: AblationLayers, *,
    epochs: int, steps: int,
    sleep_warmup: int, sleep_every: int,
) -> dict:
    cond_name = _condition_name_from_layers(layers)
    k = _K_BY_CONDITION[cond_name]
    centroids = _build_centroids(layers.B, seed)
    weights = _build_weights(layers.C)
    t0 = time.time()
    r = train_one(
        "single", seed, centroids,
        epochs=epochs, steps_per_epoch=steps,
        mix_sample_weight=weights,
        enable_ripe_head=bool(layers.is_active("D")),
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
        "condition_name": cond_name,
        "k_clusters": k,
        "centroid_mode": layers.B,
        "sample_mode": layers.C,
        "ripe_head": bool(layers.is_active("D")),
        "anchor_nearest_hues": nearest,
        "spacings": eq["spacings"],
        "is_equidistant": eq["is_equidistant"],
        "max_dev": eq["max_dev"],
        "rotation_class": rot,
        "cluster_sizes": [cluster_sizes[c] for c in range(len(nearest))],
        "perceptual_prior_match": priors,
        "wall_s": time.time() - t0,
    }


# ────────────────────────────────────────────────────────────────────────
# Entrypoint
# ────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=86000)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    ap.add_argument("--sleep-warmup", type=int, default=15)
    ap.add_argument("--sleep-every", type=int, default=5)
    ap.add_argument(
        "--out", type=Path,
        default=Path("outputs/color_cone_opponent"),
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(
        f"  PAPER §16 (S5) cone-opponent vs LMS: n_seeds={args.n_seeds}, "
        f"epochs={args.epochs}, steps={args.steps_per_epoch}"
    )
    print(f"  device={DEVICE}; protocol=pcm.diagnostics.run_causal_ablation")
    print("=" * 76)

    summary = run_causal_ablation(
        _run_one_seed,
        n_seeds=args.n_seeds,
        seed_base=args.seed_base,
        conditions=CONDITIONS,
        primary_metrics=("max_dev", "wall_s"),
        epochs=args.epochs, steps=args.steps_per_epoch,
        sleep_warmup=args.sleep_warmup, sleep_every=args.sleep_every,
    )
    summary["config"] |= vars(args) | {"out": str(args.out)}

    for cond in CONDITIONS:
        cs = summary["by_condition"][cond.name]
        rows = cs["per_seed"]
        n_equi = sum(1 for r in rows if r.get("is_equidistant"))
        rot_counter: Counter[int] = Counter()
        prior_counter: Counter[str] = Counter()
        for r in rows:
            if r.get("rotation_class") is not None:
                rot_counter[r["rotation_class"]] += 1
            for name, m in (r.get("perceptual_prior_match") or {}).items():
                if m:
                    prior_counter[name] += 1
        red_anchor_rate = sum(
            1 for r in rows
            if any(h in {0, 1, 11} for h in (r.get("anchor_nearest_hues") or []))
        ) / max(args.n_seeds, 1)
        cs["n_equidistant"] = n_equi
        cs["fraction_equidistant"] = n_equi / max(args.n_seeds, 1)
        cs["rotation_class_histogram"] = dict(rot_counter)
        cs["perceptual_prior_match_counts"] = dict(prior_counter)
        cs["fraction_anchor_in_red_wedge"] = red_anchor_rate
        print(
            f"  [{cond.name}] equidistant: {n_equi}/{args.n_seeds}, "
            f"red-wedge: {red_anchor_rate:.2f}, "
            f"prior hits: {dict(prior_counter)}"
        )

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "═" * 76)
    print("  cross-condition summary (k anchors, prior matches):")
    print(
        f"  {'condition':<16s} {'k':>3s} {'equi':>6s} {'RGB':>5s} "
        f"{'Hering4':>8s} {'CMYK':>6s} {'red_wedge':>10s}"
    )
    for cond in CONDITIONS:
        cs = summary["by_condition"][cond.name]
        priors = cs["perceptual_prior_match_counts"]
        k = _K_BY_CONDITION[cond.name]
        print(
            f"  {cond.name:<16s} {k:>3d} "
            f"{cs['n_equidistant']:>3d}/{args.n_seeds:<2d} "
            f"{priors.get('RGB', 0):>3d}/{args.n_seeds:<2d} "
            f"{priors.get('Hering4', 0):>5d}/{args.n_seeds:<2d} "
            f"{priors.get('CMYK_aligned', 0):>3d}/{args.n_seeds:<2d} "
            f"{cs['fraction_anchor_in_red_wedge']:>10.2f}"
        )
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
