"""PAPER §17 (S2) — hierarchical (HSAE-style) anchor emergence.

Tests whether running a *second-level* k-means on Tier-G's
first-level anchors produces meaningful super-anchors that align
with human-readable category boundaries:

* **Color** (12 hue ring, first-level k=6, second-level k=2): does
  the binary partition correspond to **warm/cool** (hues 0-5 vs
  6-11)?
* **Number** (100-N grid, first-level k=10, second-level k=2 or 5):
  does the partition correspond to **small/large** or **odd/even
  parity** when D-layer LastDigitHead pre-organises the lower
  anchors into the digit ring?

The hypothesis (S2 in ``docs/2026_LITERATURE_AND_PLANS.md``) is
that PCM's anchor pool — when subjected to a second clustering
step — exposes a **two-level abstraction** matching the multi-level
semantic abstraction observed in human MTL concept neurons (PLOS
Bio 2025; Nat Commun 2025 region-based feature coding) and the
parent-child feature splitting seen in Hierarchical Sparse
Autoencoders (HSAE, arxiv 2602.11881, 2026).

Falsifiability:

* ``warm_cool_NMI`` ≥ 0.5 across seeds → S2 confirmed for color.
* ``parity_NMI`` ≥ 0.4 (number, super-k=2) OR
  ``small_large_NMI`` ≥ 0.4 → S2 confirmed for number.
* Otherwise S2 falsified for that domain.

Usage::

    python -m experiments.sleep_hierarchical_anchors \\
        --domain color --n-seeds 8 \\
        --first-k 6 --second-k 2 \\
        --epochs 30 --steps-per-epoch 200 \\
        --out outputs/hsae_color_6to2

    python -m experiments.sleep_hierarchical_anchors \\
        --domain number --n-seeds 5 \\
        --first-k 10 --second-k 2 \\
        --out outputs/hsae_number_10to2
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from pcm.sleep import PROTO_CID_TEMPLATE, SleepConfig, _kmeans


__all__ = ["main", "second_level_kmeans", "nmi"]


# ─────────────────────────────────────────────────────────────────
# Information-theoretic alignment metric
# ─────────────────────────────────────────────────────────────────


def nmi(labels_a: list[int], labels_b: list[int]) -> float:
    """Normalised mutual information (geometric mean), ε-free.

    Returns ≈ 1.0 when the two label vectors agree up to permutation,
    ≈ 0.0 when independent. Uses base-2 entropies so the answer is
    in [0, 1] for any finite confusion matrix.
    """
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0
    n = len(labels_a)
    ca = Counter(labels_a)
    cb = Counter(labels_b)
    cab = Counter(zip(labels_a, labels_b))

    def _h(counter: Counter) -> float:
        h = 0.0
        for v in counter.values():
            p = v / n
            if p > 0:
                h -= p * math.log2(p)
        return h

    h_a = _h(ca)
    h_b = _h(cb)
    if h_a == 0 or h_b == 0:
        return 0.0
    h_ab = _h(cab)
    mi = h_a + h_b - h_ab
    return mi / math.sqrt(h_a * h_b)


def second_level_kmeans(
    proto_rows: torch.Tensor, second_k: int, seed: int,
) -> tuple[torch.Tensor, list[int]]:
    """Run a second-level k-means over first-level prototype rows.

    Returns ``(super_centroids (k2, D), assignments_per_proto (N,))``.
    Uses the same ``pcm.sleep._kmeans`` as the first-level pass for
    bit-identical methodology.
    """
    cfg = SleepConfig(
        k_clusters=second_k,
        kmeans_iters=64,
        distance="cosine",
        init="kmeans++",
        seed=seed,
    )
    centroids, assignments = _kmeans(proto_rows, second_k, cfg=cfg)
    return centroids, [int(a.item()) for a in assignments]


# ─────────────────────────────────────────────────────────────────
# Domain runners
# ─────────────────────────────────────────────────────────────────


def _color_one_seed(
    seed: int, *, first_k: int, second_k: int,
    epochs: int, steps: int,
    sleep_warmup: int, sleep_every: int,
) -> dict:
    """One seed of the color domain: BCD setup (LMS + green peak +
    ripe head), run sleep with first-level k=first_k, then run a
    second-level k-means on the prototype rows."""
    from experiments.color_concept_study._config import (
        EMBED_DIM, FACET_MIX, N_COLORS,
    )
    from experiments.color_concept_study.graph_builder import (
        make_lms_like_centroids,
    )
    from experiments.color_concept_study.train import train_one
    from experiments.sleep_color_primaries import _green_peak_weights

    centroids = make_lms_like_centroids(N_COLORS, EMBED_DIM, seed)
    weights = _green_peak_weights()
    t0 = time.time()
    r = train_one(
        "single", seed, centroids,
        epochs=epochs, steps_per_epoch=steps,
        mix_sample_weight=weights,
        enable_ripe_head=True,
        sleep_every=sleep_every, sleep_warmup=sleep_warmup,
        sleep_k_clusters=first_k,
        sleep_assignment="hard",
        use_abstract=False,
    )
    bs = r["bundle_state"]

    proto_rows: list[torch.Tensor] = []
    proto_idx_to_facet_id: list[str] = []
    for c in range(first_k):
        proto_id = PROTO_CID_TEMPLATE.format(facet=FACET_MIX, k=c)
        if proto_id not in bs:
            continue
        key = f"params.{FACET_MIX}"
        if key not in bs[proto_id]:
            continue
        proto_rows.append(bs[proto_id][key])
        proto_idx_to_facet_id.append(proto_id)

    if len(proto_rows) < second_k:
        return {
            "n_proto": len(proto_rows),
            "n_super": 0,
            "wall_s": time.time() - t0,
            "warning": "first-level k produced fewer prototypes than second-k",
        }

    proto_stack = torch.stack(proto_rows)
    super_cents, super_assigns = second_level_kmeans(
        proto_stack, second_k, seed,
    )

    # Map each hue → first-level proto → second-level super.
    hue_rows = torch.stack([
        bs[f"concept:color:{i}"][f"params.{FACET_MIX}"]
        for i in range(N_COLORS)
    ])
    pn = F.normalize(proto_stack.flatten(start_dim=1), dim=-1)
    hn = F.normalize(hue_rows.flatten(start_dim=1), dim=-1)
    hue_to_proto = (hn @ pn.t()).argmax(dim=1).tolist()
    hue_to_super = [super_assigns[p] for p in hue_to_proto]

    # Reference labelings: warm/cool (hues 0..5 warm, 6..11 cool),
    # parity (even/odd), red/notred ({0,1,11} red wedge).
    warm_cool = [0 if h <= 5 else 1 for h in range(N_COLORS)]
    parity = [h % 2 for h in range(N_COLORS)]
    red_wedge = [1 if h in {0, 1, 11} else 0 for h in range(N_COLORS)]

    return {
        "n_proto": len(proto_rows),
        "n_super": second_k,
        "hue_to_proto": hue_to_proto,
        "hue_to_super": hue_to_super,
        "nmi_warm_cool": nmi(hue_to_super, warm_cool),
        "nmi_parity": nmi(hue_to_super, parity),
        "nmi_red_wedge": nmi(hue_to_super, red_wedge),
        "wall_s": time.time() - t0,
    }


def _number_one_seed(
    seed: int, *, first_k: int, second_k: int,
    epochs: int, steps: int,
    sleep_warmup: int, sleep_every: int,
    n_total: int,
) -> dict:
    """One seed of the number domain (BCD: decimal cones + Zipf
    weights + LastDigitHead). Re-uses :func:`train_quad` directly,
    then runs a second-level k-means on the prototype rows held
    inside ``cg.bundle_pool["arithmetic_bias"]``."""
    import random

    from experiments.number_decimal_priors import zipf_weights
    from experiments.quad_study import enumerate_triples, train_quad

    weights = zipf_weights(n_total)
    facet = "arithmetic_bias"

    # Build train triples on the smaller "trainable" range (mimics
    # §7.4 setup: train QuadArithHead on 1..N_train, register the
    # full inventory 1..n_total via LastDigitHead).
    N_train = min(n_total, 30)
    all_triples = enumerate_triples(N_train, 1.0)
    rng = random.Random(seed)
    train_triples = []
    for op, trips in all_triples.items():
        trips = list(trips)
        rng.shuffle(trips)
        n_test = max(1, int(len(trips) * 0.15))
        for (a, b, c) in trips[n_test:]:
            train_triples.append((a, b, op, c))

    t0 = time.time()
    r = train_quad(
        N_train, 1.0, seed,
        train_triples=train_triples,
        epochs=epochs, steps_per_epoch=steps,
        centroid_mode="decimal_cones",
        digit_sample_weight=weights,
        enable_last_digit_head=True,
        n_total=n_total,
        sleep_every=sleep_every, sleep_warmup=sleep_warmup,
        sleep_k_clusters=first_k,
        sleep_assignment="hard",
        use_abstract=False,
    )
    cg = r["cg"]
    pool = cg.bundle_pool[facet]

    proto_rows: list[torch.Tensor] = []
    for c in range(first_k):
        proto_id = PROTO_CID_TEMPLATE.format(facet=facet, k=c)
        if proto_id in cg.cid_to_slot:
            slot = cg.cid_to_slot[proto_id]
            proto_rows.append(pool.data[slot].detach().cpu().clone())

    if len(proto_rows) < second_k:
        return {
            "n_proto": len(proto_rows),
            "n_super": 0,
            "wall_s": time.time() - t0,
            "warning": "first-level k produced fewer prototypes than second-k",
        }

    proto_stack = torch.stack(proto_rows)
    super_cents, super_assigns = second_level_kmeans(
        proto_stack, second_k, seed,
    )

    num_rows: list[torch.Tensor] = []
    for i in range(n_total):
        cid = f"concept:num:{i}"
        if cid in cg.cid_to_slot:
            slot = cg.cid_to_slot[cid]
            num_rows.append(pool.data[slot].detach().cpu().clone())
        else:
            num_rows.append(torch.zeros_like(proto_stack[0]))

    num_stack = torch.stack(num_rows)
    pn = F.normalize(proto_stack.flatten(start_dim=1), dim=-1)
    nn_rows = F.normalize(num_stack.flatten(start_dim=1), dim=-1)
    num_to_proto = (nn_rows @ pn.t()).argmax(dim=1).tolist()
    num_to_super = [super_assigns[p] for p in num_to_proto]

    parity = [n % 2 for n in range(n_total)]
    half = n_total // 2
    small_large = [0 if n < half else 1 for n in range(n_total)]
    last_digit = [n % 10 for n in range(n_total)]
    decade = [n // 10 for n in range(n_total)]

    return {
        "n_proto": len(proto_rows),
        "n_super": second_k,
        "num_to_proto": num_to_proto,
        "num_to_super": num_to_super,
        "nmi_parity": nmi(num_to_super, parity),
        "nmi_small_large": nmi(num_to_super, small_large),
        "nmi_last_digit": nmi(num_to_super, last_digit),
        "nmi_decade": nmi(num_to_super, decade),
        "wall_s": time.time() - t0,
    }


# ─────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["color", "number"], default="color")
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=88000)
    ap.add_argument("--first-k", type=int, default=6)
    ap.add_argument("--second-k", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--sleep-warmup", type=int, default=15)
    ap.add_argument("--sleep-every", type=int, default=5)
    ap.add_argument("--n-total", type=int, default=50,
                    help="number-domain only: |concept:num| inventory")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/hsae_anchors"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(
        f"  PAPER §17 (S2) hierarchical anchors: domain={args.domain}, "
        f"first_k={args.first_k} → second_k={args.second_k}, "
        f"n_seeds={args.n_seeds}"
    )
    print("=" * 76)

    rows: list[dict] = []
    for si in range(args.n_seeds):
        seed = args.seed_base + si
        print(f"\n── seed {seed} ──")
        if args.domain == "color":
            r = _color_one_seed(
                seed,
                first_k=args.first_k, second_k=args.second_k,
                epochs=args.epochs, steps=args.steps_per_epoch,
                sleep_warmup=args.sleep_warmup,
                sleep_every=args.sleep_every,
            )
        else:
            r = _number_one_seed(
                seed,
                first_k=args.first_k, second_k=args.second_k,
                epochs=args.epochs, steps=args.steps_per_epoch,
                sleep_warmup=args.sleep_warmup,
                sleep_every=args.sleep_every,
                n_total=args.n_total,
            )
        r["seed"] = seed
        rows.append(r)
        nmi_keys = [k for k in r if k.startswith("nmi_")]
        line = " ".join(f"{k}={r[k]:.3f}" for k in nmi_keys
                        if isinstance(r.get(k), (int, float)))
        print(f"  {line}")

    # Aggregate NMI across seeds.
    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "per_seed": rows,
        "by_metric": {},
    }
    nmi_metrics = sorted({
        k for r in rows for k in r if k.startswith("nmi_")
    })
    for m in nmi_metrics:
        vals = [r[m] for r in rows if isinstance(r.get(m), (int, float))]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
        summary["by_metric"][m] = {
            "mean": mean, "std": math.sqrt(var),
            "min": min(vals), "max": max(vals), "n": len(vals),
        }

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "═" * 76)
    print(f"  hierarchical NMI summary ({args.domain}, n={args.n_seeds}):")
    for m in nmi_metrics:
        s = summary["by_metric"].get(m)
        if s:
            tag = ""
            if s["mean"] >= 0.5:
                tag = " ← STRONG"
            elif s["mean"] >= 0.3:
                tag = " ← MODERATE"
            print(
                f"  {m:<25s} {s['mean']:+.3f}±{s['std']:.3f}"
                f" (min {s['min']:.2f}, max {s['max']:.2f}){tag}"
            )
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
