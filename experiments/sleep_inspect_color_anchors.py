"""Sleep anchor inspection on the 12-hue circular color domain.

PAPER §6.7 falsifiability test: when we run a sleep pass with
``k_clusters=k`` on the 12-hue color study, where do the anchor
prototypes land relative to the original hue ring?

Two competing hypotheses:

* **H_perceptual** — sleep abstraction recovers a *perceptual prior*
  (e.g. RGB primaries at hue {0,4,8} for k=3, or CMYK at {0,3,6,9}
  for k=4). This would mean PCM has snuck in a colour-vision-like
  bias somewhere.
* **H_taskSym** — sleep abstraction is *purely cyclic-equivariant*:
  anchors land on k equidistant hues (360° / k apart), but the
  starting offset is determined entirely by the seed. Cross-seed
  the offset distribution should be roughly uniform over the k
  rotational classes.

The script runs ``train_one("single", seed, ...)`` with sleep enabled,
inspects the prototype slot rows, finds the nearest hue index for
each anchor, and reports:

* per-seed: list of nearest-hue indices for anchors, the modular
  spacing between them, and the cluster sizes;
* aggregate: histogram of starting offsets mod k; whether the
  spacing is exactly 360°/k (pure rotational symmetry); a "hit
  rate" against several common perceptual-prior hue sets (RGB,
  CMY, RYB, CMYK).

Usage::

    python -m experiments.sleep_inspect_color_anchors \
        --k 3 --n-seeds 8 --out outputs/anchor_color_k3
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
    make_random_orthogonal_centroids,
)
from experiments.color_concept_study.train import train_one
from pcm.sleep import (
    PROTO_CID_TEMPLATE,
    iter_abstract_relations,
)


# Common perceptual-prior hue sets on a 12-hue ring (hue 0 = "red"
# by convention; deltas wrap mod 12). These are the canonical
# placements that *would* appear if PCM had a color-vision-like
# inductive bias.
PERCEPTUAL_PRIORS = {
    # Additive primaries: red / green / blue → hue 0, 4, 8
    "RGB": frozenset({0, 4, 8}),
    # Subtractive primaries: cyan / magenta / yellow → hue 6, 10, 2
    "CMY": frozenset({2, 6, 10}),
    # Painter primaries: red / yellow / blue → hue 0, 2, 8
    "RYB": frozenset({0, 2, 8}),
    # CMYK 4-way (drop K): cyan / magenta / yellow / black-as-gray
    # The four-way printer split is {C, M, Y, ...} which on a
    # 12-hue ring at 90° spacing is {0, 3, 6, 9} (any rotation thereof)
    "CMYK_aligned": frozenset({0, 3, 6, 9}),
    # Hot / cold split: warm hues (red/orange/yellow) vs cool
    # hues (cyan/blue). On a 12-hue ring at 60° spacing, this is
    # {0, 2, 4, 6, 8, 10}.
    "WarmCool6": frozenset({0, 2, 4, 6, 8, 10}),
}


def _proto_slots_for_facet(cg, facet: str) -> list[tuple[int, int]]:
    """Return ``[(k, slot_idx)]`` for prototypes registered on ``facet``."""
    out: list[tuple[int, int]] = []
    # Walk relations to discover which prototypes were emitted on this facet.
    seen_anchor_ids: set[str] = set()
    for rel_node in iter_abstract_relations(cg):
        consts = (rel_node.metadata or {}).get("constants", {}) or {}
        if consts.get("anchor_facet") != facet:
            continue
        anchor_id = consts.get("anchor_id")
        if not anchor_id or anchor_id in seen_anchor_ids:
            continue
        seen_anchor_ids.add(anchor_id)
        # anchor_id format: concept:cluster:<facet>:<k>
        prefix = PROTO_CID_TEMPLATE.format(facet=facet, k="").rstrip(":")
        if anchor_id.startswith(prefix + ":"):
            try:
                k_idx = int(anchor_id.split(":")[-1])
                slot_idx = cg.cid_to_slot[anchor_id]
                out.append((k_idx, slot_idx))
            except (ValueError, KeyError):
                continue
    return sorted(out)


def _nearest_hue_for_anchor(
    anchor_row: torch.Tensor, hue_rows: torch.Tensor
) -> tuple[int, list[float]]:
    """Return ``(nearest_hue_index, full_cosines)``.

    ``hue_rows`` is ``(N_COLORS, D)``; cosines are computed over
    L2-normalised rows.
    """
    ar = F.normalize(anchor_row.flatten().unsqueeze(0), dim=-1)
    hr = F.normalize(hue_rows.flatten(start_dim=1), dim=-1)
    cos = (ar @ hr.t()).squeeze(0)
    nearest = int(cos.argmax().item())
    return nearest, [float(x) for x in cos.tolist()]


def _modular_spacings(hues: list[int], n: int = N_COLORS) -> list[int]:
    """Sorted modular spacings between consecutive hues on the ring."""
    if not hues:
        return []
    s = sorted(hues)
    spacings: list[int] = []
    for i in range(len(s)):
        diff = (s[(i + 1) % len(s)] - s[i]) % n
        spacings.append(diff if diff > 0 else n)
    return sorted(spacings)


def _equidistant_score(hues: list[int], n: int = N_COLORS) -> dict:
    """How close are ``hues`` to perfectly equidistant on the ring?"""
    k = len(hues)
    if k < 2:
        return {"is_equidistant": False, "expected_step": None,
                "max_dev": None, "spacings": []}
    spacings = _modular_spacings(hues, n)
    expected = n / k
    max_dev = max(abs(s - expected) for s in spacings)
    return {
        "is_equidistant": all(s == int(expected) for s in spacings)
        if expected.is_integer() else False,
        "expected_step": expected,
        "max_dev": max_dev,
        "spacings": spacings,
    }


def _rotation_class(hues: list[int], n: int = N_COLORS) -> int:
    """Canonical rotation class: smallest hue mod (n / k)."""
    if not hues:
        return 0
    k = len(hues)
    period = n // k if (n % k == 0) else n  # only well-defined when n % k == 0
    s = sorted(hues)
    return min(h % period for h in s)


def _prior_match(hues: list[int]) -> dict[str, bool]:
    """For each canonical perceptual prior, does ``hues`` match
    *up to circular rotation*?"""
    h_set = frozenset(hues)
    out: dict[str, bool] = {}
    for name, prior in PERCEPTUAL_PRIORS.items():
        if len(prior) != len(h_set):
            out[name] = False
            continue
        match = False
        for r in range(N_COLORS):
            rotated = frozenset((p + r) % N_COLORS for p in prior)
            if rotated == h_set:
                match = True
                break
        out[name] = match
    return out


def _run_one_seed(seed: int, k: int, sleep_warmup: int,
                  sleep_every: int, epochs: int, steps: int) -> dict:
    centroids = make_random_orthogonal_centroids(N_COLORS, EMBED_DIM, seed)
    t0 = time.time()
    r = train_one(
        "single", seed, centroids,
        epochs=epochs, steps_per_epoch=steps,
        sleep_every=sleep_every, sleep_warmup=sleep_warmup,
        sleep_k_clusters=k,
        sleep_assignment="hard",  # cleanest geometry; doesn't affect anchor positions
        use_abstract=False,  # we only need anchors registered, not consumed
    )
    bs = r["bundle_state"]
    proto_rows = []
    proto_slots = []
    # anchor identification: by reconstructing PROTO_CID_TEMPLATE
    for c in range(k):
        proto_id = PROTO_CID_TEMPLATE.format(facet=FACET_MIX, k=c)
        if proto_id not in bs:
            continue
        key = f"params.{FACET_MIX}"
        if key not in bs[proto_id]:
            continue
        proto_rows.append(bs[proto_id][key])
        proto_slots.append(c)

    # 12 hue rows (base concepts)
    hue_rows = torch.stack([
        bs[f"concept:color:{i}"][f"params.{FACET_MIX}"] for i in range(N_COLORS)
    ])

    nearest = []
    cluster_sizes = Counter()
    # For each hue, which anchor is closest (this is the cluster member)
    hue_to_anchor: list[int] = []
    if proto_rows:
        anchor_stack = torch.stack(proto_rows)
        for i in range(N_COLORS):
            nh, _ = _nearest_hue_for_anchor(anchor_stack[i] if i < len(proto_rows)
                                            else proto_rows[0], hue_rows)
            # That's anchor i's nearest hue
            if i < len(proto_rows):
                nearest.append(nh)

        an = F.normalize(anchor_stack.flatten(start_dim=1), dim=-1)
        hn = F.normalize(hue_rows.flatten(start_dim=1), dim=-1)
        sims = hn @ an.t()  # (N_COLORS, k)
        for hue_idx in range(N_COLORS):
            assigned_cluster = int(sims[hue_idx].argmax().item())
            hue_to_anchor.append(assigned_cluster)
            cluster_sizes[assigned_cluster] += 1

    eq = _equidistant_score(nearest)
    rot = _rotation_class(nearest) if eq["is_equidistant"] else None
    priors = _prior_match(nearest)

    return {
        "seed": seed,
        "k_requested": k,
        "k_recovered": len(nearest),
        "anchor_nearest_hues": nearest,
        "spacings": eq["spacings"],
        "is_equidistant": eq["is_equidistant"],
        "expected_step": eq["expected_step"],
        "max_dev": eq["max_dev"],
        "rotation_class": rot,
        "cluster_sizes": [cluster_sizes[c] for c in range(len(nearest))],
        "hue_to_anchor": hue_to_anchor,
        "perceptual_prior_match": priors,
        "wall_s": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=80000)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    ap.add_argument("--sleep-warmup", type=int, default=15)
    ap.add_argument("--sleep-every", type=int, default=5)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/sleep_inspect_color"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  color sleep anchor inspection: k={args.k}, "
          f"n_seeds={args.n_seeds}, epochs={args.epochs}, "
          f"steps={args.steps_per_epoch}, warmup={args.sleep_warmup}, "
          f"sleep_every={args.sleep_every}")
    print(f"  device={DEVICE}")
    print("=" * 72)

    per_seed: list[dict] = []
    for si in range(args.n_seeds):
        seed = args.seed_base + si
        r = _run_one_seed(
            seed, args.k, args.sleep_warmup, args.sleep_every,
            args.epochs, args.steps_per_epoch,
        )
        per_seed.append(r)
        eq_str = "EQUI" if r["is_equidistant"] else f"max_dev={r['max_dev']:.1f}"
        priors_str = ",".join(
            name for name, m in r["perceptual_prior_match"].items() if m
        ) or "(none)"
        print(
            f"  [seed={seed}] anchors → hues {r['anchor_nearest_hues']}  "
            f"spacings={r['spacings']}  {eq_str}  "
            f"sizes={r['cluster_sizes']}  prior_match={priors_str}  "
            f"({r['wall_s']:.1f}s)"
        )

    # Aggregate
    n_equi = sum(1 for r in per_seed if r["is_equidistant"])
    rot_counter: Counter[int] = Counter()
    for r in per_seed:
        if r["rotation_class"] is not None:
            rot_counter[r["rotation_class"]] += 1
    prior_counter: Counter[str] = Counter()
    for r in per_seed:
        for name, m in r["perceptual_prior_match"].items():
            if m:
                prior_counter[name] += 1

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "n_seeds": args.n_seeds,
        "k": args.k,
        "per_seed": per_seed,
        "n_equidistant": n_equi,
        "fraction_equidistant": n_equi / max(args.n_seeds, 1),
        "rotation_class_histogram": dict(rot_counter),
        "perceptual_prior_match_counts": dict(prior_counter),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print()
    print("─" * 72)
    print(f"  equidistant 120°/k spacing : {n_equi}/{args.n_seeds} seeds")
    if n_equi == args.n_seeds and N_COLORS % args.k == 0:
        period = N_COLORS // args.k
        print(f"  rotation-class histogram (mod {period}):")
        for rc in sorted(rot_counter):
            bar = "█" * rot_counter[rc]
            print(f"    rc={rc}: {rot_counter[rc]:2d}  {bar}")
        print(f"  expected uniform under H_taskSym: ~{args.n_seeds / period:.2f}/class")
    print(f"  perceptual prior matches (up to rotation):")
    for name in PERCEPTUAL_PRIORS.keys():
        cnt = prior_counter.get(name, 0)
        print(f"    {name:16s}: {cnt}/{args.n_seeds} seeds match")
    print(f"  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
