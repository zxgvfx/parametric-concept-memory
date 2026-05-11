"""F58 — neuromorphic-deployment profile for PCM v3 / v4 operations.

Quantifies FLOPs, parameter count, memory footprint, and
sparsity for the four core PCM v3 / v4 head archetypes:

* `RelativePositionEmbedding` (System-1 retrieval, lookup)
* `SuccessorHead` + `IterativeDiffCook` (System-2 procedural,
  iterative)
* `PhysicsStateHead` + `PhysicsCook` (v4 physics rollout)
* `DualChannelPairHead` with optional RPE (the v2 baseline)

Output is a tabular spec sheet a hardware engineer can use to
estimate Loihi 2 / Akida / SpiNNaker / FPGA budget envelopes
for each operation. Numbers are reported per query (single
batch element), so multi-query batches multiply linearly.

Usage::

    python -m scripts.neuromorphic_profile \\
        --slot-dim 16 --attr-dim 8 --n-cells 100 \\
        --K-rollout 99 \\
        --out outputs/f58_profile
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from pcm.dual_channel import (
    ALiBiRelativePositionBias,
    RelativePositionEmbedding,
    SinusoidalRelativePositionEmbedding,
)
from pcm.dual_process import IterativeDiffCook, SuccessorHead
from pcm.physics import PhysicsCook, PhysicsStateHead


__all__ = ["profile_module", "main"]


# ─────────────────────────────────────────────────────────────────
# Helpers — count params, FLOPs, sparsity for nn.Modules.
# ─────────────────────────────────────────────────────────────────


@dataclass
class OpProfile:
    name: str
    n_params: int
    flops_per_query: int
    bytes_resident: int
    bytes_per_query_io: int
    sparsity: float = 0.0  # fraction of params/activations zero
    notes: str = ""
    per_iteration: bool = False  # True for cook-style ops

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_params": self.n_params,
            "flops_per_query": self.flops_per_query,
            "bytes_resident": self.bytes_resident,
            "bytes_per_query_io": self.bytes_per_query_io,
            "sparsity": self.sparsity,
            "notes": self.notes,
            "per_iteration": self.per_iteration,
        }


def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _bytes_resident(m: nn.Module, *, dtype_bytes: int = 4) -> int:
    return _count_params(m) * dtype_bytes


def _flops_linear(in_dim: int, out_dim: int, batch: int = 1) -> int:
    """2 * in_dim * out_dim per output element (mul + add)."""
    return 2 * in_dim * out_dim * batch


def _flops_relu(n_act: int, batch: int = 1) -> int:
    return n_act * batch  # 1 comparison per activation


# ─────────────────────────────────────────────────────────────────
# Per-archetype profile builders.
# ─────────────────────────────────────────────────────────────────


def profile_rpe_lookup(n_cells: int, embed_dim: int) -> OpProfile:
    """`RelativePositionEmbedding` table lookup + classifier.

    Loihi 2 mapping: Embedding row read = 1 spike per neuron
    selected; classifier = standard dense layer."""
    rpe = RelativePositionEmbedding(
        ranges=[(-(n_cells - 1), n_cells - 1)], embed_dim=embed_dim,
    )
    classifier = nn.Sequential(
        nn.Linear(embed_dim, 64), nn.ReLU(),
        nn.Linear(64, 2 * n_cells - 1),
    )
    n_disp = 2 * n_cells - 1
    # FLOPs per query: lookup (≈ free, 1 row read) +
    # classifier (Linear(embed→64) + ReLU + Linear(64→n_disp))
    flops = (
        _flops_linear(embed_dim, 64)
        + _flops_relu(64)
        + _flops_linear(64, n_disp)
    )
    n_params = _count_params(rpe) + _count_params(classifier)
    bytes_resident = (
        _bytes_resident(rpe) + _bytes_resident(classifier)
    )
    # I/O per query: 1 int delta in + n_disp logits out.
    bytes_io = 4 + n_disp * 4
    return OpProfile(
        name="RPE lookup + classifier",
        n_params=n_params,
        flops_per_query=flops,
        bytes_resident=bytes_resident,
        bytes_per_query_io=bytes_io,
        sparsity=0.0,
        notes=(
            f"Embedding({n_disp}, {embed_dim}) lookup is content-"
            "addressable on Loihi 2 (constant time) but consumes "
            "memory linearly in n_disp; for large domains consider "
            "factorising the table."
        ),
    )


def profile_successor_head(slot_dim: int, attr_dim: int) -> OpProfile:
    """`SuccessorHead` 1-step transition predictor."""
    head = SuccessorHead(
        slot_dim=slot_dim, attr_dim=attr_dim, max_step=1, hidden=64,
    )
    in_dim = 2 * slot_dim + 2 * attr_dim
    flops = (
        _flops_linear(in_dim, 64)
        + _flops_relu(64)
        + _flops_linear(64, 64)
        + _flops_relu(64)
        + _flops_linear(64, 3)  # 3-class output
    )
    return OpProfile(
        name="SuccessorHead (slot+attr → step)",
        n_params=_count_params(head),
        flops_per_query=flops,
        bytes_resident=_bytes_resident(head),
        bytes_per_query_io=in_dim * 4 + 3 * 4,
        sparsity=0.0,
        notes=(
            "Tiny dense MLP. Maps cleanly to a single Loihi 2 "
            "neuro-core (≈1024 neurons available, head uses ~131). "
            "ReLU activation is native."
        ),
        per_iteration=True,
    )


def profile_iterative_cook(
    slot_dim: int, attr_dim: int, K_avg: int,
) -> OpProfile:
    """One `IterativeDiffCook` invocation = `K_avg` SuccessorHead
    calls + cursor update arithmetic."""
    succ = profile_successor_head(slot_dim, attr_dim)
    cursor_update_flops = 1  # 1 add per step
    flops = K_avg * (succ.flops_per_query + cursor_update_flops)
    return OpProfile(
        name=f"IterativeDiffCook ({K_avg} avg iters)",
        n_params=succ.n_params,
        flops_per_query=flops,
        bytes_resident=succ.bytes_resident,
        bytes_per_query_io=succ.bytes_per_query_io,
        sparsity=0.0,
        notes=(
            f"Sequential dependency: cannot parallelise across "
            f"the {K_avg} iterations. Loihi 2 / SpiNNaker "
            "throughput depends on neuron-update tick rate "
            "(typically 1 kHz). Akida supports reservoir-style "
            "iterative loops natively."
        ),
        per_iteration=False,  # whole cook is a single 'query'
    )


def profile_physics_step(state_dim: int, force_dim: int = 0) -> OpProfile:
    """`PhysicsStateHead` 1-step continuous-state transition."""
    head = PhysicsStateHead(state_dim=state_dim, force_dim=force_dim)
    in_dim = state_dim + force_dim
    flops = (
        _flops_linear(in_dim, 64)
        + _flops_relu(64)
        + _flops_linear(64, 64)
        + _flops_relu(64)
        + _flops_linear(64, state_dim)
    )
    return OpProfile(
        name="PhysicsStateHead (state+force → Δstate)",
        n_params=_count_params(head),
        flops_per_query=flops,
        bytes_resident=_bytes_resident(head),
        bytes_per_query_io=(in_dim + state_dim) * 4,
        sparsity=0.0,
        notes=(
            "Continuous-state regressor. On neuromorphic hardware "
            "with discrete spikes, would use rate coding; ANN-to-"
            "SNN conversion is straightforward for this MLP."
        ),
        per_iteration=True,
    )


def profile_physics_cook(
    state_dim: int, force_dim: int, K_rollout: int,
) -> OpProfile:
    step = profile_physics_step(state_dim, force_dim)
    flops = K_rollout * (step.flops_per_query + state_dim)  # +1 add
    return OpProfile(
        name=f"PhysicsCook K={K_rollout}",
        n_params=step.n_params,
        flops_per_query=flops,
        bytes_resident=step.bytes_resident,
        bytes_per_query_io=(state_dim * (K_rollout + 1)) * 4,
        sparsity=0.0,
        notes=(
            "Same sequential bottleneck as IterativeDiffCook; the "
            "reward is a full trajectory. Ideal for neuromorphic "
            "hardware with native sequential reservoir dynamics."
        ),
    )


def profile_sinusoidal_rpe(
    n_axes: int, embed_dim: int, n_classes: int,
) -> OpProfile:
    rpe = SinusoidalRelativePositionEmbedding(
        n_axes=n_axes, embed_dim=embed_dim,
    )
    classifier = nn.Sequential(
        nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, n_classes),
    )
    # Sinusoidal basis: ~D sin/cos evaluations + classifier dense.
    flops = (
        embed_dim * 2  # sin + cos
        + _flops_linear(embed_dim, 128)
        + _flops_relu(128)
        + _flops_linear(128, n_classes)
    )
    return OpProfile(
        name="Sinusoidal RPE + classifier",
        n_params=_count_params(rpe) + _count_params(classifier),
        flops_per_query=flops,
        bytes_resident=_bytes_resident(classifier),
        bytes_per_query_io=4 + n_classes * 4,
        sparsity=0.0,
        notes=(
            "No lookup memory needed (basis is computed). Trades "
            "memory for compute — a good fit for FPGAs but not "
            "for low-power Loihi 2 cores."
        ),
    )


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot-dim", type=int, default=16)
    ap.add_argument("--attr-dim", type=int, default=8)
    ap.add_argument("--n-cells", type=int, default=100)
    ap.add_argument("--K-rollout", type=int, default=99)
    ap.add_argument("--state-dim", type=int, default=2)
    ap.add_argument("--force-dim", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f58_neuromorphic_profile"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    profiles: list[OpProfile] = [
        profile_rpe_lookup(n_cells=args.n_cells, embed_dim=16),
        profile_successor_head(args.slot_dim, args.attr_dim),
        profile_iterative_cook(
            args.slot_dim, args.attr_dim, K_avg=args.K_rollout,
        ),
        profile_physics_step(args.state_dim, args.force_dim),
        profile_physics_cook(
            args.state_dim, args.force_dim, args.K_rollout,
        ),
        profile_sinusoidal_rpe(
            n_axes=1, embed_dim=32, n_classes=2 * args.n_cells - 1,
        ),
    ]

    print("=" * 90)
    print(f"  PCM v3 / v4 neuromorphic profile (slot_dim={args.slot_dim}, "
          f"attr_dim={args.attr_dim}, n_cells={args.n_cells}, "
          f"K_rollout={args.K_rollout})")
    print("=" * 90)
    print(
        f"  {'op':<42s} {'params':>9s} {'FLOPs/q':>12s} "
        f"{'res(KB)':>10s} {'IO(B)':>8s}"
    )
    for p in profiles:
        kb = p.bytes_resident / 1024.0
        print(
            f"  {p.name:<42s} {p.n_params:>9d} {p.flops_per_query:>12d} "
            f"{kb:>10.2f} {p.bytes_per_query_io:>8d}"
        )

    print()
    print("  Per-op notes:")
    for p in profiles:
        print(f"  * {p.name}: {p.notes}")

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "profiles": [p.to_dict() for p in profiles],
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    # ─── headline comparisons ───
    rpe_p = next(p for p in profiles if p.name.startswith("RPE lookup"))
    cook_p = next(p for p in profiles if p.name.startswith("IterativeDiffCook"))
    print()
    print("=" * 90)
    print("  Headline tradeoffs (System 1 retrieval vs System 2 procedural):")
    print("=" * 90)
    print(
        f"  RPE lookup:  {rpe_p.flops_per_query:>10d} FLOPs, "
        f"{rpe_p.n_params:>7d} params, {rpe_p.bytes_resident / 1024.0:>7.2f} KB"
    )
    print(
        f"  Cook K={args.K_rollout:>3d}: {cook_p.flops_per_query:>10d} FLOPs, "
        f"{cook_p.n_params:>7d} params, "
        f"{cook_p.bytes_resident / 1024.0:>7.2f} KB"
    )
    flop_ratio = cook_p.flops_per_query / max(rpe_p.flops_per_query, 1)
    param_ratio = cook_p.n_params / max(rpe_p.n_params, 1)
    print(
        f"\n  cook / RPE FLOP ratio:    {flop_ratio:.1f}x  "
        f"(cook is {flop_ratio:.1f}x more compute per query)"
    )
    print(
        f"  cook / RPE param ratio:   {param_ratio:.2f}x  "
        f"(cook uses {1/param_ratio:.1f}x LESS memory)"
    )
    print()
    print(
        "  → Cook is the right choice on memory-constrained "
        "neuromorphic hardware (Loihi 2: 1 MB / chip);"
    )
    print(
        "  → RPE is the right choice on compute-constrained edge "
        "devices once it has been distilled (F53)."
    )
    print()
    print("  → Combined: RPE for in-range queries, cook for OOD")
    print("    (route_diff F54 dispatcher) gives the best of both:")
    print(
        f"       in-range queries: {rpe_p.flops_per_query} FLOPs (cheap)"
    )
    print(
        f"       OOD queries:      {cook_p.flops_per_query} FLOPs (expensive but solves)"
    )
    print()
    print(f"  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
