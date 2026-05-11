"""F62e — Does the universal operator respect group composition?

F62 showed that a trained ``UniversalCombiner + RPE`` perfectly
predicts ``b = a + Δ (mod N)`` for individual ``(a, Δ)`` pairs.
But this only proves the operator memorised the truth table;
it doesn't prove the operator learned the *group structure*.

A genuine algebra-respecting operator must satisfy:

* **Composition** — applying Δ₁ then Δ₂ equals applying Δ₁+Δ₂:
  ``T(T(a, Δ₁), Δ₂) ≈ T(a, Δ₁ + Δ₂ (mod N))``.
* **Identity** — ``T(a, 0) = a``.
* **Inverse** — ``T(T(a, Δ), -Δ) = a``.
* **Cyclic order** — applying ``+1`` exactly ``N`` times brings
  every state back to itself.

These are the **group-theory invariants** the trained operator
must satisfy if it has truly learned the cyclic group structure
of ℤ_N rather than merely fitting input-output pairs.

This experiment trains the F62 universal operator on math, then
tests four falsifiable composition invariants:

* **C1** identity: agreement with ``T(a, 0) = a`` ≥ 0.99.
* **C2** inverse: ``T(T(a, Δ), -Δ) = a`` agreement ≥ 0.95.
* **C3** binary composition: ``T(T(a, Δ₁), Δ₂) = T(a, Δ₁+Δ₂)``
  agreement ≥ 0.95.
* **C4** cyclic order: ``T^N(a, +1) = a`` for all a ≥ 0.95.

If C1–C4 all pass, the operator is a faithful representation of
ℤ_N — not just a lookup. This is the strongest test of "the
operator IS the concept" in the F62 architecture.

Usage::

    python -m experiments.operator_composition_test \\
        --N 50 --slot-dim 32 --epochs 80 \\
        --out outputs/f62e_composition
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcm.dual_channel import RelativePositionEmbedding
from experiments.cross_discipline_operator import (
    DisciplineModule, UniversalCombiner, _smaller_rpe_init,
    _sample_batch,
)


__all__ = ["main"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _train_operator(
    *, N: int, dim: int, delta_max: int,
    epochs: int, batches_per_epoch: int, batch_size: int, lr: float,
    seed: int = 0,
) -> tuple:
    torch.manual_seed(seed)
    rpe = RelativePositionEmbedding(
        ranges=[(-delta_max, delta_max)], embed_dim=dim,
    ).to(DEVICE)
    _smaller_rpe_init(rpe)
    combiner = UniversalCombiner(dim=dim).to(DEVICE)
    module = DisciplineModule(
        N=N, dim=dim, rpe=rpe, combiner=combiner,
        delta_max=delta_max,
    ).to(DEVICE)
    opt = torch.optim.AdamW(
        list(rpe.parameters()) + list(combiner.parameters())
        + list(module.slot.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    for _ in range(epochs):
        for _ in range(batches_per_epoch):
            a, delta, b = _sample_batch(N, delta_max, batch_size, DEVICE)
            logits = module(a, delta)
            loss = F.cross_entropy(logits, b)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        a, delta, b = _sample_batch(N, delta_max, 5000, DEVICE)
        train_acc = float((module(a, delta).argmax(-1) == b).float().mean())
    return module, rpe, combiner, train_acc


# ─────────────────────────────────────────────────────────────────
# Composition tests — apply T as a function and check group axioms
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def _T(module: DisciplineModule, a: torch.Tensor,
       delta: torch.Tensor) -> torch.Tensor:
    """Apply the operator: returns argmax slot prediction."""
    return module(a, delta).argmax(dim=-1)


@torch.no_grad()
def test_identity(module: DisciplineModule, N: int) -> float:
    """C1: T(a, 0) ≈ a for all a ∈ {0..N-1}."""
    a = torch.arange(N, device=DEVICE)
    delta = torch.zeros(N, dtype=torch.long, device=DEVICE)
    pred = _T(module, a, delta)
    return float((pred == a).float().mean().item())


@torch.no_grad()
def test_inverse(module: DisciplineModule, N: int,
                 delta_max: int) -> float:
    """C2: T(T(a, Δ), -Δ) ≈ a."""
    n_test = 5000
    a, delta, _ = _sample_batch(N, delta_max, n_test, DEVICE)
    b = _T(module, a, delta)
    a_recovered = _T(module, b, -delta)
    return float((a_recovered == a).float().mean().item())


@torch.no_grad()
def test_composition(module: DisciplineModule, N: int,
                     delta_max: int) -> float:
    """C3: T(T(a, Δ₁), Δ₂) ≈ T(a, (Δ₁+Δ₂) clipped to delta range)."""
    n_test = 5000
    a = torch.randint(0, N, (n_test,), device=DEVICE)
    d1 = torch.randint(-delta_max // 2, delta_max // 2 + 1,
                        (n_test,), device=DEVICE)
    d2 = torch.randint(-delta_max // 2, delta_max // 2 + 1,
                        (n_test,), device=DEVICE)
    sequential = _T(module, _T(module, a, d1), d2)
    composed_delta = (d1 + d2).clamp(-delta_max, delta_max)
    direct = _T(module, a, composed_delta)
    return float((sequential == direct).float().mean().item())


@torch.no_grad()
def test_cyclic_order(module: DisciplineModule, N: int) -> float:
    """C4: T^N(a, +1) ≈ a for all a (cyclic-group order = N)."""
    a = torch.arange(N, device=DEVICE)
    delta_p1 = torch.ones(N, dtype=torch.long, device=DEVICE)
    state = a.clone()
    for _ in range(N):
        state = _T(module, state, delta_p1)
    return float((state == a).float().mean().item())


@torch.no_grad()
def test_composition_drift(module: DisciplineModule, N: int,
                            delta_max: int, K: int = 8) -> dict:
    """Bonus: track how composition error grows over K sequential
    applications. T^K(a, +1) vs T(a, +K). Lower drift = better."""
    a = torch.arange(N, device=DEVICE)
    delta_p1 = torch.ones(N, dtype=torch.long, device=DEVICE)
    state = a.clone()
    drift = []
    for k in range(1, K + 1):
        state = _T(module, state, delta_p1)
        if k <= delta_max:
            direct_delta = torch.full((N,), k, dtype=torch.long,
                                       device=DEVICE)
            direct = _T(module, a, direct_delta)
            agreement = float((state == direct).float().mean().item())
            drift.append({"k": k, "sequential_vs_direct": agreement})
    return drift


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--slot-dim", type=int, default=32)
    ap.add_argument("--delta-max", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batches-per-epoch", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f62e_composition"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"  F62e operator composition test on Z_{args.N}")
    print("=" * 76)

    print(f"\n[1/2] Training F62-style universal operator on math...")
    t0 = time.time()
    module, rpe, combiner, train_acc = _train_operator(
        N=args.N, dim=args.slot_dim, delta_max=args.delta_max,
        epochs=args.epochs, batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr, seed=11,
    )
    print(f"    wall = {time.time()-t0:.1f}s")
    print(f"    train_acc = {train_acc:.3f}")

    print(f"\n[2/2] Testing group-theory invariants...")
    c1 = test_identity(module, args.N)
    c2 = test_inverse(module, args.N, args.delta_max)
    c3 = test_composition(module, args.N, args.delta_max)
    c4 = test_cyclic_order(module, args.N)
    drift = test_composition_drift(module, args.N, args.delta_max,
                                    K=min(8, args.delta_max))

    print(f"    C1 identity        T(a, 0) == a            : {c1:.3f}")
    print(f"    C2 inverse         T(T(a,Δ), -Δ) == a      : {c2:.3f}")
    print(f"    C3 binary compose  T(T(a,Δ1), Δ2) == T(a,Δ1+Δ2): "
          f"{c3:.3f}")
    print(f"    C4 cyclic order    T^N(a, +1) == a         : {c4:.3f}")
    print(f"    drift T^k(a, +1) vs T(a, +k):")
    for d in drift:
        print(f"      k={d['k']:>2d}: agreement = {d['sequential_vs_direct']:.3f}")

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "train_acc": train_acc,
        "C1_identity": c1,
        "C2_inverse": c2,
        "C3_binary_composition": c3,
        "C4_cyclic_order": c4,
        "drift_curve": drift,
        "verdict": {
            "C1_pass": c1 >= 0.99,
            "C2_pass": c2 >= 0.95,
            "C3_pass": c3 >= 0.95,
            "C4_pass": c4 >= 0.95,
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print(f"  F62e composition verdict:")
    print("=" * 76)
    v = summary["verdict"]
    print(f"  C1 identity (>=0.99)             : {c1:.3f}  "
          f"[{'PASS' if v['C1_pass'] else 'FAIL'}]")
    print(f"  C2 inverse  (>=0.95)             : {c2:.3f}  "
          f"[{'PASS' if v['C2_pass'] else 'FAIL'}]")
    print(f"  C3 binary composition (>=0.95)   : {c3:.3f}  "
          f"[{'PASS' if v['C3_pass'] else 'FAIL'}]")
    print(f"  C4 cyclic order (>=0.95)         : {c4:.3f}  "
          f"[{'PASS' if v['C4_pass'] else 'FAIL'}]")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
