"""F62 — Universal Operator Hypothesis across math / physics / chemistry.

User hypothesis (paraphrased): different scientific disciplines —
math, physics, chemistry — are different *muscles* consuming the
same underlying *concept structure*. A transformation learned in
one discipline should therefore apply in another, because the
algebraic backbone is shared.

This module operationalises that hypothesis as a falsifiable
experiment. We construct three structurally-isomorphic but
semantically-distinct tasks under the cyclic additive group ℤ_N:

* **Math (M)** — integer arithmetic: ``a + Δ ≡ b (mod N)`` where
  ``a, b ∈ {0, …, N-1}`` are integers.
* **Physics (P)** — discrete 1-D position lattice: ``x_t + v·dt
  ≡ x_{t+1} (mod N)`` where each cell is a position on a ring.
* **Chemistry (C)** — 1-D reaction-extent coordinate:
  ``ξ_in + Δ ≡ ξ_out (mod N)`` where each cell is a mole-fraction
  bin in a stoichiometric chain.

All three encode the same group action — the category-theoretic
view (Maruyama 2026, Mašulović 2026, "Geometric Alignment +
Functor", arxiv 2602.01992) is that there exist isomorphic
functors from a base universe ``U`` into each discipline's
embedding category. PCM v2's ``RelativePositionEmbedding`` is
the candidate implementation of that base universe: a single
learnable lookup ``Δ → vector`` shared across disciplines.

We test five conditions and verify five falsifiable invariants:

* **U1** — joint-shared training reaches per-discipline accuracy
  ≥ 0.95 (in-domain sanity).
* **U2** — joint-shared ≈ joint-separate (sharing one RPE is no
  worse than per-discipline RPEs).
* **U3** — independently-trained RPEs across disciplines have
  cosine similarity ≥ 0.85 *up to a global linear alignment*
  (Procrustes), confirming the same algebraic structure emerges
  on its own.
* **U4** — frozen RPE (trained on math only) plus a freshly
  initialised slot bundle for physics reaches ≥ 0.90 accuracy
  with **only the slot bundle trained**, well above the
  identity-shuffled ceiling of U5.
* **U5** (negative control) — frozen RPE + a *permuted*
  slot-identity layout for physics fails: accuracy ≤ 0.20.
  Confirms transfer is via shared structure, not coincidence.

This is the cleanest falsification of the user's hypothesis we
can produce inside a small testbed: if U1–U4 pass and U5 fails
(low accuracy) the universal-operator picture survives; if any
of U1–U4 fail or U5 succeeds (transfer despite shuffle), the
"shared structure" claim falls apart.

Usage::

    python -m experiments.cross_discipline_operator \\
        --N 100 --slot-dim 32 --epochs 100 \\
        --out outputs/f62_cross_discipline
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcm.dual_channel import RelativePositionEmbedding


__all__ = ["main"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────
# Tiny model: per-discipline slot embedding + (shared) RPE
# ─────────────────────────────────────────────────────────────────


class UniversalCombiner(nn.Module):
    """A learned ``(slot_a, rpe_vec) → slot_b_pred`` map.

    The naive bilinear form ``slot_a + rpe_vec`` cannot represent
    cyclic-group actions like ``a + Δ (mod N)`` when slots are
    embedded as sinusoidal-style cyclic codes (because the group
    action is a *rotation*, not a translation, in feature space).
    A small MLP recovers the missing expressivity and lets the
    same module sit in front of any discipline. This module is
    the candidate "universal operator" — the architectural piece
    F62 tests for cross-discipline reuse.
    """

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self, slot_a: torch.Tensor, rpe_vec: torch.Tensor,
    ) -> torch.Tensor:
        # Residual: lets the model trivially recover identity for Δ=0.
        return slot_a + self.net(torch.cat([slot_a, rpe_vec], dim=-1))


class DisciplineModule(nn.Module):
    """One discipline's slot bundle plus references to the shared
    RPE and Combiner. The forward predicts ``b`` from ``(a, Δ)``
    via ``Combiner(slot_a, RPE(Δ))`` followed by a dot-product
    classifier over all slot embeddings.

    Args:
        N: number of states (slot count).
        dim: embedding dimensionality.
        rpe: shared (or per-discipline) :class:`RelativePositionEmbedding`.
        combiner: shared (or per-discipline) :class:`UniversalCombiner`.
        delta_max: half-range of the displacement table.
    """

    def __init__(
        self,
        N: int, dim: int,
        rpe: RelativePositionEmbedding,
        combiner: UniversalCombiner,
        delta_max: int,
    ) -> None:
        super().__init__()
        self.N = N
        self.dim = dim
        self.rpe = rpe
        self.combiner = combiner
        self.delta_max = delta_max
        self.slot = nn.Embedding(N, dim)
        nn.init.normal_(self.slot.weight, std=1.0)

    def forward(self, a: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        slot_a = self.slot(a)
        rpe_vec = self.rpe(delta)  # (B, dim)
        slot_b_pred = self.combiner(slot_a, rpe_vec)
        all_slots = self.slot.weight  # (N, dim)
        logits = slot_b_pred @ all_slots.t()
        return logits


def _smaller_rpe_init(rpe: RelativePositionEmbedding,
                      *, scale: float = 0.3) -> None:
    """Shrink RPE row norms so the slot identity dominates the
    initial ``Combiner(slot, rpe)`` output. With this scale the
    residual-MLP combiner starts close to the identity ``slot →
    slot``, giving a useful gradient signal for both slot and rpe
    to differentiate from."""
    with torch.no_grad():
        rpe.table.weight.mul_(scale)


# ─────────────────────────────────────────────────────────────────
# Data: each discipline samples (a, Δ, b) tuples under +Δ mod N
# ─────────────────────────────────────────────────────────────────


def _sample_batch(N: int, delta_max: int, B: int, device: str) -> tuple:
    a = torch.randint(0, N, (B,), device=device)
    delta = torch.randint(-delta_max, delta_max + 1, (B,), device=device)
    b = (a + delta) % N
    return a, delta, b


# ─────────────────────────────────────────────────────────────────
# Trainers, one per condition
# ─────────────────────────────────────────────────────────────────


def _train_condition(
    *,
    discipline_names: list[str], shared_rpe: bool,
    N: int, dim: int, delta_max: int,
    epochs: int, batches_per_epoch: int, batch_size: int, lr: float,
    seed: int = 0,
    combiner_hidden: int = 128,
) -> dict:
    """Train ``len(discipline_names)`` discipline modules.

    The ``UniversalCombiner`` is **always shared** across the
    disciplines in this run — that is the architectural object
    we are testing. ``shared_rpe`` toggles whether the RPE table
    is also shared; when False each discipline gets its own RPE
    table (still using the shared combiner)."""
    torch.manual_seed(seed)
    combiner = UniversalCombiner(dim=dim, hidden=combiner_hidden).to(DEVICE)
    if shared_rpe:
        rpe = RelativePositionEmbedding(
            ranges=[(-delta_max, delta_max)], embed_dim=dim,
        ).to(DEVICE)
        _smaller_rpe_init(rpe)
        modules = nn.ModuleDict({
            name: DisciplineModule(
                N=N, dim=dim, rpe=rpe, combiner=combiner,
                delta_max=delta_max,
            ).to(DEVICE)
            for name in discipline_names
        })
        params = list(rpe.parameters()) + list(combiner.parameters())
        for m in modules.values():
            params += list(m.slot.parameters())
        rpes_for_log = {name: rpe for name in discipline_names}
    else:
        modules = nn.ModuleDict()
        rpes_for_log: dict[str, RelativePositionEmbedding] = {}
        params: list[nn.Parameter] = list(combiner.parameters())
        for name in discipline_names:
            r = RelativePositionEmbedding(
                ranges=[(-delta_max, delta_max)], embed_dim=dim,
            ).to(DEVICE)
            _smaller_rpe_init(r)
            rpes_for_log[name] = r
            m = DisciplineModule(
                N=N, dim=dim, rpe=r, combiner=combiner,
                delta_max=delta_max,
            ).to(DEVICE)
            modules[name] = m
            params += list(r.parameters()) + list(m.slot.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    history: list[dict] = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = {n: 0 for n in discipline_names}
        epoch_total = {n: 0 for n in discipline_names}
        for _ in range(batches_per_epoch):
            for name in discipline_names:
                a, delta, b = _sample_batch(N, delta_max, batch_size, DEVICE)
                logits = modules[name](a, delta)
                loss = F.cross_entropy(logits, b)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item())
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    epoch_correct[name] += (pred == b).sum().item()
                    epoch_total[name] += b.numel()
        epoch_acc = {n: epoch_correct[n] / max(epoch_total[n], 1)
                     for n in discipline_names}
        history.append({"epoch": epoch, "loss": epoch_loss, **epoch_acc})

    test_acc = {}
    for name in discipline_names:
        with torch.no_grad():
            a, delta, b = _sample_batch(N, delta_max, 5000, DEVICE)
            logits = modules[name](a, delta)
            pred = logits.argmax(dim=-1)
            test_acc[name] = float((pred == b).float().mean().item())

    return {
        "history": history,
        "test_acc": test_acc,
        "modules": modules,
        "rpes": rpes_for_log,
        "combiner": combiner,
    }


# ─────────────────────────────────────────────────────────────────
# U3 — Procrustes-aligned cosine similarity between RPE tables
# ─────────────────────────────────────────────────────────────────


def _procrustes_align(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Return the optimal orthogonal matrix R such that
    ``A R ≈ B`` in the Frobenius sense, then return the aligned
    Frobenius cosine ``<A R, B> / (||A R||·||B||)``.

    A, B both ``(M, dim)``. We use the classical solution
    ``R = U V^T`` where ``A^T B = U Σ V^T``."""
    cov = A.t() @ B
    U, _, Vt = torch.linalg.svd(cov)
    R = U @ Vt
    A_aligned = A @ R
    cos = (
        (A_aligned * B).sum()
        / (A_aligned.norm() * B.norm() + 1e-12)
    )
    return cos


def _rpe_table(rpe: RelativePositionEmbedding, delta_max: int,
               dim: int) -> torch.Tensor:
    """Materialise the RPE as a (2*delta_max+1, dim) tensor."""
    deltas = torch.arange(-delta_max, delta_max + 1, device=DEVICE)
    with torch.no_grad():
        return rpe(deltas)


# ─────────────────────────────────────────────────────────────────
# U4 — frozen-RPE transfer to a new discipline
# ─────────────────────────────────────────────────────────────────


def _train_frozen_transfer(
    source_rpe: RelativePositionEmbedding,
    source_combiner: UniversalCombiner,
    *,
    N: int, dim: int, delta_max: int,
    epochs: int, batches_per_epoch: int, batch_size: int, lr: float,
    permute_slots: bool = False,
    seed: int = 0,
) -> dict:
    """Initialise a fresh slot bundle for a new discipline. Train
    *only* the slot bundle on (a, Δ, b) triples, with the RPE
    **and** UniversalCombiner frozen at the source's learned values.
    Together (RPE + Combiner) constitute the universal operator
    under test.

    If ``permute_slots`` is True, after the slot bundle finishes
    training we permute the slot identities (i.e. shuffle the
    ``(slot_index → embedding)`` map) and re-evaluate. This is the
    U5 negative control.
    """
    torch.manual_seed(seed)
    for p in source_rpe.parameters():
        p.requires_grad_(False)
    for p in source_combiner.parameters():
        p.requires_grad_(False)

    module = DisciplineModule(
        N=N, dim=dim, rpe=source_rpe, combiner=source_combiner,
        delta_max=delta_max,
    ).to(DEVICE)
    opt = torch.optim.AdamW(
        module.slot.parameters(), lr=lr, weight_decay=1e-4,
    )
    history: list[dict] = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for _ in range(batches_per_epoch):
            a, delta, b = _sample_batch(N, delta_max, batch_size, DEVICE)
            logits = module(a, delta)
            loss = F.cross_entropy(logits, b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                epoch_correct += (pred == b).sum().item()
                epoch_total += b.numel()
        history.append({
            "epoch": epoch,
            "loss": epoch_loss,
            "train_acc": epoch_correct / max(epoch_total, 1),
        })

    with torch.no_grad():
        a, delta, b = _sample_batch(N, delta_max, 5000, DEVICE)
        logits = module(a, delta)
        clean_acc = float((logits.argmax(-1) == b).float().mean().item())

        if permute_slots:
            perm = torch.randperm(N, device=DEVICE)
            permuted_weight = module.slot.weight.data[perm].clone()
            module.slot.weight.data.copy_(permuted_weight)
            a, delta, b = _sample_batch(N, delta_max, 5000, DEVICE)
            logits = module(a, delta)
            perm_acc = float((logits.argmax(-1) == b).float().mean().item())
        else:
            perm_acc = float("nan")

    return {
        "history": history,
        "clean_test_acc": clean_acc,
        "permuted_test_acc": perm_acc,
    }


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--slot-dim", type=int, default=32)
    ap.add_argument("--delta-max", type=int, default=49)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batches-per-epoch", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--n-seeds-u3", type=int, default=3,
                    help="independent training runs for U3 RPE-similarity")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f62_cross_discipline"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(
        f"  F62 cross-discipline operator transfer "
        f"(M, P, C under Z_{args.N} additive group)"
    )
    print("=" * 76)

    DISCIPLINES = ["math", "physics", "chemistry"]

    # Condition A: joint training, shared RPE
    print("\n[A] joint training, SHARED RPE across all disciplines...")
    t0 = time.time()
    A = _train_condition(
        discipline_names=DISCIPLINES, shared_rpe=True,
        N=args.N, dim=args.slot_dim, delta_max=args.delta_max,
        epochs=args.epochs, batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr, seed=11,
    )
    print(f"    wall = {time.time()-t0:.1f}s")
    for n, acc in A["test_acc"].items():
        print(f"    {n:<10s} test_acc = {acc:.3f}")

    # Condition B: joint training, separate RPE per discipline
    print("\n[B] joint training, SEPARATE RPE per discipline...")
    t0 = time.time()
    B = _train_condition(
        discipline_names=DISCIPLINES, shared_rpe=False,
        N=args.N, dim=args.slot_dim, delta_max=args.delta_max,
        epochs=args.epochs, batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr, seed=22,
    )
    print(f"    wall = {time.time()-t0:.1f}s")
    for n, acc in B["test_acc"].items():
        print(f"    {n:<10s} test_acc = {acc:.3f}")

    # Condition C: independent training per discipline (each is its own run)
    print(f"\n[C] independent training per discipline "
          f"({args.n_seeds_u3} seeds for U3 cosine analysis)...")
    independent_runs: list[dict] = []
    for seed_idx in range(args.n_seeds_u3):
        runs_per_seed = {}
        for name in DISCIPLINES:
            r = _train_condition(
                discipline_names=[name], shared_rpe=False,
                N=args.N, dim=args.slot_dim, delta_max=args.delta_max,
                epochs=args.epochs,
                batches_per_epoch=args.batches_per_epoch,
                batch_size=args.batch_size, lr=args.lr,
                seed=100 + seed_idx,
            )
            runs_per_seed[name] = r
        independent_runs.append(runs_per_seed)
        accs = {n: runs_per_seed[n]["test_acc"][n]
                for n in DISCIPLINES}
        print(f"    seed {seed_idx}: " + "  ".join(
            f"{n}={accs[n]:.3f}" for n in DISCIPLINES))

    # U3: cosine similarity between independently-trained RPEs,
    # under Procrustes orthogonal alignment.
    print("\n[U3] Procrustes-aligned cosine similarity between "
          "independently-trained RPEs:")
    u3_rows: list[dict] = []
    for seed_idx in range(args.n_seeds_u3):
        per_disc_table = {
            n: _rpe_table(
                independent_runs[seed_idx][n]["rpes"][n],
                args.delta_max, args.slot_dim,
            ) for n in DISCIPLINES
        }
        # Pair each pair of disciplines
        for i, di in enumerate(DISCIPLINES):
            for dj in DISCIPLINES[i + 1:]:
                cos = _procrustes_align(
                    per_disc_table[di], per_disc_table[dj]
                )
                u3_rows.append({
                    "seed": seed_idx,
                    "pair": f"{di}-{dj}",
                    "procrustes_cos": float(cos.item()),
                })
                print(f"    seed {seed_idx} {di:<9s} <-> {dj:<9s}: "
                      f"cos={cos.item():.3f}")
    if u3_rows:
        u3_mean_cos = sum(r["procrustes_cos"] for r in u3_rows) / len(u3_rows)
    else:
        u3_mean_cos = float("nan")

    # Null baseline for U3: Procrustes cosine between two
    # *independently random* RPE tables. With (2*delta_max+1)
    # vectors in dim-D, random tables have Procrustes cosine
    # bounded above by ~sqrt(M/D) for M vectors. We measure to
    # confirm the trained-RPE cosine is significantly above the
    # random baseline.
    null_cos: list[float] = []
    torch.manual_seed(31337)
    for _ in range(8):
        r1 = torch.randn(2 * args.delta_max + 1, args.slot_dim, device=DEVICE)
        r2 = torch.randn(2 * args.delta_max + 1, args.slot_dim, device=DEVICE)
        null_cos.append(float(_procrustes_align(r1, r2).item()))
    u3_null_cos_mean = sum(null_cos) / len(null_cos)
    print(f"    [null baseline] random-vs-random Procrustes cos mean = "
          f"{u3_null_cos_mean:.3f}")

    # U4 / U5: frozen-RPE transfer.
    # Use the math-only branch from condition C (seed 0) as the source RPE.
    print("\n[U4] frozen-operator transfer (math -> physics, "
          "only slot bundle trains)...")
    source_rpe = independent_runs[0]["math"]["rpes"]["math"]
    source_combiner = independent_runs[0]["math"]["combiner"]
    transfer_epochs = args.epochs  # match the joint-training budget
    transfer = _train_frozen_transfer(
        source_rpe, source_combiner,
        N=args.N, dim=args.slot_dim, delta_max=args.delta_max,
        epochs=transfer_epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr, seed=999,
        permute_slots=False,
    )
    print(f"    transfer_epochs = {transfer_epochs}")
    print(f"    physics test_acc (frozen RPE+Combiner) = "
          f"{transfer['clean_test_acc']:.3f}")

    print("\n[U5] negative control - same as U4 but slot identities "
          "are PERMUTED at evaluation:")
    # Re-run a separate frozen transfer that returns both clean and permuted.
    transfer_neg = _train_frozen_transfer(
        source_rpe, source_combiner,
        N=args.N, dim=args.slot_dim, delta_max=args.delta_max,
        epochs=transfer_epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr, seed=998,
        permute_slots=True,
    )
    print(f"    physics test_acc (frozen RPE, slots PERMUTED) = "
          f"{transfer_neg['permuted_test_acc']:.3f}  (random=1/N={1.0/args.N:.3f})")

    # Verdict assembly
    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "A_joint_shared_acc": A["test_acc"],
        "B_joint_separate_acc": B["test_acc"],
        "C_independent_acc": [
            {n: r[n]["test_acc"][n] for n in DISCIPLINES}
            for r in independent_runs
        ],
        "U3_pairs": u3_rows,
        "U3_mean_procrustes_cos": u3_mean_cos,
        "U3_null_random_baseline_cos": u3_null_cos_mean,
        "U4_frozen_transfer_acc": transfer["clean_test_acc"],
        "U5_permuted_transfer_acc": transfer_neg["permuted_test_acc"],
    }

    a_min = min(A["test_acc"].values())
    b_min = min(B["test_acc"].values())
    summary["verdict"] = {
        "U1_pass": a_min >= 0.95,
        "U2_pass": (a_min >= b_min - 0.03),  # within 3pp
        # U3: Procrustes cosine between independently-trained RPEs
        # significantly exceeds the random-vs-random baseline.
        "U3_pass": (
            (not math.isnan(u3_mean_cos))
            and u3_mean_cos >= 0.85
            and u3_mean_cos >= u3_null_cos_mean + 0.20
        ),
        "U4_pass": transfer["clean_test_acc"] >= 0.90,
        "U5_pass": transfer_neg["permuted_test_acc"] <= 0.20,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print("  Universal-operator hypothesis verdict:")
    print("=" * 76)
    print(f"  U1 (joint-shared works)            : "
          f"min_acc = {a_min:.3f}  "
          f"[{'PASS' if summary['verdict']['U1_pass'] else 'FAIL'}]")
    print(f"  U2 (sharing has no penalty)        : "
          f"shared_min={a_min:.3f} vs separate_min={b_min:.3f}  "
          f"[{'PASS' if summary['verdict']['U2_pass'] else 'FAIL'}]")
    print(f"  U3 (independent RPEs are aligned)  : "
          f"trained cos = {u3_mean_cos:.3f}  "
          f"random baseline cos = {u3_null_cos_mean:.3f}  "
          f"[{'PASS' if summary['verdict']['U3_pass'] else 'FAIL'}]")
    print(f"  U4 (frozen-RPE transfer to new D)  : "
          f"acc = {transfer['clean_test_acc']:.3f}  "
          f"[{'PASS' if summary['verdict']['U4_pass'] else 'FAIL'}]")
    print(f"  U5 (permuted-slot transfer fails)  : "
          f"acc = {transfer_neg['permuted_test_acc']:.3f}  "
          f"[{'PASS' if summary['verdict']['U5_pass'] else 'FAIL'}]")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
