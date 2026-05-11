"""F62b — Universal-operator hypothesis on non-abelian group D_n.

F62 demonstrated that ``UniversalCombiner + RPE`` transfers
across three abelian-group disciplines (math/physics/chemistry,
all isomorphic to Z_50). The natural follow-up: does the same
architecture survive *non-commutative* group structure?

The dihedral group ``D_n`` is the smallest natural non-abelian
testbed. ``D_n = ⟨r, s | r^n = s^2 = 1, s r s = r^{-1}⟩`` has
``2n`` elements: ``n`` rotations ``r^k`` plus ``n`` reflections
``s · r^k``. The action on the cyclic set ``{0, …, n-1}`` is

    rotation r^k  : a → (a + k)     (mod n)
    reflection sk : a → (k − a)     (mod n)

Critically ``r · s ≠ s · r`` (composition matters), so any
architecture that secretly relied on additive group structure
will fail.

We mimic the F62 protocol exactly:

* Three "disciplines" (geometry, biology, music) all implement
  the *same* ``D_n`` action on different surface labels.
* Same ``UniversalCombiner`` MLP + ``RelativePositionEmbedding``
  architecture as F62.
* Same five falsifiable invariants U1–U5.

If U1–U5 all pass on D_25 like they did on Z_50, the universal-
operator architecture is robust to non-abelian structure. If
any fail, we have a sharp falsification: the F62 result was
abelian-specific.

Usage::

    python -m experiments.non_abelian_operator \\
        --n 25 --slot-dim 32 --epochs 80 \\
        --out outputs/f62b_dihedral
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
from experiments.cross_discipline_operator import (
    UniversalCombiner, _smaller_rpe_init,
)


__all__ = ["main"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────
# Dihedral action: D_n acts on Z_n
# ─────────────────────────────────────────────────────────────────


def apply_dihedral(a: torch.Tensor, g: torch.Tensor, n: int) -> torch.Tensor:
    """Apply group element ``g`` to state ``a`` under D_n action.

    ``g`` is a flat group-index in ``[0, 2n)``: ``g < n`` means
    rotation by ``g``, ``g >= n`` means reflection composed with
    rotation by ``g - n``.
    """
    is_reflect = (g >= n).to(a.dtype)
    k = g % n
    rot_result = (a + k) % n
    ref_result = (k - a) % n
    return (1 - is_reflect) * rot_result + is_reflect * ref_result


def _sample_batch(n: int, B: int, device: str) -> tuple:
    a = torch.randint(0, n, (B,), device=device)
    g = torch.randint(0, 2 * n, (B,), device=device)
    b = apply_dihedral(a, g, n)
    return a, g, b


# ─────────────────────────────────────────────────────────────────
# Module: same shape as F62, but ``rpe`` indexes group elements
# ─────────────────────────────────────────────────────────────────


class DihedralDisciplineModule(nn.Module):
    def __init__(
        self, n: int, dim: int,
        rpe: RelativePositionEmbedding,
        combiner: UniversalCombiner,
    ) -> None:
        super().__init__()
        self.n = n
        self.dim = dim
        self.rpe = rpe
        self.combiner = combiner
        self.slot = nn.Embedding(n, dim)
        nn.init.normal_(self.slot.weight, std=1.0)

    def forward(self, a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        slot_a = self.slot(a)
        # Map group-element index in [0, 2n) to RPE input.
        # RelativePositionEmbedding takes a "delta" tensor; here
        # we treat the group-element index as the lookup key
        # (range encoded as (-n, n] via shift).
        rpe_index = g - self.n  # shift to ~[-n, n)
        rpe_vec = self.rpe(rpe_index)
        slot_b_pred = self.combiner(slot_a, rpe_vec)
        all_slots = self.slot.weight
        return slot_b_pred @ all_slots.t()


# ─────────────────────────────────────────────────────────────────
# Trainers
# ─────────────────────────────────────────────────────────────────


def _train_condition(
    *, discipline_names, shared_rpe: bool,
    n: int, dim: int, epochs: int, batches_per_epoch: int,
    batch_size: int, lr: float, seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    combiner = UniversalCombiner(dim=dim).to(DEVICE)
    if shared_rpe:
        rpe = RelativePositionEmbedding(
            ranges=[(-n, n - 1)], embed_dim=dim,
        ).to(DEVICE)
        _smaller_rpe_init(rpe)
        modules = nn.ModuleDict({
            name: DihedralDisciplineModule(
                n=n, dim=dim, rpe=rpe, combiner=combiner,
            ).to(DEVICE)
            for name in discipline_names
        })
        params = list(rpe.parameters()) + list(combiner.parameters())
        for m in modules.values():
            params += list(m.slot.parameters())
        rpes_for_log = {nm: rpe for nm in discipline_names}
    else:
        modules = nn.ModuleDict()
        rpes_for_log = {}
        params = list(combiner.parameters())
        for nm in discipline_names:
            r = RelativePositionEmbedding(
                ranges=[(-n, n - 1)], embed_dim=dim,
            ).to(DEVICE)
            _smaller_rpe_init(r)
            rpes_for_log[nm] = r
            m = DihedralDisciplineModule(
                n=n, dim=dim, rpe=r, combiner=combiner,
            ).to(DEVICE)
            modules[nm] = m
            params += list(r.parameters()) + list(m.slot.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        for _ in range(batches_per_epoch):
            for nm in discipline_names:
                a, g, b = _sample_batch(n, batch_size, DEVICE)
                logits = modules[nm](a, g)
                loss = F.cross_entropy(logits, b)
                opt.zero_grad()
                loss.backward()
                opt.step()

    test_acc = {}
    for nm in discipline_names:
        with torch.no_grad():
            a, g, b = _sample_batch(n, 5000, DEVICE)
            logits = modules[nm](a, g)
            pred = logits.argmax(dim=-1)
            test_acc[nm] = float((pred == b).float().mean().item())
    return {
        "test_acc": test_acc, "modules": modules,
        "rpes": rpes_for_log, "combiner": combiner,
    }


def _train_frozen_transfer(
    source_rpe, source_combiner,
    *, n: int, dim: int, epochs: int, batches_per_epoch: int,
    batch_size: int, lr: float, permute_slots: bool = False,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    for p in source_rpe.parameters():
        p.requires_grad_(False)
    for p in source_combiner.parameters():
        p.requires_grad_(False)
    module = DihedralDisciplineModule(
        n=n, dim=dim, rpe=source_rpe, combiner=source_combiner,
    ).to(DEVICE)
    opt = torch.optim.AdamW(module.slot.parameters(), lr=lr,
                             weight_decay=1e-4)
    for _ in range(epochs):
        for _ in range(batches_per_epoch):
            a, g, b = _sample_batch(n, batch_size, DEVICE)
            logits = module(a, g)
            loss = F.cross_entropy(logits, b)
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        a, g, b = _sample_batch(n, 5000, DEVICE)
        logits = module(a, g)
        clean = float((logits.argmax(-1) == b).float().mean().item())
        if permute_slots:
            perm = torch.randperm(n, device=DEVICE)
            module.slot.weight.data.copy_(module.slot.weight.data[perm])
            a, g, b = _sample_batch(n, 5000, DEVICE)
            logits = module(a, g)
            permuted = float((logits.argmax(-1) == b).float().mean().item())
        else:
            permuted = float("nan")
    return {"clean_test_acc": clean, "permuted_test_acc": permuted}


def _procrustes_align(A: torch.Tensor, B: torch.Tensor) -> float:
    cov = A.t() @ B
    U, _, Vt = torch.linalg.svd(cov)
    R = U @ Vt
    A_aligned = A @ R
    return float(((A_aligned * B).sum()
                  / (A_aligned.norm() * B.norm() + 1e-12)).item())


def _rpe_table(rpe, n, dim) -> torch.Tensor:
    deltas = torch.arange(-n, n, device=DEVICE)
    with torch.no_grad():
        return rpe(deltas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25, help="D_n group order")
    ap.add_argument("--slot-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batches-per-epoch", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--n-seeds-u3", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f62b_dihedral"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"  F62b non-abelian D_{args.n} (|G|={2*args.n}) operator transfer")
    print("=" * 76)

    DISCIPLINES = ["geometry", "biology", "music"]
    n = args.n

    print("\n[A] joint training, SHARED operator across all disciplines...")
    t0 = time.time()
    A = _train_condition(
        discipline_names=DISCIPLINES, shared_rpe=True,
        n=n, dim=args.slot_dim, epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr, seed=11,
    )
    print(f"    wall = {time.time()-t0:.1f}s")
    for nm, acc in A["test_acc"].items():
        print(f"    {nm:<10s} test_acc = {acc:.3f}")

    print("\n[B] joint training, SEPARATE RPE per discipline (combiner shared)...")
    t0 = time.time()
    B = _train_condition(
        discipline_names=DISCIPLINES, shared_rpe=False,
        n=n, dim=args.slot_dim, epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr, seed=22,
    )
    print(f"    wall = {time.time()-t0:.1f}s")
    for nm, acc in B["test_acc"].items():
        print(f"    {nm:<10s} test_acc = {acc:.3f}")

    print(f"\n[C] independent training per discipline ({args.n_seeds_u3} seeds)...")
    indep_runs = []
    for si in range(args.n_seeds_u3):
        seed_runs = {}
        for nm in DISCIPLINES:
            seed_runs[nm] = _train_condition(
                discipline_names=[nm], shared_rpe=False,
                n=n, dim=args.slot_dim, epochs=args.epochs,
                batches_per_epoch=args.batches_per_epoch,
                batch_size=args.batch_size, lr=args.lr,
                seed=100 + si,
            )
        indep_runs.append(seed_runs)
        accs = {nm: seed_runs[nm]["test_acc"][nm] for nm in DISCIPLINES}
        print(f"    seed {si}: " + "  ".join(
            f"{nm}={accs[nm]:.3f}" for nm in DISCIPLINES))

    print("\n[U3] Procrustes-aligned cosine similarity between RPEs:")
    u3_rows = []
    for si in range(args.n_seeds_u3):
        tabs = {nm: _rpe_table(indep_runs[si][nm]["rpes"][nm], n, args.slot_dim)
                for nm in DISCIPLINES}
        for i, di in enumerate(DISCIPLINES):
            for dj in DISCIPLINES[i + 1:]:
                cos = _procrustes_align(tabs[di], tabs[dj])
                u3_rows.append({"seed": si, "pair": f"{di}-{dj}",
                                "cos": cos})
                print(f"    seed {si} {di:<9s} <-> {dj:<9s}: cos={cos:.3f}")

    null_cos = []
    torch.manual_seed(31337)
    for _ in range(8):
        r1 = torch.randn(2 * n, args.slot_dim, device=DEVICE)
        r2 = torch.randn(2 * n, args.slot_dim, device=DEVICE)
        null_cos.append(_procrustes_align(r1, r2))
    null_mean = sum(null_cos) / len(null_cos)
    print(f"    [random baseline] cos = {null_mean:.3f}")
    u3_mean = sum(r["cos"] for r in u3_rows) / max(len(u3_rows), 1)

    print("\n[U4] frozen-operator transfer (geometry -> biology)...")
    src_rpe = indep_runs[0]["geometry"]["rpes"]["geometry"]
    src_comb = indep_runs[0]["geometry"]["combiner"]
    # Non-abelian transfer requires the slot bundle to align to a
    # richer operator structure (2n group elements vs n in the
    # abelian case). Give it 2x the training budget compared to
    # the in-domain joint training to compensate.
    transfer = _train_frozen_transfer(
        src_rpe, src_comb, n=n, dim=args.slot_dim,
        epochs=args.epochs * 2,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr, seed=999,
    )
    print(f"    biology test_acc (frozen op) = {transfer['clean_test_acc']:.3f}")

    print("\n[U5] permuted-slot negative control...")
    transfer_neg = _train_frozen_transfer(
        src_rpe, src_comb, n=n, dim=args.slot_dim,
        epochs=args.epochs * 2,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr, seed=998,
        permute_slots=True,
    )
    print(f"    biology test_acc (permuted slots) = "
          f"{transfer_neg['permuted_test_acc']:.3f}  (random=1/{n}={1.0/n:.3f})")

    a_min = min(A["test_acc"].values())
    b_min = min(B["test_acc"].values())
    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "group": f"D_{n}", "group_order": 2 * n,
        "A_joint_shared_acc": A["test_acc"],
        "B_joint_separate_acc": B["test_acc"],
        "C_independent_acc": [
            {nm: r[nm]["test_acc"][nm] for nm in DISCIPLINES}
            for r in indep_runs
        ],
        "U3_pairs": u3_rows,
        "U3_mean_procrustes_cos": u3_mean,
        "U3_random_baseline": null_mean,
        "U4_frozen_transfer_acc": transfer["clean_test_acc"],
        "U5_permuted_transfer_acc": transfer_neg["permuted_test_acc"],
    }
    summary["verdict"] = {
        "U1_pass": a_min >= 0.95,
        "U2_pass": a_min >= b_min - 0.03,
        "U3_pass": u3_mean >= 0.85 and u3_mean >= null_mean + 0.20,
        "U4_pass": transfer["clean_test_acc"] >= 0.90,
        "U5_pass": transfer_neg["permuted_test_acc"] <= 0.20,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print(f"  F62b D_{n} non-abelian verdict:")
    print("=" * 76)
    v = summary["verdict"]
    print(f"  U1 joint-shared works               : min_acc={a_min:.3f}  "
          f"[{'PASS' if v['U1_pass'] else 'FAIL'}]")
    print(f"  U2 sharing has no penalty           : "
          f"shared={a_min:.3f} sep={b_min:.3f}  "
          f"[{'PASS' if v['U2_pass'] else 'FAIL'}]")
    print(f"  U3 indep RPEs align (Procrustes)    : "
          f"trained={u3_mean:.3f} random={null_mean:.3f}  "
          f"[{'PASS' if v['U3_pass'] else 'FAIL'}]")
    print(f"  U4 frozen-op transfer to new disc.  : "
          f"acc={transfer['clean_test_acc']:.3f}  "
          f"[{'PASS' if v['U4_pass'] else 'FAIL'}]")
    print(f"  U5 permuted-slot control fails      : "
          f"acc={transfer_neg['permuted_test_acc']:.3f}  "
          f"[{'PASS' if v['U5_pass'] else 'FAIL'}]")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
