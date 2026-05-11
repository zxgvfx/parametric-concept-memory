"""F55 — functional RPE vs PCM dual-process cook on number length-OOD.

The 2026 RoPE-as-phase-modulation paper (arxiv 2602.10959) and
the ALiBi 2022 paper both claim length extrapolation, but on
*sequence* tasks where OOD is "longer context window" and
correctness is implicit (next-token cross-entropy). For
*pair-input arithmetic* — predict the signed integer
``b - a`` given concept embeddings — the question is sharper:
does a continuous-function RPE generalise the **classifier**
beyond the displacement range it was trained on?

This experiment puts four heads on the same number domain and
the same length-OOD split:

* ``lookup``    — :class:`pcm.dual_channel.RelativePositionEmbedding`
                  (the F48 baseline; saturates within train range,
                  fails OOD).
* ``sinusoidal`` — :class:`pcm.dual_channel.SinusoidalRelativePositionEmbedding`
                  + a small ReLU classifier. Continuous basis, no
                  train-range cap, but classifier still has to
                  generalise.
* ``alibi``     — :class:`pcm.dual_channel.ALiBiRelativePositionBias`
                  + a per-class learned bias (no displacement-
                  specific capacity). Strong inductive bias toward
                  small Δ; expected to fail at hard arithmetic
                  but a useful "minimum-information" lower bound.
* ``cook``      — PCM v3 :class:`pcm.dual_process.IterativeDiffCook`
                  + :class:`pcm.dual_process.SuccessorHead`. The
                  System-2 procedural path (saturated K=99 in F51).

All four heads see the *same* inner training distribution
(|Δ| ≤ ``train_max``). All four are evaluated on the same outer
test grid (|Δ| up to ``n_total - 1``).

Falsifiability: the experiment falsifies the claim "functional
RPE is the right answer to PCM number length-OOD" if either
sinusoidal or alibi fails to match cook on the OOD K range.

Usage::

    python -m experiments.functional_rpe_vs_cook \\
        --n-seeds 3 --epochs 15 --steps-per-epoch 200 \\
        --n-total 100 --train-max 19 \\
        --out outputs/f55_functional_rpe_vs_cook
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcm.concept_graph import ConceptGraph
from pcm.dual_channel import (
    ALiBiRelativePositionBias,
    RelativePositionEmbedding,
    SinusoidalRelativePositionEmbedding,
    collapse_dual_channel,
    register_dual_channel_facet,
)
from pcm.dual_process import IterativeDiffCook, SuccessorHead


__all__ = ["main"]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SLOT_DIM = 16
ATTR_DIM = 8
BASE_FACET = "num"


# ─────────────────────────────────────────────────────────────────
# Heads
# ─────────────────────────────────────────────────────────────────


class LookupRPEHead(nn.Module):
    def __init__(self, n_total: int, embed_dim: int = 16):
        super().__init__()
        self.n_total = n_total
        self.rpe = RelativePositionEmbedding(
            ranges=[(-(n_total - 1), n_total - 1)], embed_dim=embed_dim,
        )
        self.cls = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Linear(64, 2 * n_total - 1),
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.cls(self.rpe(delta))


class SinusoidalHead(nn.Module):
    def __init__(self, n_total: int, embed_dim: int = 32):
        super().__init__()
        self.n_total = n_total
        self.rpe = SinusoidalRelativePositionEmbedding(
            n_axes=1, embed_dim=embed_dim,
        )
        self.cls = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(),
            nn.Linear(128, 2 * n_total - 1),
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.cls(self.rpe(delta))


class ALiBiHead(nn.Module):
    """ALiBi as a pair head requires a per-class bias map. We
    use a small linear from |Δ| to logits, regularised by the
    ALiBi linear-decay shape via additive bias."""

    def __init__(self, n_total: int):
        super().__init__()
        self.n_total = n_total
        self.alibi = ALiBiRelativePositionBias(n_axes=1)
        # Tiny per-class learned bias (linear function of |Δ|).
        n_classes = 2 * n_total - 1
        self.cls = nn.Linear(2, n_classes)  # (Δ, |Δ|) → logits

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        d = delta.float().unsqueeze(-1)
        ad = d.abs()
        x = torch.cat([d, ad], dim=-1)
        logits = self.cls(x)
        # ALiBi additive bias scaled to all classes.
        bias = self.alibi(delta)  # (B, 1)
        return logits + bias


# ─────────────────────────────────────────────────────────────────
# Per-seed runner
# ─────────────────────────────────────────────────────────────────


def _build_cg(n_total: int) -> tuple[ConceptGraph, list[str]]:
    cg = ConceptGraph(feat_dim=SLOT_DIM)
    cids = []
    for i in range(n_total):
        cid = f"concept:num:{i}"
        cg.register_concept(node_id=cid, label=f"NUM_{i}", scope="BASE",
                            provenance=f"f55:{i}")
        cids.append(cid)
    register_dual_channel_facet(
        cg, BASE_FACET, slot_dim=SLOT_DIM, attr_dim=ATTR_DIM,
    )
    return cg, cids


def _train_rpe_head(
    head: nn.Module, n_total: int, train_max: int, seed: int,
    epochs: int, steps_per_epoch: int, batch_size: int = 64, lr: float = 5e-3,
) -> None:
    """Train a Δ-input head on |Δ| ≤ train_max."""
    rng = random.Random(seed)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    head.train()
    for epoch in range(epochs):
        for _ in range(steps_per_epoch):
            batch_d = []
            for _ in range(batch_size):
                a = rng.randrange(n_total)
                d = rng.randint(-train_max, train_max)
                b = max(0, min(n_total - 1, a + d))
                batch_d.append(b - a)
            delta = torch.tensor(batch_d, device=DEVICE, dtype=torch.long)
            tgt = torch.tensor(
                [d + (n_total - 1) for d in batch_d],
                device=DEVICE,
            )
            logits = head(delta)
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()


def _eval_rpe_head(
    head: nn.Module, n_total: int, eval_Ks: list[int],
) -> dict[int, float]:
    acc: dict[int, float] = {}
    head.eval()
    with torch.no_grad():
        for K in eval_Ks:
            valid = [a for a in range(n_total) if 0 <= a + K < n_total]
            sample = valid[:min(40, len(valid))]
            if not sample:
                acc[K] = float("nan")
                continue
            deltas = torch.tensor(
                [K] * len(sample), device=DEVICE, dtype=torch.long,
            )
            tgt = torch.tensor(
                [K + (n_total - 1)] * len(sample), device=DEVICE,
            )
            preds = head(deltas).argmax(-1)
            acc[K] = float((preds == tgt).float().mean().item())
    return acc


def _train_cook_pipeline(
    seed: int, n_total: int, train_max: int,
    epochs: int, steps_per_epoch: int,
) -> tuple[IterativeDiffCook, dict[int, float]]:
    """Mirror of the F51 PoC: train SuccessorHead on |Δ|=1, build
    cook, evaluate at each eval_K."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    cg, cids = _build_cg(n_total)
    with torch.no_grad():
        collapse_dual_channel(
            cg, caller="warmup", base_facet=BASE_FACET, concept_ids=cids,
            slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
            tick=0, device=DEVICE,
        )
    succ = SuccessorHead(
        slot_dim=SLOT_DIM, attr_dim=ATTR_DIM, max_step=1, hidden=64,
    ).to(DEVICE)
    opt = torch.optim.AdamW(
        list(succ.parameters()) + list(cg.iter_bundle_parameters()),
        lr=5e-3, weight_decay=1e-4,
    )

    n_max = 1
    for epoch in range(epochs):
        succ.train()
        for step_i in range(steps_per_epoch):
            batch_a, batch_b, batch_step = [], [], []
            for _ in range(64):
                a = rng.randrange(n_total)
                d = rng.randint(-train_max, train_max)
                b = max(0, min(n_total - 1, a + d))
                actual_d = b - a
                if actual_d > 0:
                    s = min(actual_d, n_max)
                elif actual_d < 0:
                    s = max(actual_d, -n_max)
                else:
                    s = 0
                batch_a.append(a); batch_b.append(b); batch_step.append(s)
            ids_a = [cids[a] for a in batch_a]
            ids_b = [cids[b] for b in batch_b]
            slot_a, attr_a = collapse_dual_channel(
                cg, caller="succ-a", base_facet=BASE_FACET,
                concept_ids=ids_a,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=epoch * 10000 + step_i, device=DEVICE,
            )
            slot_b, attr_b = collapse_dual_channel(
                cg, caller="succ-b", base_facet=BASE_FACET,
                concept_ids=ids_b,
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
                tick=epoch * 10000 + step_i + 1, device=DEVICE,
            )
            tgt = torch.tensor([s + 1 for s in batch_step], device=DEVICE)
            logits = succ(slot_a, slot_b, attr_a, attr_b)
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
    succ.eval()

    def _lookup(idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cid = cids[idx]
        slot, attr = collapse_dual_channel(
            cg, caller="cook-lookup", base_facet=BASE_FACET,
            concept_ids=[cid],
            slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
            tick=99999, device=DEVICE,
        )
        return slot[0], attr[0]

    cook = IterativeDiffCook(
        successor_head=succ, identity_lookup=_lookup,
        max_iters=2 * n_total, cursor_min=0, cursor_max=n_total - 1,
        with_attr=True,
    )
    return cook, {}  # cook eval done in the main loop


def _run_one(
    seed: int, *, n_total: int, train_max: int,
    epochs: int, steps_per_epoch: int,
) -> dict:
    eval_Ks = [1, 5, 10, 19, 25, 35, 50, 70, 99]
    eval_Ks = [k for k in eval_Ks if k <= n_total - 1]

    out: dict = {"seed": seed}

    # Lookup RPE
    torch.manual_seed(seed)
    head_lookup = LookupRPEHead(n_total=n_total, embed_dim=16).to(DEVICE)
    t0 = time.time()
    _train_rpe_head(
        head_lookup, n_total, train_max, seed, epochs, steps_per_epoch,
    )
    out["lookup_acc"] = _eval_rpe_head(head_lookup, n_total, eval_Ks)
    out["lookup_wall_s"] = time.time() - t0

    # Sinusoidal RPE
    torch.manual_seed(seed)
    head_sin = SinusoidalHead(n_total=n_total, embed_dim=32).to(DEVICE)
    t0 = time.time()
    _train_rpe_head(
        head_sin, n_total, train_max, seed, epochs, steps_per_epoch,
    )
    out["sinusoidal_acc"] = _eval_rpe_head(head_sin, n_total, eval_Ks)
    out["sinusoidal_wall_s"] = time.time() - t0

    # ALiBi
    torch.manual_seed(seed)
    head_alibi = ALiBiHead(n_total=n_total).to(DEVICE)
    t0 = time.time()
    _train_rpe_head(
        head_alibi, n_total, train_max, seed, epochs, steps_per_epoch,
    )
    out["alibi_acc"] = _eval_rpe_head(head_alibi, n_total, eval_Ks)
    out["alibi_wall_s"] = time.time() - t0

    # Cook (PCM v3 dual-process)
    torch.manual_seed(seed)
    t0 = time.time()
    cook, _ = _train_cook_pipeline(
        seed, n_total, train_max, epochs, steps_per_epoch,
    )
    cook_acc: dict[int, float] = {}
    for K in eval_Ks:
        valid = [a for a in range(n_total) if 0 <= a + K < n_total]
        sample = valid[:min(40, len(valid))]
        hits = 0
        for a in sample:
            diff_pred, _ = cook(a, a + K)
            if diff_pred == K:
                hits += 1
        cook_acc[K] = hits / max(len(sample), 1)
    out["cook_acc"] = cook_acc
    out["cook_wall_s"] = time.time() - t0
    out["eval_Ks"] = eval_Ks
    return out


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-total", type=int, default=100)
    ap.add_argument("--train-max", type=int, default=19)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=99100)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f55_functional_rpe_vs_cook"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(
        f"  F55 functional RPE vs cook: N={args.n_total}, "
        f"train_max=|Δ|≤{args.train_max}, n_seeds={args.n_seeds}"
    )
    print("=" * 76)

    rows: list[dict] = []
    for si in range(args.n_seeds):
        seed = args.seed_base + si
        r = _run_one(
            seed, n_total=args.n_total, train_max=args.train_max,
            epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
        )
        rows.append(r)
        # Compact per-seed line.
        K_OOD = max(r["eval_Ks"])
        print(
            f"  [seed={seed}] K={K_OOD}  "
            f"lookup={r['lookup_acc'][K_OOD]:.3f}  "
            f"sin={r['sinusoidal_acc'][K_OOD]:.3f}  "
            f"alibi={r['alibi_acc'][K_OOD]:.3f}  "
            f"cook={r['cook_acc'][K_OOD]:.3f}"
        )

    # ─── aggregate ───
    all_Ks = sorted(set(rows[0]["eval_Ks"]))
    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "per_seed": rows,
        "by_head": {},
    }
    for head_name in ("lookup", "sinusoidal", "alibi", "cook"):
        per_K = {}
        for K in all_Ks:
            vals = [r[f"{head_name}_acc"][K] for r in rows
                    if isinstance(r[f"{head_name}_acc"].get(K), (int, float))]
            if not vals:
                continue
            m = sum(vals) / len(vals)
            sd = math.sqrt(
                sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)
            )
            per_K[K] = {"mean": m, "std": sd, "n": len(vals)}
        summary["by_head"][head_name] = per_K

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "═" * 76)
    print(f"  Cross-head accuracy curve (n_seeds={args.n_seeds}):")
    header = f"  {'K':>4s}"
    for h in ("lookup", "sinusoidal", "alibi", "cook"):
        header += f" {h:>14s}"
    print(header)
    for K in all_Ks:
        row = f"  {K:>4d}"
        for h in ("lookup", "sinusoidal", "alibi", "cook"):
            cell = summary["by_head"][h].get(K, {})
            m = cell.get("mean", float("nan"))
            sd = cell.get("std", float("nan"))
            row += f"  {m:+.3f}±{sd:.2f}"
        print(row)

    K_OOD = max(all_Ks)
    cook_OOD = summary["by_head"]["cook"][K_OOD]["mean"]
    sin_OOD = summary["by_head"]["sinusoidal"][K_OOD]["mean"]
    alibi_OOD = summary["by_head"]["alibi"][K_OOD]["mean"]
    print(
        f"\n  K={K_OOD} OOD verdict:  "
        f"cook={cook_OOD:.3f}  sin={sin_OOD:.3f}  alibi={alibi_OOD:.3f}"
    )
    if cook_OOD >= max(sin_OOD, alibi_OOD) + 0.20:
        print("  → cook is strictly stronger on length-OOD (≥ +0.20 over best functional RPE)")
    elif max(sin_OOD, alibi_OOD) >= cook_OOD + 0.20:
        print("  → functional RPE strictly stronger (rare expected outcome)")
    else:
        print("  → comparable; use task-specific factors (RT, simplicity) to choose")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
