"""Number-domain training + per-seed run for paper §B.1.

Trains a dual-muscle (AddHead + CmpHead) model on
``concept:ans:1..7`` with random-orthogonal centroids, then sweeps the
four swap conditions (baseline, arith-only, ord-only, both) and
reports per-condition accuracy on the full enumerated pair set,
grouped by "involves swap pair" vs "not involving".
"""
from __future__ import annotations

import random
import time

import torch
import torch.nn.functional as F

from pcm.concept_graph import ConceptGraph
from pcm.heads.arithmetic_head_v2 import ArithmeticHeadV2
from pcm.heads.comparison_head import ComparisonHead

from experiments.color_concept_study import make_random_orthogonal_centroids

from ._config import (
    DEVICE,
    NUM_BATCH_SIZE,
    NUM_BIAS_DIM,
    NUM_EMBED_DIM,
    NUM_EPOCHS,
    NUM_LR,
    NUM_N_MAX,
    NUM_N_MIN,
    NUM_ORD_DIM,
    NUM_STEPS_PER_EPOCH,
    NUM_SWAP_A,
    NUM_SWAP_B,
)
from .swap_ops import swap_all_facets, swap_bundle_facet


__all__ = [
    "build_number_graph",
    "train_number_dual",
    "eval_number_add",
    "eval_number_cmp",
    "run_number_seed",
]


# ─── Graph builder + samplers ───────────────────────────────────────


def build_number_graph() -> ConceptGraph:
    cg = ConceptGraph(feat_dim=NUM_EMBED_DIM)
    for n in range(NUM_N_MIN, NUM_N_MAX + 1):
        cg.register_concept(
            node_id=f"concept:ans:{n}",
            label=f"ANS_{n}",
            scope="BASE",
            provenance=f"cf_swap:n={n}",
        )
    return cg


def _sample_arith(rng: random.Random, bs: int):
    a_l, b_l, op_l, c_l = [], [], [], []
    for _ in range(bs):
        op = "add" if rng.random() < 0.5 else "sub"
        if op == "add":
            a = rng.randint(NUM_N_MIN, NUM_N_MAX)
            b = rng.randint(NUM_N_MIN, NUM_N_MAX - a + NUM_N_MIN)
            while a + b > NUM_N_MAX or a + b < NUM_N_MIN:
                a = rng.randint(NUM_N_MIN, NUM_N_MAX)
                b = rng.randint(NUM_N_MIN, NUM_N_MAX)
            c = a + b
        else:
            a = rng.randint(NUM_N_MIN + 1, NUM_N_MAX)
            b = rng.randint(NUM_N_MIN, a - 1)
            c = a - b
        a_l.append(a); b_l.append(b); op_l.append(op); c_l.append(c)
    return a_l, b_l, op_l, c_l


def _sample_cmp(rng: random.Random, bs: int):
    a_l, b_l, lab = [], [], []
    for _ in range(bs):
        a = rng.randint(NUM_N_MIN, NUM_N_MAX)
        b = rng.randint(NUM_N_MIN, NUM_N_MAX)
        a_l.append(a); b_l.append(b)
        lab.append(0 if a < b else (1 if a == b else 2))
    return a_l, b_l, lab


def _op_onehot(ops: list[str]) -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0] if o == "add" else [0.0, 1.0] for o in ops], device=DEVICE
    )


# ─── Training + eval ────────────────────────────────────────────────


def train_number_dual(
    seed: int,
    centroids: torch.Tensor,
    epochs: int = NUM_EPOCHS,
    steps_per_epoch: int = NUM_STEPS_PER_EPOCH,
) -> dict:
    """Train an AddHead + CmpHead dual-muscle model from scratch."""
    torch.manual_seed(seed)
    rng = random.Random(seed)

    cg = build_number_graph()
    head_add = ArithmeticHeadV2(embed_dim=NUM_EMBED_DIM, bias_dim=NUM_BIAS_DIM).to(DEVICE)
    head_cmp = ComparisonHead(
        embed_dim=NUM_EMBED_DIM, facet_dim=NUM_ORD_DIM, hidden_dim=64
    ).to(DEVICE)

    with torch.no_grad():
        for n in range(NUM_N_MIN, NUM_N_MAX + 1):
            c = cg.concepts[f"concept:ans:{n}"]
            c.collapse("ArithmeticHeadV2", "arithmetic_bias",
                       (NUM_BIAS_DIM,), tick=0, device=DEVICE)
            c.collapse("ComparisonHead", "ordinal_offset",
                       (NUM_ORD_DIM,), tick=0, device=DEVICE)
    cg.bundles_to(torch.device(DEVICE))

    params = (
        list(head_add.parameters())
        + list(head_cmp.parameters())
        + list(cg.iter_bundle_parameters())
    )
    opt = torch.optim.AdamW(params, lr=NUM_LR, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        head_add.train(); head_cmp.train()
        for step in range(steps_per_epoch):
            a_l, b_l, op_l, c_l = _sample_arith(rng, NUM_BATCH_SIZE)
            ids_a = [f"concept:ans:{n}" for n in a_l]
            ids_b = [f"concept:ans:{n}" for n in b_l]
            op = _op_onehot(op_l)
            tgt = torch.tensor([c - NUM_N_MIN for c in c_l], device=DEVICE)
            dummy = torch.zeros(NUM_BATCH_SIZE, NUM_EMBED_DIM, device=DEVICE)
            pred = head_add(dummy, dummy, op, ids_a, ids_b, cg,
                            tick=epoch * 10000 + step)
            loss_a = F.cross_entropy(pred @ centroids.t(), tgt)

            ca_l, cb_l, clab = _sample_cmp(rng, NUM_BATCH_SIZE)
            cids_a = [f"concept:ans:{n}" for n in ca_l]
            cids_b = [f"concept:ans:{n}" for n in cb_l]
            ctgt = torch.tensor(clab, device=DEVICE)
            logits_c = head_cmp(None, None, cids_a, cids_b, cg,
                                tick=epoch * 10000 + step)
            loss_c = F.cross_entropy(logits_c, ctgt)

            opt.zero_grad()
            (loss_a + loss_c).backward()
            opt.step()

    head_add.eval(); head_cmp.eval()
    return {"cg": cg, "head_add": head_add, "head_cmp": head_cmp,
            "centroids": centroids}


def _involves(a: int, b: int, swap_ns: tuple[int, int]) -> bool:
    return a in swap_ns or b in swap_ns


@torch.no_grad()
def eval_number_add(
    cg: ConceptGraph,
    head_add: ArithmeticHeadV2,
    centroids: torch.Tensor,
    swap_ns: tuple[int, int],
) -> dict:
    """Enumerate all (a, b, op) with c=a±b ∈ [N_MIN, N_MAX]; group by
    involvement with the swap pair.
    """
    head_add.eval()
    records = {"all": [0, 0], "involving_swap": [0, 0], "not_involving": [0, 0]}
    triples: list[tuple[int, int, str, int]] = []
    for a in range(NUM_N_MIN, NUM_N_MAX + 1):
        for b in range(NUM_N_MIN, NUM_N_MAX + 1):
            for op in ("add", "sub"):
                c = a + b if op == "add" else a - b
                if c < NUM_N_MIN or c > NUM_N_MAX:
                    continue
                triples.append((a, b, op, c))
    BS = 64
    for i in range(0, len(triples), BS):
        batch = triples[i:i + BS]
        ids_a = [f"concept:ans:{t[0]}" for t in batch]
        ids_b = [f"concept:ans:{t[1]}" for t in batch]
        op = _op_onehot([t[2] for t in batch])
        tgt = torch.tensor([t[3] - NUM_N_MIN for t in batch], device=DEVICE)
        dummy = torch.zeros(len(batch), NUM_EMBED_DIM, device=DEVICE)
        pred = head_add(dummy, dummy, op, ids_a, ids_b, cg)
        hit = (pred @ centroids.t()).argmax(-1).eq(tgt)
        for t, h in zip(batch, hit.tolist()):
            key = "involving_swap" if _involves(t[0], t[1], swap_ns) else "not_involving"
            records[key][0] += int(h); records[key][1] += 1
            records["all"][0] += int(h); records["all"][1] += 1
    return {k: {"acc": v[0] / max(v[1], 1), "n": v[1]} for k, v in records.items()}


@torch.no_grad()
def eval_number_cmp(
    cg: ConceptGraph,
    head_cmp: ComparisonHead,
    swap_ns: tuple[int, int],
) -> dict:
    head_cmp.eval()
    records = {"all": [0, 0], "involving_swap": [0, 0], "not_involving": [0, 0]}
    triples: list[tuple[int, int, int]] = []
    for a in range(NUM_N_MIN, NUM_N_MAX + 1):
        for b in range(NUM_N_MIN, NUM_N_MAX + 1):
            lab = 0 if a < b else (1 if a == b else 2)
            triples.append((a, b, lab))
    BS = 64
    for i in range(0, len(triples), BS):
        batch = triples[i:i + BS]
        ids_a = [f"concept:ans:{t[0]}" for t in batch]
        ids_b = [f"concept:ans:{t[1]}" for t in batch]
        tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
        logits = head_cmp(None, None, ids_a, ids_b, cg)
        hit = logits.argmax(-1).eq(tgt)
        for t, h in zip(batch, hit.tolist()):
            key = "involving_swap" if _involves(t[0], t[1], swap_ns) else "not_involving"
            records[key][0] += int(h); records[key][1] += 1
            records["all"][0] += int(h); records["all"][1] += 1
    return {k: {"acc": v[0] / max(v[1], 1), "n": v[1]} for k, v in records.items()}


# ─── Per-seed run ───────────────────────────────────────────────────


def run_number_seed(seed: int, epochs: int, steps_per_epoch: int) -> dict:
    swap_ns = (NUM_SWAP_A, NUM_SWAP_B)
    cid_a = f"concept:ans:{NUM_SWAP_A}"
    cid_b = f"concept:ans:{NUM_SWAP_B}"
    centroids = make_random_orthogonal_centroids(
        NUM_N_MAX - NUM_N_MIN + 1, NUM_EMBED_DIM, seed
    )
    t0 = time.time()
    trained = train_number_dual(seed, centroids, epochs, steps_per_epoch)
    cg = trained["cg"]; head_add = trained["head_add"]; head_cmp = trained["head_cmp"]

    base_add = eval_number_add(cg, head_add, centroids, swap_ns)
    base_cmp = eval_number_cmp(cg, head_cmp, swap_ns)

    swap_bundle_facet(cg, cid_a, cid_b, "arithmetic_bias")
    A_add = eval_number_add(cg, head_add, centroids, swap_ns)
    A_cmp = eval_number_cmp(cg, head_cmp, swap_ns)
    swap_bundle_facet(cg, cid_a, cid_b, "arithmetic_bias")  # undo

    swap_bundle_facet(cg, cid_a, cid_b, "ordinal_offset")
    B_add = eval_number_add(cg, head_add, centroids, swap_ns)
    B_cmp = eval_number_cmp(cg, head_cmp, swap_ns)
    swap_bundle_facet(cg, cid_a, cid_b, "ordinal_offset")

    swap_all_facets(cg, cid_a, cid_b)
    C_add = eval_number_add(cg, head_add, centroids, swap_ns)
    C_cmp = eval_number_cmp(cg, head_cmp, swap_ns)
    swap_all_facets(cg, cid_a, cid_b)

    return {
        "seed": seed,
        "swap_pair": list(swap_ns),
        "baseline": {"add": base_add, "cmp": base_cmp},
        "swap_arith_only": {"add": A_add, "cmp": A_cmp},
        "swap_ord_only":   {"add": B_add, "cmp": B_cmp},
        "swap_both":       {"add": C_add, "cmp": C_cmp},
        "wall_s": time.time() - t0,
    }
