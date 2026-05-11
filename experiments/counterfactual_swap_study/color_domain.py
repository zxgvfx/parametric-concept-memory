"""Color-domain training + per-seed run for paper §B.2.

Trains a dual-muscle (MixHead + AdjHead) model on
``concept:color:0..11``; sweeps the 4 swap conditions; reports per-
condition accuracy on the enumerated mixing/adjacency triples.

Reuses the heavy lifting (``ColorMixingHead``, ``ColorAdjacencyHead``,
``build_color_graph``, ``enumerate_mixing_triples``,
``enumerate_adjacency_triples``, ``make_random_orthogonal_centroids``)
from :mod:`experiments.color_concept_study` so the two studies stay in
lock-step on dimensions, batch size, and training schedule.
"""
from __future__ import annotations

import random
import time

import torch
import torch.nn.functional as F

from pcm.concept_graph import ConceptGraph

from experiments.color_concept_study import (
    ADJ_DIM,
    BIAS_DIM as COLOR_BIAS_DIM,
    CALLER_ADJ,
    CALLER_MIX,
    EMBED_DIM as COLOR_EMBED_DIM,
    FACET_ADJ,
    FACET_MIX,
    N_COLORS,
    ColorAdjacencyHead,
    ColorMixingHead,
    build_color_graph,
    enumerate_adjacency_triples,
    enumerate_mixing_triples,
    make_random_orthogonal_centroids,
)

from ._config import COLOR_SWAP_A, COLOR_SWAP_B, DEVICE
from .swap_ops import swap_all_facets, swap_bundle_facet


__all__ = [
    "train_color_dual",
    "eval_color_mix",
    "eval_color_adj",
    "run_color_seed",
]


def train_color_dual(
    seed: int,
    centroids: torch.Tensor,
    epochs: int = 30,
    steps_per_epoch: int = 200,
) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    mix_triples = enumerate_mixing_triples(N_COLORS)
    adj_triples = enumerate_adjacency_triples(N_COLORS)

    cg = build_color_graph(N_COLORS)
    head_mix = ColorMixingHead().to(DEVICE)
    head_adj = ColorAdjacencyHead().to(DEVICE)

    with torch.no_grad():
        for i in range(N_COLORS):
            c = cg.concepts[f"concept:color:{i}"]
            c.collapse(CALLER_MIX, FACET_MIX, (COLOR_BIAS_DIM,),
                       tick=0, device=DEVICE, init="normal_small")
            c.collapse(CALLER_ADJ, FACET_ADJ, (ADJ_DIM,),
                       tick=0, device=DEVICE, init="normal_small")
    cg.bundles_to(torch.device(DEVICE))

    params = (
        list(head_mix.parameters())
        + list(head_adj.parameters())
        + list(cg.iter_bundle_parameters())
    )
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        head_mix.train(); head_adj.train()
        for step_i in range(steps_per_epoch):
            batch = [mix_triples[rng.randrange(len(mix_triples))]
                     for _ in range(32)]
            ids_a = [f"concept:color:{t[0]}" for t in batch]
            ids_b = [f"concept:color:{t[1]}" for t in batch]
            tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
            pred = head_mix(ids_a, ids_b, cg, tick=epoch * 10000 + step_i)
            loss_m = F.cross_entropy(pred @ centroids.t(), tgt)

            batch2 = [adj_triples[rng.randrange(len(adj_triples))]
                      for _ in range(32)]
            ids_a2 = [f"concept:color:{t[0]}" for t in batch2]
            ids_b2 = [f"concept:color:{t[1]}" for t in batch2]
            tgt2 = torch.tensor([t[2] for t in batch2], device=DEVICE)
            logits = head_adj(ids_a2, ids_b2, cg, tick=epoch * 10000 + step_i)
            loss_a = F.cross_entropy(logits, tgt2)

            opt.zero_grad()
            (loss_m + loss_a).backward()
            opt.step()

    head_mix.eval(); head_adj.eval()
    return {"cg": cg, "head_mix": head_mix, "head_adj": head_adj,
            "centroids": centroids}


def _involves(a: int, b: int, swap_ns: tuple[int, int]) -> bool:
    return a in swap_ns or b in swap_ns


@torch.no_grad()
def eval_color_mix(
    cg: ConceptGraph,
    head_mix: ColorMixingHead,
    centroids: torch.Tensor,
    swap_ns: tuple[int, int],
) -> dict:
    head_mix.eval()
    triples = enumerate_mixing_triples(N_COLORS)
    records = {"all": [0, 0], "involving_swap": [0, 0], "not_involving": [0, 0]}
    BS = 64
    for i in range(0, len(triples), BS):
        batch = triples[i:i + BS]
        ids_a = [f"concept:color:{t[0]}" for t in batch]
        ids_b = [f"concept:color:{t[1]}" for t in batch]
        tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
        pred = head_mix(ids_a, ids_b, cg)
        hit = (pred @ centroids.t()).argmax(-1).eq(tgt)
        for t, h in zip(batch, hit.tolist()):
            key = "involving_swap" if _involves(t[0], t[1], swap_ns) else "not_involving"
            records[key][0] += int(h); records[key][1] += 1
            records["all"][0] += int(h); records["all"][1] += 1
    return {k: {"acc": v[0] / max(v[1], 1), "n": v[1]} for k, v in records.items()}


@torch.no_grad()
def eval_color_adj(
    cg: ConceptGraph,
    head_adj: ColorAdjacencyHead,
    swap_ns: tuple[int, int],
) -> dict:
    head_adj.eval()
    triples = enumerate_adjacency_triples(N_COLORS)
    records = {"all": [0, 0], "involving_swap": [0, 0], "not_involving": [0, 0]}
    BS = 64
    for i in range(0, len(triples), BS):
        batch = triples[i:i + BS]
        ids_a = [f"concept:color:{t[0]}" for t in batch]
        ids_b = [f"concept:color:{t[1]}" for t in batch]
        tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
        logits = head_adj(ids_a, ids_b, cg)
        hit = logits.argmax(-1).eq(tgt)
        for t, h in zip(batch, hit.tolist()):
            key = "involving_swap" if _involves(t[0], t[1], swap_ns) else "not_involving"
            records[key][0] += int(h); records[key][1] += 1
            records["all"][0] += int(h); records["all"][1] += 1
    return {k: {"acc": v[0] / max(v[1], 1), "n": v[1]} for k, v in records.items()}


def run_color_seed(seed: int, epochs: int, steps_per_epoch: int) -> dict:
    swap_ns = (COLOR_SWAP_A, COLOR_SWAP_B)
    cid_a = f"concept:color:{COLOR_SWAP_A}"
    cid_b = f"concept:color:{COLOR_SWAP_B}"
    centroids = make_random_orthogonal_centroids(N_COLORS, COLOR_EMBED_DIM, seed)
    t0 = time.time()
    trained = train_color_dual(seed, centroids, epochs, steps_per_epoch)
    cg = trained["cg"]; head_mix = trained["head_mix"]; head_adj = trained["head_adj"]

    base_mix = eval_color_mix(cg, head_mix, centroids, swap_ns)
    base_adj = eval_color_adj(cg, head_adj, swap_ns)

    swap_bundle_facet(cg, cid_a, cid_b, FACET_MIX)
    A_mix = eval_color_mix(cg, head_mix, centroids, swap_ns)
    A_adj = eval_color_adj(cg, head_adj, swap_ns)
    swap_bundle_facet(cg, cid_a, cid_b, FACET_MIX)

    swap_bundle_facet(cg, cid_a, cid_b, FACET_ADJ)
    B_mix = eval_color_mix(cg, head_mix, centroids, swap_ns)
    B_adj = eval_color_adj(cg, head_adj, swap_ns)
    swap_bundle_facet(cg, cid_a, cid_b, FACET_ADJ)

    swap_all_facets(cg, cid_a, cid_b)
    C_mix = eval_color_mix(cg, head_mix, centroids, swap_ns)
    C_adj = eval_color_adj(cg, head_adj, swap_ns)
    swap_all_facets(cg, cid_a, cid_b)

    return {
        "seed": seed,
        "swap_pair": list(swap_ns),
        "baseline":      {"mix": base_mix, "adj": base_adj},
        "swap_mix_only": {"mix": A_mix,    "adj": A_adj},
        "swap_adj_only": {"mix": B_mix,    "adj": B_adj},
        "swap_both":     {"mix": C_mix,    "adj": C_adj},
        "wall_s": time.time() - t0,
    }
