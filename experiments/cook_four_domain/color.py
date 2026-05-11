"""§5 color domain — dual mix + adj on 12 hues."""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from pcm import ConceptGraph, GraphEvaluator
from pcm.heads import HeadAsBackbone, make_cook_subgraph

from ._common import DEVICE, rho_linear


__all__ = ["train_color"]


def train_color(use_cook: bool, seed: int, epochs: int, steps: int) -> dict:
    from experiments.color_concept_study import (
        ADJ_DIM, BIAS_DIM, ColorAdjacencyHead, ColorMixingHead,
        EMBED_DIM, FACET_ADJ, FACET_MIX, N_COLORS,
        circular_dist, enumerate_adjacency_triples, enumerate_mixing_triples,
        make_random_orthogonal_centroids,
    )

    torch.manual_seed(seed)
    rng = random.Random(seed)
    mix_triples = enumerate_mixing_triples(N_COLORS)
    adj_triples = enumerate_adjacency_triples(N_COLORS)

    cg = ConceptGraph(feat_dim=EMBED_DIM, initial_capacity=16)
    cids = []
    for i in range(N_COLORS):
        cid = f"concept:color:{i}"
        cg.register_concept(node_id=cid, label=cid, scope="BASE",
                            provenance=f"color_cook:hue={i}")
        cids.append(cid)

    head_mix = ColorMixingHead().to(DEVICE)
    head_adj = ColorAdjacencyHead().to(DEVICE)
    with torch.no_grad():
        head_mix(cids, cids, cg)
        head_adj(cids, cids, cg)
    cg.bundles_to(torch.device(DEVICE))
    centroids = make_random_orthogonal_centroids(N_COLORS, EMBED_DIM, seed)

    if use_cook:
        ev = GraphEvaluator(cg)
        make_cook_subgraph(
            cg, node_id="muscle.cook.color_mix",
            inputs=["ids_a", "ids_b"],
            facet_collapses=[("ids_a", FACET_MIX), ("ids_b", FACET_MIX)],
            backbone=HeadAsBackbone(head_mix),
            backbone_key="color_mix_mlp", evaluator=ev,
        )
        make_cook_subgraph(
            cg, node_id="muscle.cook.color_adj",
            inputs=["ids_a", "ids_b"],
            facet_collapses=[("ids_a", FACET_ADJ), ("ids_b", FACET_ADJ)],
            backbone=HeadAsBackbone(head_adj),
            backbone_key="color_adj_mlp", evaluator=ev,
        )

    params = (list(head_mix.parameters()) + list(head_adj.parameters())
              + list(cg.iter_bundle_parameters()))
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    cg.register_optimizer(opt)

    for epoch in range(1, epochs + 1):
        head_mix.train(); head_adj.train()
        for step in range(steps):
            batch = [mix_triples[rng.randrange(len(mix_triples))] for _ in range(32)]
            ids_a = [f"concept:color:{t[0]}" for t in batch]
            ids_b = [f"concept:color:{t[1]}" for t in batch]
            tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
            if use_cook:
                pred = ev.eval(
                    "muscle.cook.color_mix",
                    bindings={"ids_a": ids_a, "ids_b": ids_b},
                    tick=epoch * 10000 + step,
                )
            else:
                pred = head_mix(ids_a, ids_b, cg, tick=epoch * 10000 + step)
            loss_m = F.cross_entropy(pred @ centroids.t(), tgt)

            batch2 = [adj_triples[rng.randrange(len(adj_triples))] for _ in range(32)]
            ids_a2 = [f"concept:color:{t[0]}" for t in batch2]
            ids_b2 = [f"concept:color:{t[1]}" for t in batch2]
            tgt2 = torch.tensor([t[2] for t in batch2], device=DEVICE)
            if use_cook:
                logits = ev.eval(
                    "muscle.cook.color_adj",
                    bindings={"ids_a": ids_a2, "ids_b": ids_b2},
                    tick=epoch * 10000 + step,
                )
            else:
                logits = head_adj(ids_a2, ids_b2, cg, tick=epoch * 10000 + step)
            loss_a = F.cross_entropy(logits, tgt2)
            opt.zero_grad(); (loss_m + loss_a).backward(); opt.step()

    head_mix.eval(); head_adj.eval()
    hits_mix = 0
    with torch.no_grad():
        for i in range(0, len(mix_triples), 64):
            batch = mix_triples[i:i + 64]
            ids_a = [f"concept:color:{t[0]}" for t in batch]
            ids_b = [f"concept:color:{t[1]}" for t in batch]
            tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
            if use_cook:
                pred = ev.eval("muscle.cook.color_mix",
                               bindings={"ids_a": ids_a, "ids_b": ids_b})
            else:
                pred = head_mix(ids_a, ids_b, cg)
            hits_mix += (pred @ centroids.t()).argmax(-1).eq(tgt).sum().item()

    bs = {cid: {k: v.detach().cpu()
                for k, v in cg.concepts[cid].bundle.state_dict().items()}
          for cid in cids}
    rho_circ = rho_linear(bs, cids, FACET_MIX,
                           lambda i, j: -circular_dist(i, j, N_COLORS))
    return {"acc": hits_mix / len(mix_triples), "rho_circular": rho_circ}
