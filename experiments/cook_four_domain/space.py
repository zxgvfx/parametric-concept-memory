"""§6.2 space domain — dual move + dist on 5×5 grid."""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from pcm import ConceptGraph, GraphEvaluator
from pcm.heads import HeadAsBackbone, make_cook_subgraph

from ._common import DEVICE, rho_linear


__all__ = ["train_space"]


def train_space(use_cook: bool, seed: int, epochs: int, steps: int) -> dict:
    from experiments.space_concept_study import (
        DistanceHead, EMBED_DIM, FACET_DIST, FACET_MOVE,
        MoveHead, N_COLS, N_ROWS,
        cid_of, enumerate_distance_triples, enumerate_move_triples,
        l1_dist, rc_of_idx,
    )

    torch.manual_seed(seed)
    rng = random.Random(seed)
    move_triples = enumerate_move_triples()
    dist_triples = enumerate_distance_triples()

    cg = ConceptGraph(feat_dim=EMBED_DIM, initial_capacity=64)
    cids = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            cid = cid_of(r, c)
            cg.register_concept(node_id=cid, label=cid, scope="BASE",
                                provenance=f"space_cook:r={r},c={c}")
            cids.append(cid)

    head_move = MoveHead().to(DEVICE)
    head_dist = DistanceHead().to(DEVICE)
    with torch.no_grad():
        head_move(cids, cids, cg)
        head_dist(cids, cids, cg)
    cg.bundles_to(torch.device(DEVICE))

    if use_cook:
        ev = GraphEvaluator(cg)
        make_cook_subgraph(
            cg, node_id="muscle.cook.space_move",
            inputs=["ids_a", "ids_b"],
            facet_collapses=[("ids_a", FACET_MOVE), ("ids_b", FACET_MOVE)],
            backbone=HeadAsBackbone(head_move),
            backbone_key="space_move_mlp", evaluator=ev,
        )
        make_cook_subgraph(
            cg, node_id="muscle.cook.space_dist",
            inputs=["ids_a", "ids_b"],
            facet_collapses=[("ids_a", FACET_DIST), ("ids_b", FACET_DIST)],
            backbone=HeadAsBackbone(head_dist),
            backbone_key="space_dist_mlp", evaluator=ev,
        )

    params = (list(head_move.parameters()) + list(head_dist.parameters())
              + list(cg.iter_bundle_parameters()))
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    cg.register_optimizer(opt)

    for epoch in range(1, epochs + 1):
        head_move.train(); head_dist.train()
        for step in range(steps):
            batch = [move_triples[rng.randrange(len(move_triples))] for _ in range(32)]
            ids_a = [cid_of(*t[0]) for t in batch]
            ids_b = [cid_of(*t[1]) for t in batch]
            tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
            if use_cook:
                pred = ev.eval(
                    "muscle.cook.space_move",
                    bindings={"ids_a": ids_a, "ids_b": ids_b},
                    tick=epoch * 10000 + step,
                )
            else:
                pred = head_move(ids_a, ids_b, cg, tick=epoch * 10000 + step)
            loss_m = F.cross_entropy(pred, tgt)

            batch2 = [dist_triples[rng.randrange(len(dist_triples))] for _ in range(32)]
            ids_a2 = [cid_of(*t[0]) for t in batch2]
            ids_b2 = [cid_of(*t[1]) for t in batch2]
            tgt2 = torch.tensor([t[2] for t in batch2], device=DEVICE)
            if use_cook:
                logits = ev.eval(
                    "muscle.cook.space_dist",
                    bindings={"ids_a": ids_a2, "ids_b": ids_b2},
                    tick=epoch * 10000 + step,
                )
            else:
                logits = head_dist(ids_a2, ids_b2, cg, tick=epoch * 10000 + step)
            loss_d = F.cross_entropy(logits, tgt2)
            opt.zero_grad(); (loss_m + loss_d).backward(); opt.step()

    head_move.eval(); head_dist.eval()
    hits = 0
    with torch.no_grad():
        for i in range(0, len(move_triples), 64):
            batch = move_triples[i:i + 64]
            ids_a = [cid_of(*t[0]) for t in batch]
            ids_b = [cid_of(*t[1]) for t in batch]
            tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
            if use_cook:
                pred = ev.eval("muscle.cook.space_move",
                               bindings={"ids_a": ids_a, "ids_b": ids_b})
            else:
                pred = head_move(ids_a, ids_b, cg)
            hits += pred.argmax(-1).eq(tgt).sum().item()

    bs = {cid: {k: v.detach().cpu()
                for k, v in cg.concepts[cid].bundle.state_dict().items()}
          for cid in cids}
    rho_L1 = rho_linear(
        bs, cids, FACET_MOVE,
        lambda i, j: -l1_dist(rc_of_idx(i), rc_of_idx(j)),
    )
    return {"acc": hits / len(move_triples), "rho_L1": rho_L1}
