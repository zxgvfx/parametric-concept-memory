"""§6.3 phoneme domain — triple voicing + manner + place on 20 phonemes."""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from pcm import ConceptGraph, GraphEvaluator
from pcm.heads import HeadAsBackbone, make_cook_subgraph

from ._common import DEVICE, rho_linear


__all__ = ["train_phoneme"]


def train_phoneme(use_cook: bool, seed: int, epochs: int, steps: int) -> dict:
    from experiments.phoneme_concept_study import (
        FACET_M, FACET_P, FACET_V, N_PH,
        build_manner_head, build_place_head, build_voicing_head,
        cid_of, feat_of,
    )

    torch.manual_seed(seed)
    rng = random.Random(seed)

    cg = ConceptGraph(initial_capacity=32)
    cids = [cid_of(i) for i in range(N_PH)]
    for cid in cids:
        cg.register_concept(node_id=cid, label=cid, scope="BASE",
                            provenance=f"phoneme_cook:{cid}")

    head_v = build_voicing_head().to(DEVICE)
    head_m = build_manner_head().to(DEVICE)
    head_p = build_place_head().to(DEVICE)
    with torch.no_grad():
        head_v(cids, cg); head_m(cids, cg); head_p(cids, cg)
    cg.bundles_to(torch.device(DEVICE))

    if use_cook:
        ev = GraphEvaluator(cg)
        for nid_suffix, head, facet in [
            ("voice", head_v, FACET_V),
            ("manner", head_m, FACET_M),
            ("place", head_p, FACET_P),
        ]:
            make_cook_subgraph(
                cg, node_id=f"muscle.cook.phoneme_{nid_suffix}",
                inputs=["ids"],
                facet_collapses=[("ids", facet)],
                backbone=HeadAsBackbone(head),
                backbone_key=f"phoneme_{nid_suffix}_mlp", evaluator=ev,
            )

    params: list = []
    for h in (head_v, head_m, head_p):
        params += list(h.parameters())
    params += list(cg.iter_bundle_parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    cg.register_optimizer(opt)

    for epoch in range(1, epochs + 1):
        head_v.train(); head_m.train(); head_p.train()
        for step in range(steps):
            idx_batch = [rng.randrange(N_PH) for _ in range(32)]
            ids = [cid_of(i) for i in idx_batch]
            t_total = 0.0
            for axis, head, name in [
                (0, head_v, "voice"),
                (1, head_m, "manner"),
                (2, head_p, "place"),
            ]:
                tgt = torch.tensor([feat_of(i)[axis] for i in idx_batch],
                                   device=DEVICE)
                if use_cook:
                    logits = ev.eval(
                        f"muscle.cook.phoneme_{name}",
                        bindings={"ids": ids},
                        tick=epoch * 10000 + step,
                    )
                else:
                    logits = head(ids, cg, tick=epoch * 10000 + step)
                t_total = t_total + F.cross_entropy(logits, tgt)
            opt.zero_grad(); t_total.backward(); opt.step()

    head_v.eval(); head_m.eval(); head_p.eval()
    accs: dict[str, float] = {}
    with torch.no_grad():
        for axis, head, name in [
            (0, head_v, "voice"),
            (1, head_m, "manner"),
            (2, head_p, "place"),
        ]:
            if use_cook:
                logits = ev.eval(f"muscle.cook.phoneme_{name}",
                                 bindings={"ids": cids})
            else:
                logits = head(cids, cg)
            tgt = torch.tensor([feat_of(i)[axis] for i in range(N_PH)],
                               device=DEVICE)
            accs[name] = float(logits.argmax(-1).eq(tgt).sum().item()) / N_PH

    bs = {cid: {k: v.detach().cpu()
                for k, v in cg.concepts[cid].bundle.state_dict().items()}
          for cid in cids}
    rho_v = rho_linear(
        bs, cids, FACET_V,
        lambda i, j: 1.0 if feat_of(i)[0] == feat_of(j)[0] else 0.0,
    )
    return {"accs": accs, "rho_same_v": rho_v}
