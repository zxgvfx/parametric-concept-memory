"""§4 number domain — dual-muscle (ArithmeticHeadV2 + ComparisonHead, N=7).

Trains the same head twice (direct vs cook) under shared parameters and
identical RNG. The cook path runs through ``GraphEvaluator.eval`` over
two ``parametric_muscle_subgraph`` ConceptNodes.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from pcm import ConceptGraph, GraphEvaluator
from pcm.heads import HeadAsBackbone, make_cook_subgraph

from ._common import DEVICE, rho_linear


__all__ = ["train_number"]


def train_number(use_cook: bool, seed: int, epochs: int, steps: int) -> dict:
    from pcm.heads import ArithmeticHeadV2, ComparisonHead

    torch.manual_seed(seed)
    rng = random.Random(seed)

    cg = ConceptGraph(initial_capacity=16)
    cids = []
    for n in range(1, 8):
        cid = f"concept:ans:{n}"
        cg.register_concept(node_id=cid, label=cid, scope="BASE",
                            provenance=f"number_cook:n={n}")
        cids.append(cid)

    head_add = ArithmeticHeadV2(embed_dim=128, bias_dim=64).to(DEVICE)
    head_cmp = ComparisonHead(embed_dim=128, facet_dim=8, hidden_dim=64).to(DEVICE)

    op = torch.tensor([[1.0, 0.0]] * 7, device=DEVICE)
    zeros = torch.zeros(7, 128, device=DEVICE)
    with torch.no_grad():
        head_add(zeros, zeros, op, cids, cids, cg)
        head_cmp(None, None, cids, cids, cg)
    cg.bundles_to(torch.device(DEVICE))

    g = torch.Generator().manual_seed(seed + 1)
    A = torch.randn(128, 7, generator=g)
    Q, _ = torch.linalg.qr(A)
    centroids = F.normalize(Q.t(), dim=-1).to(DEVICE)

    if use_cook:
        ev = GraphEvaluator(cg)
        make_cook_subgraph(
            cg, node_id="muscle.cook.arith_v2",
            inputs=["ids_a", "ids_b", "op_onehot"],
            facet_collapses=[("ids_a", "arithmetic_bias"),
                             ("ids_b", "arithmetic_bias")],
            extra_inputs=["op_onehot"],
            backbone=HeadAsBackbone(head_add),
            backbone_key="arith_v2_mlp", evaluator=ev,
        )
        make_cook_subgraph(
            cg, node_id="muscle.cook.comparison",
            inputs=["ids_a", "ids_b"],
            facet_collapses=[("ids_a", "ordinal_offset"),
                             ("ids_b", "ordinal_offset")],
            backbone=HeadAsBackbone(head_cmp),
            backbone_key="comparison_mlp", evaluator=ev,
        )

    params = (list(head_add.parameters()) + list(head_cmp.parameters())
              + list(cg.iter_bundle_parameters()))
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    cg.register_optimizer(opt)

    def sample_arith(bs: int):
        a_l, b_l, op_l, c_l = [], [], [], []
        for _ in range(bs):
            o = "add" if rng.random() < 0.5 else "sub"
            if o == "add":
                a = rng.randint(1, 6); b = rng.randint(1, 7 - a); c = a + b
            else:
                a = rng.randint(2, 7); b = rng.randint(1, a - 1); c = a - b
            a_l.append(a); b_l.append(b); op_l.append(o); c_l.append(c)
        return a_l, b_l, op_l, c_l

    def sample_cmp(bs: int):
        a_l, b_l, lab = [], [], []
        for _ in range(bs):
            a = rng.randint(1, 7); b = rng.randint(1, 7)
            a_l.append(a); b_l.append(b)
            lab.append(0 if a < b else (1 if a == b else 2))
        return a_l, b_l, lab

    for epoch in range(1, epochs + 1):
        head_add.train(); head_cmp.train()
        for step in range(steps):
            a_l, b_l, op_l, c_l = sample_arith(32)
            ids_a = [f"concept:ans:{n}" for n in a_l]
            ids_b = [f"concept:ans:{n}" for n in b_l]
            op = torch.tensor([[1.0, 0.0] if o == "add" else [0.0, 1.0]
                               for o in op_l], device=DEVICE)
            tgt = torch.tensor([c - 1 for c in c_l], device=DEVICE)
            dummy = torch.zeros(32, 128, device=DEVICE)
            if use_cook:
                pred = ev.eval(
                    "muscle.cook.arith_v2",
                    bindings={"ids_a": ids_a, "ids_b": ids_b, "op_onehot": op},
                    tick=epoch * 10000 + step,
                )
            else:
                pred = head_add(dummy, dummy, op, ids_a, ids_b, cg,
                                tick=epoch * 10000 + step)
            loss_a = F.cross_entropy(pred @ centroids.t(), tgt)

            ca_l, cb_l, clab = sample_cmp(32)
            cids_a = [f"concept:ans:{n}" for n in ca_l]
            cids_b = [f"concept:ans:{n}" for n in cb_l]
            ctgt = torch.tensor(clab, device=DEVICE)
            if use_cook:
                logits_c = ev.eval(
                    "muscle.cook.comparison",
                    bindings={"ids_a": cids_a, "ids_b": cids_b},
                    tick=epoch * 10000 + step,
                )
            else:
                logits_c = head_cmp(None, None, cids_a, cids_b, cg,
                                    tick=epoch * 10000 + step)
            loss_c = F.cross_entropy(logits_c, ctgt)
            opt.zero_grad(); (loss_a + loss_c).backward(); opt.step()

    head_add.eval(); head_cmp.eval()
    hits = total = 0
    rng2 = random.Random(seed + 1234)
    with torch.no_grad():
        for _ in range(40):
            a_l, b_l, op_l, c_l = [], [], [], []
            for _ in range(20):
                o = "add" if rng2.random() < 0.5 else "sub"
                if o == "add":
                    a = rng2.randint(1, 6); b = rng2.randint(1, 7 - a); c = a + b
                else:
                    a = rng2.randint(2, 7); b = rng2.randint(1, a - 1); c = a - b
                a_l.append(a); b_l.append(b); op_l.append(o); c_l.append(c)
            ids_a = [f"concept:ans:{n}" for n in a_l]
            ids_b = [f"concept:ans:{n}" for n in b_l]
            op = torch.tensor([[1.0, 0.0] if o == "add" else [0.0, 1.0]
                               for o in op_l], device=DEVICE)
            tgt = torch.tensor([c - 1 for c in c_l], device=DEVICE)
            dummy = torch.zeros(20, 128, device=DEVICE)
            if use_cook:
                pred = ev.eval(
                    "muscle.cook.arith_v2",
                    bindings={"ids_a": ids_a, "ids_b": ids_b, "op_onehot": op},
                )
            else:
                pred = head_add(dummy, dummy, op, ids_a, ids_b, cg)
            hits += (pred @ centroids.t()).argmax(-1).eq(tgt).sum().item()
            total += 20

    bs = {cid: {k: v.detach().cpu()
                for k, v in cg.concepts[cid].bundle.state_dict().items()}
          for cid in cids}
    return {
        "acc": hits / max(total, 1),
        "rho_linear": rho_linear(bs, cids, "arithmetic_bias",
                                  lambda i, j: -abs(i - j)),
    }
