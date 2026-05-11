"""experiments/cook_demo.py - Tier-D cook regression for paper §4.

Runs the same single-seed N=7 arithmetic experiment two ways:

1. **Tier-A direct forward** via :meth:`pcm.heads.ArithmeticHeadV2.forward`.
2. **Tier-D cook path** via the ``parametric_muscle_subgraph`` registered
   by :func:`pcm.heads.build_arith_v2_cook` (generic factory built on top
   of :mod:`pcm.heads.cook_factory`), evaluated through
   :class:`pcm.GraphEvaluator`.

Both runs share an identical ``ConceptGraph`` initial state and identical
``Linear`` weights (``build_arith_v2_cook`` copies fc1/fc2/fc3 from the
provided head into a freshly-allocated :class:`MLPBackbone`). With the
same RNG seed they should reach the same final ``acc`` and the same
``rho_linear`` to within 0.05 (claim D1 from
``docs/PCM_NODE_AS_FUNCTION_DESIGN.md`` §6).

Smoke run completes in well under 30s on CPU; CUDA path runs in ~10s.

Output: ``outputs/cook_demo/summary.json``.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from pcm import ConceptGraph
from pcm.heads import ArithmeticHeadV2, build_arith_v2_cook


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Tiny shared dataset utilities (paper §4 random-orthogonal setup, N=7).


def _random_orthogonal(n_classes: int, dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, n_classes, generator=g)
    Q, _ = torch.linalg.qr(A)
    return F.normalize(Q.t(), dim=-1).to(DEVICE)


def _sample(rng: random.Random, bs: int, n_max: int = 7):
    a_l, b_l, op_l, c_l = [], [], [], []
    for _ in range(bs):
        op = "add" if rng.random() < 0.5 else "sub"
        if op == "add":
            a = rng.randint(1, n_max - 1)
            b = rng.randint(1, n_max - a)
            c = a + b
        else:
            a = rng.randint(2, n_max)
            b = rng.randint(1, a - 1)
            c = a - b
        a_l.append(a); b_l.append(b); op_l.append(op); c_l.append(c)
    return a_l, b_l, op_l, c_l


def _op_onehot(ops: list[str]) -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0] if o == "add" else [0.0, 1.0] for o in ops],
        device=DEVICE,
    )


def _build_cg() -> tuple[ConceptGraph, list[str]]:
    cg = ConceptGraph(initial_capacity=16)
    cids = []
    for n in range(1, 8):
        cid = f"concept:ans:{n}"
        cg.register_concept(node_id=cid, label=f"ANS_{n}", scope="BASE",
                            provenance=f"cook_demo:n={n}")
        cids.append(cid)
    return cg, cids


def _rho_linear(cg: ConceptGraph, cids: list[str]) -> float:
    pool = cg.bundle_pool["arithmetic_bias"].detach()
    rows = torch.stack([
        pool[cg.cid_to_slot[cid]] for cid in cids
    ])
    rows = F.normalize(rows, dim=-1)
    cos = (rows @ rows.t()).cpu().numpy()
    n = len(cids)
    iu = [(i, j) for i in range(n) for j in range(n) if i != j]
    cos_vals = [float(cos[i, j]) for i, j in iu]
    d_lin = [-abs(i - j) for i, j in iu]
    return float(spearmanr(cos_vals, d_lin)[0])


# Two training paths.


def _train_direct(seed: int, epochs: int, steps: int) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    cg, cids = _build_cg()
    head = ArithmeticHeadV2(embed_dim=128, bias_dim=64).to(DEVICE)
    centroids = _random_orthogonal(7, 128, seed=seed + 1)
    with torch.no_grad():
        zeros = torch.zeros(7, 128, device=DEVICE)
        op = torch.tensor([[1.0, 0.0]] * 7, device=DEVICE)
        head(zeros, zeros, op, cids, cids, cg)
    cg.bundles_to(torch.device(DEVICE))
    params = list(head.parameters()) + list(cg.iter_bundle_parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    cg.register_optimizer(opt)

    for epoch in range(1, epochs + 1):
        head.train()
        for step in range(steps):
            a_l, b_l, op_l, c_l = _sample(rng, 32)
            ids_a = [f"concept:ans:{n}" for n in a_l]
            ids_b = [f"concept:ans:{n}" for n in b_l]
            op = _op_onehot(op_l)
            tgt = torch.tensor([c - 1 for c in c_l], device=DEVICE)
            dummy = torch.zeros(32, 128, device=DEVICE)
            pred = head(dummy, dummy, op, ids_a, ids_b, cg,
                        tick=epoch * 10000 + step)
            loss = F.cross_entropy(pred @ centroids.t(), tgt)
            opt.zero_grad(); loss.backward(); opt.step()

    head.eval()
    hits = total = 0
    rng2 = random.Random(seed + 1234)
    with torch.no_grad():
        for _ in range(40):
            a_l, b_l, op_l, c_l = _sample(rng2, 20)
            ids_a = [f"concept:ans:{n}" for n in a_l]
            ids_b = [f"concept:ans:{n}" for n in b_l]
            op = _op_onehot(op_l)
            tgt = torch.tensor([c - 1 for c in c_l], device=DEVICE)
            dummy = torch.zeros(20, 128, device=DEVICE)
            pred = head(dummy, dummy, op, ids_a, ids_b, cg)
            hits += (pred @ centroids.t()).argmax(-1).eq(tgt).sum().item()
            total += 20
    return {
        "path": "direct",
        "acc": hits / max(total, 1),
        "rho_linear": _rho_linear(cg, cids),
        "head": head,
        "cg": cg,
    }


def _train_cook(seed: int, epochs: int, steps: int) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    cg, cids = _build_cg()
    # We still build a Tier-A head first (to seed the pool with the *same*
    # RNG-determined small-init rows as the direct run) and then ask the
    # generic cook factory to mirror its weights into the cook backbone.
    head_v2 = ArithmeticHeadV2(embed_dim=128, bias_dim=64).to(DEVICE)
    centroids = _random_orthogonal(7, 128, seed=seed + 1)
    with torch.no_grad():
        zeros = torch.zeros(7, 128, device=DEVICE)
        op = torch.tensor([[1.0, 0.0]] * 7, device=DEVICE)
        head_v2(zeros, zeros, op, cids, cids, cg)
    cg.bundles_to(torch.device(DEVICE))

    node_id, backbone, evaluator = build_arith_v2_cook(cg, head_v2)

    params = list(backbone.parameters()) + list(cg.iter_bundle_parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    cg.register_optimizer(opt)

    for epoch in range(1, epochs + 1):
        backbone.train()
        for step in range(steps):
            a_l, b_l, op_l, c_l = _sample(rng, 32)
            ids_a = [f"concept:ans:{n}" for n in a_l]
            ids_b = [f"concept:ans:{n}" for n in b_l]
            op = _op_onehot(op_l)
            tgt = torch.tensor([c - 1 for c in c_l], device=DEVICE)
            pred = evaluator.eval(
                node_id,
                bindings={"ids_a": ids_a, "ids_b": ids_b, "op_onehot": op},
                tick=epoch * 10000 + step,
            )
            loss = F.cross_entropy(pred @ centroids.t(), tgt)
            opt.zero_grad(); loss.backward(); opt.step()

    backbone.eval()
    hits = total = 0
    rng2 = random.Random(seed + 1234)
    with torch.no_grad():
        for _ in range(40):
            a_l, b_l, op_l, c_l = _sample(rng2, 20)
            ids_a = [f"concept:ans:{n}" for n in a_l]
            ids_b = [f"concept:ans:{n}" for n in b_l]
            op = _op_onehot(op_l)
            tgt = torch.tensor([c - 1 for c in c_l], device=DEVICE)
            pred = evaluator.eval(
                node_id,
                bindings={"ids_a": ids_a, "ids_b": ids_b, "op_onehot": op},
            )
            hits += (pred @ centroids.t()).argmax(-1).eq(tgt).sum().item()
            total += 20
    return {
        "path": "cook",
        "acc": hits / max(total, 1),
        "rho_linear": _rho_linear(cg, cids),
        "backbone": backbone,
        "cg": cg,
    }


# Comparison entry point.


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="short epochs/steps")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=Path("outputs/cook_demo"))
    args = ap.parse_args()
    epochs, steps = (4, 60) if args.smoke else (12, 120)

    t0 = time.time()
    print(f"[cook_demo] device={DEVICE} seed={args.seed} epochs={epochs} "
          f"steps={steps}")

    direct = _train_direct(args.seed, epochs, steps)
    cook = _train_cook(args.seed, epochs, steps)

    delta_rho = abs(direct["rho_linear"] - cook["rho_linear"])
    delta_acc = abs(direct["acc"] - cook["acc"])
    dt = time.time() - t0

    summary = {
        "config": {
            "seed": args.seed, "epochs": epochs, "steps": steps,
            "device": DEVICE,
        },
        "direct": {"acc": direct["acc"], "rho_linear": direct["rho_linear"]},
        "cook":   {"acc": cook["acc"],   "rho_linear": cook["rho_linear"]},
        "delta":  {"rho_linear": delta_rho, "acc": delta_acc},
        "wall_s": dt,
        "claim_d1_passed": delta_rho < 0.05 and delta_acc < 0.05,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print("=" * 72)
    print(f"  Tier-A direct  : acc={direct['acc']:.3f}  rho_linear={direct['rho_linear']:+.4f}")
    print(f"  Tier-D cook    : acc={cook['acc']:.3f}  rho_linear={cook['rho_linear']:+.4f}")
    print("-" * 72)
    print(f"  delta_acc = {delta_acc:.4f}  |  delta_rho = {delta_rho:.4f}")
    print(f"  D1 claim   = {'PASS' if summary['claim_d1_passed'] else 'FAIL'}")
    print(f"  wall = {dt:.1f}s  ->  {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
