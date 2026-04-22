"""counterfactual_swap_study.py — Appendix B: bundle = concept identity 的因果证据.

以往的证据 (shuffle / ρ / cross-facet alignment) 都是**相关性**层面: 训练完
bundle 几何映射到 task 结构. 本实验做**因果介入**: 训练完后把两个 concept
的 bundle 直接互换, 看下游推理是否跟着"语义互换", 以此证明:

    bundle params ≡ concept identity itself (not just a correlate)

同时用 **facet-specific swap** 做 double-dissociation:
  - 仅交换 arithmetic_bias (数字域) → AddHead 输出按 swap 换语义, CmpHead 不受影响.
  - 仅交换 ordinal_offset  (数字域) → CmpHead 按 swap 换语义, AddHead 不受影响.
  - 仅交换 mixing_bias     (颜色域) → MixHead 按 swap 换, AdjHead 不受影响.
  - 仅交换 adjacency_offset(颜色域) → AdjHead 按 swap 换, MixHead 不受影响.

实现要点:
  * 为了解耦, 数字域使用 **random orthogonal centroids** (跟 color 域一致),
    不再依赖 NumerosityEncoder ckpt. Robustness study 已证明两种 centroid
    都诱导同样的 bundle 结构.
  * swap 是 in-place on ``bundle.params[facet].data`` (保留 Parameter 对象,
    避免破坏 optimizer / autograd 图).
  * Metric: 在 **全枚举** pair 上求 accuracy, 按 "涉及 swap pair" vs
    "不涉及 swap pair" 分组, 得到 swap-specific drop.

产出: outputs/counterfactual_swap/{summary.json}

运行:
    python -m experiments.counterfactual_swap_study \\
        --n-seeds 3

Smoke:
    python -m experiments.counterfactual_swap_study --smoke
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcm.concept_graph import ConceptGraph
from pcm.heads.arithmetic_head_v2 import ArithmeticHeadV2
from pcm.heads.comparison_head import ComparisonHead
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ═════════════════════════════════════════════════════════════════════
# Number-domain config
# ═════════════════════════════════════════════════════════════════════
NUM_EMBED_DIM = 128
NUM_BIAS_DIM = 64
NUM_ORD_DIM = 8
NUM_N_MIN = 1
NUM_N_MAX = 7
NUM_BATCH_SIZE = 32
NUM_LR = 1e-3
NUM_EPOCHS = 12
NUM_STEPS_PER_EPOCH = 120

# swap pair: pick two concepts with max inter-distance that still allow
# plenty of (a, b) pairs on each side (避开 boundary 1 和 7 以便 add/sub 都能采样)
NUM_SWAP_A = 3
NUM_SWAP_B = 5

# ═════════════════════════════════════════════════════════════════════
# Color-domain config (swap hue 2 ↔ hue 8, near-opposite but not antipodal)
# ═════════════════════════════════════════════════════════════════════
COLOR_SWAP_A = 2
COLOR_SWAP_B = 8  # circular_dist = min(6, 6) = 6 — 是 antipodal (d=N/2=6)
# antipodal 对 mixing 来说 mid 是 undefined, 但 swap bundle 本身没问题.
# 不过用 antipodal 对会导致 mixing_triples 不含 (2,8). 用 (2,5) 更稳妥:
COLOR_SWAP_B = 5  # circular_dist=3, 不是 antipodal, mixing triples 包含


# ═════════════════════════════════════════════════════════════════════
# Number-domain infra
# ═════════════════════════════════════════════════════════════════════


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
            # ensure a+b in [N_MIN, N_MAX]
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


def train_number_dual(
    seed: int,
    centroids: torch.Tensor,
    epochs: int = NUM_EPOCHS,
    steps_per_epoch: int = NUM_STEPS_PER_EPOCH,
) -> dict:
    """训练一个 AddHead + CmpHead 的 dual-muscle 模型."""
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
            pred = head_add(dummy, dummy, op, ids_a, ids_b, cg, tick=epoch * 10000 + step)
            loss_a = F.cross_entropy(pred @ centroids.t(), tgt)

            ca_l, cb_l, clab = _sample_cmp(rng, NUM_BATCH_SIZE)
            cids_a = [f"concept:ans:{n}" for n in ca_l]
            cids_b = [f"concept:ans:{n}" for n in cb_l]
            ctgt = torch.tensor(clab, device=DEVICE)
            logits_c = head_cmp(None, None, cids_a, cids_b, cg, tick=epoch * 10000 + step)
            loss_c = F.cross_entropy(logits_c, ctgt)

            opt.zero_grad()
            (loss_a + loss_c).backward()
            opt.step()

    head_add.eval(); head_cmp.eval()
    return {"cg": cg, "head_add": head_add, "head_cmp": head_cmp, "centroids": centroids}


# ─── Number-domain eval (全枚举 pair, 按 swap pair 分组) ────────────


def _involves(a: int, b: int, swap_ns: tuple[int, int]) -> bool:
    return a in swap_ns or b in swap_ns


@torch.no_grad()
def eval_number_add(
    cg: ConceptGraph,
    head_add: ArithmeticHeadV2,
    centroids: torch.Tensor,
    swap_ns: tuple[int, int],
) -> dict:
    """枚举全部 (a, b, op) 使得 c=a±b ∈ [N_MIN, N_MAX]. 按 involves swap 分组."""
    head_add.eval()
    records = {"all": [0, 0], "involving_swap": [0, 0], "not_involving": [0, 0]}
    # collect triples
    triples: list[tuple[int, int, str, int]] = []
    for a in range(NUM_N_MIN, NUM_N_MAX + 1):
        for b in range(NUM_N_MIN, NUM_N_MAX + 1):
            for op in ("add", "sub"):
                if op == "add":
                    c = a + b
                else:
                    c = a - b
                if c < NUM_N_MIN or c > NUM_N_MAX:
                    continue
                triples.append((a, b, op, c))

    # batched eval
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

    def _acc(kv):
        hits, tot = kv
        return hits / max(tot, 1)
    return {k: {"acc": _acc(v), "n": v[1]} for k, v in records.items()}


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

    def _acc(kv):
        return kv[0] / max(kv[1], 1)
    return {k: {"acc": _acc(v), "n": v[1]} for k, v in records.items()}


# ═════════════════════════════════════════════════════════════════════
# Color-domain training (reusing color_concept_study primitives)
# ═════════════════════════════════════════════════════════════════════


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
    return {"cg": cg, "head_mix": head_mix, "head_adj": head_adj, "centroids": centroids}


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

    def _acc(kv): return kv[0] / max(kv[1], 1)
    return {k: {"acc": _acc(v), "n": v[1]} for k, v in records.items()}


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

    def _acc(kv): return kv[0] / max(kv[1], 1)
    return {k: {"acc": _acc(v), "n": v[1]} for k, v in records.items()}


# ═════════════════════════════════════════════════════════════════════
# The swap itself — in-place on Parameter.data
# ═════════════════════════════════════════════════════════════════════


def swap_bundle_facet(
    cg: ConceptGraph,
    cid_a: str,
    cid_b: str,
    facet: str,
) -> None:
    """把 cid_a / cid_b 在指定 facet 上的 bundle param data 互换 (in-place)."""
    pa = cg.concepts[cid_a].bundle.params[facet]
    pb = cg.concepts[cid_b].bundle.params[facet]
    tmp = pa.data.clone()
    pa.data.copy_(pb.data)
    pb.data.copy_(tmp)


def swap_all_facets(cg: ConceptGraph, cid_a: str, cid_b: str) -> None:
    facets = set(cg.concepts[cid_a].bundle.params.keys())
    facets &= set(cg.concepts[cid_b].bundle.params.keys())
    for f in facets:
        swap_bundle_facet(cg, cid_a, cid_b, f)


# ═════════════════════════════════════════════════════════════════════
# Per-seed number experiment
# ═════════════════════════════════════════════════════════════════════


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

    # baseline
    base_add = eval_number_add(cg, head_add, centroids, swap_ns)
    base_cmp = eval_number_cmp(cg, head_cmp, swap_ns)

    # condition A: arithmetic_bias only
    swap_bundle_facet(cg, cid_a, cid_b, "arithmetic_bias")
    A_add = eval_number_add(cg, head_add, centroids, swap_ns)
    A_cmp = eval_number_cmp(cg, head_cmp, swap_ns)
    # undo
    swap_bundle_facet(cg, cid_a, cid_b, "arithmetic_bias")

    # condition B: ordinal_offset only
    swap_bundle_facet(cg, cid_a, cid_b, "ordinal_offset")
    B_add = eval_number_add(cg, head_add, centroids, swap_ns)
    B_cmp = eval_number_cmp(cg, head_cmp, swap_ns)
    swap_bundle_facet(cg, cid_a, cid_b, "ordinal_offset")

    # condition C: swap both
    swap_all_facets(cg, cid_a, cid_b)
    C_add = eval_number_add(cg, head_add, centroids, swap_ns)
    C_cmp = eval_number_cmp(cg, head_cmp, swap_ns)
    swap_all_facets(cg, cid_a, cid_b)   # undo

    dt = time.time() - t0
    return {
        "seed": seed,
        "swap_pair": list(swap_ns),
        "baseline": {"add": base_add, "cmp": base_cmp},
        "swap_arith_only": {"add": A_add, "cmp": A_cmp},
        "swap_ord_only":   {"add": B_add, "cmp": B_cmp},
        "swap_both":       {"add": C_add, "cmp": C_cmp},
        "wall_s": dt,
    }


# ═════════════════════════════════════════════════════════════════════
# Per-seed color experiment
# ═════════════════════════════════════════════════════════════════════


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

    dt = time.time() - t0
    return {
        "seed": seed,
        "swap_pair": list(swap_ns),
        "baseline":      {"mix": base_mix, "adj": base_adj},
        "swap_mix_only": {"mix": A_mix,    "adj": A_adj},
        "swap_adj_only": {"mix": B_mix,    "adj": B_adj},
        "swap_both":     {"mix": C_mix,    "adj": C_adj},
        "wall_s": dt,
    }


# ═════════════════════════════════════════════════════════════════════
# Aggregation
# ═════════════════════════════════════════════════════════════════════


def _agg_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    m = sum(values) / len(values)
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / max(len(values) - 1, 1))
    return {"mean": m, "std": sd, "min": min(values), "max": max(values), "n": len(values)}


def _fmt_row(label: str, v: dict) -> str:
    a = v["acc"]
    return f"{label:<22}  acc={a*100:5.1f}%  (n={v['n']})"


def _print_number_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("NUMBER DOMAIN — per-condition accuracy, averaged over seeds")
    print("=" * 70)
    def _mean(path):
        vs = []
        for r in rows:
            node = r
            for k in path:
                node = node[k]
            vs.append(node["acc"])
        return sum(vs) / len(vs)

    conds = ("baseline", "swap_arith_only", "swap_ord_only", "swap_both")
    print(f"{'cond':<20} | AddHead inv | AddHead not | CmpHead inv | CmpHead not")
    print("-" * 78)
    for c in conds:
        add_inv = _mean((c, "add", "involving_swap"))
        add_not = _mean((c, "add", "not_involving"))
        cmp_inv = _mean((c, "cmp", "involving_swap"))
        cmp_not = _mean((c, "cmp", "not_involving"))
        print(f"{c:<20} |  {add_inv*100:6.1f}%   |  {add_not*100:6.1f}%   "
              f"|  {cmp_inv*100:6.1f}%   |  {cmp_not*100:6.1f}%")


def _print_color_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("COLOR DOMAIN — per-condition accuracy, averaged over seeds")
    print("=" * 70)
    def _mean(path):
        vs = []
        for r in rows:
            node = r
            for k in path:
                node = node[k]
            vs.append(node["acc"])
        return sum(vs) / len(vs)

    conds = ("baseline", "swap_mix_only", "swap_adj_only", "swap_both")
    print(f"{'cond':<20} | MixHead inv | MixHead not | AdjHead inv | AdjHead not")
    print("-" * 78)
    for c in conds:
        mix_inv = _mean((c, "mix", "involving_swap"))
        mix_not = _mean((c, "mix", "not_involving"))
        adj_inv = _mean((c, "adj", "involving_swap"))
        adj_not = _mean((c, "adj", "not_involving"))
        print(f"{c:<20} |  {mix_inv*100:6.1f}%   |  {mix_not*100:6.1f}%   "
              f"|  {adj_inv*100:6.1f}%   |  {adj_not*100:6.1f}%")


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--num-steps", type=int, default=NUM_STEPS_PER_EPOCH)
    ap.add_argument("--color-epochs", type=int, default=30)
    ap.add_argument("--color-steps", type=int, default=200)
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed, reduced epochs, sanity check only")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="domains to skip: 'number' / 'color'")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/counterfactual_swap"))
    args = ap.parse_args()

    if args.smoke:
        args.n_seeds = 1
        args.num_epochs = 6
        args.num_steps = 60
        args.color_epochs = 10
        args.color_steps = 80

    args.out.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "config": {
            "device": DEVICE,
            "n_seeds": args.n_seeds,
            "num_epochs": args.num_epochs,
            "num_steps": args.num_steps,
            "color_epochs": args.color_epochs,
            "color_steps": args.color_steps,
            "num_swap_pair": [NUM_SWAP_A, NUM_SWAP_B],
            "color_swap_pair": [COLOR_SWAP_A, COLOR_SWAP_B],
        }
    }

    if "number" not in args.skip:
        print("=" * 70)
        print(f"NUMBER DOMAIN — swap concept:ans:{NUM_SWAP_A} ↔ concept:ans:{NUM_SWAP_B}")
        print("=" * 70)
        rows = []
        for s in range(args.n_seeds):
            seed = 1000 + s
            r = run_number_seed(seed, args.num_epochs, args.num_steps)
            rows.append(r)
            b = r["baseline"]; a = r["swap_arith_only"]; o = r["swap_ord_only"]
            print(f"[seed={seed}] baseline add_all={b['add']['all']['acc']*100:.1f}% "
                  f"cmp_all={b['cmp']['all']['acc']*100:.1f}% | "
                  f"swap-arith: add_inv={a['add']['involving_swap']['acc']*100:.1f}% "
                  f"cmp_inv={a['cmp']['involving_swap']['acc']*100:.1f}% | "
                  f"swap-ord: add_inv={o['add']['involving_swap']['acc']*100:.1f}% "
                  f"cmp_inv={o['cmp']['involving_swap']['acc']*100:.1f}% "
                  f"({r['wall_s']:.1f}s)")
        summary["number_domain"] = {"per_seed": rows}
        _print_number_summary(rows)

    if "color" not in args.skip:
        print("=" * 70)
        print(f"COLOR DOMAIN — swap concept:color:{COLOR_SWAP_A} ↔ concept:color:{COLOR_SWAP_B}")
        print("=" * 70)
        rows = []
        for s in range(args.n_seeds):
            seed = 2000 + s
            r = run_color_seed(seed, args.color_epochs, args.color_steps)
            rows.append(r)
            b = r["baseline"]; a = r["swap_mix_only"]; o = r["swap_adj_only"]
            print(f"[seed={seed}] baseline mix_all={b['mix']['all']['acc']*100:.1f}% "
                  f"adj_all={b['adj']['all']['acc']*100:.1f}% | "
                  f"swap-mix: mix_inv={a['mix']['involving_swap']['acc']*100:.1f}% "
                  f"adj_inv={a['adj']['involving_swap']['acc']*100:.1f}% | "
                  f"swap-adj: mix_inv={o['mix']['involving_swap']['acc']*100:.1f}% "
                  f"adj_inv={o['adj']['involving_swap']['acc']*100:.1f}% "
                  f"({r['wall_s']:.1f}s)")
        summary["color_domain"] = {"per_seed": rows}
        _print_color_summary(rows)

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"\n✓ summary written → {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
