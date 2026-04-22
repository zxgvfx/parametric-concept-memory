"""color_concept_study.py — A1: ConceptGraph + D91/D92 在颜色 domain 的验证.

核心问题: 数字 domain 看到的 concept bundle 几何 + 跨 muscle 对齐
(SINGLE_VS_DUAL_MUSCLE_FINDING) 是**ordinal/linear 特例**, 还是**通用
concept memory 性质**?

颜色是最小反例 domain: 拓扑是 **环** (hue wheel) 而不是 linear. 如果
bundle 仍然 emerge 某种非平凡结构, 且 cross-facet 对齐仍然成立,
D91/D92 就从"数字特例"升级为"通用 parametric concept memory".

实验:
- **12 colors** 均匀分布在 hue wheel 上 (circular topology).
- **ColorMixingHead** (消费 `mixing_bias`): (a, b) → circular midpoint c.
- **ColorAdjacencyHead** (消费 `adjacency_offset`): (a, b) → 3-class bucket
  of circular distance (adjacent / near / far).
- Centroid: **random orthogonal**, 剥离一切 color-similarity 先验.

关键指标:
- **ρ_circular**: Spearman(off-diag cos, −circular_dist).
- **ρ_linear**:   Spearman(off-diag cos, −|i−j|). linear control, should be
  *less* than ρ_circular if circular emerges.
- **cross-facet alignment**: Spearman(vec(cos_mix), vec(cos_adj)).
- **shuffle counterfactual** E2: concept_id → bundle 随机重映射, ρ 应塌缩.
- **permutation test** E4: cross-facet alignment p-value.

运行:
    python -m experiments.color_concept_study --n-seeds 5
"""
from __future__ import annotations

import argparse
import cmath
import json
import math
import random
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, ttest_ind

from pcm.concept_graph import ConceptGraph

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── 实验超参 ──────────────────────────────────────────────────────────
N_COLORS = 12
EMBED_DIM = 128
BIAS_DIM = 64
ADJ_DIM = 8
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 30
STEPS_PER_EPOCH = 200


# ─── Circular topology helpers ────────────────────────────────────────


def circular_dist(i: int, j: int, N: int = N_COLORS) -> int:
    d = abs(i - j)
    return min(d, N - d)


def mix_pair(i: int, j: int, N: int = N_COLORS) -> Optional[int]:
    """Circular midpoint. 对 opposite pair (d = N/2) 返回 None (跳过, 歧义)."""
    if circular_dist(i, j, N) == N // 2:
        return None
    angle_i = 2 * math.pi * i / N
    angle_j = 2 * math.pi * j / N
    z = cmath.exp(1j * angle_i) + cmath.exp(1j * angle_j)
    if abs(z) < 1e-8:
        return None
    mid_angle = (cmath.phase(z)) % (2 * math.pi)
    return int(round(mid_angle * N / (2 * math.pi))) % N


def enumerate_mixing_triples(N: int = N_COLORS) -> list[tuple[int, int, int]]:
    triples: list[tuple[int, int, int]] = []
    for a in range(N):
        for b in range(N):
            if a == b:
                continue
            c = mix_pair(a, b, N)
            if c is None:
                continue
            triples.append((a, b, c))
    return triples


def enumerate_adjacency_triples(N: int = N_COLORS) -> list[tuple[int, int, int]]:
    """3-class: 0=adjacent (d=1), 1=near (d=2-3), 2=far (d≥4)."""
    triples: list[tuple[int, int, int]] = []
    for a in range(N):
        for b in range(N):
            if a == b:
                continue
            d = circular_dist(a, b, N)
            if d == 1:
                lab = 0
            elif d <= 3:
                lab = 1
            else:
                lab = 2
            triples.append((a, b, lab))
    return triples


# ─── Muscles (消费 bundle facets) ─────────────────────────────────────


CALLER_MIX = "ColorMixingHead"
FACET_MIX = "mixing_bias"
CALLER_ADJ = "ColorAdjacencyHead"
FACET_ADJ = "adjacency_offset"


class ColorMixingHead(nn.Module):
    """2 bundle → predicted bundle (会被 cosine-argmax 映射到 color class).

    类似 ArithmeticHeadV2, 但无 op_onehot (mixing 是对称的单一 op).
    """

    def __init__(self, embed_dim: int = EMBED_DIM, bias_dim: int = BIAS_DIM) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.bias_dim = bias_dim
        self.fc1 = nn.Linear(2 * bias_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        concept_ids_a: list[str],
        concept_ids_b: list[str],
        cg: ConceptGraph,
        tick: int = 0,
    ) -> torch.Tensor:
        ba = self._collapse(concept_ids_a, cg, tick)
        bb = self._collapse(concept_ids_b, cg, tick)
        x = torch.cat([ba, bb], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def _collapse(self, ids: list[str], cg: ConceptGraph, tick: int) -> torch.Tensor:
        device = next(self.parameters()).device
        rows = []
        for cid in ids:
            cc = cg.concepts[cid].collapse(
                caller=CALLER_MIX, facet=FACET_MIX, shape=(self.bias_dim,),
                tick=tick, init="normal_small", device=device,
            )
            rows.append(cc.as_tensor())
        return torch.stack(rows, dim=0)


class ColorAdjacencyHead(nn.Module):
    """2 bundle → 3-class logits."""

    def __init__(self, facet_dim: int = ADJ_DIM, hidden: int = 64,
                 n_classes: int = 3) -> None:
        super().__init__()
        self.facet_dim = facet_dim
        self.fc1 = nn.Linear(2 * facet_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(
        self,
        concept_ids_a: list[str],
        concept_ids_b: list[str],
        cg: ConceptGraph,
        tick: int = 0,
    ) -> torch.Tensor:
        oa = self._collapse(concept_ids_a, cg, tick)
        ob = self._collapse(concept_ids_b, cg, tick)
        x = torch.cat([oa, ob], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def _collapse(self, ids: list[str], cg: ConceptGraph, tick: int) -> torch.Tensor:
        device = next(self.parameters()).device
        rows = []
        for cid in ids:
            cc = cg.concepts[cid].collapse(
                caller=CALLER_ADJ, facet=FACET_ADJ, shape=(self.facet_dim,),
                tick=tick, init="normal_small", device=device,
            )
            rows.append(cc.as_tensor())
        return torch.stack(rows, dim=0)


# ─── Infrastructure ───────────────────────────────────────────────────


def build_color_graph(n_colors: int = N_COLORS) -> ConceptGraph:
    cg = ConceptGraph(feat_dim=EMBED_DIM)
    for i in range(n_colors):
        cg.register_concept(
            node_id=f"concept:color:{i}",
            label=f"COLOR_{i}",
            scope="BASE",
            provenance=f"color_study:hue={i}/{n_colors}",
        )
    return cg


def make_random_orthogonal_centroids(K: int, dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, K, generator=g)
    Q, _ = torch.linalg.qr(A)
    return F.normalize(Q.t(), dim=-1).to(DEVICE)


def _apply_shuffle(ids: list[str], sm: dict[int, int] | None) -> list[str]:
    if sm is None:
        return ids
    out = []
    for cid in ids:
        k = int(cid.split(":")[-1])
        out.append(f"concept:color:{sm[k]}")
    return out


# ─── Training ─────────────────────────────────────────────────────────


def train_one(
    mode: str,                       # "single" (mix only) / "dual" (mix + adj)
    seed: int,
    centroids: torch.Tensor,
    shuffle_map: dict[int, int] | None = None,
    epochs: int = EPOCHS,
    steps_per_epoch: int = STEPS_PER_EPOCH,
) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    mix_triples = enumerate_mixing_triples(N_COLORS)
    adj_triples = enumerate_adjacency_triples(N_COLORS)

    cg = build_color_graph(N_COLORS)
    head_mix = ColorMixingHead().to(DEVICE)
    head_adj: ColorAdjacencyHead | None = None
    if mode == "dual":
        head_adj = ColorAdjacencyHead().to(DEVICE)

    with torch.no_grad():
        for i in range(N_COLORS):
            c = cg.concepts[f"concept:color:{i}"]
            c.collapse(CALLER_MIX, FACET_MIX, (BIAS_DIM,),
                       tick=0, device=DEVICE, init="normal_small")
            if mode == "dual":
                c.collapse(CALLER_ADJ, FACET_ADJ, (ADJ_DIM,),
                           tick=0, device=DEVICE, init="normal_small")
    cg.bundles_to(torch.device(DEVICE))

    params = list(head_mix.parameters()) + list(cg.iter_bundle_parameters())
    if head_adj is not None:
        params = (list(head_mix.parameters())
                  + list(head_adj.parameters())
                  + list(cg.iter_bundle_parameters()))
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        head_mix.train()
        if head_adj is not None:
            head_adj.train()
        for step_i in range(steps_per_epoch):
            batch = [mix_triples[rng.randrange(len(mix_triples))]
                     for _ in range(BATCH_SIZE)]
            ids_a = _apply_shuffle(
                [f"concept:color:{t[0]}" for t in batch], shuffle_map)
            ids_b = _apply_shuffle(
                [f"concept:color:{t[1]}" for t in batch], shuffle_map)
            tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
            pred = head_mix(ids_a, ids_b, cg,
                            tick=epoch * 10000 + step_i)
            loss_mix = F.cross_entropy(pred @ centroids.t(), tgt)
            total = loss_mix

            if head_adj is not None:
                batch2 = [adj_triples[rng.randrange(len(adj_triples))]
                          for _ in range(BATCH_SIZE)]
                ids_a2 = _apply_shuffle(
                    [f"concept:color:{t[0]}" for t in batch2], shuffle_map)
                ids_b2 = _apply_shuffle(
                    [f"concept:color:{t[1]}" for t in batch2], shuffle_map)
                tgt2 = torch.tensor([t[2] for t in batch2], device=DEVICE)
                logits = head_adj(ids_a2, ids_b2, cg,
                                  tick=epoch * 10000 + step_i)
                total = total + F.cross_entropy(logits, tgt2)

            opt.zero_grad(); total.backward(); opt.step()

    # final evaluation
    head_mix.eval()
    if head_adj is not None:
        head_adj.eval()
    hits_mix = 0
    with torch.no_grad():
        for i in range(0, len(mix_triples), 64):
            batch = mix_triples[i:i + 64]
            ids_a = _apply_shuffle(
                [f"concept:color:{t[0]}" for t in batch], shuffle_map)
            ids_b = _apply_shuffle(
                [f"concept:color:{t[1]}" for t in batch], shuffle_map)
            tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
            pred = head_mix(ids_a, ids_b, cg)
            hits_mix += (pred @ centroids.t()).argmax(-1).eq(tgt).sum().item()
    mix_acc = hits_mix / len(mix_triples)

    adj_acc: Optional[float] = None
    if head_adj is not None:
        hits_adj = 0
        with torch.no_grad():
            for i in range(0, len(adj_triples), 64):
                batch = adj_triples[i:i + 64]
                ids_a = _apply_shuffle(
                    [f"concept:color:{t[0]}" for t in batch], shuffle_map)
                ids_b = _apply_shuffle(
                    [f"concept:color:{t[1]}" for t in batch], shuffle_map)
                tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
                logits = head_adj(ids_a, ids_b, cg)
                hits_adj += logits.argmax(-1).eq(tgt).sum().item()
        adj_acc = hits_adj / len(adj_triples)

    bundle_state = {
        cid: {k: v.detach().cpu() for k, v in c.bundle.state_dict().items()}
        for cid, c in cg.concepts.items()
    }
    return {
        "mode": mode,
        "seed": seed,
        "shuffled": shuffle_map is not None,
        "mix_acc": mix_acc,
        "adj_acc": adj_acc,
        "bundle_state": bundle_state,
    }


# ─── Metrics ──────────────────────────────────────────────────────────


def _cos_matrix(bs: dict, facet: str) -> torch.Tensor:
    rows = []
    for i in range(N_COLORS):
        cid = f"concept:color:{i}"
        rows.append(bs[cid][f"params.{facet}"])
    M = F.normalize(torch.stack(rows), dim=-1)
    return M @ M.t()


def _rho_circular(cos: torch.Tensor) -> float:
    mask = ~torch.eye(N_COLORS, dtype=torch.bool)
    off = cos[mask].numpy()
    d = torch.tensor(
        [[-circular_dist(i, j) for j in range(N_COLORS)] for i in range(N_COLORS)],
        dtype=torch.float,
    )
    return float(spearmanr(off, d[mask].numpy())[0])


def _rho_linear(cos: torch.Tensor) -> float:
    mask = ~torch.eye(N_COLORS, dtype=torch.bool)
    off = cos[mask].numpy()
    d = torch.tensor(
        [[-abs(i - j) for j in range(N_COLORS)] for i in range(N_COLORS)],
        dtype=torch.float,
    )
    return float(spearmanr(off, d[mask].numpy())[0])


def _cross_facet_align(bs: dict, f1: str, f2: str) -> float:
    ca = _cos_matrix(bs, f1)
    cb = _cos_matrix(bs, f2)
    mask = ~torch.eye(N_COLORS, dtype=torch.bool)
    return float(spearmanr(ca[mask].numpy(), cb[mask].numpy())[0])


# ─── Experiments ──────────────────────────────────────────────────────


def run_e1_multi_seed(n_seeds: int, seed_base: int = 1000) -> dict:
    rows = []
    for i in range(n_seeds):
        seed = seed_base + i
        centroids = make_random_orthogonal_centroids(N_COLORS, EMBED_DIM, seed)
        t0 = time.time()
        s = train_one("single", seed, centroids)
        d = train_one("dual",   seed, centroids)
        rho_s_c = _rho_circular(_cos_matrix(s["bundle_state"], FACET_MIX))
        rho_s_l = _rho_linear(_cos_matrix(s["bundle_state"], FACET_MIX))
        rho_d_c = _rho_circular(_cos_matrix(d["bundle_state"], FACET_MIX))
        rho_d_adj = _rho_circular(_cos_matrix(d["bundle_state"], FACET_ADJ))
        align = _cross_facet_align(d["bundle_state"], FACET_MIX, FACET_ADJ)
        dt = time.time() - t0
        print(f"[E1 seed={seed}] single mix_acc={s['mix_acc']:.3f}  "
              f"ρ_circ={rho_s_c:+.3f}  ρ_lin={rho_s_l:+.3f}  | "
              f"dual mix_acc={d['mix_acc']:.3f} adj_acc={d['adj_acc']:.3f}  "
              f"ρ_mix_circ={rho_d_c:+.3f}  ρ_adj_circ={rho_d_adj:+.3f}  "
              f"align={align:+.3f}  ({dt:.1f}s)")
        rows.append({
            "seed": seed,
            "single_mix_acc": s["mix_acc"],
            "single_rho_circular": rho_s_c,
            "single_rho_linear": rho_s_l,
            "dual_mix_acc": d["mix_acc"],
            "dual_adj_acc": d["adj_acc"],
            "dual_rho_mix_circular": rho_d_c,
            "dual_rho_adj_circular": rho_d_adj,
            "dual_cross_facet_align": align,
            "wall_s": dt,
        })

    def _stats(xs):
        xs = list(xs)
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
        return {"mean": m, "std": sd, "min": min(xs), "max": max(xs)}

    circ_s = [r["single_rho_circular"] for r in rows]
    circ_d = [r["dual_rho_mix_circular"] for r in rows]
    tstat, pval = ttest_ind(circ_s, circ_d, equal_var=False)
    return {
        "n_seeds": n_seeds,
        "per_seed": rows,
        "single_rho_circular": _stats(circ_s),
        "single_rho_linear":   _stats([r["single_rho_linear"] for r in rows]),
        "dual_rho_mix_circular": _stats(circ_d),
        "dual_rho_adj_circular": _stats([r["dual_rho_adj_circular"] for r in rows]),
        "dual_cross_facet_align": _stats([r["dual_cross_facet_align"] for r in rows]),
        "welch_single_vs_dual": {"t": float(tstat), "p_value": float(pval)},
    }


def run_e2_shuffled(n_seeds: int, seed_base: int = 2000) -> dict:
    rows = []
    for i in range(n_seeds):
        seed = seed_base + i
        centroids = make_random_orthogonal_centroids(N_COLORS, EMBED_DIM, seed)
        perm = list(range(N_COLORS))
        random.Random(seed).shuffle(perm)
        sm = {k: perm[k] for k in range(N_COLORS)}
        t0 = time.time()
        s = train_one("single", seed, centroids, shuffle_map=sm)
        rho_c = _rho_circular(_cos_matrix(s["bundle_state"], FACET_MIX))
        rho_l = _rho_linear(_cos_matrix(s["bundle_state"], FACET_MIX))
        dt = time.time() - t0
        print(f"[E2 seed={seed} shuffled] mix_acc={s['mix_acc']:.3f}  "
              f"ρ_circ={rho_c:+.3f} ρ_lin={rho_l:+.3f}  ({dt:.1f}s)")
        rows.append({
            "seed": seed, "shuffle_map": sm,
            "mix_acc": s["mix_acc"],
            "rho_circular_raw_order": rho_c,
            "rho_linear_raw_order": rho_l,
            "wall_s": dt,
        })

    def _stats(xs):
        xs = [abs(x) for x in xs]
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
        return {"mean": m, "std": sd, "min": min(xs), "max": max(xs)}

    return {
        "n_seeds": n_seeds,
        "per_seed": rows,
        "abs_rho_circular_stats": _stats(
            [r["rho_circular_raw_order"] for r in rows]),
        "notes": "训练时 shuffle concept_id→bundle. ρ 是对 RAW circular order 测. "
                 "若 shuffle 破坏身份, |ρ| 应 ≈ 0.",
    }


def run_e4_permutation(dual_bundle: dict, n_perm: int = 1000) -> dict:
    cos_m = _cos_matrix(dual_bundle, FACET_MIX)
    cos_a = _cos_matrix(dual_bundle, FACET_ADJ)
    mask = ~torch.eye(N_COLORS, dtype=torch.bool)
    om = cos_m[mask].numpy()
    oa = cos_a[mask].numpy()
    observed = float(spearmanr(om, oa)[0])

    rng = random.Random(0)
    ge = 0
    null = []
    for _ in range(n_perm):
        perm = list(range(N_COLORS))
        rng.shuffle(perm)
        cos_a_perm = cos_a[perm][:, perm]
        oap = cos_a_perm[mask].numpy()
        r = float(spearmanr(om, oap)[0])
        null.append(r)
        if abs(r) >= abs(observed):
            ge += 1
    p = (ge + 1) / (n_perm + 1)
    return {
        "observed_cross_facet_rho": observed,
        "n_permutations": n_perm,
        "p_value": p,
        "null_mean": sum(null) / len(null),
        "null_std": math.sqrt(sum((r - sum(null) / len(null)) ** 2 for r in null)
                              / max(len(null) - 1, 1)),
        "conclusion": ("significant (p<0.01)" if p < 0.01
                       else "significant (p<0.05)" if p < 0.05
                       else "not significant"),
    }


# ─── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    ap.add_argument("--out", type=Path, default=Path("outputs/color_concept"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], help="e1 / e2 / e4")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.n_seeds = 1

    summary: dict = {
        "n_seeds": args.n_seeds,
        "device": DEVICE,
        "n_colors": N_COLORS,
    }

    if "e1" not in args.skip:
        print("=" * 60); print("E1: Multi-seed (single vs dual)"); print("=" * 60)
        summary["E1_multi_seed"] = run_e1_multi_seed(args.n_seeds)

    if "e2" not in args.skip:
        print("=" * 60); print("E2: Shuffled concept counterfactual"); print("=" * 60)
        summary["E2_shuffled"] = run_e2_shuffled(args.n_seeds)

    if "e4" not in args.skip:
        print("=" * 60); print("E4: Cross-facet permutation test"); print("=" * 60)
        centroids = make_random_orthogonal_centroids(N_COLORS, EMBED_DIM, 1000)
        d = train_one("dual", 1000, centroids)
        summary["E4_permutation"] = run_e4_permutation(d["bundle_state"])

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60); print("SUMMARY"); print("=" * 60)
    if "E1_multi_seed" in summary:
        e1 = summary["E1_multi_seed"]
        sc = e1["single_rho_circular"]; sl = e1["single_rho_linear"]
        dc = e1["dual_rho_mix_circular"]; al = e1["dual_cross_facet_align"]
        print(f"E1  single ρ_circ = {sc['mean']:+.3f} ± {sc['std']:.3f}  "
              f"ρ_lin = {sl['mean']:+.3f}")
        print(f"E1  dual   ρ_mix_circ = {dc['mean']:+.3f} ± {dc['std']:.3f}")
        print(f"E1  dual   cross-facet align = {al['mean']:+.3f} ± {al['std']:.3f}")
    if "E2_shuffled" in summary:
        e2 = summary["E2_shuffled"]
        s = e2["abs_rho_circular_stats"]
        print(f"E2  |ρ_circ| shuffled = {s['mean']:.3f} ± {s['std']:.3f}")
    if "E4_permutation" in summary:
        e4 = summary["E4_permutation"]
        print(f"E4  observed = {e4['observed_cross_facet_rho']:+.3f}  "
              f"null_mean = {e4['null_mean']:+.3f}  p = {e4['p_value']:.3f}  "
              f"→ {e4['conclusion']}")


if __name__ == "__main__":
    main()
