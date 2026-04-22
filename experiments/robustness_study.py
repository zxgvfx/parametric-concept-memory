"""robustness_study.py — 验证 SINGLE_VS_DUAL_MUSCLE_FINDING 不是单次 seed 偶然.

跑四个实验:
  E1. Multi-seed (N=10): single / dual ρ 的 mean ± std, Welch t-test
  E2. Shuffled-concept 反事实 (破坏 identity): 训练时把 concept_id 随机映射到
      别的 bundle, 让 bundle 与数量不再对应. 如果 ρ 仍然高, 说明"数量序"是
      任意的, D91/D92 的"身份"判据失效.
  E3. N-scan: n_max ∈ {5, 7, 9}, 看 ρ 是否随 N 稳定
  E4. Permutation test: 对 cross-facet 一致性 ρ(vec(cos_arith), vec(cos_ord))
      做 1000-perm null-distribution, 给出 p-value

产出: outputs/robustness/{summary.json, raw.json, tables.md}

运行:
    python -m experiments.robustness_study \\
        --encoder-ckpt outputs/ans_encoder/final.pt --n-seeds 10
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, ttest_ind

from pcm.heads.arithmetic_head_v2 import ArithmeticHeadV2
from pcm.heads.comparison_head import ComparisonHead
from pcm.heads.numerosity_encoder import (
    DatasetConfig,
    NumerosityEncoder,
    generate_dot_canvas,
)
from experiments._graph_builder import build_ans_graph

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 12
STEPS_PER_EPOCH = 120
BATCH_SIZE = 32
LR = 1e-3
BIAS_DIM = 64
ORD_DIM = 8


# ── 数据采样 (跟 train_* 同源) ─────────────────────────────────────────


def _sample_arith(cfg: DatasetConfig, bs: int, rng: torch.Generator):
    a_l, b_l, op_l, c_l = [], [], [], []
    for _ in range(bs):
        op = "add" if torch.rand(1, generator=rng).item() < 0.5 else "sub"
        if op == "add":
            a = int(torch.randint(cfg.n_min, cfg.n_max, (1,), generator=rng).item())
            b = int(torch.randint(cfg.n_min, cfg.n_max + 1 - a, (1,), generator=rng).item())
            c = a + b
        else:
            a = int(torch.randint(cfg.n_min + 1, cfg.n_max + 1, (1,), generator=rng).item())
            b = int(torch.randint(cfg.n_min, a, (1,), generator=rng).item())
            c = a - b
        a_l.append(a); b_l.append(b); op_l.append(op); c_l.append(c)
    return a_l, b_l, op_l, c_l


def _sample_cmp(cfg: DatasetConfig, bs: int, rng: torch.Generator):
    a_l, b_l, lab = [], [], []
    for _ in range(bs):
        a = int(torch.randint(cfg.n_min, cfg.n_max + 1, (1,), generator=rng).item())
        b = int(torch.randint(cfg.n_min, cfg.n_max + 1, (1,), generator=rng).item())
        a_l.append(a); b_l.append(b)
        lab.append(0 if a < b else (1 if a == b else 2))
    return a_l, b_l, lab


def _op_onehot(ops: list[str]) -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0] if o == "add" else [0.0, 1.0] for o in ops], device=DEVICE
    )


def _compute_centroids(enc: NumerosityEncoder, cfg: DatasetConfig, n_per: int = 200) -> torch.Tensor:
    """Centroid 在 CPU 算 (generate_dot_canvas 只给 CPU), 然后一次性搬到 DEVICE."""
    enc_cpu = enc.cpu()
    enc_cpu.eval()
    rng = torch.Generator().manual_seed(0)
    rows = []
    with torch.no_grad():
        for n in range(cfg.n_min, cfg.n_max + 1):
            xs = torch.stack([generate_dot_canvas(n, cfg, rng) for _ in range(n_per)])
            rows.append(F.normalize(enc_cpu(xs).mean(0), dim=-1))
    return torch.stack(rows).to(DEVICE)


# ── 训练 primitives ────────────────────────────────────────────────────


def _apply_shuffle_map(ids: list[str], shuffle_map: dict[int, int] | None) -> list[str]:
    """把 ``concept:ans:n`` 按 shuffle_map 映射到 ``concept:ans:shuffle_map[n]``.

    shuffle_map 给 E2 反事实用: concept_id -> 被随机重新分配的 bundle_id. 若为 None,
    原样返回.
    """
    if shuffle_map is None:
        return ids
    out = []
    for cid in ids:
        n = int(cid.split(":")[-1])
        out.append(f"concept:ans:{shuffle_map[n]}")
    return out


def train_one(
    enc_ckpt: dict,
    mode: str,                        # "single" / "dual"
    seed: int,
    cfg: DatasetConfig,
    centroids: torch.Tensor,
    shuffle_map: dict[int, int] | None = None,
) -> dict:
    """训练 single 或 dual 一次, 返回 {bundle_state, final_*, meta}."""
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    cg = build_ans_graph(cfg.n_min, cfg.n_max, include_void=True)
    head_add = ArithmeticHeadV2(embed_dim=128, bias_dim=BIAS_DIM).to(DEVICE)
    head_cmp: ComparisonHead | None = None
    if mode == "dual":
        head_cmp = ComparisonHead(embed_dim=128, facet_dim=ORD_DIM, hidden_dim=64).to(DEVICE)

    with torch.no_grad():
        for n in range(cfg.n_min, cfg.n_max + 1):
            c = cg.concepts[f"concept:ans:{n}"]
            c.collapse("ArithmeticHeadV2", "arithmetic_bias", (BIAS_DIM,), tick=0, device=DEVICE)
            if mode == "dual":
                c.collapse("ComparisonHead", "ordinal_offset", (ORD_DIM,), tick=0, device=DEVICE)
    cg.bundles_to(torch.device(DEVICE))

    params = list(head_add.parameters()) + list(cg.iter_bundle_parameters())
    if head_cmp is not None:
        params = list(head_add.parameters()) + list(head_cmp.parameters()) + list(cg.iter_bundle_parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    for epoch in range(1, EPOCHS + 1):
        head_add.train()
        if head_cmp is not None:
            head_cmp.train()
        for step in range(STEPS_PER_EPOCH):
            a_l, b_l, op_l, c_l = _sample_arith(cfg, BATCH_SIZE, rng)
            ids_a = _apply_shuffle_map([f"concept:ans:{n}" for n in a_l], shuffle_map)
            ids_b = _apply_shuffle_map([f"concept:ans:{n}" for n in b_l], shuffle_map)
            op = _op_onehot(op_l)
            tgt = torch.tensor(c_l, device=DEVICE) - cfg.n_min
            dummy = torch.zeros(BATCH_SIZE, 128, device=DEVICE)
            pred = head_add(dummy, dummy, op, ids_a, ids_b, cg, tick=epoch * 10000 + step)
            la = F.cross_entropy(pred @ centroids.t(), tgt)
            total_loss = la
            if head_cmp is not None:
                ca_l, cb_l, clab = _sample_cmp(cfg, BATCH_SIZE, rng)
                cids_a = _apply_shuffle_map([f"concept:ans:{n}" for n in ca_l], shuffle_map)
                cids_b = _apply_shuffle_map([f"concept:ans:{n}" for n in cb_l], shuffle_map)
                ctgt = torch.tensor(clab, device=DEVICE)
                logits_c = head_cmp(None, None, cids_a, cids_b, cg, tick=epoch * 10000 + step)
                total_loss = la + F.cross_entropy(logits_c, ctgt)
            opt.zero_grad(); total_loss.backward(); opt.step()

    head_add.eval()
    if head_cmp is not None:
        head_cmp.eval()
    hits = total_n = 0
    with torch.no_grad():
        for _ in range(50):
            a_l, b_l, op_l, c_l = _sample_arith(cfg, 20, rng)
            ids_a = _apply_shuffle_map([f"concept:ans:{n}" for n in a_l], shuffle_map)
            ids_b = _apply_shuffle_map([f"concept:ans:{n}" for n in b_l], shuffle_map)
            op = _op_onehot(op_l)
            tgt = torch.tensor(c_l, device=DEVICE) - cfg.n_min
            dummy = torch.zeros(20, 128, device=DEVICE)
            pred = head_add(dummy, dummy, op, ids_a, ids_b, cg)
            hits += (pred @ centroids.t()).argmax(-1).eq(tgt).sum().item()
            total_n += 20
    add_acc = hits / max(total_n, 1)

    bundle_state = {cid: {k: v.detach().cpu() for k, v in c.bundle.state_dict().items()}
                    for cid, c in cg.concepts.items()}
    return {
        "mode": mode,
        "seed": seed,
        "shuffled": shuffle_map is not None,
        "add_acc": add_acc,
        "bundle_state": bundle_state,
        "concepts": sorted(cg.concepts.keys()),
    }


# ── Metrics ───────────────────────────────────────────────────────────


def _cos_matrix(bundle_state: dict, facet: str, ns: list[int]) -> torch.Tensor:
    rows = [bundle_state[f"concept:ans:{n}"][f"params.{facet}"] for n in ns]
    M = F.normalize(torch.stack(rows), dim=-1)
    return M @ M.t()


def _rho_vs_order(cos: torch.Tensor, ns: list[int]) -> float:
    n = cos.size(0)
    mask = ~torch.eye(n, dtype=torch.bool)
    off = cos[mask].numpy()
    neg_d = torch.tensor([[-abs(ns[i]-ns[j]) for j in range(n)] for i in range(n)], dtype=torch.float)
    return float(spearmanr(off, neg_d[mask].numpy())[0])


def _cross_facet_alignment(bs: dict, facet_a: str, facet_b: str, ns: list[int]) -> float:
    ca = _cos_matrix(bs, facet_a, ns)
    cb = _cos_matrix(bs, facet_b, ns)
    mask = ~torch.eye(len(ns), dtype=torch.bool)
    return float(spearmanr(ca[mask].numpy(), cb[mask].numpy())[0])


def ns_for_cfg(cfg: DatasetConfig) -> list[int]:
    return list(range(cfg.n_min, cfg.n_max + 1))


# ── Experiments ───────────────────────────────────────────────────────


def run_e1_multi_seed(enc_ckpt: dict, n_seeds: int, seed_base: int = 1000) -> dict:
    """E1: N=n_seeds 个不同 seed, 对比 single vs dual ρ 分布."""
    cfg = DatasetConfig(**enc_ckpt["ds_cfg"])
    enc = NumerosityEncoder(); enc.load_state_dict(enc_ckpt["encoder_state"]); enc.eval()
    for p in enc.parameters(): p.requires_grad_(False)
    centroids = _compute_centroids(enc, cfg)
    ns = ns_for_cfg(cfg)

    rows = []
    for i in range(n_seeds):
        seed = seed_base + i
        t0 = time.time()
        s = train_one(enc_ckpt, "single", seed, cfg, centroids)
        d = train_one(enc_ckpt, "dual",   seed, cfg, centroids)
        rho_s = _rho_vs_order(_cos_matrix(s["bundle_state"], "arithmetic_bias", ns), ns)
        rho_d = _rho_vs_order(_cos_matrix(d["bundle_state"], "arithmetic_bias", ns), ns)
        rho_d_ord = _rho_vs_order(_cos_matrix(d["bundle_state"], "ordinal_offset", ns), ns)
        align = _cross_facet_alignment(d["bundle_state"], "arithmetic_bias", "ordinal_offset", ns)
        dt = time.time() - t0
        print(f"[E1 seed={seed}] single_rho={rho_s:.4f}  dual_rho={rho_d:.4f}  "
              f"dual_ord={rho_d_ord:.4f}  align={align:.4f}  ({dt:.1f}s)")
        rows.append({
            "seed": seed,
            "single_rho_arith": rho_s, "single_add_acc": s["add_acc"],
            "dual_rho_arith": rho_d, "dual_rho_ord": rho_d_ord,
            "dual_cross_facet_align": align, "dual_add_acc": d["add_acc"],
            "wall_s": dt,
        })

    def _stats(xs):
        xs = list(xs)
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x-m)**2 for x in xs) / max(len(xs)-1, 1))
        return {"mean": m, "std": sd, "min": min(xs), "max": max(xs)}

    single = [r["single_rho_arith"] for r in rows]
    dual   = [r["dual_rho_arith"] for r in rows]
    dual_o = [r["dual_rho_ord"] for r in rows]
    align  = [r["dual_cross_facet_align"] for r in rows]
    tstat, pval = ttest_ind(single, dual, equal_var=False)

    return {
        "n_seeds": n_seeds,
        "per_seed": rows,
        "single_rho_arith": _stats(single),
        "dual_rho_arith":   _stats(dual),
        "dual_rho_ordinal": _stats(dual_o),
        "dual_cross_facet_alignment": _stats(align),
        "welch_t_test_single_vs_dual_arith": {
            "t": float(tstat), "p_value": float(pval),
            "conclusion": "significant" if pval < 0.05 else "not significant",
        },
    }


def run_e2_shuffled(enc_ckpt: dict, n_seeds: int, seed_base: int = 2000) -> dict:
    """E2: 反事实. 训练时把 concept_id → bundle 的映射随机打乱, ρ 应大幅下降."""
    cfg = DatasetConfig(**enc_ckpt["ds_cfg"])
    enc = NumerosityEncoder(); enc.load_state_dict(enc_ckpt["encoder_state"]); enc.eval()
    for p in enc.parameters(): p.requires_grad_(False)
    centroids = _compute_centroids(enc, cfg)
    ns = ns_for_cfg(cfg)

    rows = []
    for i in range(n_seeds):
        seed = seed_base + i
        shuffled = list(ns)
        random.Random(seed).shuffle(shuffled)
        sm = dict(zip(ns, shuffled))  # e.g. {1:3, 2:7, 3:1, ...}
        t0 = time.time()
        s = train_one(enc_ckpt, "single", seed, cfg, centroids, shuffle_map=sm)
        rho_s = _rho_vs_order(_cos_matrix(s["bundle_state"], "arithmetic_bias", ns), ns)
        dt = time.time() - t0
        print(f"[E2 seed={seed} shuffle={sm}] rho={rho_s:.4f} acc={s['add_acc']:.3f} ({dt:.1f}s)")
        rows.append({
            "seed": seed, "shuffle_map": sm,
            "single_rho_arith_on_raw_order": rho_s,
            "single_add_acc": s["add_acc"],
            "wall_s": dt,
        })

    def _stats(xs):
        xs = list(xs)
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x-m)**2 for x in xs) / max(len(xs)-1, 1))
        return {"mean": m, "std": sd, "min": min(xs), "max": max(xs)}

    abs_rho = [abs(r["single_rho_arith_on_raw_order"]) for r in rows]
    return {
        "n_seeds": n_seeds,
        "per_seed": rows,
        "abs_rho_stats": _stats(abs_rho),
        "notes": "在 shuffle_map 下训练, ρ 是对 RAW natural order 测的. 若 shuffle 破坏了身份, ρ 应 ≈ 0.",
    }


def run_e3_n_scan(enc_ckpt: dict, n_seeds: int, n_maxes: Iterable[int] = (5, 7, 9)) -> dict:
    """E3: N-scan. 改 cfg.n_max, 看 single ρ 是否稳定."""
    base_cfg = DatasetConfig(**enc_ckpt["ds_cfg"])
    enc = NumerosityEncoder(); enc.load_state_dict(enc_ckpt["encoder_state"]); enc.eval()
    for p in enc.parameters(): p.requires_grad_(False)

    result: dict[str, dict] = {}
    for n_max in n_maxes:
        cfg = dataclasses.replace(base_cfg, n_max=n_max)
        centroids = _compute_centroids(enc, cfg)
        ns = ns_for_cfg(cfg)
        rhos = []
        rows = []
        for i in range(n_seeds):
            seed = 3000 + n_max * 100 + i
            t0 = time.time()
            s = train_one(enc_ckpt, "single", seed, cfg, centroids)
            rho = _rho_vs_order(_cos_matrix(s["bundle_state"], "arithmetic_bias", ns), ns)
            rhos.append(rho)
            rows.append({"seed": seed, "rho": rho, "acc": s["add_acc"], "wall_s": time.time()-t0})
            print(f"[E3 N={n_max} seed={seed}] rho={rho:.4f} acc={s['add_acc']:.3f}")
        result[f"N={n_max}"] = {
            "per_seed": rows,
            "mean_rho": sum(rhos)/len(rhos),
            "std_rho": math.sqrt(sum((r - sum(rhos)/len(rhos))**2 for r in rhos)/max(len(rhos)-1,1)),
        }
    return result


def run_e4_permutation(dual_bundle: dict, ns: list[int], n_perm: int = 1000) -> dict:
    """E4: 对 arith↔ord cross-facet alignment 做 permutation test."""
    cos_a = _cos_matrix(dual_bundle, "arithmetic_bias", ns)
    cos_o = _cos_matrix(dual_bundle, "ordinal_offset", ns)
    mask = ~torch.eye(len(ns), dtype=torch.bool)
    oa = cos_a[mask].numpy()
    ob = cos_o[mask].numpy()
    observed = float(spearmanr(oa, ob)[0])

    rng = random.Random(0)
    ge = 0
    null_rhos = []
    for _ in range(n_perm):
        # 打乱 facet_b 对应的 concept 标签 (行列同时打乱)
        perm = list(range(len(ns)))
        rng.shuffle(perm)
        cos_o_perm = cos_o[perm][:, perm]
        ob_perm = cos_o_perm[mask].numpy()
        r = float(spearmanr(oa, ob_perm)[0])
        null_rhos.append(r)
        if abs(r) >= abs(observed):
            ge += 1
    p = (ge + 1) / (n_perm + 1)
    return {
        "observed_cross_facet_rho": observed,
        "n_permutations": n_perm,
        "p_value": p,
        "null_mean": sum(null_rhos)/len(null_rhos),
        "null_std": math.sqrt(sum((r-sum(null_rhos)/len(null_rhos))**2 for r in null_rhos)/max(len(null_rhos)-1,1)),
        "conclusion": "significant (p<0.01)" if p < 0.01 else ("significant (p<0.05)" if p < 0.05 else "not significant"),
    }


# ── 主入口 ─────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-ckpt", type=Path, required=True)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("outputs/robustness"))
    ap.add_argument("--skip", nargs="*", default=[], help="跳过的实验 (e1/e2/e3/e4)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    enc_ckpt = torch.load(args.encoder_ckpt, map_location="cpu", weights_only=False)

    summary: dict = {
        "n_seeds": args.n_seeds,
        "device": DEVICE,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "encoder_ckpt": str(args.encoder_ckpt),
    }

    if "e1" not in args.skip:
        print("=" * 60); print("E1: Multi-seed (single vs dual)"); print("=" * 60)
        summary["E1_multi_seed"] = run_e1_multi_seed(enc_ckpt, args.n_seeds)

    if "e2" not in args.skip:
        print("=" * 60); print("E2: Shuffled-concept 反事实"); print("=" * 60)
        summary["E2_shuffled"] = run_e2_shuffled(enc_ckpt, args.n_seeds)

    if "e3" not in args.skip:
        print("=" * 60); print("E3: N-scan"); print("=" * 60)
        n_seeds_e3 = min(args.n_seeds, 5)
        summary["E3_n_scan"] = run_e3_n_scan(enc_ckpt, n_seeds_e3)

    if "e4" not in args.skip and "E1_multi_seed" in summary:
        print("=" * 60); print("E4: Permutation test (用 E1 dual seed0)"); print("=" * 60)
        # 用 E1 第一个 dual seed 重新训一次, 取 bundle_state
        cfg = DatasetConfig(**enc_ckpt["ds_cfg"])
        enc = NumerosityEncoder(); enc.load_state_dict(enc_ckpt["encoder_state"]); enc.eval()
        for p in enc.parameters(): p.requires_grad_(False)
        centroids = _compute_centroids(enc, cfg)
        d = train_one(enc_ckpt, "dual", 1000, cfg, centroids)
        summary["E4_permutation"] = run_e4_permutation(d["bundle_state"], ns_for_cfg(cfg), n_perm=1000)

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60); print("SUMMARY"); print("=" * 60)
    if "E1_multi_seed" in summary:
        e1 = summary["E1_multi_seed"]
        s = e1["single_rho_arith"]; d = e1["dual_rho_arith"]
        print(f"E1  single ρ = {s['mean']:.4f} ± {s['std']:.4f}  (min {s['min']:.3f}, max {s['max']:.3f})")
        print(f"E1  dual   ρ = {d['mean']:.4f} ± {d['std']:.4f}  (min {d['min']:.3f}, max {d['max']:.3f})")
        t = e1["welch_t_test_single_vs_dual_arith"]
        print(f"E1  Welch t = {t['t']:.3f}, p = {t['p_value']:.4f} → {t['conclusion']}")
        al = e1["dual_cross_facet_alignment"]
        print(f"E1  cross-facet align (dual) = {al['mean']:.4f} ± {al['std']:.4f}")
    if "E2_shuffled" in summary:
        e2 = summary["E2_shuffled"]
        print(f"E2  |ρ| shuffled = {e2['abs_rho_stats']['mean']:.4f} ± {e2['abs_rho_stats']['std']:.4f}")
    if "E3_n_scan" in summary:
        for key, v in summary["E3_n_scan"].items():
            print(f"E3  {key}: ρ = {v['mean_rho']:.4f} ± {v['std_rho']:.4f}")
    if "E4_permutation" in summary:
        e4 = summary["E4_permutation"]
        print(f"E4  observed = {e4['observed_cross_facet_rho']:.4f}, "
              f"null mean = {e4['null_mean']:.4f}, p = {e4['p_value']:.4f} → {e4['conclusion']}")


if __name__ == "__main__":
    main()
