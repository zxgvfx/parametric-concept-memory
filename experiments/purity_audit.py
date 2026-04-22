"""purity_audit.py — 四路 ablation 验证 robustness 结果不是因为信息泄漏.

审计的潜在"信息渠道" (详见 docs/research/PURITY_AUDIT.md):

  C1  预训练 NumerosityEncoder 权重 → centroids 已有 ordinal 结构
  C6  "normal_small" init (std=0.01) 让所有 bundle 初始相似
  C8  E2 shuffle 只是打乱了坐标系, 没真破坏 identity
  C4  ``concept:ans:N`` 的 ID 字符串本身含数字

四个 Assay:

  A1  Random-centroid: 用随机正交 centroid 代替 encoder centroid,
      重训 single + dual. 如果 ρ 仍然 >> 0, 证明 coherence 不是 encoder
      泄漏, 而是"学习 arithmetic 必然诱导 ordinal" 的结构性结论.

  A2  Shuffle-inverse: E2 方式训练 (shuffle_map), 但既测 natural-order ρ
      又测 shuffle-inverse ρ (把 bundle 按 shuffle_map 还原到正确坐标).
      如果 inverse ρ 恢复到 ≈ 0.95, 说明 bundle 其实学到了正确的 identity
      只是 coordinate 被打乱 → 反而**更强地**支持 H5'.
      如果 inverse ρ 仍然低, 说明 shuffle 真的破坏了 identity 学习.

  A3  Init-scale: 把 ParamBundle init 改为 ``normal`` (std=1.0), 让 bundle
      起点就有大幅分散. 如果 ρ 仍然高, 证明 coherence 不是 "near-zero
      init + 扰动放大" 这个 trivial 机制.

  A4  Random-ID: 把 concept_id 从 ``concept:ans:1..7`` 改成 ``concept:ans:<random_hex>``,
      保持 identity 映射. 如果 ρ 仍然高, 证明 ID 字符串本身没信息泄漏.

产出:
  outputs/purity_audit/summary.json
  outputs/purity_audit/report.md

运行:
  python -m experiments.purity_audit \\
      --encoder-ckpt outputs/ans_encoder/final.pt --n-seeds 5
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
import uuid
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from pcm.concept_graph import ConceptGraph
from pcm.heads.arithmetic_head_v2 import ArithmeticHeadV2
from pcm.heads.comparison_head import ComparisonHead
from pcm.heads.numerosity_encoder import (
    DatasetConfig,
    NumerosityEncoder,
    generate_dot_canvas,
)
from experiments.robustness_study import (
    BATCH_SIZE,
    BIAS_DIM,
    DEVICE,
    EPOCHS,
    LR,
    ORD_DIM,
    STEPS_PER_EPOCH,
    _compute_centroids,
    _cos_matrix,
    _op_onehot,
    _rho_vs_order,
    _sample_arith,
    _sample_cmp,
)

# ────────────────────────────────────────────────────────────────────────
# 1. 随机 centroid 生成 (A1 用)
# ────────────────────────────────────────────────────────────────────────


def make_random_orthogonal_centroids(n_classes: int, dim: int, seed: int) -> torch.Tensor:
    """生成 n_classes 个**随机正交**单位向量, 作为 arithmetic 分类 target.

    这故意剥离任何"数量序"信息 (cos(c_i, c_j) ≈ 0 for i ≠ j), 模拟
    一个没有内禀 ordinal 的 supervision target.
    """
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, n_classes, generator=g)
    Q, _ = torch.linalg.qr(A)   # (dim, n_classes)
    return F.normalize(Q.t(), dim=-1).to(DEVICE)


def make_random_gaussian_centroids(n_classes: int, dim: int, seed: int) -> torch.Tensor:
    """生成 n_classes 个独立 L2-normalized 高斯向量 (近似正交, 对照)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_classes, dim, generator=g)
    return F.normalize(X, dim=-1).to(DEVICE)


# ────────────────────────────────────────────────────────────────────────
# 2. 自定义 graph builder (A4 用)
# ────────────────────────────────────────────────────────────────────────


def build_graph_with_id_fn(n_min: int, n_max: int, id_fn: Callable[[int], str]) -> tuple[ConceptGraph, dict[int, str]]:
    """按 id_fn(n) 生成 concept_id. 返回 (cg, {n -> concept_id})."""
    cg = ConceptGraph(feat_dim=128)
    id_map: dict[int, str] = {}
    for n in range(n_min, n_max + 1):
        cid = id_fn(n)
        cg.register_concept(
            node_id=cid,
            label=f"ANS_{n}",
            scope="BASE",
            provenance=f"purity_audit:n={n}",
        )
        id_map[n] = cid
    return cg, id_map


# ────────────────────────────────────────────────────────────────────────
# 3. 通用训练函数 (支持所有 ablation)
# ────────────────────────────────────────────────────────────────────────


def purity_train_one(
    mode: str,                                    # "single" / "dual"
    seed: int,
    cfg: DatasetConfig,
    centroids: torch.Tensor,
    *,
    id_fn: Callable[[int], str] | None = None,
    shuffle_map: dict[int, int] | None = None,
    init_strategy: str = "normal_small",
) -> dict:
    """通用单轮训练. 返回 bundle_state (按 ``n`` 键存) 方便后续各种度量.

    Returns:
      dict:
        - mode, seed
        - add_acc (final)
        - bundle_by_n: {n: {"arithmetic_bias": tensor, "ordinal_offset": tensor?}}
          (按自然数 n 索引, 不依赖 concept_id 具体形式)
        - id_map: {n: concept_id_used}
        - shuffled: bool
    """
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    id_fn = id_fn or (lambda n: f"concept:ans:{n}")
    cg, id_map = build_graph_with_id_fn(cfg.n_min, cfg.n_max, id_fn)

    head_add = ArithmeticHeadV2(embed_dim=128, bias_dim=BIAS_DIM).to(DEVICE)
    head_cmp: ComparisonHead | None = None
    if mode == "dual":
        head_cmp = ComparisonHead(embed_dim=128, facet_dim=ORD_DIM, hidden_dim=64).to(DEVICE)

    with torch.no_grad():
        for n in range(cfg.n_min, cfg.n_max + 1):
            c = cg.concepts[id_map[n]]
            c.collapse("ArithmeticHeadV2", "arithmetic_bias",
                       (BIAS_DIM,), tick=0, device=DEVICE, init=init_strategy)
            if mode == "dual":
                c.collapse("ComparisonHead", "ordinal_offset",
                           (ORD_DIM,), tick=0, device=DEVICE, init=init_strategy)
    cg.bundles_to(torch.device(DEVICE))

    params = list(head_add.parameters()) + list(cg.iter_bundle_parameters())
    if head_cmp is not None:
        params = list(head_add.parameters()) + list(head_cmp.parameters()) + list(cg.iter_bundle_parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    def map_ids(ns_list: list[int]) -> list[str]:
        if shuffle_map is None:
            return [id_map[n] for n in ns_list]
        return [id_map[shuffle_map[n]] for n in ns_list]

    for epoch in range(1, EPOCHS + 1):
        head_add.train()
        if head_cmp is not None:
            head_cmp.train()
        for step in range(STEPS_PER_EPOCH):
            a_l, b_l, op_l, c_l = _sample_arith(cfg, BATCH_SIZE, rng)
            ids_a = map_ids(a_l)
            ids_b = map_ids(b_l)
            op = _op_onehot(op_l)
            tgt = torch.tensor(c_l, device=DEVICE) - cfg.n_min
            dummy = torch.zeros(BATCH_SIZE, 128, device=DEVICE)
            pred = head_add(dummy, dummy, op, ids_a, ids_b, cg, tick=epoch * 10000 + step)
            la = F.cross_entropy(pred @ centroids.t(), tgt)
            total_loss = la
            if head_cmp is not None:
                ca_l, cb_l, clab = _sample_cmp(cfg, BATCH_SIZE, rng)
                cids_a = map_ids(ca_l)
                cids_b = map_ids(cb_l)
                ctgt = torch.tensor(clab, device=DEVICE)
                logits_c = head_cmp(None, None, cids_a, cids_b, cg, tick=epoch * 10000 + step)
                total_loss = la + F.cross_entropy(logits_c, ctgt)
            opt.zero_grad(); total_loss.backward(); opt.step()

    # eval
    head_add.eval()
    if head_cmp is not None:
        head_cmp.eval()
    hits = total_n = 0
    with torch.no_grad():
        for _ in range(50):
            a_l, b_l, op_l, c_l = _sample_arith(cfg, 20, rng)
            ids_a = map_ids(a_l)
            ids_b = map_ids(b_l)
            op = _op_onehot(op_l)
            tgt = torch.tensor(c_l, device=DEVICE) - cfg.n_min
            dummy = torch.zeros(20, 128, device=DEVICE)
            pred = head_add(dummy, dummy, op, ids_a, ids_b, cg)
            hits += (pred @ centroids.t()).argmax(-1).eq(tgt).sum().item()
            total_n += 20
    add_acc = hits / max(total_n, 1)

    bundle_by_n: dict[int, dict] = {}
    for n in range(cfg.n_min, cfg.n_max + 1):
        cid = id_map[n]
        state = cg.concepts[cid].bundle.state_dict()
        entry: dict = {}
        for k, v in state.items():
            # k looks like "params.arithmetic_bias" or "params.ordinal_offset"
            facet = k.split(".", 1)[-1]
            entry[facet] = v.detach().cpu().clone()
        bundle_by_n[n] = entry

    return {
        "mode": mode,
        "seed": seed,
        "shuffled": shuffle_map is not None,
        "init": init_strategy,
        "add_acc": add_acc,
        "bundle_by_n": bundle_by_n,
        "id_map": {str(n): cid for n, cid in id_map.items()},
    }


# ────────────────────────────────────────────────────────────────────────
# 4. 度量工具 (按 n 索引)
# ────────────────────────────────────────────────────────────────────────


def _cos_matrix_by_n(bundle_by_n: dict[int, dict], facet: str, ns: list[int]) -> torch.Tensor:
    rows = [bundle_by_n[n][facet] for n in ns]
    M = F.normalize(torch.stack(rows), dim=-1)
    return M @ M.t()


def rho_by_n(bundle_by_n: dict[int, dict], facet: str, ns: list[int]) -> float:
    cos = _cos_matrix_by_n(bundle_by_n, facet, ns)
    return _rho_vs_order(cos, ns)


def rho_with_inverse_remap(
    bundle_by_n: dict[int, dict],
    facet: str,
    ns: list[int],
    shuffle_map: dict[int, int],
) -> float:
    """A2 核心: 用 shuffle_map 把 bundle 还原到正确的数量坐标后再测 ρ.

    shuffle_map 是 "自然数 n -> 训练中替代的 bundle_id". 所以训练后
    bundle[n] 实际承载的是"数量 shuffle_map^-1(n)"的语义. 要测它是否
    真的学到了数量序, 我们按 shuffle_map 重排: 对自然序 i, 取 bundle[sm[i]].
    """
    remapped: dict[int, dict] = {i: bundle_by_n[shuffle_map[i]] for i in ns}
    return rho_by_n(remapped, facet, ns)


def _stats(xs: list[float]) -> dict:
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
    return {"mean": m, "std": sd, "min": min(xs), "max": max(xs), "n": len(xs)}


# ────────────────────────────────────────────────────────────────────────
# 5. 四个 Assay
# ────────────────────────────────────────────────────────────────────────


def assay_a1_random_centroids(enc_ckpt: dict, n_seeds: int, cent_type: str) -> dict:
    """A1: 用随机 centroid (正交或高斯) 替代 encoder centroid, 重训 single + dual."""
    cfg = DatasetConfig(**enc_ckpt["ds_cfg"])
    ns = list(range(cfg.n_min, cfg.n_max + 1))
    n_classes = 2 * cfg.n_max - cfg.n_min + 1  # 0..2*n_max-n_min, arith output range

    rows = []
    for i in range(n_seeds):
        seed = 10000 + i
        if cent_type == "orthogonal":
            C = make_random_orthogonal_centroids(n_classes, 128, seed)
        elif cent_type == "gaussian":
            C = make_random_gaussian_centroids(n_classes, 128, seed)
        else:
            raise ValueError(cent_type)
        t0 = time.time()
        s = purity_train_one("single", seed, cfg, C)
        d = purity_train_one("dual",   seed, cfg, C)
        rho_s = rho_by_n(s["bundle_by_n"], "arithmetic_bias", ns)
        rho_d = rho_by_n(d["bundle_by_n"], "arithmetic_bias", ns)
        rho_d_ord = rho_by_n(d["bundle_by_n"], "ordinal_offset", ns)
        dt = time.time() - t0
        print(f"[A1/{cent_type} seed={seed}] single_rho={rho_s:+.4f}  "
              f"dual_rho={rho_d:+.4f}  dual_ord={rho_d_ord:+.4f}  "
              f"single_acc={s['add_acc']:.3f}  dual_acc={d['add_acc']:.3f}  ({dt:.1f}s)")
        rows.append({
            "seed": seed,
            "single_rho_arith": rho_s, "dual_rho_arith": rho_d,
            "dual_rho_ord": rho_d_ord,
            "single_add_acc": s["add_acc"], "dual_add_acc": d["add_acc"],
            "wall_s": dt,
        })
    return {
        "centroid_type": cent_type,
        "n_seeds": n_seeds,
        "per_seed": rows,
        "single_rho_arith": _stats([abs(r["single_rho_arith"]) for r in rows]),
        "dual_rho_arith":   _stats([abs(r["dual_rho_arith"]) for r in rows]),
        "dual_rho_ord":     _stats([abs(r["dual_rho_ord"]) for r in rows]),
        "interpretation":
            "若 |ρ| 仍 > 0.8 → ordinal 来自 arithmetic 任务结构, 非 encoder 污染. "
            "若 |ρ| 崩到 < 0.3 → encoder centroid 是主要驱动.",
    }


def assay_a2_shuffle_inverse(enc_ckpt: dict, n_seeds: int) -> dict:
    """A2: shuffle_map 训练, 同时测 natural-order ρ 和 inverse-remapped ρ."""
    cfg = DatasetConfig(**enc_ckpt["ds_cfg"])
    enc = NumerosityEncoder(); enc.load_state_dict(enc_ckpt["encoder_state"]); enc.eval()
    for p in enc.parameters(): p.requires_grad_(False)
    centroids = _compute_centroids(enc, cfg)
    ns = list(range(cfg.n_min, cfg.n_max + 1))

    rows = []
    for i in range(n_seeds):
        seed = 20000 + i
        shuffled = list(ns)
        random.Random(seed).shuffle(shuffled)
        sm = dict(zip(ns, shuffled))
        t0 = time.time()
        r = purity_train_one("single", seed, cfg, centroids, shuffle_map=sm)
        rho_nat = rho_by_n(r["bundle_by_n"], "arithmetic_bias", ns)
        rho_inv = rho_with_inverse_remap(r["bundle_by_n"], "arithmetic_bias", ns, sm)
        dt = time.time() - t0
        print(f"[A2 seed={seed} sm={sm}] nat_rho={rho_nat:+.4f}  inv_rho={rho_inv:+.4f}  "
              f"acc={r['add_acc']:.3f}  ({dt:.1f}s)")
        rows.append({
            "seed": seed, "shuffle_map": sm,
            "rho_natural_order": rho_nat,
            "rho_inverse_remapped": rho_inv,
            "add_acc": r["add_acc"],
            "wall_s": dt,
        })
    return {
        "n_seeds": n_seeds,
        "per_seed": rows,
        "rho_natural": _stats([abs(r["rho_natural_order"]) for r in rows]),
        "rho_inverse_remapped": _stats([abs(r["rho_inverse_remapped"]) for r in rows]),
        "interpretation":
            "若 inverse ρ >> natural ρ (e.g. 0.95 vs 0.2) → shuffle 只打乱坐标, "
            "bundle 仍然学到了数量 identity → H5' 更强. "
            "若两者都低 → shuffle 真的破坏了 identity 学习, E2 结论为保守的下界.",
    }


def assay_a3_init_scale(enc_ckpt: dict, n_seeds: int) -> dict:
    """A3: 对比 init='normal_small' (baseline) vs init='normal' vs init='zero'."""
    cfg = DatasetConfig(**enc_ckpt["ds_cfg"])
    enc = NumerosityEncoder(); enc.load_state_dict(enc_ckpt["encoder_state"]); enc.eval()
    for p in enc.parameters(): p.requires_grad_(False)
    centroids = _compute_centroids(enc, cfg)
    ns = list(range(cfg.n_min, cfg.n_max + 1))

    all_strategies = ["normal_small", "normal", "zero"]
    out: dict[str, dict] = {}
    for strat in all_strategies:
        rows = []
        for i in range(n_seeds):
            seed = 30000 + i
            t0 = time.time()
            r = purity_train_one("single", seed, cfg, centroids, init_strategy=strat)
            rho = rho_by_n(r["bundle_by_n"], "arithmetic_bias", ns)
            dt = time.time() - t0
            print(f"[A3 init={strat} seed={seed}] rho={rho:+.4f}  acc={r['add_acc']:.3f}  ({dt:.1f}s)")
            rows.append({"seed": seed, "rho": rho, "acc": r["add_acc"], "wall_s": dt})
        out[strat] = {
            "per_seed": rows,
            "abs_rho": _stats([abs(r["rho"]) for r in rows]),
            "acc": _stats([r["acc"] for r in rows]),
        }
    out["interpretation"] = (
        "若三种 init 的 |ρ| 都很高 (>0.9) → coherence 与初始 scale 无关, "
        "是梯度流的结构性结果. 若只有 normal_small 高 → 可能是 near-zero "
        "初始化 + small-perturbation 的 trivial 涌现."
    )
    return out


def assay_a4_random_id(enc_ckpt: dict, n_seeds: int) -> dict:
    """A4: 用随机 UUID 作为 concept_id, 保持 n ↔ bundle 一对一."""
    cfg = DatasetConfig(**enc_ckpt["ds_cfg"])
    enc = NumerosityEncoder(); enc.load_state_dict(enc_ckpt["encoder_state"]); enc.eval()
    for p in enc.parameters(): p.requires_grad_(False)
    centroids = _compute_centroids(enc, cfg)
    ns = list(range(cfg.n_min, cfg.n_max + 1))

    rows = []
    for i in range(n_seeds):
        seed = 40000 + i
        rnd = random.Random(seed)
        uuids = [uuid.UUID(int=rnd.getrandbits(128)).hex[:12] for _ in ns]
        # id_fn 按 n 返回固定 uuid, 保持 1:1
        def id_fn(n, _u=uuids, _ns=ns):
            return f"concept:opaque:{_u[_ns.index(n)]}"
        t0 = time.time()
        r = purity_train_one("single", seed, cfg, centroids, id_fn=id_fn)
        rho = rho_by_n(r["bundle_by_n"], "arithmetic_bias", ns)
        dt = time.time() - t0
        print(f"[A4 seed={seed} id_prefix=opaque] rho={rho:+.4f}  acc={r['add_acc']:.3f}  ({dt:.1f}s)")
        rows.append({
            "seed": seed, "rho": rho, "acc": r["add_acc"],
            "sample_id": uuids[0], "wall_s": dt,
        })
    return {
        "n_seeds": n_seeds,
        "per_seed": rows,
        "abs_rho": _stats([abs(r["rho"]) for r in rows]),
        "acc": _stats([r["acc"] for r in rows]),
        "interpretation":
            "若 |ρ| ≈ baseline (0.95+) → ID 字符串本身无信息泄漏, "
            "identity 纯粹由训练时 n↔bundle 的对应关系建立.",
    }


# ────────────────────────────────────────────────────────────────────────
# 6. 主入口 + 报告
# ────────────────────────────────────────────────────────────────────────


def render_report(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# Purity Audit Report")
    lines.append("")
    lines.append(f"- encoder_ckpt: `{summary['encoder_ckpt']}`")
    lines.append(f"- n_seeds: {summary['n_seeds']}")
    lines.append(f"- device: {summary['device']}")
    lines.append(f"- epochs: {summary['epochs']}, steps/epoch: {summary['steps_per_epoch']}")
    lines.append("")

    if "A1_random_orthogonal" in summary:
        a = summary["A1_random_orthogonal"]
        lines.append("## A1 · Random Orthogonal Centroids (encoder 污染测试)")
        lines.append("")
        lines.append(f"- single |ρ| = **{a['single_rho_arith']['mean']:.4f} ± {a['single_rho_arith']['std']:.4f}**")
        lines.append(f"- dual   |ρ| = **{a['dual_rho_arith']['mean']:.4f} ± {a['dual_rho_arith']['std']:.4f}**")
        lines.append(f"- dual   |ρ_ord| = {a['dual_rho_ord']['mean']:.4f} ± {a['dual_rho_ord']['std']:.4f}")
        lines.append("")

    if "A1_random_gaussian" in summary:
        a = summary["A1_random_gaussian"]
        lines.append("## A1b · Random Gaussian Centroids (对照)")
        lines.append("")
        lines.append(f"- single |ρ| = **{a['single_rho_arith']['mean']:.4f} ± {a['single_rho_arith']['std']:.4f}**")
        lines.append(f"- dual   |ρ| = **{a['dual_rho_arith']['mean']:.4f} ± {a['dual_rho_arith']['std']:.4f}**")
        lines.append("")

    if "A2_shuffle_inverse" in summary:
        a = summary["A2_shuffle_inverse"]
        lines.append("## A2 · Shuffle-Inverse (shuffle 是坐标破坏还是 identity 破坏)")
        lines.append("")
        lines.append(f"- natural-order |ρ|   = {a['rho_natural']['mean']:.4f} ± {a['rho_natural']['std']:.4f}")
        lines.append(f"- inverse-remap |ρ|  = **{a['rho_inverse_remapped']['mean']:.4f} ± {a['rho_inverse_remapped']['std']:.4f}**")
        lines.append("")

    if "A3_init_scale" in summary:
        a = summary["A3_init_scale"]
        lines.append("## A3 · Init-Scale (near-zero init trivial 解释)")
        lines.append("")
        for strat in ("normal_small", "normal", "zero"):
            if strat in a:
                s = a[strat]["abs_rho"]; acc = a[strat]["acc"]
                lines.append(f"- init=`{strat}`: |ρ| = {s['mean']:.4f} ± {s['std']:.4f}, acc = {acc['mean']:.3f} ± {acc['std']:.3f}")
        lines.append("")

    if "A4_random_id" in summary:
        a = summary["A4_random_id"]
        lines.append("## A4 · Random Concept-ID (ID 字符串泄漏)")
        lines.append("")
        lines.append(f"- |ρ| = **{a['abs_rho']['mean']:.4f} ± {a['abs_rho']['std']:.4f}**")
        lines.append(f"- acc = {a['acc']['mean']:.3f} ± {a['acc']['std']:.3f}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-ckpt", type=Path, required=True)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("outputs/purity_audit"))
    ap.add_argument("--skip", nargs="*", default=[], help="跳过 (a1/a1b/a2/a3/a4)")
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

    if "a1" not in args.skip:
        print("=" * 60); print("A1: Random Orthogonal Centroids"); print("=" * 60)
        summary["A1_random_orthogonal"] = assay_a1_random_centroids(enc_ckpt, args.n_seeds, "orthogonal")

    if "a1b" not in args.skip:
        print("=" * 60); print("A1b: Random Gaussian Centroids"); print("=" * 60)
        summary["A1_random_gaussian"] = assay_a1_random_centroids(enc_ckpt, args.n_seeds, "gaussian")

    if "a2" not in args.skip:
        print("=" * 60); print("A2: Shuffle-Inverse"); print("=" * 60)
        summary["A2_shuffle_inverse"] = assay_a2_shuffle_inverse(enc_ckpt, args.n_seeds)

    if "a3" not in args.skip:
        print("=" * 60); print("A3: Init-Scale"); print("=" * 60)
        summary["A3_init_scale"] = assay_a3_init_scale(enc_ckpt, args.n_seeds)

    if "a4" not in args.skip:
        print("=" * 60); print("A4: Random Concept-ID"); print("=" * 60)
        summary["A4_random_id"] = assay_a4_random_id(enc_ckpt, args.n_seeds)

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (args.out / "report.md").write_text(render_report(summary))
    print("\nWrote:")
    print(f"  {args.out/'summary.json'}")
    print(f"  {args.out/'report.md'}")


if __name__ == "__main__":
    main()
