"""Four ablation assays (A1 / A2 / A3 / A4) for the purity audit."""
from __future__ import annotations

import random
import time
import uuid

from pcm.heads.numerosity_encoder import DatasetConfig, NumerosityEncoder

from experiments.robustness_study import _compute_centroids

from .centroids import (
    make_random_gaussian_centroids,
    make_random_orthogonal_centroids,
)
from .metrics import _stats, rho_by_n, rho_with_inverse_remap
from .train import purity_train_one


__all__ = [
    "assay_a1_random_centroids",
    "assay_a2_shuffle_inverse",
    "assay_a3_init_scale",
    "assay_a4_random_id",
]


def assay_a1_random_centroids(enc_ckpt: dict, n_seeds: int, cent_type: str) -> dict:
    """A1: 用随机 centroid (正交或高斯) 替代 encoder centroid, 重训 single + dual."""
    cfg = DatasetConfig(**enc_ckpt["ds_cfg"])
    ns = list(range(cfg.n_min, cfg.n_max + 1))
    n_classes = 2 * cfg.n_max - cfg.n_min + 1  # arith output range

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
    for p in enc.parameters():
        p.requires_grad_(False)
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
    for p in enc.parameters():
        p.requires_grad_(False)
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
    for p in enc.parameters():
        p.requires_grad_(False)
    centroids = _compute_centroids(enc, cfg)
    ns = list(range(cfg.n_min, cfg.n_max + 1))

    rows = []
    for i in range(n_seeds):
        seed = 40000 + i
        rnd = random.Random(seed)
        uuids = [uuid.UUID(int=rnd.getrandbits(128)).hex[:12] for _ in ns]

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
