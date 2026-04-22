"""scale_study.py — 扩大数字范围, 观察 bundle 几何随 N 的演化.

研究问题:

  Q1. ρ (数量序 coherence) 随 N 增大是否稳定? 还是会崩塌?
  Q2. bundle 的内在"数字线"是 linear 的还是 log-scale 的 (Weber's law)?
  Q3. add-only / sub-only / add+sub mixed 三种任务驱动是否给出同构几何?

实验设置:

  - 用 random-orthogonal centroids (purity audit A1 已证明 ρ ≡ real centroid),
    绕过 NumerosityEncoder 只训到 N=7 的限制, 允许任意扩展 N.
  - N ∈ {7, 15, 30}.
  - 3 个 setup: ``mix`` (add 50% + sub 50%, baseline), ``add_only``, ``sub_only``.
  - n_seeds=3 per (N, setup).

产出:
  outputs/scale_study/{summary.json, report.md}

运行:
  python -m experiments.scale_study --n-seeds 3

关键度量:

  - ``rho_linear``: spearman(cos, -|n-m|)           — 线性数字线
  - ``rho_log``:    spearman(cos, -|log n - log m|) — 对数数字线 (Weber)
  - ``rho_sqrt``:   spearman(cos, -sqrt(|n-m|))     — power-law 替代
  - ``cross_setup_rho``: spearman of vec(cos_A) vs vec(cos_B) — 任务组合之间
    几何是否同构 (add-only vs sub-only vs mix).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from pcm.heads.arithmetic_head_v2 import ArithmeticHeadV2
from pcm.heads.numerosity_encoder import DatasetConfig
from experiments.purity_audit import (
    build_graph_with_id_fn,
    make_random_orthogonal_centroids,
)
from experiments.robustness_study import (
    BATCH_SIZE,
    BIAS_DIM,
    DEVICE,
    LR,
    _op_onehot,
)

OpFilter = Literal["mix", "add", "sub"]


# ────────────────────────────────────────────────────────────────────────
# 按 scale / filter 定制的数据采样 (允许 n_max 自由扩展)
# ────────────────────────────────────────────────────────────────────────


def _sample_arith_filtered(
    cfg: DatasetConfig, bs: int, rng: torch.Generator, op_filter: OpFilter
):
    """采样 (a, b, op, c) batch. op_filter 控制 add/sub/mix."""
    a_l, b_l, op_l, c_l = [], [], [], []
    for _ in range(bs):
        if op_filter == "mix":
            op = "add" if torch.rand(1, generator=rng).item() < 0.5 else "sub"
        else:
            op = op_filter
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


# ────────────────────────────────────────────────────────────────────────
# 训练 (支持 scale / op_filter / 动态 epoch)
# ────────────────────────────────────────────────────────────────────────


def train_scale_one(
    cfg: DatasetConfig,
    seed: int,
    *,
    op_filter: OpFilter,
    centroids: torch.Tensor,
    epochs: int = 12,
    steps_per_epoch: int = 120,
    id_fn: Callable[[int], str] | None = None,
) -> dict:
    """单 muscle arithmetic, 支持 add/sub/mix."""
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    id_fn = id_fn or (lambda n: f"concept:ans:{n}")
    cg, id_map = build_graph_with_id_fn(cfg.n_min, cfg.n_max, id_fn)

    head_add = ArithmeticHeadV2(embed_dim=128, bias_dim=BIAS_DIM).to(DEVICE)
    with torch.no_grad():
        for n in range(cfg.n_min, cfg.n_max + 1):
            cg.concepts[id_map[n]].collapse(
                "ArithmeticHeadV2", "arithmetic_bias",
                (BIAS_DIM,), tick=0, device=DEVICE, init="normal_small",
            )
    cg.bundles_to(torch.device(DEVICE))

    params = list(head_add.parameters()) + list(cg.iter_bundle_parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        head_add.train()
        for step in range(steps_per_epoch):
            a_l, b_l, op_l, c_l = _sample_arith_filtered(cfg, BATCH_SIZE, rng, op_filter)
            ids_a = [id_map[n] for n in a_l]
            ids_b = [id_map[n] for n in b_l]
            op = _op_onehot(op_l)
            tgt = torch.tensor(c_l, device=DEVICE) - cfg.n_min
            dummy = torch.zeros(BATCH_SIZE, 128, device=DEVICE)
            pred = head_add(dummy, dummy, op, ids_a, ids_b, cg, tick=epoch * 10000 + step)
            loss = F.cross_entropy(pred @ centroids.t(), tgt)
            opt.zero_grad(); loss.backward(); opt.step()

    head_add.eval()
    hits = total_n = 0
    with torch.no_grad():
        for _ in range(80):
            a_l, b_l, op_l, c_l = _sample_arith_filtered(cfg, 20, rng, op_filter)
            ids_a = [id_map[n] for n in a_l]
            ids_b = [id_map[n] for n in b_l]
            op = _op_onehot(op_l)
            tgt = torch.tensor(c_l, device=DEVICE) - cfg.n_min
            dummy = torch.zeros(20, 128, device=DEVICE)
            pred = head_add(dummy, dummy, op, ids_a, ids_b, cg)
            hits += (pred @ centroids.t()).argmax(-1).eq(tgt).sum().item()
            total_n += 20
    acc = hits / max(total_n, 1)

    bundle_by_n = {
        n: cg.concepts[id_map[n]].bundle.state_dict()["params.arithmetic_bias"].detach().cpu().clone()
        for n in range(cfg.n_min, cfg.n_max + 1)
    }
    return {
        "N": cfg.n_max - cfg.n_min + 1,
        "seed": seed,
        "op_filter": op_filter,
        "acc": acc,
        "bundle_by_n": bundle_by_n,
    }


# ────────────────────────────────────────────────────────────────────────
# 度量: Weber's law 检测
# ────────────────────────────────────────────────────────────────────────


def _cos_matrix(bundle_by_n: dict[int, torch.Tensor], ns: list[int]) -> torch.Tensor:
    M = F.normalize(torch.stack([bundle_by_n[n] for n in ns]), dim=-1)
    return M @ M.t()


def compute_rho_variants(bundle_by_n: dict[int, torch.Tensor], ns: list[int]) -> dict:
    """对三种数字线假设分别计算 spearman ρ (higher = better fit)."""
    cos = _cos_matrix(bundle_by_n, ns)
    mask = ~torch.eye(len(ns), dtype=torch.bool)
    y = cos[mask].numpy()

    # linear: -|Δn|
    d_lin = torch.tensor([[-abs(i - j) for j in ns] for i in ns], dtype=torch.float)
    rho_lin = float(spearmanr(y, d_lin[mask].numpy())[0])

    # log: -|log n - log m|
    d_log = torch.tensor(
        [[-abs(math.log(i) - math.log(j)) for j in ns] for i in ns],
        dtype=torch.float,
    )
    rho_log = float(spearmanr(y, d_log[mask].numpy())[0])

    # sqrt: -sqrt|Δn|
    d_sqrt = torch.tensor([[-math.sqrt(abs(i - j)) for j in ns] for i in ns], dtype=torch.float)
    rho_sqrt = float(spearmanr(y, d_sqrt[mask].numpy())[0])

    return {"rho_linear": rho_lin, "rho_log": rho_log, "rho_sqrt": rho_sqrt}


def cross_setup_cos_rho(
    bundle_a: dict[int, torch.Tensor],
    bundle_b: dict[int, torch.Tensor],
    ns: list[int],
) -> float:
    """两种 op_filter 训出 bundle 的 cos matrix 间的 spearman 相关."""
    cos_a = _cos_matrix(bundle_a, ns)
    cos_b = _cos_matrix(bundle_b, ns)
    mask = ~torch.eye(len(ns), dtype=torch.bool)
    return float(spearmanr(cos_a[mask].numpy(), cos_b[mask].numpy())[0])


# ────────────────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────────────────


def _stats(xs: list[float]) -> dict:
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
    return {"mean": m, "std": sd, "min": min(xs), "max": max(xs), "n": len(xs)}


def run_one_config(
    N: int, op_filter: OpFilter, n_seeds: int, epochs: int, steps_per_epoch: int
) -> dict:
    cfg = DatasetConfig(n_min=1, n_max=N)
    ns = list(range(1, N + 1))
    n_classes = N
    rows = []
    bundles_by_seed: dict[int, dict] = {}
    for i in range(n_seeds):
        seed = 50000 + N * 100 + {"mix": 0, "add": 1, "sub": 2}[op_filter] * 10 + i
        centroids = make_random_orthogonal_centroids(n_classes, 128, seed)
        t0 = time.time()
        r = train_scale_one(
            cfg, seed, op_filter=op_filter, centroids=centroids,
            epochs=epochs, steps_per_epoch=steps_per_epoch,
        )
        variants = compute_rho_variants(r["bundle_by_n"], ns)
        dt = time.time() - t0
        print(
            f"[N={N:2d} op={op_filter:3s} seed={seed}] acc={r['acc']:.3f}  "
            f"ρ_lin={variants['rho_linear']:+.4f}  "
            f"ρ_log={variants['rho_log']:+.4f}  "
            f"ρ_sqrt={variants['rho_sqrt']:+.4f}  ({dt:.1f}s)"
        )
        rows.append({"seed": seed, "acc": r["acc"], **variants, "wall_s": dt})
        bundles_by_seed[seed] = r["bundle_by_n"]
    return {
        "N": N,
        "op_filter": op_filter,
        "per_seed": rows,
        "rho_linear": _stats([r["rho_linear"] for r in rows]),
        "rho_log": _stats([r["rho_log"] for r in rows]),
        "rho_sqrt": _stats([r["rho_sqrt"] for r in rows]),
        "acc": _stats([r["acc"] for r in rows]),
        "bundles_by_seed": bundles_by_seed,  # 中间产物, 不写 json
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--ns", type=int, nargs="+", default=[7, 15, 30])
    ap.add_argument("--out", type=Path, default=Path("outputs/scale_study"))
    ap.add_argument("--smoke", action="store_true", help="只跑 N=7 mix 1 seed")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.ns = [7]
        args.n_seeds = 1

    # 大 N 给更多训练 (combinations 数 O(N²))
    def epochs_for(N: int) -> tuple[int, int]:
        if N <= 7:
            return 12, 120
        elif N <= 15:
            return 16, 160
        else:
            return 24, 200

    summary: dict = {
        "ns": args.ns,
        "n_seeds": args.n_seeds,
        "device": DEVICE,
        "setups": ["mix", "add", "sub"],
        "by_scale": {},
    }

    for N in args.ns:
        epochs, steps = epochs_for(N)
        print("=" * 70)
        print(f"  N = {N} (epochs={epochs}, steps_per_epoch={steps})")
        print("=" * 70)
        per_setup: dict = {}
        for op_filter in ("mix", "add", "sub"):
            per_setup[op_filter] = run_one_config(N, op_filter, args.n_seeds, epochs, steps)

        # cross-setup cos matrix similarity (任务不变性)
        ns = list(range(1, N + 1))
        def _avg_cos(bundles_by_seed: dict) -> dict[int, torch.Tensor]:
            # 对 seed 平均 bundle (先 L2 norm, 再平均)
            avg: dict[int, torch.Tensor] = {}
            seeds = list(bundles_by_seed.keys())
            for n in ns:
                stacked = torch.stack([
                    F.normalize(bundles_by_seed[s][n], dim=-1) for s in seeds
                ])
                avg[n] = stacked.mean(0)
            return avg
        mix_avg = _avg_cos(per_setup["mix"]["bundles_by_seed"])
        add_avg = _avg_cos(per_setup["add"]["bundles_by_seed"])
        sub_avg = _avg_cos(per_setup["sub"]["bundles_by_seed"])
        cross = {
            "mix_vs_add": cross_setup_cos_rho(mix_avg, add_avg, ns),
            "mix_vs_sub": cross_setup_cos_rho(mix_avg, sub_avg, ns),
            "add_vs_sub": cross_setup_cos_rho(add_avg, sub_avg, ns),
        }
        print(f"  [cross-setup invariance N={N}] "
              f"mix↔add={cross['mix_vs_add']:+.4f}  "
              f"mix↔sub={cross['mix_vs_sub']:+.4f}  "
              f"add↔sub={cross['add_vs_sub']:+.4f}")

        # 剥掉 bundle tensor (太大, 不存 json)
        for k in per_setup:
            per_setup[k].pop("bundles_by_seed", None)
        summary["by_scale"][f"N={N}"] = {
            "per_setup": per_setup,
            "cross_setup_rho": cross,
            "epochs": epochs,
            "steps_per_epoch": steps,
        }

    # 写 summary.json
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    # 写 report.md
    (args.out / "report.md").write_text(render_report(summary))
    print("\nWrote:")
    print(f"  {args.out/'summary.json'}")
    print(f"  {args.out/'report.md'}")


def render_report(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# Scale Study · 数字范围扩大 + Weber's law 检测")
    lines.append("")
    lines.append(f"- n_seeds: {summary['n_seeds']}")
    lines.append(f"- device: {summary['device']}")
    lines.append(f"- scales: {summary['ns']}")
    lines.append(f"- setups: {', '.join(summary['setups'])}")
    lines.append("")
    lines.append("## 主表")
    lines.append("")
    lines.append("| N | setup | acc | ρ_linear | ρ_log | ρ_sqrt |")
    lines.append("|---|---|---|---|---|---|")
    for N_key, v in summary["by_scale"].items():
        for op, s in v["per_setup"].items():
            acc = s["acc"]; rl = s["rho_linear"]; rg = s["rho_log"]; rq = s["rho_sqrt"]
            lines.append(
                f"| {N_key} | `{op}` | {acc['mean']:.3f} ± {acc['std']:.3f} | "
                f"{rl['mean']:+.4f} ± {rl['std']:.4f} | "
                f"{rg['mean']:+.4f} ± {rg['std']:.4f} | "
                f"{rq['mean']:+.4f} ± {rq['std']:.4f} |"
            )
    lines.append("")
    lines.append("## 任务不变性 (cross-setup cos matrix ρ)")
    lines.append("")
    lines.append("| N | mix ↔ add | mix ↔ sub | add ↔ sub |")
    lines.append("|---|---|---|---|")
    for N_key, v in summary["by_scale"].items():
        c = v["cross_setup_rho"]
        lines.append(
            f"| {N_key} | {c['mix_vs_add']:+.4f} | "
            f"{c['mix_vs_sub']:+.4f} | {c['add_vs_sub']:+.4f} |"
        )
    lines.append("")
    lines.append("## 解读")
    lines.append("")
    lines.append("- **ρ 最高的假设**胜出 — 那个就是 bundle 学出的数字线形态.")
    lines.append("- 若 `ρ_log > ρ_linear` 且差距大 → **Weber's law** 涌现 (log-spaced).")
    lines.append("- 若 N↑ 时 ρ 仍稳定 → coherence 是结构性, 不依赖具体 scale.")
    lines.append("- 若 `cross_setup_rho` ≈ 1 → bundle 几何与任务 (add/sub/mix) 无关, 只由 |Δn| 决定.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
