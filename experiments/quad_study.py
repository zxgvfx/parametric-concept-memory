"""quad_study.py — 四则运算 (add/sub/mul/div) + pair-OOD 外推测试.

与 ``scale_study`` 的区别:

  - 扩展到四则运算, op_onehot 从 2d → 4d
  - 约束所有运算结果 ``c ∈ [1, N]``, 保证 concept 空间闭合
  - **Pair-wise OOD**: 训练时 hold-out k% 的 (a, b, op) 三元组,
    测试 held-out 组合的准确率 — 这是唯一能回答"能否预测未见组合"的
    compositional generalization 实验.
  - 支持 "小数离散化" 变体: 概念按 ``step`` 离散 (e.g. 0.5 → 61 concept)

架构限制声明:

  - "预测训练中未见过的**单个数字**" 在 D91/D92 不可能: concept bundle 是
    per-concept 参数, 未注册 concept 根本没 bundle. 真正能测的泛化是
    "未见过的 (a, b, op) **组合**".
  - "真小数" 需要 D93 级架构升级 (bundle 生成器 / concept codebook).
    本脚本通过离散化到固定精度 step 实现近似, 验证架构是否对
    "细粒度 concept 网格" 仍能学出线性数字线.

运行:

  python -m experiments.quad_study --n-seeds 3 --ns 30
  python -m experiments.quad_study --n-seeds 2 --ns 100
  python -m experiments.quad_study --n-seeds 2 --ns 30 --step 0.5
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from experiments.purity_audit import (
    build_graph_with_id_fn,
    make_random_orthogonal_centroids,
)
from experiments.robustness_study import BATCH_SIZE, BIAS_DIM, DEVICE, LR

OPS = ["add", "sub", "mul", "div"]
OP_IDX = {op: i for i, op in enumerate(OPS)}


# ────────────────────────────────────────────────────────────────────────
# 四则运算 triple 枚举 (c 必须落在 concept 网格上)
# ────────────────────────────────────────────────────────────────────────


def _on_grid(x: float, step: float) -> bool:
    """x 是否落在 grid {step, 2*step, ..., N*step} 上."""
    q = x / step
    return abs(q - round(q)) < 1e-6 and round(q) >= 1


def enumerate_triples(N: int, step: float = 1.0) -> dict[str, list[tuple[float, float, float]]]:
    """枚举所有合法的 (a, b, c) triple per op, 保证 a,b,c ∈ grid 且 ≤ N*step."""
    vals = [i * step for i in range(1, N + 1)]
    by_op: dict[str, list[tuple[float, float, float]]] = {op: [] for op in OPS}
    max_val = N * step
    for a in vals:
        for b in vals:
            # add
            c = a + b
            if c <= max_val + 1e-6 and _on_grid(c, step):
                by_op["add"].append((a, b, c))
            # sub
            c = a - b
            if c >= step - 1e-6 and _on_grid(c, step):
                by_op["sub"].append((a, b, c))
            # mul
            c = a * b
            if c <= max_val + 1e-6 and _on_grid(c, step):
                by_op["mul"].append((a, b, c))
            # div
            if b > 1e-9:
                c = a / b
                if step - 1e-6 <= c <= max_val + 1e-6 and _on_grid(c, step):
                    by_op["div"].append((a, b, c))
    return by_op


# ────────────────────────────────────────────────────────────────────────
# Head 支持 4d op_onehot
# ────────────────────────────────────────────────────────────────────────


class QuadArithHead(nn.Module):
    """四则肌肉: 消费 arithmetic_bias facet + 4d op_onehot."""

    def __init__(self, embed_dim: int = 128, bias_dim: int = BIAS_DIM, n_ops: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.bias_dim = bias_dim
        self.n_ops = n_ops
        in_dim = 2 * bias_dim + n_ops
        self.fc1 = nn.Linear(in_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        op_onehot: torch.Tensor,
        concept_ids_a: list[str],
        concept_ids_b: list[str],
        cg,
        tick: int = 0,
    ) -> torch.Tensor:
        bias_a = self._collapse_batch(concept_ids_a, cg, tick)
        bias_b = self._collapse_batch(concept_ids_b, cg, tick)
        x = torch.cat([bias_a, bias_b, op_onehot], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def _collapse_batch(self, ids: list[str], cg, tick: int) -> torch.Tensor:
        device = next(self.parameters()).device
        rows = []
        for cid in ids:
            if cid not in cg.concepts:
                raise KeyError(f"concept {cid!r} missing")
            cc = cg.concepts[cid].collapse(
                caller="QuadArithHead",
                facet="arithmetic_bias",
                shape=(self.bias_dim,),
                tick=tick,
                init="normal_small",
                device=device,
            )
            rows.append(cc.as_tensor())
        return torch.stack(rows, dim=0)


# ────────────────────────────────────────────────────────────────────────
# 训练 + eval
# ────────────────────────────────────────────────────────────────────────


def value_to_concept_id(v: float, step: float) -> str:
    """把 grid 上的数转成 concept_id. 整数用 `concept:ans:N`, 小数用 `concept:ans:N_half`."""
    if step == 1.0:
        return f"concept:ans:{int(round(v))}"
    idx = int(round(v / step))
    return f"concept:ans:s{idx}"   # s{idx}: step index


def value_to_tgt(v: float, step: float) -> int:
    """把 c 映射到 centroid class index (0..N-1)."""
    return int(round(v / step)) - 1


def op_onehot_tensor(op_names: list[str]) -> torch.Tensor:
    t = torch.zeros(len(op_names), len(OPS), device=DEVICE)
    for i, op in enumerate(op_names):
        t[i, OP_IDX[op]] = 1.0
    return t


def train_quad(
    N: int,
    step: float,
    seed: int,
    *,
    train_triples: list[tuple[float, float, str, float]],
    epochs: int,
    steps_per_epoch: int,
    batch_size: int = BATCH_SIZE,
    balanced_op: bool = True,
) -> dict:
    """train_triples: list of (a, b, op, c)."""
    torch.manual_seed(seed)
    rng_np = random.Random(seed)
    rng = torch.Generator().manual_seed(seed)

    by_op_triples: dict[str, list] = {op: [] for op in OPS}
    for t in train_triples:
        by_op_triples[t[2]].append(t)
    ops_with_data = [op for op in OPS if by_op_triples[op]]

    # Build concept graph with all grid points (train + ood)
    grid_vals = [i * step for i in range(1, N + 1)]
    cg, _ = build_graph_with_id_fn(
        1, N, id_fn=lambda idx, _step=step: value_to_concept_id(idx * _step, _step)
    )
    # ↑ `build_graph_with_id_fn` iterates n in range(1, N+1), so id_fn(n) = grid_vals[n-1]'s id
    # = value_to_concept_id(n * step, step). Good.
    # centroid: N classes (idx 0..N-1 correspond to values 1*step..N*step)
    centroids = make_random_orthogonal_centroids(N, 128, seed)

    head = QuadArithHead().to(DEVICE)
    with torch.no_grad():
        for v in grid_vals:
            cid = value_to_concept_id(v, step)
            cg.concepts[cid].collapse(
                "QuadArithHead", "arithmetic_bias",
                (BIAS_DIM,), tick=0, device=DEVICE, init="normal_small",
            )
    cg.bundles_to(torch.device(DEVICE))

    params = list(head.parameters()) + list(cg.iter_bundle_parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    n_triples = len(train_triples)
    for epoch in range(1, epochs + 1):
        head.train()
        for step_i in range(steps_per_epoch):
            if balanced_op and len(ops_with_data) > 1:
                # 每 batch 在 ops 之间均衡 (防止 mul/div 稀疏 op 被淹没)
                batch = []
                per_op = max(1, batch_size // len(ops_with_data))
                for op in ops_with_data:
                    pool = by_op_triples[op]
                    for _ in range(per_op):
                        batch.append(pool[rng_np.randrange(len(pool))])
                # fill remainder
                while len(batch) < batch_size:
                    op = ops_with_data[rng_np.randrange(len(ops_with_data))]
                    pool = by_op_triples[op]
                    batch.append(pool[rng_np.randrange(len(pool))])
            else:
                idxs = [rng_np.randrange(n_triples) for _ in range(batch_size)]
                batch = [train_triples[i] for i in idxs]
            a_l = [t[0] for t in batch]
            b_l = [t[1] for t in batch]
            op_l = [t[2] for t in batch]
            c_l = [t[3] for t in batch]
            ids_a = [value_to_concept_id(a, step) for a in a_l]
            ids_b = [value_to_concept_id(b, step) for b in b_l]
            tgt = torch.tensor([value_to_tgt(c, step) for c in c_l], device=DEVICE)
            op = op_onehot_tensor(op_l)
            pred = head(op, ids_a, ids_b, cg, tick=epoch * 10000 + step_i)
            loss = F.cross_entropy(pred @ centroids.t(), tgt)
            opt.zero_grad(); loss.backward(); opt.step()

    # Read bundle_by_grid_idx
    bundle_by_idx: dict[int, torch.Tensor] = {}
    for i, v in enumerate(grid_vals):
        cid = value_to_concept_id(v, step)
        bundle_by_idx[i + 1] = cg.concepts[cid].bundle.state_dict()[
            "params.arithmetic_bias"
        ].detach().cpu().clone()

    return {
        "head": head, "cg": cg, "centroids": centroids,
        "bundle_by_idx": bundle_by_idx, "grid_vals": grid_vals,
        "step": step, "N": N, "seed": seed,
    }


@torch.no_grad()
def eval_on_triples(
    head: QuadArithHead,
    cg,
    centroids: torch.Tensor,
    triples: list[tuple[float, float, str, float]],
    step: float,
    batch_size: int = 64,
) -> dict[str, float]:
    """返回 per-op accuracy."""
    head.eval()
    by_op: dict[str, list[int]] = {op: [] for op in OPS}
    for i in range(0, len(triples), batch_size):
        batch = triples[i : i + batch_size]
        a_l = [t[0] for t in batch]
        b_l = [t[1] for t in batch]
        op_l = [t[2] for t in batch]
        c_l = [t[3] for t in batch]
        ids_a = [value_to_concept_id(a, step) for a in a_l]
        ids_b = [value_to_concept_id(b, step) for b in b_l]
        tgt = torch.tensor([value_to_tgt(c, step) for c in c_l], device=DEVICE)
        op = op_onehot_tensor(op_l)
        pred = head(op, ids_a, ids_b, cg)
        pred_idx = (pred @ centroids.t()).argmax(-1)
        hits = pred_idx.eq(tgt).cpu().tolist()
        for h, o in zip(hits, op_l):
            by_op[o].append(int(h))
    return {op: (sum(hits) / max(len(hits), 1)) if hits else float("nan")
            for op, hits in by_op.items()}


# ────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────


def rho_variants_on_grid(bundle_by_idx: dict[int, torch.Tensor]) -> dict[str, float]:
    """按 grid index (不是 value) 计算 ρ_linear / ρ_log."""
    ns = sorted(bundle_by_idx.keys())
    M = F.normalize(torch.stack([bundle_by_idx[n] for n in ns]), dim=-1)
    cos = M @ M.t()
    mask = ~torch.eye(len(ns), dtype=torch.bool)
    y = cos[mask].numpy()
    d_lin = torch.tensor([[-abs(i - j) for j in ns] for i in ns], dtype=torch.float)
    d_log = torch.tensor(
        [[-abs(math.log(i) - math.log(j)) for j in ns] for i in ns],
        dtype=torch.float,
    )
    return {
        "rho_linear": float(spearmanr(y, d_lin[mask].numpy())[0]),
        "rho_log":    float(spearmanr(y, d_log[mask].numpy())[0]),
    }


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
    return {"mean": m, "std": sd, "min": min(xs), "max": max(xs), "n": len(xs)}


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────


def run_one(
    N: int, step: float, ood_ratio: float, n_seeds: int,
    epochs: int, steps_per_epoch: int,
) -> dict:
    all_triples = enumerate_triples(N, step)
    by_op_counts = {op: len(trips) for op, trips in all_triples.items()}
    print(f"  triples per op: {by_op_counts}")

    per_seed_rows = []
    bundles_by_seed: dict[int, dict] = {}
    for si in range(n_seeds):
        seed = 60000 + N * 100 + int(step * 10) * 10 + si
        rng = random.Random(seed)
        train_triples = []
        test_triples = []
        for op, trips in all_triples.items():
            trips = list(trips)
            rng.shuffle(trips)
            n_test = max(1, int(len(trips) * ood_ratio))
            for (a, b, c) in trips[:n_test]:
                test_triples.append((a, b, op, c))
            for (a, b, c) in trips[n_test:]:
                train_triples.append((a, b, op, c))

        t0 = time.time()
        r = train_quad(
            N, step, seed,
            train_triples=train_triples,
            epochs=epochs, steps_per_epoch=steps_per_epoch,
        )
        train_acc = eval_on_triples(
            r["head"], r["cg"], r["centroids"], train_triples, step
        )
        ood_acc = eval_on_triples(
            r["head"], r["cg"], r["centroids"], test_triples, step
        )
        rho = rho_variants_on_grid(r["bundle_by_idx"])
        dt = time.time() - t0
        overall_train = sum(train_acc.values()) / 4
        overall_ood = sum(ood_acc.values()) / 4
        print(
            f"[N={N} step={step} seed={seed}] "
            f"train_acc={overall_train:.3f} ood_acc={overall_ood:.3f} "
            f"ρ_lin={rho['rho_linear']:+.3f} ρ_log={rho['rho_log']:+.3f} ({dt:.1f}s)"
        )
        print(f"   per-op train: {train_acc}")
        print(f"   per-op ood:   {ood_acc}")
        per_seed_rows.append({
            "seed": seed,
            "n_train_triples": len(train_triples),
            "n_ood_triples": len(test_triples),
            "train_acc": train_acc,
            "ood_acc": ood_acc,
            **rho,
            "wall_s": dt,
        })
        bundles_by_seed[seed] = r["bundle_by_idx"]

    # aggregate
    def _collect(key: str, op: str = "") -> list[float]:
        out = []
        for r in per_seed_rows:
            v = r[key][op] if op else r[key]
            if not (isinstance(v, float) and math.isnan(v)):
                out.append(v)
        return out

    agg = {
        "N": N, "step": step, "ood_ratio": ood_ratio,
        "per_seed": per_seed_rows,
        "train_acc_per_op": {op: _stats([r["train_acc"][op] for r in per_seed_rows]) for op in OPS},
        "ood_acc_per_op":   {op: _stats([r["ood_acc"][op]   for r in per_seed_rows]) for op in OPS},
        "rho_linear": _stats(_collect("rho_linear")),
        "rho_log":    _stats(_collect("rho_log")),
        "triples_per_op": by_op_counts,
    }
    return agg


def render_report(summary: dict) -> str:
    lines = ["# Quad-Arithmetic + Pair-OOD Report", ""]
    lines.append(f"- device: {summary['device']}, n_seeds: {summary['n_seeds']}, "
                 f"ood_ratio: {summary['ood_ratio']}")
    lines.append("")
    lines.append("## 主表: per-op train acc / OOD acc")
    lines.append("")
    lines.append("| config | n_triples | add train / ood | sub train / ood | "
                 "mul train / ood | div train / ood | ρ_lin | ρ_log |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cfg_key, v in summary["by_config"].items():
        counts = v["triples_per_op"]
        total = sum(counts.values())
        row = [cfg_key, str(total)]
        for op in OPS:
            t = v["train_acc_per_op"][op]; o = v["ood_acc_per_op"][op]
            row.append(f"{t['mean']:.3f} / {o['mean']:.3f}")
        row.append(f"{v['rho_linear']['mean']:+.3f}")
        row.append(f"{v['rho_log']['mean']:+.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[30])
    ap.add_argument("--step", type=float, default=1.0,
                    help="concept 精度 (1.0=整数, 0.5=半整数)")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--ood-ratio", type=float, default=0.15)
    ap.add_argument("--out", type=Path, default=Path("outputs/quad_study"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.ns = [30]
        args.n_seeds = 1

    def schedule(N: int) -> tuple[int, int]:
        # 大 N 多个 triple → 需要更多训练量 (balanced op sampling 已补偿)
        if N <= 30: return 30, 240
        if N <= 60: return 50, 320
        return 80, 500

    summary: dict = {
        "device": DEVICE,
        "n_seeds": args.n_seeds,
        "ood_ratio": args.ood_ratio,
        "step": args.step,
        "by_config": {},
    }

    for N in args.ns:
        epochs, steps = schedule(N)
        print("=" * 72)
        print(f"  N = {N}, step = {args.step}, epochs = {epochs}, "
              f"steps_per_epoch = {steps}, ood = {args.ood_ratio}")
        print("=" * 72)
        cfg_key = f"N={N},step={args.step}"
        summary["by_config"][cfg_key] = run_one(
            N, args.step, args.ood_ratio, args.n_seeds, epochs, steps
        )

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (args.out / "report.md").write_text(render_report(summary))
    print("\nWrote:")
    print(f"  {args.out/'summary.json'}")
    print(f"  {args.out/'report.md'}")


if __name__ == "__main__":
    main()
