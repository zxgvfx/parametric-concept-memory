"""emergent_base10_study.py — Step 1 of the pure-emergence program.

诊断: D91 flat 架构 (每数字一个独立 ConceptNode, 每个独立 bundle) 下,
单靠加减法训练信号, base-10 结构能否自发涌现?

对比 D93a (手写 slot + carry): D93a 把 "base-10 分解" 作为架构先验植入,
外推 100% 完美. 本实验完全移除这个先验, 看 bundle 自己的 geometry 会
不会出现以下任何一个 base-10 signal:

  1. **10-周期性**: cos(bundle_n, bundle_{n+10}) 显著 > 线性插值 (周期 spike).
  2. **单位数共享**: cos(bundle_n, bundle_m) where n%10 == m%10 显著 >
     当 n%10 ≠ m%10 时 (units-digit cluster).
  3. **位数分层**: 1-9 vs 10-19 vs 20-29 ... 在 t-SNE 下分层 clustering.

如果**没有**任何 signal → information-theoretic 上这个 scale 不够, 证明
D93a 的先验是**必需的** (在 Percept 当前数据/compute 规模下).
如果**有** signal → 说明人手写的 base-10 并非唯一出路, 视觉 / 任务 压力
足以诱导架构自发发现位值.

本 Step 1 只用 add/sub + dot canvas centroid (random orthogonal surrogate),
不加入视觉符号 glyph (留给 Step 2).

运行:

  python -m experiments.emergent_base10_study --smoke
  python -m experiments.emergent_base10_study --N 100 --n-seeds 3
  python -m experiments.emergent_base10_study --scan 50 100 200
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
from experiments.quad_study import (
    QuadArithHead,
    enumerate_triples,
)
from experiments.robustness_study import (
    BATCH_SIZE, BIAS_DIM, DEVICE, LR,
)

# restrict to add/sub for clean test
OPS = ["add", "sub"]
OP_IDX = {op: i for i, op in enumerate(OPS)}


def op_onehot_tensor(ops: list[str]) -> torch.Tensor:
    t = torch.zeros(len(ops), len(OPS), device=DEVICE)
    for i, op in enumerate(ops):
        t[i, OP_IDX[op]] = 1.0
    return t


# ────────────────────────────────────────────────────────────────────────
# Training (reuse QuadArithHead with n_ops=2)
# ────────────────────────────────────────────────────────────────────────


def train_flat_and_extract(
    N: int, seed: int, *, epochs: int, steps_per_epoch: int,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Train D91 flat (one ConceptNode per integer 1..N) + one muscle
    consuming arithmetic_bias. No slot/carry/digit structure. Return
    bundle_by_n + train_acc."""
    torch.manual_seed(seed)
    rng = random.Random(seed)

    by_op = enumerate_triples(N, step=1.0)
    train_triples: list[tuple[int, int, str, int]] = []
    for op in OPS:
        for (a, b, c) in by_op[op]:
            train_triples.append((int(a), int(b), op, int(c)))
    by_op_triples = {op: [t for t in train_triples if t[2] == op] for op in OPS}

    cg, _ = build_graph_with_id_fn(1, N, id_fn=lambda n: f"concept:ans:{n}")
    centroids = make_random_orthogonal_centroids(N, 128, seed)

    head = QuadArithHead(n_ops=2).to(DEVICE)

    with torch.no_grad():
        for n in range(1, N + 1):
            cid = f"concept:ans:{n}"
            cg.concepts[cid].collapse(
                "QuadArithHead", "arithmetic_bias", (BIAS_DIM,),
                tick=0, device=DEVICE, init="normal_small",
            )
    cg.bundles_to(torch.device(DEVICE))

    params = list(head.parameters()) + list(cg.iter_bundle_parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        head.train()
        for step_i in range(steps_per_epoch):
            batch = []
            per_op = batch_size // 2
            for op in OPS:
                pool = by_op_triples[op]
                for _ in range(per_op):
                    batch.append(pool[rng.randrange(len(pool))])
            while len(batch) < batch_size:
                op = OPS[rng.randrange(len(OPS))]
                pool = by_op_triples[op]
                batch.append(pool[rng.randrange(len(pool))])

            ids_a = [f"concept:ans:{t[0]}" for t in batch]
            ids_b = [f"concept:ans:{t[1]}" for t in batch]
            op_l = [t[2] for t in batch]
            tgt = torch.tensor([t[3] - 1 for t in batch], device=DEVICE,
                               dtype=torch.long)
            op_oh = op_onehot_tensor(op_l)
            pred = head(op_oh, ids_a, ids_b, cg,
                        tick=epoch * 10000 + step_i)
            loss = F.cross_entropy(pred @ centroids.t(), tgt)
            opt.zero_grad(); loss.backward(); opt.step()

    # train acc (sanity on a subsample)
    rng2 = random.Random(seed + 1)
    rng2.shuffle(train_triples)
    sample = train_triples[: min(2000, len(train_triples))]
    head.eval()
    hits = 0
    with torch.no_grad():
        for i in range(0, len(sample), 64):
            batch = sample[i: i + 64]
            ids_a = [f"concept:ans:{t[0]}" for t in batch]
            ids_b = [f"concept:ans:{t[1]}" for t in batch]
            op_l = [t[2] for t in batch]
            tgt = torch.tensor([t[3] - 1 for t in batch], device=DEVICE)
            pred = head(op_onehot_tensor(op_l), ids_a, ids_b, cg)
            hits += pred.matmul(centroids.t()).argmax(-1).eq(tgt).sum().item()
    train_acc = hits / len(sample)

    # extract bundles
    bundle_by_n = {}
    for n in range(1, N + 1):
        cid = f"concept:ans:{n}"
        bundle_by_n[n] = cg.concepts[cid].bundle.state_dict()[
            "params.arithmetic_bias"
        ].detach().cpu().clone()

    return {"bundle_by_n": bundle_by_n, "N": N, "seed": seed,
            "train_acc": train_acc}


# ────────────────────────────────────────────────────────────────────────
# Base-10 emergence analysis
# ────────────────────────────────────────────────────────────────────────


def analyze_base10_emergence(bundle_by_n: dict[int, torch.Tensor]) -> dict:
    """Return key periodicity indicators.

    出现 base-10 emergence 的 tell-tale signals:
      - `10_periodic_spike`: shift-10 avg cos 显著高于 shift-9, 11 的插值.
      - `units_periodicity_effect`: same-units-digit cos > diff-units-digit cos.
      - `tens_periodicity_effect`: same-tens-digit cos > diff-tens-digit cos.
    """
    ns = sorted(bundle_by_n.keys())
    N = len(ns)
    M = F.normalize(torch.stack([bundle_by_n[n] for n in ns]), dim=-1)
    cos = (M @ M.t()).cpu()

    # 1. shift scan (distance-periodicity)
    shift_stats = {}
    for shift in range(1, min(25, N)):
        vals = []
        for i, n in enumerate(ns):
            if (n + shift) in bundle_by_n:
                j = ns.index(n + shift)
                vals.append(cos[i, j].item())
        if vals:
            shift_stats[shift] = sum(vals) / len(vals)

    # 2. 10-periodic spike (vs linear interp at shift=10)
    if 9 in shift_stats and 10 in shift_stats and 11 in shift_stats:
        interp = (shift_stats[9] + shift_stats[11]) / 2
        spike_10 = shift_stats[10] - interp
    else:
        spike_10 = float("nan")
    if 4 in shift_stats and 5 in shift_stats and 6 in shift_stats:
        interp = (shift_stats[4] + shift_stats[6]) / 2
        spike_5 = shift_stats[5] - interp
    else:
        spike_5 = float("nan")
    if 19 in shift_stats and 20 in shift_stats and 21 in shift_stats:
        interp = (shift_stats[19] + shift_stats[21]) / 2
        spike_20 = shift_stats[20] - interp
    else:
        spike_20 = float("nan")

    # 3. Units-digit sharing: same_units vs diff_units
    same_u, diff_u = [], []
    for i, a in enumerate(ns):
        for j, b in enumerate(ns):
            if j <= i:
                continue
            c = cos[i, j].item()
            if a % 10 == b % 10:
                same_u.append(c)
            else:
                diff_u.append(c)
    same_u_mean = sum(same_u) / len(same_u) if same_u else 0.0
    diff_u_mean = sum(diff_u) / len(diff_u) if diff_u else 0.0

    # 4. Tens-digit sharing
    same_t, diff_t = [], []
    for i, a in enumerate(ns):
        for j, b in enumerate(ns):
            if j <= i:
                continue
            c = cos[i, j].item()
            if (a // 10) == (b // 10):
                same_t.append(c)
            else:
                diff_t.append(c)
    same_t_mean = sum(same_t) / len(same_t) if same_t else 0.0
    diff_t_mean = sum(diff_t) / len(diff_t) if diff_t else 0.0

    # 5. ρ_linear (distance ordinality, already known result)
    d_lin = torch.tensor([[-abs(i - j) for j in ns] for i in ns],
                         dtype=torch.float)
    mask = ~torch.eye(len(ns), dtype=torch.bool)
    rho_linear = float(spearmanr(cos[mask].numpy(), d_lin[mask].numpy())[0])

    # 6. Linear-detrended residual base-10 signal.
    # 先拟合 cos ≈ a * (-|Δ|) + b (纯 linear ordinal trend), 算 residual
    # 在 residual 上看 units-digit sharing, 这是剥离 distance confound
    # 后唯一能显示 "emergent base-10" 的 signal.
    import numpy as np
    pairs_d, pairs_cos = [], []
    for i, a in enumerate(ns):
        for j, b in enumerate(ns):
            if j <= i:
                continue
            pairs_d.append(-abs(a - b))
            pairs_cos.append(cos[i, j].item())
    pairs_d = np.array(pairs_d); pairs_cos = np.array(pairs_cos)
    slope, intercept = np.polyfit(pairs_d, pairs_cos, 1)
    residuals = pairs_cos - (slope * pairs_d + intercept)

    # 对 residual 做 units / tens 分层统计
    res_same_u, res_diff_u = [], []
    res_same_t, res_diff_t = [], []
    idx = 0
    for i, a in enumerate(ns):
        for j, b in enumerate(ns):
            if j <= i:
                continue
            r = residuals[idx]
            if a % 10 == b % 10:
                res_same_u.append(r)
            else:
                res_diff_u.append(r)
            if (a // 10) == (b // 10):
                res_same_t.append(r)
            else:
                res_diff_t.append(r)
            idx += 1

    def _avg(xs): return float(np.mean(xs)) if len(xs) else float("nan")
    residual_units_effect = _avg(res_same_u) - _avg(res_diff_u)
    residual_tens_effect = _avg(res_same_t) - _avg(res_diff_t)

    # Permutation p-value for residual_units_effect
    rng_perm = np.random.RandomState(0)
    null_effects = []
    labels = np.array([1 if a % 10 == b % 10 else 0
                       for i, a in enumerate(ns) for j, b in enumerate(ns)
                       if j > i])
    assert len(labels) == len(residuals)
    for _ in range(200):
        rng_perm.shuffle(labels)
        null_same = residuals[labels == 1]
        null_diff = residuals[labels == 0]
        null_effects.append(null_same.mean() - null_diff.mean())
    null_effects = np.array(null_effects)
    p_val = float(np.mean(np.abs(null_effects) >= abs(residual_units_effect)))

    return {
        "shift_stats": shift_stats,
        "cos_same_units": same_u_mean, "cos_diff_units": diff_u_mean,
        "units_periodicity_effect_raw": same_u_mean - diff_u_mean,
        "cos_same_tens": same_t_mean, "cos_diff_tens": diff_t_mean,
        "tens_periodicity_effect_raw": same_t_mean - diff_t_mean,
        "10_periodic_spike": spike_10,
        "5_periodic_spike": spike_5,
        "20_periodic_spike": spike_20,
        "rho_linear": rho_linear,
        "residual_units_effect": residual_units_effect,
        "residual_tens_effect": residual_tens_effect,
        "residual_units_pvalue": p_val,
        "linear_slope": float(slope),
        "linear_intercept": float(intercept),
    }


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────


def run_one(N: int, seed: int, epochs: int, steps_per_epoch: int,
            batch_size: int = BATCH_SIZE) -> dict:
    t0 = time.time()
    trained = train_flat_and_extract(
        N, seed, epochs=epochs, steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
    )
    analysis = analyze_base10_emergence(trained["bundle_by_n"])
    dt = time.time() - t0
    return {
        "seed": seed, "N": N,
        "train_acc": trained["train_acc"],
        **analysis,
        "bundle_by_n": {n: b.tolist() for n, b in trained["bundle_by_n"].items()},
        "wall_s": dt,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--scan", type=int, nargs="*", default=None,
                    help="If set, scan N values instead of single N")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps-per-epoch", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--out", type=Path, default=Path("outputs/emergent_base10"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.n_seeds = 1
        args.epochs = 20
        args.steps_per_epoch = 200

    N_list = args.scan if args.scan else [args.N]
    summary = {"configs": []}

    for N in N_list:
        rows = []
        for si in range(args.n_seeds):
            seed = 80000 + N * 10 + si
            row = run_one(N, seed, args.epochs, args.steps_per_epoch,
                          batch_size=args.batch_size)
            # drop heavy bundle list from summary json
            row_light = {k: v for k, v in row.items() if k != "bundle_by_n"}
            rows.append(row_light)
            # save full bundles per seed
            (args.out / f"bundles_N{N}_seed{seed}.json").write_text(
                json.dumps({n: b for n, b in row["bundle_by_n"].items()})
            )
            print(f"[N={N} seed={seed}] train={row['train_acc']:.3f}  "
                  f"ρ_lin={row['rho_linear']:+.3f}  "
                  f"spike10={row['10_periodic_spike']:+.4f}  "
                  f"spike5={row['5_periodic_spike']:+.4f}  "
                  f"resΔunits={row['residual_units_effect']:+.4f}  "
                  f"p={row['residual_units_pvalue']:.3f}  "
                  f"resΔtens={row['residual_tens_effect']:+.4f}  "
                  f"({row['wall_s']:.1f}s)")
        # aggregate across seeds
        def _mean(k):
            vals = [r[k] for r in rows if isinstance(r[k], float)
                    and not math.isnan(r[k])]
            return sum(vals) / len(vals) if vals else float("nan")
        agg = {
            "N": N,
            "rho_linear": _mean("rho_linear"),
            "10_periodic_spike": _mean("10_periodic_spike"),
            "5_periodic_spike": _mean("5_periodic_spike"),
            "20_periodic_spike": _mean("20_periodic_spike"),
            "residual_units_effect": _mean("residual_units_effect"),
            "residual_units_pvalue": _mean("residual_units_pvalue"),
            "residual_tens_effect": _mean("residual_tens_effect"),
            "train_acc": _mean("train_acc"),
            "per_seed": rows,
        }
        summary["configs"].append(agg)

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
