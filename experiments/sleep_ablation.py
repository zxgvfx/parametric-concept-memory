"""Tier-G sleep on/off ablation (V2 / V3 验证脚手架).

直接复用 :mod:`experiments.quad_study` 的 ``train_quad`` /
``rho_variants_on_grid`` / ``eval_on_triples``，对同一组 seed 跑三遍：

* group **A** — ``sleep_every=None`` (baseline, Tier-A/B/C/D bit-identical)
* group **B** — ``sleep_every=K`` (sleep 跑但 head 不消费抽象节点)
* group **C** — ``sleep_every=K, use_abstract=True``
  (sleep + head 通过 ``anchor + residual`` 读取, 改进 1)

输出：

1. 每 seed 的 (rho_linear / rho_log / OOD acc / sleep silhouette);
2. 跨 seed 聚合的 mean ± std;
3. 三组对照的 verdict 表 (默认门槛 ``rho_log_baseline - 0.02 ≤ x``).

跑法 (smoke, 单 seed):

    python -m experiments.sleep_ablation --N 30 --epochs 6 --steps 60 \
        --sleep-every 2 --n-seeds 1 --out outputs/sleep_ablation_smoke

跑法 (论文小配置 5 seed):

    python -m experiments.sleep_ablation --N 30 --n-seeds 5 \
        --sleep-every 5 --out outputs/sleep_ablation
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from experiments.quad_study import (
    DEVICE,
    enumerate_triples,
    eval_on_triples,
    rho_variants_on_grid,
    train_quad,
)


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
    return {"mean": m, "std": sd, "min": min(xs), "max": max(xs), "n": len(xs)}


def _split_triples(N: int, step: float, ood_ratio: float, seed: int):
    all_triples = enumerate_triples(N, step)
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
    return train_triples, test_triples


def _run_one(
    N: int, step: float, seed: int, *,
    train_triples, test_triples,
    epochs: int, steps_per_epoch: int,
    sleep_every: int | None,
    sleep_warmup: int = 0,
    sleep_force_recluster: bool = False,
    sleep_k_clusters: int | str = "auto",
    sleep_anchor_ema: float = 1.0,
    sleep_assignment: str = "hard",
    sleep_soft_tau: float = 0.5,
    use_abstract: bool = False,
) -> dict:
    t0 = time.time()
    r = train_quad(
        N, step, seed,
        train_triples=train_triples,
        epochs=epochs, steps_per_epoch=steps_per_epoch,
        sleep_every=sleep_every,
        sleep_warmup=sleep_warmup,
        sleep_force_recluster=sleep_force_recluster,
        sleep_k_clusters=sleep_k_clusters,
        sleep_anchor_ema=sleep_anchor_ema,
        sleep_assignment=sleep_assignment,
        sleep_soft_tau=sleep_soft_tau,
        use_abstract=use_abstract,
    )
    train_acc = eval_on_triples(
        r["head"], r["cg"], r["centroids"], train_triples, step
    )
    ood_acc = eval_on_triples(
        r["head"], r["cg"], r["centroids"], test_triples, step
    )
    rho = rho_variants_on_grid(r["bundle_by_idx"])

    silhouette_curve: list[float] = []
    for rep in r.get("sleep_reports", []):
        for f in rep.get("facets", []):
            silhouette_curve.append(float(f.get("silhouette", 0.0)))

    return {
        "seed": seed,
        "wall_s": time.time() - t0,
        "train_acc_overall": sum(train_acc.values()) / max(len(train_acc), 1),
        "ood_acc_overall": sum(ood_acc.values()) / max(len(ood_acc), 1),
        "train_acc_per_op": train_acc,
        "ood_acc_per_op": ood_acc,
        "rho_linear": rho["rho_linear"],
        "rho_log": rho["rho_log"],
        "n_sleep_reports": len(r.get("sleep_reports", [])),
        "silhouette_per_facet_per_pass": silhouette_curve,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--ood-ratio", type=float, default=0.15)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--sleep-every", type=int, default=2)
    ap.add_argument("--sleep-warmup", type=int, default=0,
                    help="sleep 前先训练多少个 epoch (让 baseline 结构形成)")
    ap.add_argument("--force-recluster", action="store_true",
                    help="每次 sleep 都重聚类 (默认 G6 幂等只聚一次)")
    ap.add_argument("--k-clusters", default="auto",
                    help="cluster 数; int 或 'auto' (default auto = ⌈√N/2⌉)")
    ap.add_argument("--anchor-ema", type=float, default=1.0,
                    help="S1 EMA blend factor; 1.0 = legacy hard overwrite, "
                         "<1 = Online Codebook style slow update")
    ap.add_argument("--assignment", choices=["hard", "soft"], default="hard",
                    help="S2 cluster assignment routing")
    ap.add_argument("--soft-tau", type=float, default=0.5,
                    help="S2 softmax temperature when --assignment=soft")
    ap.add_argument("--rho-tolerance", type=float, default=0.02,
                    help="允许 sleep 比 baseline 在 ρ 上至多回退多少仍判为 pass")
    ap.add_argument("--skip-c", action="store_true",
                    help="只跑 A/B 两组, 跳过 C(use_abstract=True) 路径")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/sleep_ablation"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  V2/V3 sleep ablation: N={args.N} step={args.step} "
          f"epochs={args.epochs} steps_per_epoch={args.steps} "
          f"n_seeds={args.n_seeds} sleep_every={args.sleep_every}")
    print(f"  device={DEVICE}")
    print(f"  groups: A=no-sleep  B=sleep+direct  "
          f"C=sleep+abstract({'on' if not args.skip_c else 'OFF'})")
    print("=" * 72)

    rows_a, rows_b, rows_c = [], [], []
    for si in range(args.n_seeds):
        seed = 70000 + args.N * 100 + si
        train_triples, test_triples = _split_triples(
            args.N, args.step, args.ood_ratio, seed
        )

        a = _run_one(
            args.N, args.step, seed,
            train_triples=train_triples, test_triples=test_triples,
            epochs=args.epochs, steps_per_epoch=args.steps,
            sleep_every=None,
        )
        k_arg = (int(args.k_clusters) if str(args.k_clusters).isdigit()
                 else args.k_clusters)
        b = _run_one(
            args.N, args.step, seed,
            train_triples=train_triples, test_triples=test_triples,
            epochs=args.epochs, steps_per_epoch=args.steps,
            sleep_every=args.sleep_every,
            sleep_warmup=args.sleep_warmup,
            sleep_force_recluster=args.force_recluster,
            sleep_k_clusters=k_arg,
            sleep_anchor_ema=args.anchor_ema,
            sleep_assignment=args.assignment,
            sleep_soft_tau=args.soft_tau,
        )
        rows_a.append(a)
        rows_b.append(b)

        if not args.skip_c:
            c = _run_one(
                args.N, args.step, seed,
                train_triples=train_triples, test_triples=test_triples,
                epochs=args.epochs, steps_per_epoch=args.steps,
                sleep_every=args.sleep_every,
                sleep_warmup=args.sleep_warmup,
                sleep_force_recluster=args.force_recluster,
                sleep_k_clusters=k_arg,
                sleep_anchor_ema=args.anchor_ema,
                sleep_assignment=args.assignment,
                sleep_soft_tau=args.soft_tau,
                use_abstract=True,
            )
            rows_c.append(c)
            print(
                f"[seed={seed}] "
                f"A ρ_log={a['rho_log']:+.3f} ood={a['ood_acc_overall']:.3f} | "
                f"B ρ_log={b['rho_log']:+.3f} ood={b['ood_acc_overall']:.3f} | "
                f"C ρ_log={c['rho_log']:+.3f} ood={c['ood_acc_overall']:.3f}"
            )
        else:
            print(
                f"[seed={seed}] "
                f"A ρ_log={a['rho_log']:+.3f} ood={a['ood_acc_overall']:.3f} | "
                f"B ρ_log={b['rho_log']:+.3f} ood={b['ood_acc_overall']:.3f}"
            )

    rho_a = [r["rho_log"] for r in rows_a]
    rho_b = [r["rho_log"] for r in rows_b]
    ood_a = [r["ood_acc_overall"] for r in rows_a]
    ood_b = [r["ood_acc_overall"] for r in rows_b]

    summary = {
        "device": DEVICE,
        "config": vars(args) | {"out": str(args.out)},
        "A_no_sleep": {
            "per_seed": rows_a,
            "rho_log": _stats(rho_a),
            "rho_linear": _stats([r["rho_linear"] for r in rows_a]),
            "ood_acc": _stats(ood_a),
        },
        "B_sleep_direct": {
            "per_seed": rows_b,
            "rho_log": _stats(rho_b),
            "rho_linear": _stats([r["rho_linear"] for r in rows_b]),
            "ood_acc": _stats(ood_b),
            "silhouette_all": _stats([
                s for r in rows_b for s in r["silhouette_per_facet_per_pass"]
            ]) if any(r["silhouette_per_facet_per_pass"] for r in rows_b)
            else {"n": 0},
        },
    }

    a_mean = summary["A_no_sleep"]["rho_log"]["mean"]
    b_mean = summary["B_sleep_direct"]["rho_log"]["mean"]
    delta_ba = b_mean - a_mean

    verdict = {
        "rho_log_delta_b_minus_a": delta_ba,
        "rho_tolerance": args.rho_tolerance,
        "B_no_regression_vs_A": bool(b_mean >= a_mean - args.rho_tolerance),
        "B_strictly_better_than_A": bool(b_mean > a_mean),
    }

    if rows_c:
        rho_c = [r["rho_log"] for r in rows_c]
        ood_c = [r["ood_acc_overall"] for r in rows_c]
        summary["C_sleep_abstract"] = {
            "per_seed": rows_c,
            "rho_log": _stats(rho_c),
            "rho_linear": _stats([r["rho_linear"] for r in rows_c]),
            "ood_acc": _stats(ood_c),
        }
        c_mean = summary["C_sleep_abstract"]["rho_log"]["mean"]
        verdict["rho_log_delta_c_minus_a"] = c_mean - a_mean
        verdict["C_no_regression_vs_A"] = bool(
            c_mean >= a_mean - args.rho_tolerance
        )
        verdict["C_strictly_better_than_A"] = bool(c_mean > a_mean)
        verdict["C_strictly_better_than_B"] = bool(c_mean > b_mean)

    summary["verdict"] = verdict

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print()
    print("─" * 72)
    print(f"  ρ_log A = {a_mean:+.3f} ± {summary['A_no_sleep']['rho_log']['std']:.3f}")
    print(f"  ρ_log B = {b_mean:+.3f} ± {summary['B_sleep_direct']['rho_log']['std']:.3f}"
          f"   ΔB-A = {delta_ba:+.3f}")
    if rows_c:
        c_mean = summary["C_sleep_abstract"]["rho_log"]["mean"]
        print(f"  ρ_log C = {c_mean:+.3f} ± "
              f"{summary['C_sleep_abstract']['rho_log']['std']:.3f}"
              f"   ΔC-A = {c_mean - a_mean:+.3f}   ΔC-B = {c_mean - b_mean:+.3f}")
        print(f"  ood   A = {summary['A_no_sleep']['ood_acc']['mean']:.3f}  "
              f"B = {summary['B_sleep_direct']['ood_acc']['mean']:.3f}  "
              f"C = {summary['C_sleep_abstract']['ood_acc']['mean']:.3f}")
    else:
        print(f"  ood   A = {summary['A_no_sleep']['ood_acc']['mean']:.3f}  "
              f"B = {summary['B_sleep_direct']['ood_acc']['mean']:.3f}")
    print(f"  verdict: {json.dumps(verdict, ensure_ascii=False)}")
    print(f"  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
