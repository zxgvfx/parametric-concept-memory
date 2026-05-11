"""F4 — Tier-G 四域 ablation: number / color / space / phoneme.

每个域跑两组 (5 seed):

* **A** — no sleep (baseline, Tier-A/B/C/D bit-identical)
* **C** — sleep + ``use_abstract=True``, ``assignment="soft"``,
  ``k_clusters ≈ N-1``, ``soft_tau=0.1``, ``sleep_warmup=12``,
  ``sleep_every=4``  (best config from quad-domain S2 ablation)

跨域可比指标采用 **task accuracy**:

* number — overall train accuracy on quad triples
* color  — single-mode ``mix_acc``
* space  — single-mode ``move_acc``
* phoneme — single-axis voicing accuracy ``accs['v']``

每域跑 single-mode 已足够诊断 sleep+abstract 是否带来下游收益; dual /
triple 模式留待未来扩展.

跑法 (smoke 1 seed)::

    python -m experiments.sleep_ablation_four_domain --n-seeds 1 \
        --out outputs/f4_smoke

跑法 (5 seed paper-grade)::

    python -m experiments.sleep_ablation_four_domain --n-seeds 5 \
        --out outputs/f4_5seed
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
    return {"mean": m, "std": sd, "min": min(xs), "max": max(xs), "n": len(xs)}


# ────────────────────────────────────────────────────────────────────────
# Per-domain runners (small wrappers around train_one)
# ────────────────────────────────────────────────────────────────────────


def _run_number(seed: int, *, mode: str, k_override: int | None,
                soft_tau: float, warmup: int, every: int,
                ood_ratio: float = 0.15) -> dict:
    """Number domain via experiments.quad_study.train_quad."""
    import random
    import torch
    from experiments.quad_study import (
        enumerate_triples, eval_on_triples, rho_variants_on_grid, train_quad,
    )
    N, step = 30, 1.0
    rng = random.Random(seed)
    all_triples = enumerate_triples(N, step)
    train_triples, test_triples = [], []
    for op, trips in all_triples.items():
        trips = list(trips)
        rng.shuffle(trips)
        n_test = max(1, int(len(trips) * ood_ratio))
        for (a, b, c) in trips[:n_test]:
            test_triples.append((a, b, op, c))
        for (a, b, c) in trips[n_test:]:
            train_triples.append((a, b, op, c))

    if mode == "A":
        sleep_kw = dict(sleep_every=None, use_abstract=False)
    else:
        sleep_kw = dict(
            sleep_every=every, sleep_warmup=warmup,
            sleep_k_clusters=k_override or 28,
            sleep_assignment="soft", sleep_soft_tau=soft_tau,
            use_abstract=True,
        )
    t0 = time.time()
    r = train_quad(
        N, step, seed,
        train_triples=train_triples,
        epochs=20, steps_per_epoch=80,
        **sleep_kw,
    )
    train_acc = eval_on_triples(
        r["head"], r["cg"], r["centroids"], train_triples, step
    )
    ood_acc = eval_on_triples(
        r["head"], r["cg"], r["centroids"], test_triples, step
    )
    rho = rho_variants_on_grid(r["bundle_by_idx"])
    return {
        "seed": seed, "wall_s": time.time() - t0,
        "task_acc": sum(train_acc.values()) / max(len(train_acc), 1),
        "ood_acc": sum(ood_acc.values()) / max(len(ood_acc), 1),
        "rho": rho.get("rho_log"),
        "rho_lin": rho.get("rho_linear"),
    }


def _run_color(seed: int, *, mode: str, k_override: int | None,
               soft_tau: float, warmup: int, every: int,
               ood_ratio: float = 0.0) -> dict:
    from experiments.color_concept_study._config import EMBED_DIM, N_COLORS
    from experiments.color_concept_study.graph_builder import (
        make_random_orthogonal_centroids,
    )
    from experiments.color_concept_study.metrics import (
        _cos_matrix, _rho_circular,
    )
    from experiments.color_concept_study.train import train_one

    centroids = make_random_orthogonal_centroids(N_COLORS, EMBED_DIM, seed)
    if mode == "A":
        sleep_kw = dict(sleep_every=None, use_abstract=False)
    else:
        sleep_kw = dict(
            sleep_every=every, sleep_warmup=warmup,
            sleep_k_clusters=k_override or (N_COLORS - 2),
            sleep_assignment="soft", sleep_soft_tau=soft_tau,
            use_abstract=True,
        )
    t0 = time.time()
    r = train_one("single", seed, centroids,
                  ood_ratio=ood_ratio, **sleep_kw)
    rho_circ = _rho_circular(_cos_matrix(r["bundle_state"], "mixing_bias"))
    return {
        "seed": seed, "wall_s": time.time() - t0,
        "task_acc": r["mix_acc"],
        "ood_acc": r.get("mix_ood_acc"),
        "rho": rho_circ,
    }


def _run_space(seed: int, *, mode: str, k_override: int | None,
               soft_tau: float, warmup: int, every: int,
               ood_ratio: float = 0.0) -> dict:
    from experiments.space_concept_study._config import N_CELLS
    from experiments.space_concept_study.metrics import (
        _cos_matrix, _rho_L1,
    )
    from experiments.space_concept_study.train import train_one

    if mode == "A":
        sleep_kw = dict(sleep_every=None, use_abstract=False)
    else:
        sleep_kw = dict(
            sleep_every=every, sleep_warmup=warmup,
            sleep_k_clusters=k_override or (N_CELLS - 2),
            sleep_assignment="soft", sleep_soft_tau=soft_tau,
            use_abstract=True,
        )
    t0 = time.time()
    r = train_one("single", seed, ood_ratio=ood_ratio, **sleep_kw)
    rho_l1 = _rho_L1(_cos_matrix(r["bundle_state"], "motion_bias"))
    return {
        "seed": seed, "wall_s": time.time() - t0,
        "task_acc": r["move_acc"],
        "ood_acc": r.get("move_ood_acc"),
        "rho": rho_l1,
    }


def _run_phoneme(seed: int, *, mode: str, k_override: int | None,
                 soft_tau: float, warmup: int, every: int,
                 ood_ratio: float = 0.0) -> dict:
    """Phoneme N is too small per axis (2/4/4) for OOD splitting to be
    meaningful — ood_ratio is silently ignored here."""
    from experiments.phoneme_concept_study.metrics import (
        _cos_matrix, _rho_hamming_total,
    )
    from experiments.phoneme_concept_study.train import train_one

    if mode == "A":
        sleep_kw = dict(sleep_every=None, use_abstract=False)
    else:
        sleep_kw = dict(
            sleep_every=every, sleep_warmup=warmup,
            sleep_k_clusters="auto",
            sleep_assignment="soft", sleep_soft_tau=soft_tau,
            use_abstract=True,
        )
    t0 = time.time()
    r = train_one("triple", seed, **sleep_kw)
    rho_v = _rho_hamming_total(_cos_matrix(r["bundle_state"], "voice_bias"))
    return {
        "seed": seed, "wall_s": time.time() - t0,
        "task_acc": r["accs"]["v"],
        "ood_acc": None,
        "task_acc_m": r["accs"]["m"],
        "task_acc_p": r["accs"]["p"],
        "rho": rho_v,
    }


DOMAINS = {
    "number":  _run_number,
    "color":   _run_color,
    "space":   _run_space,
    "phoneme": _run_phoneme,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--soft-tau", type=float, default=0.1)
    ap.add_argument("--sleep-every", type=int, default=4)
    ap.add_argument("--sleep-warmup", type=int, default=12)
    ap.add_argument("--ood-ratio", type=float, default=0.0,
                    help="F5: hold out fraction of triples for OOD eval "
                         "(0 = legacy full-train); applies to number/color/space")
    ap.add_argument("--k-override", type=int, default=None,
                    help="Override sleep_k_clusters for all domains; "
                         "default = N-2 per domain (or 'auto' for phoneme)")
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS.keys()),
                    help="Subset of domains to run")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f4_four_domain"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  F4 four-domain ablation: n_seeds={args.n_seeds}, "
          f"soft_tau={args.soft_tau}, every={args.sleep_every}, "
          f"warmup={args.sleep_warmup}")
    print(f"  domains: {args.domains}")
    print("=" * 72)

    summary: dict = {
        "config": vars(args) | {"out": str(args.out)},
        "by_domain": {},
    }

    for domain in args.domains:
        if domain not in DOMAINS:
            print(f"  ! skipping unknown domain {domain!r}")
            continue
        runner = DOMAINS[domain]
        print(f"\n── domain: {domain} ──")
        rows_a, rows_c = [], []
        for si in range(args.n_seeds):
            seed = 90000 + si
            a = runner(seed, mode="A", k_override=args.k_override,
                       soft_tau=args.soft_tau,
                       warmup=args.sleep_warmup, every=args.sleep_every,
                       ood_ratio=args.ood_ratio)
            c = runner(seed, mode="C", k_override=args.k_override,
                       soft_tau=args.soft_tau,
                       warmup=args.sleep_warmup, every=args.sleep_every,
                       ood_ratio=args.ood_ratio)
            rows_a.append(a)
            rows_c.append(c)
            ood_a = a.get("ood_acc")
            ood_c = c.get("ood_acc")
            ood_str = (
                f" oodA={ood_a:.3f} oodC={ood_c:.3f}"
                if (ood_a is not None and ood_c is not None) else ""
            )
            print(
                f"  [seed={seed}] "
                f"A acc={a['task_acc']:.3f} ρ={a['rho']:+.3f} | "
                f"C acc={c['task_acc']:.3f} ρ={c['rho']:+.3f}{ood_str} "
                f"(wallA={a['wall_s']:.1f}s, wallC={c['wall_s']:.1f}s)"
            )

        a_acc = [r["task_acc"] for r in rows_a]
        c_acc = [r["task_acc"] for r in rows_c]
        a_rho = [r["rho"] for r in rows_a]
        c_rho = [r["rho"] for r in rows_c]
        delta_acc = [c_acc[i] - a_acc[i] for i in range(len(a_acc))]
        delta_rho = [c_rho[i] - a_rho[i] for i in range(len(a_rho))]

        s_a_acc, s_c_acc = _stats(a_acc), _stats(c_acc)
        s_a_rho, s_c_rho = _stats(a_rho), _stats(c_rho)
        d_acc, d_rho = _stats(delta_acc), _stats(delta_rho)

        ood_a = [r.get("ood_acc") for r in rows_a]
        ood_c = [r.get("ood_acc") for r in rows_c]
        has_ood = all(x is not None for x in ood_a + ood_c)
        if has_ood:
            d_ood = _stats([ood_c[i] - ood_a[i] for i in range(len(ood_a))])
            s_a_ood = _stats(ood_a)
            s_c_ood = _stats(ood_c)
        else:
            d_ood = s_a_ood = s_c_ood = None

        domain_summary = {
            "per_seed_A": rows_a,
            "per_seed_C": rows_c,
            "task_acc_A": s_a_acc,
            "task_acc_C": s_c_acc,
            "rho_A": s_a_rho,
            "rho_C": s_c_rho,
            "delta_task_acc": d_acc,
            "delta_rho": d_rho,
            "ood_acc_A": s_a_ood,
            "ood_acc_C": s_c_ood,
            "delta_ood_acc": d_ood,
            "verdict": {
                "C_no_acc_regression": d_acc["mean"] >= -0.02,
                "C_no_rho_regression": d_rho["mean"] >= -0.02,
                "C_pareto_better_train_rho": (
                    d_acc["mean"] > 0 and d_rho["mean"] > 0
                ),
                "C_pareto_better_ood_rho": (
                    has_ood and d_ood["mean"] > 0 and d_rho["mean"] > 0
                ),
            },
        }
        summary["by_domain"][domain] = domain_summary
        print(
            f"  → A acc={s_a_acc['mean']:.3f}±{s_a_acc['std']:.3f}  "
            f"C acc={s_c_acc['mean']:.3f}±{s_c_acc['std']:.3f}  "
            f"Δ={d_acc['mean']:+.3f}"
        )
        if has_ood:
            print(
                f"  → A OOD={s_a_ood['mean']:.3f}±{s_a_ood['std']:.3f}  "
                f"C OOD={s_c_ood['mean']:.3f}±{s_c_ood['std']:.3f}  "
                f"Δ={d_ood['mean']:+.3f}"
            )
        print(
            f"  → A ρ  ={s_a_rho['mean']:+.3f}±{s_a_rho['std']:.3f}  "
            f"C ρ  ={s_c_rho['mean']:+.3f}±{s_c_rho['std']:.3f}  "
            f"Δ={d_rho['mean']:+.3f}"
        )
        print(f"  → verdict: {domain_summary['verdict']}")

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"\nwrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
