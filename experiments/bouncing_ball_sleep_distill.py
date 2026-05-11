"""F59 — physics sleep cache: bouncing-ball cook → lookup distillation.

Builds on `experiments/bouncing_ball_poc.py` (F57 cook training)
and adds a sleep phase that distils the cook's K-step rollouts
into a single-shot ``StateLookupHead``. The empirical question:

    Can a lookup head, trained only on cook rollouts of horizon
    ≤ K_train_max, predict K-step terminal states for all
    K ∈ [1, K_train_max] in a single forward pass — replacing
    the K-step sequential cook with a 1-shot retrieval?

Predicted outcome (mirrors F53 for discrete cook):

* phase 1 (lookup random init): MSE ≫ 0
* phase 2 (post-distill): MSE ≪ 0, **lookup matches cook on
  in-range K**

This closes the v4 dual-process loop: PhysicsCook = System 2
procedural rollout, StateLookupHead = System 1 retrieval cache,
and `distill_physics_cook_to_lookup` is the sleep consolidation.

Usage::

    python -m experiments.bouncing_ball_sleep_distill \\
        --n-seeds 3 --epochs 30 --steps-per-epoch 200 \\
        --K-train-max 50 --K-eval-max 50 \\
        --distill-steps 600 \\
        --out outputs/f59_sleep_distill
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from pcm.physics import (
    PhysicsCook,
    PhysicsStateHead,
    StateLookupHead,
    distill_physics_cook_to_lookup,
    physics_step_loss,
)

from experiments.bouncing_ball_poc import (
    ground_truth_rollout,
    ground_truth_step,
)


__all__ = ["main"]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _train_physics_head(
    head: PhysicsStateHead, *,
    g: float, dt: float, L: float,
    epochs: int, steps_per_epoch: int,
    batch_size: int = 64, lr: float = 5e-3, seed: int = 0,
) -> float:
    """Train PhysicsStateHead on 1-step ground-truth transitions.
    Returns wall time."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    head.train()
    t0 = time.time()
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            x = torch.empty(batch_size, device=DEVICE).uniform_(0, L)
            v = torch.empty(batch_size, device=DEVICE).uniform_(-6, 6)
            state = torch.stack([x, v], dim=-1)
            true_next = ground_truth_step(state, g=g, dt=dt, L=L)
            pred_delta = head(state)
            loss = physics_step_loss(pred_delta, true_next, state)
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    return time.time() - t0


def _run_one(
    seed: int, *,
    g: float = -1.0, dt: float = 0.1, L: float = 10.0,
    epochs: int = 30, steps_per_epoch: int = 200,
    K_train_max: int = 50,
    distill_steps: int = 600,
    n_distill_states: int = 100,
    distill_K_set: list[int] | None = None,
) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    # ─── 1) Train physics head ───
    head = PhysicsStateHead(
        state_dim=2, force_dim=0, hidden=64, dt=dt,
    ).to(DEVICE)
    train_wall = _train_physics_head(
        head, g=g, dt=dt, L=L,
        epochs=epochs, steps_per_epoch=steps_per_epoch, seed=seed,
    )

    cook = PhysicsCook(
        head, max_iters=K_train_max + 10,
        state_clip=(
            torch.tensor([0.0, -10.0], device=DEVICE),
            torch.tensor([L, 10.0], device=DEVICE),
        ),
        wall_reflect=True, diverge_threshold=1e3,
    )

    # ─── 2) Phase-1 eval: lookup at random init ───
    lookup = StateLookupHead(
        state_dim=2, max_K=K_train_max, hidden=64,
    ).to(DEVICE)

    eval_Ks = [1, 5, 10, 20, K_train_max]
    eval_Ks = sorted(set(k for k in eval_Ks if k <= K_train_max))

    @torch.no_grad()
    def _eval_lookup_vs_cook(stage_name: str) -> dict:
        """For each test K, compute MSE of lookup's 1-shot prediction
        and cook's K-step prediction against ground truth."""
        n_test = 50
        results: dict[int, dict[str, float]] = {}
        for K in eval_Ks:
            x = torch.empty(n_test, device=DEVICE).uniform_(2, L - 2)
            v = torch.empty(n_test, device=DEVICE).uniform_(-3, 3)
            s0 = torch.stack([x, v], dim=-1)
            # Ground truth K-step.
            traj_true = ground_truth_rollout(s0, K, g=g, dt=dt, L=L)
            terminal_true = traj_true[-1]
            # Cook K-step.
            traj_cook, _ = cook(s0, K=K)
            terminal_cook = traj_cook[-1]
            cook_mse = float(
                (terminal_cook - terminal_true).pow(2).sum(dim=-1).mean().item()
            )
            # Lookup 1-shot.
            K_t = torch.tensor([K] * n_test, device=DEVICE)
            terminal_lookup = lookup(s0, K_t)
            lookup_mse = float(
                (terminal_lookup - terminal_true).pow(2).sum(dim=-1).mean().item()
            )
            results[K] = {
                "lookup_mse": lookup_mse,
                "cook_mse": cook_mse,
            }
        return {"label": stage_name, "per_K": results}

    phase1_eval = _eval_lookup_vs_cook("phase1_pre_distill")

    # ─── 3) Phase-2: distillation ───
    if distill_K_set is None:
        distill_K_set = list(range(1, K_train_max + 1, 2))  # every 2

    # Sample initial states uniformly across the bouncing-ball
    # state space.
    s_x = torch.empty(n_distill_states, device=DEVICE).uniform_(2, L - 2)
    s_v = torch.empty(n_distill_states, device=DEVICE).uniform_(-3, 3)
    sample_states = torch.stack([s_x, s_v], dim=-1)

    distill_t0 = time.time()
    distill_report = distill_physics_cook_to_lookup(
        cook=cook,
        lookup=lookup,
        sample_initial_states=sample_states,
        sample_Ks=distill_K_set,
        n_steps=distill_steps,
        batch_size=64,
        rng_seed=seed,
    )
    distill_wall = time.time() - distill_t0

    # ─── 4) Phase-2 eval ───
    phase2_eval = _eval_lookup_vs_cook("phase2_post_distill")

    return {
        "seed": seed,
        "train_wall_s": train_wall,
        "distill_wall_s": distill_wall,
        "K_train_max": K_train_max,
        "phase1": phase1_eval,
        "phase2": phase2_eval,
        "distill": {
            "n_pairs": distill_report.n_pairs_distilled,
            "n_steps": distill_report.n_steps,
            "initial_loss": distill_report.initial_loss,
            "final_loss": distill_report.final_loss,
            "diverge_rate": distill_report.cook_oracle_diverge_rate,
        },
    }


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=99400)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--K-train-max", type=int, default=50)
    ap.add_argument("--distill-steps", type=int, default=600)
    ap.add_argument("--n-distill-states", type=int, default=100)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f59_sleep_distill"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(
        f"  F59 physics sleep cache: K_train_max={args.K_train_max}, "
        f"n_seeds={args.n_seeds}, distill_steps={args.distill_steps}"
    )
    print("=" * 76)

    rows: list[dict] = []
    for si in range(args.n_seeds):
        seed = args.seed_base + si
        r = _run_one(
            seed,
            epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
            K_train_max=args.K_train_max,
            distill_steps=args.distill_steps,
            n_distill_states=args.n_distill_states,
        )
        rows.append(r)
        K_max = args.K_train_max
        p1 = r["phase1"]["per_K"][K_max]
        p2 = r["phase2"]["per_K"][K_max]
        print(
            f"  [seed={seed}] K={K_max} "
            f"phase1 lookup_mse={p1['lookup_mse']:.2f}  "
            f"phase2 lookup_mse={p2['lookup_mse']:.2f}  "
            f"cook_mse={p2['cook_mse']:.2f}  "
            f"distill {r['distill']['initial_loss']:.2f}->"
            f"{r['distill']['final_loss']:.2f}"
        )

    # ─── aggregate ───
    def _stats_per_K(K: int, key: str, root: list[dict],
                     phase: str) -> dict:
        vals = [r[phase]["per_K"][K][key] for r in root
                if K in r[phase]["per_K"]]
        if not vals:
            return {"n": 0}
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals)
                       / max(len(vals) - 1, 1))
        return {"mean": m, "std": sd, "min": min(vals), "max": max(vals),
                "n": len(vals)}

    eval_Ks = [1, 5, 10, 20, args.K_train_max]
    eval_Ks = sorted(set(k for k in eval_Ks if k <= args.K_train_max))
    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "per_seed": rows,
        "by_K": {},
    }
    for K in eval_Ks:
        summary["by_K"][K] = {
            "phase1_lookup_mse": _stats_per_K(K, "lookup_mse", rows, "phase1"),
            "phase2_lookup_mse": _stats_per_K(K, "lookup_mse", rows, "phase2"),
            "cook_mse": _stats_per_K(K, "cook_mse", rows, "phase2"),
        }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print("  F59 sleep-cache summary:")
    print(f"  {'K':>4s}  {'p1 lookup MSE':>16s}  "
          f"{'p2 lookup MSE':>16s}  {'cook MSE':>14s}")
    for K in eval_Ks:
        p1 = summary["by_K"][K]["phase1_lookup_mse"]
        p2 = summary["by_K"][K]["phase2_lookup_mse"]
        c = summary["by_K"][K]["cook_mse"]
        print(
            f"  {K:>4d}  "
            f"{p1.get('mean', float('nan')):>10.2f}+-"
            f"{p1.get('std', float('nan')):.2f}  "
            f"{p2.get('mean', float('nan')):>10.2f}+-"
            f"{p2.get('std', float('nan')):.2f}  "
            f"{c.get('mean', float('nan')):>8.2f}+-"
            f"{c.get('std', float('nan')):.2f}"
        )

    # F59 PASS condition: post-distill lookup MSE matches or beats
    # cook MSE on at least the largest in-range K, AND falls
    # significantly from phase1 init.
    K_check = args.K_train_max
    p1_m = summary["by_K"][K_check]["phase1_lookup_mse"].get(
        "mean", float("nan"))
    p2_m = summary["by_K"][K_check]["phase2_lookup_mse"].get(
        "mean", float("nan"))
    c_m = summary["by_K"][K_check]["cook_mse"].get("mean", float("nan"))
    drop = p1_m - p2_m
    print(
        f"\n  K={K_check} lookup MSE drop: {p1_m:.2f} -> {p2_m:.2f}  "
        f"(reduction {drop:.2f})"
    )
    print(f"  K={K_check} cook MSE:        {c_m:.2f}")
    f59_pass = drop > 0.5 * p1_m and p2_m < 5.0 * c_m
    print(
        f"\n  F59 verdict: "
        f"{'[PASS]' if f59_pass else '[FAIL]'} "
        f"(target lookup MSE drop > 50%, post-distill within 5x cook)"
    )
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
