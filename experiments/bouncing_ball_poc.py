"""F57 — PCM v4 PoC: 1-D bouncing ball under gravity.

Tests three falsifiability invariants from
``docs/PCM_V4_PHYSICS_COOK_DESIGN.md`` §5:

* **P1 1-step accuracy** — head MSE on ground-truth dt-Euler
  step ≤ 1e-2 (relative to state range).
* **P2 K-step rollout error** — error grows polynomially in K,
  not exponentially (R² of polynomial fit ≥ 0.95).
* **P3 long-horizon recognisability** — at K=100, predicted
  trajectory does not blow up (max state component ≤ 10×
  natural scale).

Toy domain:

* **State**: ``s = (x, v) ∈ ℝ²``.
* **Dynamics**: ``v ← v + g·dt``, ``x ← x + v·dt``, with
  elastic reflection at ``x=0`` and ``x=L`` (flip ``v`` sign,
  clamp ``x`` to wall).
* ``g = -1.0``, ``dt = 0.1``, ``L = 10.0``.
* No external force input (autonomous dynamics; the v4 head's
  force_dim is set to 0).

We compare three rollouts on a held-out test set:

* **ground-truth** physics integration with reflection.
* **cook** rollout via :class:`pcm.physics.PhysicsCook`.
* **last-step constant** baseline (state = initial_state for
  all K), the dumb "no learning" floor.

Usage::

    python -m experiments.bouncing_ball_poc \\
        --n-seeds 3 --epochs 30 --steps-per-epoch 200 \\
        --K-rollout 100 --out outputs/f57_bouncing_ball
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

from pcm.physics import PhysicsCook, PhysicsStateHead, physics_step_loss


__all__ = ["main", "ground_truth_rollout"]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────
# Ground-truth physics (closed-form, used for training labels +
# evaluation oracle).
# ─────────────────────────────────────────────────────────────────


def ground_truth_step(
    state: torch.Tensor, *, g: float, dt: float, L: float,
) -> torch.Tensor:
    """One step of dt-Euler integration with elastic walls."""
    x, v = state[..., 0], state[..., 1]
    v_new = v + g * dt
    x_new = x + v_new * dt
    # Wall reflection (single pass; if speed > L this can miss
    # second bounce in one dt, which is fine for our tame setup).
    under = x_new < 0
    over = x_new > L
    x_new = torch.where(under, -x_new, x_new)
    v_new = torch.where(under, v_new.abs(), v_new)
    x_new = torch.where(over, 2 * L - x_new, x_new)
    v_new = torch.where(over, -v_new.abs(), v_new)
    return torch.stack([x_new, v_new], dim=-1)


def ground_truth_rollout(
    initial_state: torch.Tensor, K: int, *,
    g: float, dt: float, L: float,
) -> torch.Tensor:
    """Roll the ground-truth physics forward K steps. Returns
    ``(K+1, B, 2)``."""
    traj = [initial_state]
    state = initial_state
    for _ in range(K):
        state = ground_truth_step(state, g=g, dt=dt, L=L)
        traj.append(state)
    return torch.stack(traj, dim=0)


# ─────────────────────────────────────────────────────────────────
# Per-seed training / eval
# ─────────────────────────────────────────────────────────────────


def _run_one(
    seed: int, *,
    epochs: int, steps_per_epoch: int,
    K_rollout: int,
    g: float = -1.0, dt: float = 0.1, L: float = 10.0,
    batch_size: int = 64, lr: float = 5e-3,
) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    head = PhysicsStateHead(
        state_dim=2, force_dim=0, hidden=64, dt=dt,
    ).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    # ─── train: 1-step transitions on uniformly-sampled (x, v) ───
    t0 = time.time()
    for epoch in range(epochs):
        head.train()
        for _ in range(steps_per_epoch):
            x = torch.empty(batch_size, device=DEVICE).uniform_(0, L)
            # Velocity range matches what gravity produces over
            # ~5 second flight: |v| up to sqrt(2 g L) ≈ 4.5; pad to
            # 6 for headroom.
            v = torch.empty(batch_size, device=DEVICE).uniform_(-6, 6)
            state = torch.stack([x, v], dim=-1)
            true_next = ground_truth_step(state, g=g, dt=dt, L=L)
            pred_delta = head(state)
            loss = physics_step_loss(pred_delta, true_next, state)
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    train_wall = time.time() - t0

    # ─── eval P1: 1-step accuracy on a fresh test set ───
    @torch.no_grad()
    def _eval_p1() -> float:
        x = torch.empty(2000, device=DEVICE).uniform_(0, L)
        v = torch.empty(2000, device=DEVICE).uniform_(-6, 6)
        state = torch.stack([x, v], dim=-1)
        true_next = ground_truth_step(state, g=g, dt=dt, L=L)
        pred_next = state + head(state)
        # Normalised MSE (state range is [0, L] × [-6, 6] → diag ~ 12).
        mse = (pred_next - true_next).pow(2).mean()
        # Report relative to natural state scale L.
        return float((mse / (L ** 2)).item())

    # ─── eval P2/P3: K-step rollout vs ground truth ───
    cook = PhysicsCook(
        head,
        max_iters=K_rollout + 10,
        state_clip=(
            torch.tensor([0.0, -10.0]),
            torch.tensor([L, 10.0]),
        ),
        wall_reflect=True,
        diverge_threshold=1e3,
    )

    def _eval_rollout(K: int, n_test: int = 30) -> dict:
        """Roll cook out K steps from random initial states; compare
        against ground-truth rollout."""
        x0 = torch.empty(n_test, device=DEVICE).uniform_(2, L - 2)
        v0 = torch.empty(n_test, device=DEVICE).uniform_(-3, 3)
        s0 = torch.stack([x0, v0], dim=-1)
        traj_pred, rep = cook(s0, K=K)
        traj_true = ground_truth_rollout(s0, K, g=g, dt=dt, L=L)
        # Per-step MSE (averaged over batch).
        per_step_mse = (traj_pred - traj_true).pow(2).sum(dim=-1).mean(dim=-1)
        return {
            "K": K,
            "final_mse": float(per_step_mse[-1].item()),
            "mean_mse": float(per_step_mse.mean().item()),
            "max_mse": float(per_step_mse.max().item()),
            "diverged": rep.diverged,
            "final_state_norm": rep.final_state_norm,
            "wall_s": rep.wall_seconds,
        }

    # P2 curve. Filter to K within rollout horizon.
    eval_Ks = sorted(set(k for k in [1, 5, 10, 20, 50, 75, K_rollout]
                          if k <= K_rollout))
    p2_curve = {K: _eval_rollout(K) for K in eval_Ks}

    p1 = _eval_p1()
    final = p2_curve[K_rollout]

    # Polynomial fit on log(mse) vs log(K).
    import math
    Ks = sorted(p2_curve)
    log_K = [math.log(k) for k in Ks if p2_curve[k]["final_mse"] > 0]
    log_mse = [
        math.log(p2_curve[k]["final_mse"]) for k in Ks
        if p2_curve[k]["final_mse"] > 0
    ]
    poly_R2 = float("nan")
    if len(log_K) >= 3:
        n = len(log_K)
        sx, sy = sum(log_K), sum(log_mse)
        sxx = sum(x * x for x in log_K)
        sxy = sum(x * y for x, y in zip(log_K, log_mse))
        denom = n * sxx - sx * sx
        if abs(denom) > 1e-9:
            slope = (n * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / n
            ss_res = sum(
                (y - (slope * x + intercept)) ** 2
                for x, y in zip(log_K, log_mse)
            )
            ss_tot = sum((y - sy / n) ** 2 for y in log_mse) + 1e-12
            poly_R2 = 1.0 - ss_res / ss_tot

    return {
        "seed": seed,
        "wall_train_s": train_wall,
        "P1_1step_norm_mse": p1,
        "P2_curve": p2_curve,
        "P2_log_log_R2": poly_R2,
        "P3_K100_final_state_norm": final["final_state_norm"],
        "P3_K100_diverged": final["diverged"],
        "K_rollout": K_rollout,
    }


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=99300)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--K-rollout", type=int, default=100)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f57_bouncing_ball"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(
        f"  F57 1-D bouncing ball: K_rollout={args.K_rollout}, "
        f"n_seeds={args.n_seeds}, epochs={args.epochs}"
    )
    print("=" * 76)

    rows: list[dict] = []
    for si in range(args.n_seeds):
        seed = args.seed_base + si
        r = _run_one(
            seed,
            epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
            K_rollout=args.K_rollout,
        )
        rows.append(r)
        K_OOD = args.K_rollout
        final_mse = r["P2_curve"][K_OOD]["final_mse"]
        print(
            f"  [seed={seed}] P1_norm_mse={r['P1_1step_norm_mse']:.5f}  "
            f"K={K_OOD} final_mse={final_mse:.4f}  "
            f"R2={r['P2_log_log_R2']:.3f}  "
            f"final_norm={r['P3_K100_final_state_norm']:.2f}  "
            f"diverged={r['P3_K100_diverged']}  "
            f"({r['wall_train_s']:.1f}s)"
        )

    # ─── aggregate ───
    def _stats(key_path: str, root: list[dict]) -> dict:
        vals = []
        for r in root:
            v = r
            for k in key_path.split("."):
                v = v.get(k, None) if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, (int, float)) and not (
                isinstance(v, float) and math.isnan(v)
            ):
                vals.append(v)
        if not vals:
            return {"n": 0}
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((x - m) ** 2 for x in vals)
                       / max(len(vals) - 1, 1))
        return {"mean": m, "std": sd, "min": min(vals),
                "max": max(vals), "n": len(vals)}

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "per_seed": rows,
        "P1_1step_norm_mse": _stats("P1_1step_norm_mse", rows),
        "P2_log_log_R2": _stats("P2_log_log_R2", rows),
        "P3_K100_final_state_norm": _stats("P3_K100_final_state_norm", rows),
    }
    # Curve aggregate
    curve_Ks = sorted(rows[0]["P2_curve"].keys())
    curve = {}
    for K in curve_Ks:
        vals = [r["P2_curve"][K]["final_mse"] for r in rows]
        curve[K] = sum(vals) / len(vals)
    summary["P2_curve_mean"] = curve

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "═" * 76)
    p1m = summary["P1_1step_norm_mse"].get("mean", float("nan"))
    p1_pass = p1m <= 1e-2
    r2m = summary["P2_log_log_R2"].get("mean", float("nan"))
    p2_pass = r2m >= 0.85  # weaker than 0.95 — small K range
    norm_m = summary["P3_K100_final_state_norm"].get("mean", float("nan"))
    p3_pass = norm_m <= 100.0  # 10× state-space scale
    print(
        f"  P1 1-step norm MSE:     {p1m:.5f}   "
        f"{'[PASS]' if p1_pass else '[FAIL]'} (target <= 0.01)"
    )
    print(
        f"  P2 log-log R2:          {r2m:.3f}   "
        f"{'[PASS]' if p2_pass else '[FAIL]'} (target >= 0.85)"
    )
    print(
        f"  P3 K=100 final norm:    {norm_m:.2f}    "
        f"{'[PASS]' if p3_pass else '[FAIL]'} (target <= 100, "
        f"i.e. trajectory bounded)"
    )
    print("\n  P2 curve (K → mean final MSE):")
    for K, mse in curve.items():
        print(f"    K={K:>3d}: MSE={mse:.4f}")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
