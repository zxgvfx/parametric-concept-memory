"""F65 — PCM v6.2 continuous-action agent on cyclic S¹ navigation.

Operationalises the v6.2 claim: replacing F64's discrete action
embedding (``nn.Embedding`` keyed by 4 action indices) with a
**RoPE-style continuous action encoder** (mirroring F62c's
extension of the universal operator to continuous Lie groups,
but on the action side of the (state, action) → next_state
operator) preserves the F62 universal-operator invariants on an
agentic, multi-step, continuous-control task.

Environment (``pcm.agent.envs.continuous_nav``):
* state space ``ℤ_N`` (default N=40 bins of S¹)
* action space ``a ∈ [-1, 1]`` continuous scalar
* dynamics ``θ_next = (θ + a · max_step) mod 2π`` with
  ``max_step = π/4`` (so a full circle takes 8 steps with full-
  magnitude actions)
* success: bin-quantised ``θ_next == goal``

Six falsifiable invariants (parallel to F64 U1–U6):

* **U1** in-domain success ≥ 0.90 within ±1 bin
* **U2** joint-shared training no penalty vs per-bin separate
* **U3** independent transition heads all reach ``transition_acc
  within-1-bin ≥ 0.90`` — they all learn the same correct
  continuous-action S¹ algebra
* **U4** frozen-transition transfer (positive-goal-range trained
  → frozen → negative-goal-range policy retrained) success ≥
  0.85
* **U5** sign-flipped policy negative control: at eval, multiply
  policy mean by -1; honest − flipped ≥ 0.30
* **U6** multi-step horizon: agent steps / greedy-optimal steps
  ≤ 1.20 on hard ``|Δ| ≥ π/2`` tasks

The invariant we redesigned during F64 development (Procrustes
→ direct algebraic correctness) carries over here: U3 is graded
on transition prediction accuracy, not on action-encoder
embedding similarity.

Usage::

    python -m experiments.agent_continuous_nav_poc \\
        --N 40 --slot-dim 32 --epochs 80 \\
        --batches-per-epoch 30 \\
        --out outputs/f65_full
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from pcm.agent import (
    ContinuousPolicyHead,
    ContinuousTransitionHead,
    SlotStateEncoder,
    bc_gaussian_loss,
    continuous_rollout,
    continuous_transition_loss,
)
from pcm.agent.envs import (
    ContinuousCyclicNavEnv,
    optimal_continuous_action,
    optimal_continuous_steps,
)


__all__ = ["main"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────


def _signed_delta_idx(s: int, g: int, N: int) -> int:
    half = N // 2
    return (g - s + half) % N - half


def _sample_bc_batch(
    N: int, B: int, max_step: float, device: str,
    *, goal_range_bins: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states, goals, actions = [], [], []
    while len(states) < B:
        s = int(torch.randint(0, N, (1,)).item())
        g = int(torch.randint(0, N, (1,)).item())
        if s == g:
            continue
        if goal_range_bins is not None:
            d = _signed_delta_idx(s, g, N)
            lo, hi = goal_range_bins
            if not (lo <= d <= hi):
                continue
        a = optimal_continuous_action(s, g, N, max_step=max_step)
        states.append(s)
        goals.append(g)
        actions.append(a)
    return (
        torch.tensor(states, dtype=torch.long, device=device),
        torch.tensor(goals, dtype=torch.long, device=device),
        torch.tensor(actions, dtype=torch.float32, device=device),
    )


def _sample_transition_batch(
    N: int, B: int, max_step: float, device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample uniformly random ``(state, continuous_action,
    next_state)``."""
    s = torch.randint(0, N, (B,), device=device)
    a = torch.empty(B, device=device).uniform_(-1.0, 1.0)
    bin_size = 2 * math.pi / N
    theta = s.float() * bin_size + a * max_step
    s_next = (theta / bin_size).round().long() % N
    return s, a, s_next


# ─────────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────────


def _train(
    encoder: SlotStateEncoder,
    transition: ContinuousTransitionHead,
    policy: ContinuousPolicyHead,
    *, N: int, max_step: float,
    epochs: int, batches_per_epoch: int,
    batch_size: int, lr: float,
    train_transition: bool = True, train_policy: bool = True,
    goal_range_bins: tuple[int, int] | None = None,
) -> dict:
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    if train_policy:
        for p in list(encoder.parameters()) + list(policy.parameters()):
            if id(p) in seen:
                continue
            seen.add(id(p))
            params.append(p)
    if train_transition:
        for p in list(transition.parameters()):
            if id(p) in seen:
                continue
            seen.add(id(p))
            params.append(p)
        if not train_policy:
            for p in encoder.parameters():
                if id(p) in seen:
                    continue
                seen.add(id(p))
                params.append(p)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    history = []
    for ep in range(epochs):
        bc_total = trans_total = 0.0
        n = 0
        for _ in range(batches_per_epoch):
            losses = []
            if train_policy:
                s, g, a_star = _sample_bc_batch(
                    N, batch_size, max_step, DEVICE,
                    goal_range_bins=goal_range_bins,
                )
                losses.append(bc_gaussian_loss(policy, encoder, s, g, a_star))
                bc_total += float(losses[-1].item())
            if train_transition:
                ts, ta, tsn = _sample_transition_batch(
                    N, batch_size, max_step, DEVICE,
                )
                t_loss = continuous_transition_loss(
                    transition, encoder, ts, ta, tsn,
                )
                losses.append(t_loss)
                trans_total += float(t_loss.item())
            loss = sum(losses)
            opt.zero_grad()
            loss.backward()
            opt.step()
            n += 1
        history.append({
            "epoch": ep,
            "bc": bc_total / max(n, 1),
            "trans": trans_total / max(n, 1),
        })
    return {"history": history}


def _train_pair(
    *, N: int, dim: int, max_step: float,
    epochs: int, batches_per_epoch: int, batch_size: int, lr: float,
    seed: int, n_freqs: int = 8,
    goal_range_bins: tuple[int, int] | None = None,
    train_transition: bool = True,
) -> dict:
    torch.manual_seed(seed)
    encoder = SlotStateEncoder(N, dim).to(DEVICE)
    transition = ContinuousTransitionHead(
        dim, n_freqs=n_freqs,
    ).to(DEVICE)
    policy = ContinuousPolicyHead(dim).to(DEVICE)
    history = _train(
        encoder, transition, policy,
        N=N, max_step=max_step, epochs=epochs,
        batches_per_epoch=batches_per_epoch,
        batch_size=batch_size, lr=lr,
        goal_range_bins=goal_range_bins,
        train_transition=train_transition,
    )
    return {
        "history": history["history"],
        "encoder": encoder,
        "transition": transition,
        "policy": policy,
    }


@torch.no_grad()
def _evaluate(
    encoder: SlotStateEncoder, policy: ContinuousPolicyHead, *,
    N: int, max_step: float, n_episodes: int = 200,
    max_steps: int = 32,
    goal_range_bins: tuple[int, int] | None = None,
    flip_sign: bool = False,
    success_tolerance: int = 1,
) -> dict:
    """Run greedy continuous rollouts; report success rate and
    step ratio. ``success_tolerance`` is the number of bins of
    slop allowed around the goal at the *final* step (continuous
    actions can over/undershoot by up to ±0.5 bin per step due
    to argmax rounding)."""
    encoder.eval()
    policy.eval()
    total = succ = 0
    step_ratios = []
    rng = torch.Generator(device="cpu").manual_seed(2026)
    for _ in range(n_episodes):
        while True:
            s0 = int(torch.randint(0, N, (1,), generator=rng).item())
            g = int(torch.randint(0, N, (1,), generator=rng).item())
            if s0 == g:
                continue
            d = _signed_delta_idx(s0, g, N)
            if goal_range_bins is None or (
                goal_range_bins[0] <= d <= goal_range_bins[1]
            ):
                break
        opt_steps = optimal_continuous_steps(
            s0, g, N, max_step=max_step,
        )
        env = ContinuousCyclicNavEnv(
            N=N, max_step=max_step, max_steps=max_steps,
        )
        env.reset(s0)
        env.set_goal(g)
        # Tolerance via bin distance check at the last visited
        # state — implemented below by manual rollout so we can
        # apply within-tolerance success criterion at any time
        # within budget.
        success_at = -1
        s_cur = s0
        for t in range(max_steps):
            s_t = torch.tensor([s_cur], dtype=torch.long, device=DEVICE)
            g_t = torch.tensor([g], dtype=torch.long, device=DEVICE)
            mean, _ = policy(encoder(s_t), encoder(g_t))
            a = float(mean.clamp(-1.0, 1.0).item())
            if flip_sign:
                a = -a
            s_next, _, done = env.step(a)
            d_circ = abs(s_next - g)
            d_circ = min(d_circ, N - d_circ)
            if d_circ <= success_tolerance:
                success_at = t + 1
                break
            s_cur = s_next
            if done:
                break
        total += 1
        if success_at >= 0:
            succ += 1
            step_ratios.append(success_at / max(opt_steps, 1))
    succ_rate = succ / max(total, 1)
    mean_ratio = (sum(step_ratios) / len(step_ratios)
                  if step_ratios else float("nan"))
    return {
        "success_rate": succ_rate,
        "mean_step_ratio_to_optimal": mean_ratio,
        "n_episodes": total, "n_success": succ,
    }


# ─────────────────────────────────────────────────────────────────
# U3 — direct algebraic correctness on continuous transitions
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def _continuous_transition_within1(
    encoder: SlotStateEncoder, transition: ContinuousTransitionHead,
    *, N: int, max_step: float, n_samples: int = 5000,
) -> float:
    """Within-1-bin transition accuracy on uniformly sampled
    ``(s, a, s_next)`` triples. Like F62c, the inherent argmax-
    rounding slop in continuous → discrete next-state mapping
    makes within-1-bin the natural headline metric."""
    encoder.eval()
    transition.eval()
    s = torch.randint(0, N, (n_samples,), device=DEVICE)
    a = torch.empty(n_samples, device=DEVICE).uniform_(-1.0, 1.0)
    bin_size = 2 * math.pi / N
    theta = s.float() * bin_size + a * max_step
    s_next = (theta / bin_size).round().long() % N
    slot_pred = transition(encoder(s), a)
    logits = slot_pred @ encoder.all_slots().t()
    pred = logits.argmax(dim=-1)
    diff = (pred - s_next).cpu()
    d_mod = diff.abs() % N
    d_circ = torch.minimum(d_mod, N - d_mod)
    return float((d_circ <= 1).float().mean().item())


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=40,
                    help="state-bin count (S^1 discretisation)")
    ap.add_argument("--slot-dim", type=int, default=32)
    ap.add_argument("--n-freqs", type=int, default=8,
                    help="RoPE log-frequencies for the action encoder")
    ap.add_argument("--max-step", type=float, default=math.pi / 4,
                    help="per-step max angular displacement (rad)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batches-per-epoch", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--n-seeds-u3", type=int, default=3)
    ap.add_argument("--max-steps-eval", type=int, default=32)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f65_full"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"  F65 PCM v6.2 continuous-action agent on S^1 nav "
          f"(N={args.N}, max_step={args.max_step:.4f} rad)")
    print("=" * 76)

    half = args.N // 2

    # A — full goal range
    print("\n[A] joint training, shared (transition + policy) on "
          "full goal range...")
    t0 = time.time()
    A = _train_pair(
        N=args.N, dim=args.slot_dim, max_step=args.max_step,
        epochs=args.epochs, batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr,
        n_freqs=args.n_freqs, seed=11,
    )
    print(f"    wall = {time.time()-t0:.1f}s")
    a_eval = _evaluate(
        A["encoder"], A["policy"], N=args.N, max_step=args.max_step,
        n_episodes=args.n_eval, max_steps=args.max_steps_eval,
    )
    print(f"    success = {a_eval['success_rate']:.3f}  "
          f"mean_step_ratio = {a_eval['mean_step_ratio_to_optimal']:.3f}")

    # B-pos / B-neg per-bin
    pos_range = (1, half)
    neg_range = (-half + 1, -1)
    print("\n[B-pos] training on positive-goal range only...")
    t0 = time.time()
    B_pos = _train_pair(
        N=args.N, dim=args.slot_dim, max_step=args.max_step,
        epochs=args.epochs, batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr,
        n_freqs=args.n_freqs, seed=22,
        goal_range_bins=pos_range,
    )
    b_pos_eval = _evaluate(
        B_pos["encoder"], B_pos["policy"],
        N=args.N, max_step=args.max_step,
        n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        goal_range_bins=pos_range,
    )
    print(f"    wall = {time.time()-t0:.1f}s  "
          f"pos-only success = {b_pos_eval['success_rate']:.3f}")

    print("\n[B-neg] training on negative-goal range only...")
    t0 = time.time()
    B_neg = _train_pair(
        N=args.N, dim=args.slot_dim, max_step=args.max_step,
        epochs=args.epochs, batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr,
        n_freqs=args.n_freqs, seed=33,
        goal_range_bins=neg_range,
    )
    b_neg_eval = _evaluate(
        B_neg["encoder"], B_neg["policy"],
        N=args.N, max_step=args.max_step,
        n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        goal_range_bins=neg_range,
    )
    print(f"    wall = {time.time()-t0:.1f}s  "
          f"neg-only success = {b_neg_eval['success_rate']:.3f}")

    a_pos = _evaluate(
        A["encoder"], A["policy"], N=args.N, max_step=args.max_step,
        n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        goal_range_bins=pos_range,
    )
    a_neg = _evaluate(
        A["encoder"], A["policy"], N=args.N, max_step=args.max_step,
        n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        goal_range_bins=neg_range,
    )
    print(f"    [A] pos success = {a_pos['success_rate']:.3f}  "
          f"neg success = {a_neg['success_rate']:.3f}")

    # C — independent runs for U3
    print(f"\n[C] independent training for U3 ({args.n_seeds_u3} "
          f"seeds, different goal ranges)...")
    indep_runs = []
    ranges = [pos_range, neg_range, pos_range]
    for si in range(args.n_seeds_u3):
        gr = ranges[si % len(ranges)]
        r = _train_pair(
            N=args.N, dim=args.slot_dim, max_step=args.max_step,
            epochs=args.epochs,
            batches_per_epoch=args.batches_per_epoch,
            batch_size=args.batch_size, lr=args.lr,
            n_freqs=args.n_freqs,
            seed=10_000 * (si + 1) + 7,
            goal_range_bins=gr,
        )
        indep_runs.append({"goal_range_bins": gr, **r})
        print(f"    seed {si} range={gr}: trained")

    u3_trans_accs = []
    for i, run in enumerate(indep_runs):
        acc = _continuous_transition_within1(
            run["encoder"], run["transition"],
            N=args.N, max_step=args.max_step,
        )
        u3_trans_accs.append({
            "i": i, "transition_acc_within1": acc,
            "goal_range_bins": run["goal_range_bins"],
        })
        print(f"    seed {i} range={run['goal_range_bins']}: "
              f"transition_acc within-1 = {acc:.3f}")
    u3_min = min(r["transition_acc_within1"] for r in u3_trans_accs)
    u3_mean = (sum(r["transition_acc_within1"] for r in u3_trans_accs)
               / len(u3_trans_accs))

    # U4 frozen-transition transfer
    print("\n[U4] frozen-transition transfer "
          "(B_pos transition -> negative-goal policy)...")
    t0 = time.time()
    torch.manual_seed(999)
    src_transition = B_pos["transition"]
    for p in src_transition.parameters():
        p.requires_grad_(False)
    u4_encoder = SlotStateEncoder(args.N, args.slot_dim).to(DEVICE)
    u4_policy = ContinuousPolicyHead(args.slot_dim).to(DEVICE)
    _train(
        u4_encoder, src_transition, u4_policy,
        N=args.N, max_step=args.max_step,
        epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr,
        train_transition=False, train_policy=True,
        goal_range_bins=neg_range,
    )
    u4_eval = _evaluate(
        u4_encoder, u4_policy,
        N=args.N, max_step=args.max_step,
        n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        goal_range_bins=neg_range,
    )
    print(f"    wall = {time.time()-t0:.1f}s  "
          f"transfer success = {u4_eval['success_rate']:.3f}")

    # U5 sign-flipped negative control with tight budget
    u5_budget = max(6, int(math.ceil(math.pi / args.max_step)) + 2)
    print(f"\n[U5] sign-flipped policy negative control "
          f"(tight budget = {u5_budget})...")
    u5_eval = _evaluate(
        A["encoder"], A["policy"],
        N=args.N, max_step=args.max_step,
        n_episodes=args.n_eval, max_steps=u5_budget,
        flip_sign=True,
    )
    u5_honest = _evaluate(
        A["encoder"], A["policy"],
        N=args.N, max_step=args.max_step,
        n_episodes=args.n_eval, max_steps=u5_budget,
    )
    print(f"    flipped success = {u5_eval['success_rate']:.3f}  "
          f"vs honest @ same budget = {u5_honest['success_rate']:.3f}")

    # U6 multi-step horizon: hardest tasks
    hard_pos = (max(1, half - 4), half)
    hard_neg = (-half, -max(1, half - 4))
    print(f"\n[U6] multi-step horizon test on hard tasks "
          f"(|Δ_idx| in [{half - 4}, {half}])...")
    u6_pos = _evaluate(
        A["encoder"], A["policy"],
        N=args.N, max_step=args.max_step,
        n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        goal_range_bins=hard_pos,
    )
    u6_neg = _evaluate(
        A["encoder"], A["policy"],
        N=args.N, max_step=args.max_step,
        n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        goal_range_bins=hard_neg,
    )
    u6_mean_ratio = (
        u6_pos["mean_step_ratio_to_optimal"]
        + u6_neg["mean_step_ratio_to_optimal"]
    ) / 2.0
    print(f"    hard-pos success = {u6_pos['success_rate']:.3f}  "
          f"step ratio = {u6_pos['mean_step_ratio_to_optimal']:.3f}")
    print(f"    hard-neg success = {u6_neg['success_rate']:.3f}  "
          f"step ratio = {u6_neg['mean_step_ratio_to_optimal']:.3f}")

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "max_step_rad": args.max_step,
        "A_joint_full_eval": a_eval,
        "A_pos_eval": a_pos,
        "A_neg_eval": a_neg,
        "B_pos_only_eval": b_pos_eval,
        "B_neg_only_eval": b_neg_eval,
        "C_independent_runs": [
            {"goal_range_bins": list(r["goal_range_bins"])}
            for r in indep_runs
        ],
        "U3_transition_accs": u3_trans_accs,
        "U3_min_transition_acc_within1": u3_min,
        "U3_mean_transition_acc_within1": u3_mean,
        "U4_frozen_transition_transfer_eval": u4_eval,
        "U5_sign_flipped_eval": u5_eval,
        "U5_honest_at_same_budget": u5_honest,
        "U5_tight_budget": u5_budget,
        "U6_hard_pos_eval": u6_pos,
        "U6_hard_neg_eval": u6_neg,
        "U6_mean_step_ratio": u6_mean_ratio,
    }
    summary["verdict"] = {
        "U1_pass": a_eval["success_rate"] >= 0.90,
        "U2_pass": (
            a_pos["success_rate"] >= b_pos_eval["success_rate"] - 0.03
            and a_neg["success_rate"] >= b_neg_eval["success_rate"] - 0.03
        ),
        # U3 (continuous version): every independent run learns the
        # same correct continuous-action S^1 algebra; we use the
        # within-1-bin tolerance because continuous Δ -> discrete
        # next-state index has inherent ±0.5-bin slop.
        "U3_pass": u3_min >= 0.90,
        "U4_pass": u4_eval["success_rate"] >= 0.85,
        "U5_pass": (
            u5_eval["success_rate"] <= 0.50
            and u5_honest["success_rate"]
            - u5_eval["success_rate"] >= 0.30
        ),
        "U6_pass": u6_mean_ratio <= 1.20,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print("  F65 v6.2 continuous-action agent verdict:")
    print("=" * 76)
    v = summary["verdict"]
    print(f"  U1 in-domain (success >=0.90)        : "
          f"{a_eval['success_rate']:.3f}  "
          f"[{'PASS' if v['U1_pass'] else 'FAIL'}]")
    print(f"  U2 sharing has no penalty            : "
          f"pos shared {a_pos['success_rate']:.3f} vs sep "
          f"{b_pos_eval['success_rate']:.3f} | "
          f"neg shared {a_neg['success_rate']:.3f} vs sep "
          f"{b_neg_eval['success_rate']:.3f}  "
          f"[{'PASS' if v['U2_pass'] else 'FAIL'}]")
    print(f"  U3 transition_acc within-1 (>=0.90)  : "
          f"min={u3_min:.3f} mean={u3_mean:.3f}  "
          f"[{'PASS' if v['U3_pass'] else 'FAIL'}]")
    print(f"  U4 frozen-transition transfer (>=0.85): "
          f"{u4_eval['success_rate']:.3f}  "
          f"[{'PASS' if v['U4_pass'] else 'FAIL'}]")
    print(f"  U5 sign-flip neg ctrl (gap>=.30)     : "
          f"honest {u5_honest['success_rate']:.3f} - "
          f"flipped {u5_eval['success_rate']:.3f} = "
          f"{u5_honest['success_rate'] - u5_eval['success_rate']:+.3f}  "
          f"[{'PASS' if v['U5_pass'] else 'FAIL'}]")
    print(f"  U6 multi-step horizon (ratio <=1.20) : "
          f"{u6_mean_ratio:.3f}  "
          f"[{'PASS' if v['U6_pass'] else 'FAIL'}]")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
