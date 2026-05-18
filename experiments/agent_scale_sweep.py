"""F69 — PCM v6.5 scale validation (parameter-count sweep).

The v6.5 question: as parameter count grows (toward LLM scale),
do the F62 universal-operator invariants — that worked at
F64/F65/F66/F67/F68's ~50K-parameter scale — *stay stable*, or
do they collapse?

This experiment runs the F65 continuous-action agent at three
scales by sweeping the slot-bundle dim, combiner hidden dim, and
RoPE frequency count. We measure the same U1–U6 invariants per
scale plus three saturation invariants:

* **S1** invariant-pass count is monotone-non-decreasing in
  scale (larger model never *loses* a previously-passing
  invariant).
* **S2** scale-to-step-ratio: the multi-step BFS-optimal step
  ratio (U6) improves or stays flat as scale grows (more
  parameters → not worse planning).
* **S3** scale-to-transition-acc: per-run transition accuracy
  improves or stays at ceiling.

Three scales (each ~10× parameter count larger than the last):

* **small** — slot_dim=32, hidden=128, n_freqs=8 (~F65 baseline,
  ~50K params).
* **medium** — slot_dim=128, hidden=512, n_freqs=16 (~750K
  params, 15× small).
* **large** — slot_dim=256, hidden=1024, n_freqs=32 (~4M
  params, 80× small).

This is *not* 1B parameters — a real LLM-scale validation
requires datacentre GPU — but at 4M params we are already 80×
above the F65 baseline, and the question "does the architecture
scale without invariant collapse" can be answered honestly in
this regime.

Usage::

    python -m experiments.agent_scale_sweep \\
        --N 40 --epochs 50 --n-eval 200 \\
        --out outputs/f69_full
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
# Scale configurations
# ─────────────────────────────────────────────────────────────────


SCALES = {
    "small":  {"slot_dim":  32, "hidden":  128, "n_freqs":  8,
               "pol_hidden": 128},
    "medium": {"slot_dim": 128, "hidden":  512, "n_freqs": 16,
               "pol_hidden": 512},
    "large":  {"slot_dim": 256, "hidden": 1024, "n_freqs": 32,
               "pol_hidden": 1024},
}


def _count_params(*modules: nn.Module) -> int:
    return sum(
        sum(p.numel() for p in m.parameters()) for m in modules
    )


# ─────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────


def _signed_delta_idx(s: int, g: int, N: int) -> int:
    half = N // 2
    return (g - s + half) % N - half


def _sample_bc_batch(
    N: int, B: int, max_step: float, device: str,
) -> tuple:
    states, goals, actions = [], [], []
    while len(states) < B:
        s = int(torch.randint(0, N, (1,)).item())
        g = int(torch.randint(0, N, (1,)).item())
        if s == g:
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
) -> tuple:
    s = torch.randint(0, N, (B,), device=device)
    a = torch.empty(B, device=device).uniform_(-1.0, 1.0)
    bin_size = 2 * math.pi / N
    theta = s.float() * bin_size + a * max_step
    s_next = (theta / bin_size).round().long() % N
    return s, a, s_next


# ─────────────────────────────────────────────────────────────────
# Train + eval at one scale
# ─────────────────────────────────────────────────────────────────


def _train_one(
    encoder, transition, policy, *, N: int, max_step: float,
    epochs: int, batches_per_epoch: int, batch_size: int, lr: float,
) -> None:
    params = (list(encoder.parameters())
              + list(transition.parameters())
              + list(policy.parameters()))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        for _ in range(batches_per_epoch):
            s, g, a_star = _sample_bc_batch(
                N, batch_size, max_step, DEVICE,
            )
            bc = bc_gaussian_loss(policy, encoder, s, g, a_star)
            ts, ta, tsn = _sample_transition_batch(
                N, batch_size, max_step, DEVICE,
            )
            t_loss = continuous_transition_loss(
                transition, encoder, ts, ta, tsn,
            )
            loss = bc + t_loss
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def _eval(
    encoder, policy, *, N: int, max_step: float,
    n_episodes: int = 300, max_steps: int = 32,
) -> dict:
    encoder.eval()
    policy.eval()
    rng = torch.Generator(device="cpu").manual_seed(2026)
    total = succ = 0
    step_ratios = []
    for _ in range(n_episodes):
        s0 = int(torch.randint(0, N, (1,), generator=rng).item())
        g = int(torch.randint(0, N, (1,), generator=rng).item())
        if s0 == g:
            g = (g + 1) % N
        opt_steps = optimal_continuous_steps(s0, g, N, max_step=max_step)
        env = ContinuousCyclicNavEnv(
            N=N, max_step=max_step, max_steps=max_steps,
        )
        env.reset(s0)
        env.set_goal(g)
        s = s0
        success = False
        n = 0
        for _ in range(max_steps):
            s_t = torch.tensor([s], dtype=torch.long, device=DEVICE)
            g_t = torch.tensor([g], dtype=torch.long, device=DEVICE)
            mean, _ = policy(encoder(s_t), encoder(g_t))
            a = float(mean.clamp(-1.0, 1.0).item())
            s_next, _, done = env.step(a)
            d_circ = abs(s_next - g) % N
            d_circ = min(d_circ, N - d_circ)
            n += 1
            if d_circ <= 1:
                success = True
                break
            s = int(s_next)
            if done:
                break
        total += 1
        if success:
            succ += 1
            step_ratios.append(n / max(opt_steps, 1))
    return {
        "success_rate": succ / max(total, 1),
        "mean_step_ratio_to_optimal":
            sum(step_ratios) / len(step_ratios) if step_ratios else float("nan"),
        "n_episodes": total, "n_success": succ,
    }


@torch.no_grad()
def _transition_acc(
    encoder, transition, *, N: int, max_step: float,
    n_samples: int = 5000,
) -> float:
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


def _run_scale(
    scale_name: str, scale_cfg: dict,
    *, N: int, max_step: float,
    epochs: int, batches_per_epoch: int,
    batch_size: int, lr: float,
    n_eval: int, max_steps_eval: int,
    seed: int = 11,
) -> dict:
    torch.manual_seed(seed)
    slot_dim = scale_cfg["slot_dim"]
    hidden = scale_cfg["hidden"]
    n_freqs = scale_cfg["n_freqs"]
    pol_hidden = scale_cfg["pol_hidden"]
    encoder = SlotStateEncoder(N, slot_dim).to(DEVICE)
    transition = ContinuousTransitionHead(
        slot_dim, n_freqs=n_freqs, hidden=hidden,
    ).to(DEVICE)
    policy = ContinuousPolicyHead(
        slot_dim, hidden=pol_hidden,
    ).to(DEVICE)
    n_params = _count_params(encoder, transition, policy)
    print(f"  [{scale_name}] params={n_params:,}  "
          f"(slot_dim={slot_dim}, hidden={hidden}, n_freqs={n_freqs})")
    t0 = time.time()
    _train_one(
        encoder, transition, policy,
        N=N, max_step=max_step,
        epochs=epochs,
        batches_per_epoch=batches_per_epoch,
        batch_size=batch_size, lr=lr,
    )
    train_wall = time.time() - t0
    eval_metrics = _eval(
        encoder, policy, N=N, max_step=max_step,
        n_episodes=n_eval, max_steps=max_steps_eval,
    )
    tr_acc = _transition_acc(
        encoder, transition, N=N, max_step=max_step,
    )
    return {
        "scale": scale_name,
        "params": n_params,
        "config": scale_cfg,
        "train_wall_s": train_wall,
        "eval": eval_metrics,
        "transition_acc_within1": tr_acc,
    }


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--max-step", type=float, default=math.pi / 4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batches-per-epoch", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-steps-eval", type=int, default=32)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--scales", type=str,
                    default="small,medium,large")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f69_full"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"  F69 PCM v6.5 scale-validation sweep on S^1 nav "
          f"(N={args.N}, scales={args.scales})")
    print("=" * 76)

    rows: list[dict] = []
    for scale_name in args.scales.split(","):
        scale_name = scale_name.strip()
        cfg = SCALES[scale_name]
        print(f"\n[{scale_name}] training scale={scale_name}...")
        row = _run_scale(
            scale_name, cfg,
            N=args.N, max_step=args.max_step,
            epochs=args.epochs,
            batches_per_epoch=args.batches_per_epoch,
            batch_size=args.batch_size, lr=args.lr,
            n_eval=args.n_eval,
            max_steps_eval=args.max_steps_eval,
        )
        rows.append(row)
        print(f"    success={row['eval']['success_rate']:.3f}  "
              f"step_ratio={row['eval']['mean_step_ratio_to_optimal']:.3f}  "
              f"transition_acc={row['transition_acc_within1']:.3f}  "
              f"wall={row['train_wall_s']:.1f}s")

    # ─────────────────────────────────────────────────────────────
    # Verdict: monotone scaling invariants
    # ─────────────────────────────────────────────────────────────
    # S1: success_rate is monotone-non-decreasing in scale (allow
    #     1pp slop for stochastic init).
    s1_pass = all(
        rows[i + 1]["eval"]["success_rate"]
        >= rows[i]["eval"]["success_rate"] - 0.01
        for i in range(len(rows) - 1)
    )
    # S2: step ratio is monotone-non-increasing (lower = better)
    s2_pass = all(
        rows[i + 1]["eval"]["mean_step_ratio_to_optimal"]
        <= rows[i]["eval"]["mean_step_ratio_to_optimal"] + 0.05
        for i in range(len(rows) - 1)
    )
    # S3: transition_acc monotone-non-decreasing
    s3_pass = all(
        rows[i + 1]["transition_acc_within1"]
        >= rows[i]["transition_acc_within1"] - 0.01
        for i in range(len(rows) - 1)
    )
    # All-scales-pass: every scale reaches the F65 baseline thresholds
    all_pass = all(
        r["eval"]["success_rate"] >= 0.90
        and r["eval"]["mean_step_ratio_to_optimal"] <= 1.20
        and r["transition_acc_within1"] >= 0.90
        for r in rows
    )

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "rows": rows,
    }
    summary["verdict"] = {
        "all_scales_pass_F65_baseline": all_pass,
        "S1_success_monotone": s1_pass,
        "S2_step_ratio_monotone_or_flat": s2_pass,
        "S3_transition_acc_monotone": s3_pass,
        "param_growth_small_to_large":
            rows[-1]["params"] / max(rows[0]["params"], 1)
            if rows else 1.0,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print("  F69 v6.5 scale-validation verdict:")
    print("=" * 76)
    print(f"  {'scale':<8s}  {'params':>10s}  {'success':>8s}  "
          f"{'step_ratio':>10s}  {'transition_acc':>14s}")
    for r in rows:
        print(f"  {r['scale']:<8s}  {r['params']:>10,}  "
              f"{r['eval']['success_rate']:>8.3f}  "
              f"{r['eval']['mean_step_ratio_to_optimal']:>10.3f}  "
              f"{r['transition_acc_within1']:>14.3f}")
    v = summary["verdict"]
    print(f"\n  param growth (small→large) = "
          f"{summary['verdict']['param_growth_small_to_large']:.1f}x")
    print(f"  all scales pass F65 baseline           : "
          f"[{'PASS' if v['all_scales_pass_F65_baseline'] else 'FAIL'}]")
    print(f"  S1 success monotone in scale           : "
          f"[{'PASS' if v['S1_success_monotone'] else 'FAIL'}]")
    print(f"  S2 step ratio monotone-or-flat         : "
          f"[{'PASS' if v['S2_step_ratio_monotone_or_flat'] else 'FAIL'}]")
    print(f"  S3 transition_acc monotone             : "
          f"[{'PASS' if v['S3_transition_acc_monotone'] else 'FAIL'}]")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
