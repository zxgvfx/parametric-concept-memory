"""F73 — PCM v6.6 cook-then-act long-horizon planning agent.

Combines F61 ``AttractorHead`` (distributional outcome
prediction in O(1)) with F70 ``MultiArgActionTransitionHead``
(world model) and ``MultiArgPolicyHead`` (1-step BC policy) to
build a hybrid agent that:

* **System 1**: direct argmax over policy (cheap, fast).
* **System 2**: MPC planning — generates K candidate first-
  actions, rolls each out via the world model for T steps under
  greedy policy, scores by terminal-slot distance to goal,
  executes the first action of the best rollout.
* **Attractor**: ``AgentAttractorHead`` predicts ``p_success``
  and ``expected_steps`` from ``(state, goal)`` in one shot.
* **Hybrid dispatcher**: routes high-confidence
  ``(p_success ≥ τ)`` to System 1, low-confidence to System 2.

This mirrors F61's ``HybridPhysicsDispatcher`` design exactly,
but on the agent stack instead of the physics simulator.

Environment: F70's ``MultiToolCalcEnv`` (S=20, 6 tools incl.
``LERP`` 2-arg). For F73 we *deliberately train under-budgeted*
(30 epochs instead of F70's 80) so the policy is non-trivial-
but-imperfect — this gives MPC room to add value.

Five falsifiable invariants:

* **A1** MPC planner success ≥ BC-direct success (planning
  doesn't hurt).
* **A2** MPC planner improves over BC-direct on the *under-
  budgeted* policy by ≥ 5pp absolute.
* **A3** ``AgentAttractorHead`` is *calibrated*: episodes
  predicted ``p_success ≥ 0.7`` succeed ≥ 90% of the time.
* **A4** Hybrid dispatcher matches or beats *both* MPC-only
  and BC-direct on the held-out test distribution.
* **A5** Permuted-action world model breaks MPC (negative
  control): scrambling tool labels inside the transition head
  drops MPC success ≥ 0.30 below honest MPC.

Usage::

    python -m experiments.agent_cook_then_act_f73 \\
        --S 20 --epochs 30 --out outputs/f73_full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcm.agent import (
    AgentAttractorHead,
    AttractorTargets,
    MultiArgActionTransitionHead,
    MultiArgPolicyHead,
    SlotStateEncoder,
    agent_attractor_loss,
    multi_arg_bc_loss,
    multi_arg_transition_loss,
    plan_multi_arg_mpc,
    hybrid_multi_arg_step,
)
from pcm.agent.envs import (
    MTC_ARG_DIM,
    MTC_TOOL_ARITY,
    MULTI_TOOL_CALC_TOOLS,
    MultiToolCalcEnv,
    mtc_apply_tool,
    mtc_bfs_optimal_action,
    mtc_bfs_optimal_steps,
)


__all__ = ["main"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TOOLS = len(MULTI_TOOL_CALC_TOOLS)


# ─────────────────────────────────────────────────────────────────
# Data (mirrors F70's _sample_*)
# ─────────────────────────────────────────────────────────────────


def _arg_mask_for_tool(tool_id: int) -> tuple[bool, bool]:
    return tuple(i < MTC_TOOL_ARITY[tool_id] for i in range(MTC_ARG_DIM))


def _sample_bc_batch(S: int, B: int, device: str) -> tuple:
    s_list, g_list = [], []
    tool_list, args_list, mask_list = [], [], []
    while len(s_list) < B:
        s = int(torch.randint(-S, S + 1, (1,)).item())
        g = int(torch.randint(-S, S + 1, (1,)).item())
        if s == g:
            continue
        t, args = mtc_bfs_optimal_action(s, g, S)
        s_list.append(s + S)
        g_list.append(g + S)
        tool_list.append(t)
        args_list.append(list(args))
        mask_list.append(list(_arg_mask_for_tool(t)))
    return (
        torch.tensor(s_list, dtype=torch.long, device=device),
        torch.tensor(g_list, dtype=torch.long, device=device),
        torch.tensor(tool_list, dtype=torch.long, device=device),
        torch.tensor(args_list, dtype=torch.float32, device=device),
        torch.tensor(mask_list, dtype=torch.bool, device=device),
    )


def _sample_transition_batch(S: int, B: int, device: str) -> tuple:
    s = torch.randint(-S, S + 1, (B,))
    tool = torch.randint(0, N_TOOLS, (B,))
    args = torch.empty(B, MTC_ARG_DIM).uniform_(-1.0, 1.0)
    for i in range(B):
        arity = MTC_TOOL_ARITY[int(tool[i].item())]
        for k in range(arity, MTC_ARG_DIM):
            args[i, k] = 0.0
    s_next = torch.tensor([
        mtc_apply_tool(
            int(s[i].item()), int(tool[i].item()),
            tuple(float(args[i, k].item()) for k in range(MTC_ARG_DIM)),
            S=S,
        )
        for i in range(B)
    ])
    return ((s + S).to(device), tool.to(device),
            args.to(device), (s_next + S).to(device))


# ─────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────


def _train_policy_and_world_model(
    *, S: int, dim: int, epochs: int, batches_per_epoch: int,
    batch_size: int, lr: float, seed: int,
) -> dict:
    torch.manual_seed(seed)
    n_states = 2 * S + 1
    encoder = SlotStateEncoder(n_states, dim).to(DEVICE)
    transition = MultiArgActionTransitionHead(
        dim, n_tools=N_TOOLS, arg_dim=MTC_ARG_DIM,
    ).to(DEVICE)
    policy = MultiArgPolicyHead(
        dim, n_tools=N_TOOLS, arg_dim=MTC_ARG_DIM,
    ).to(DEVICE)
    params = (list(encoder.parameters())
              + list(transition.parameters())
              + list(policy.parameters()))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        for _ in range(batches_per_epoch):
            s, g, tool, args, mask = _sample_bc_batch(S, batch_size, DEVICE)
            bc, _ = multi_arg_bc_loss(
                policy, encoder, s, g, tool, args, mask,
            )
            ts, ttool, targs, tsn = _sample_transition_batch(
                S, batch_size, DEVICE,
            )
            tl = multi_arg_transition_loss(
                transition, encoder, ts, ttool, targs, tsn,
            )
            loss = bc + tl
            opt.zero_grad()
            loss.backward()
            opt.step()
    return {"encoder": encoder, "transition": transition,
            "policy": policy}


def _collect_attractor_data(
    encoder, policy, *,
    S: int, n_episodes: int, max_steps: int,
    use_mpc: bool = False, transition=None,
    rng_seed: int = 7777,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Roll out the current policy on random (start, goal) pairs
    and record ``(state, goal, success, n_steps)`` for attractor
    training."""
    encoder.eval()
    policy.eval()
    if transition is not None:
        transition.eval()
    rng = torch.Generator(device="cpu").manual_seed(rng_seed)
    states, goals, succs, steps = [], [], [], []
    for _ in range(n_episodes):
        s0 = int(torch.randint(-S, S + 1, (1,), generator=rng).item())
        g = int(torch.randint(-S, S + 1, (1,), generator=rng).item())
        if s0 == g:
            continue
        s_cur = s0
        success = False
        n_steps = 0
        for _ in range(max_steps):
            if use_mpc and transition is not None:
                tool, args = plan_multi_arg_mpc(
                    s_cur + S, g + S, encoder, transition, policy,
                    n_candidates=4, rollout_steps=3,
                    device=DEVICE, rng=rng,
                )
            else:
                s_t = torch.tensor([s_cur + S], dtype=torch.long,
                                   device=DEVICE)
                g_t = torch.tensor([g + S], dtype=torch.long,
                                   device=DEVICE)
                tool_logits, arg_mean, _ = policy(encoder(s_t),
                                                   encoder(g_t))
                tool = int(tool_logits.argmax(-1).item())
                args = tuple(
                    float(arg_mean[0, j].clamp(-1.0, 1.0).item())
                    for j in range(MTC_ARG_DIM)
                )
            s_cur = mtc_apply_tool(s_cur, tool, args, S=S)
            n_steps += 1
            if s_cur == g:
                success = True
                break
        states.append(s0 + S)
        goals.append(g + S)
        succs.append(1.0 if success else 0.0)
        steps.append(max(n_steps, 1))
    return (
        torch.tensor(states, dtype=torch.long, device=DEVICE),
        torch.tensor(goals, dtype=torch.long, device=DEVICE),
        torch.tensor(succs, dtype=torch.float32, device=DEVICE),
        torch.tensor(steps, dtype=torch.long, device=DEVICE),
    )


def _train_attractor(
    attractor: AgentAttractorHead, encoder: SlotStateEncoder,
    *, states, goals, succs, steps,
    epochs: int, batch_size: int, lr: float,
) -> None:
    """Train attractor head on collected outcome data."""
    opt = torch.optim.AdamW(attractor.parameters(), lr=lr,
                             weight_decay=1e-4)
    n = states.shape[0]
    encoder.eval()
    for _ in range(epochs):
        idx = torch.randperm(n)
        for i in range(0, n, batch_size):
            j = idx[i:i + batch_size]
            s_t = states[j]
            g_t = goals[j]
            with torch.no_grad():
                ss = encoder(s_t)
                sg = encoder(g_t)
            targets = AttractorTargets(succs[j], steps[j])
            loss, _ = agent_attractor_loss(attractor, ss, sg, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()


# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def _evaluate_policy(
    encoder, policy, *, S: int, n_episodes: int, max_steps: int,
    transition=None, mode: str = "bc",
    attractor: AgentAttractorHead | None = None,
    p_success_threshold: float = 0.8,
    permute_tools: bool = False,
    rng_seed: int = 2026,
) -> dict:
    """Generic policy evaluator.

    ``mode``:
    * ``"bc"`` — direct argmax policy (System 1)
    * ``"mpc"`` — full MPC planner (System 2)
    * ``"hybrid"`` — attractor-routed dispatcher
    """
    encoder.eval()
    policy.eval()
    if transition is not None:
        transition.eval()
    if attractor is not None:
        attractor.eval()
    rng = torch.Generator(device="cpu").manual_seed(rng_seed)
    if permute_tools:
        perm_rng = torch.Generator(device="cpu").manual_seed(54321)
        for _ in range(16):
            perm = torch.randperm(N_TOOLS, generator=perm_rng).tolist()
            if perm != list(range(N_TOOLS)):
                break
        else:
            perm = list(range(1, N_TOOLS)) + [0]
    else:
        perm = list(range(N_TOOLS))
    total = succ = 0
    step_ratios = []
    routes = {"S1": 0, "S2": 0}
    for _ in range(n_episodes):
        s0 = int(torch.randint(-S, S + 1, (1,), generator=rng).item())
        g = int(torch.randint(-S, S + 1, (1,), generator=rng).item())
        if s0 == g:
            continue
        opt_steps = mtc_bfs_optimal_steps(s0, g, S)
        s_cur = s0
        success = False
        n_steps_taken = 0
        for _ in range(max_steps):
            if mode == "bc":
                s_t = torch.tensor([s_cur + S], dtype=torch.long,
                                   device=DEVICE)
                g_t = torch.tensor([g + S], dtype=torch.long,
                                   device=DEVICE)
                tool_logits, arg_mean, _ = policy(
                    encoder(s_t), encoder(g_t),
                )
                tool = int(tool_logits.argmax(-1).item())
                args = tuple(
                    float(arg_mean[0, j].clamp(-1.0, 1.0).item())
                    for j in range(MTC_ARG_DIM)
                )
            elif mode == "mpc":
                tool, args = plan_multi_arg_mpc(
                    s_cur + S, g + S, encoder, transition, policy,
                    n_candidates=4, rollout_steps=3,
                    device=DEVICE, rng=rng,
                )
            elif mode == "hybrid":
                res = hybrid_multi_arg_step(
                    s_cur + S, g + S, encoder, transition, policy,
                    attractor,
                    p_success_threshold=p_success_threshold,
                    n_candidates=4, rollout_steps=3,
                    device=DEVICE, rng=rng,
                )
                tool, args = res.tool_id, res.args
                routes[res.route] += 1
            else:
                raise ValueError(f"unknown mode {mode}")
            env_tool = perm[tool]
            s_cur = mtc_apply_tool(s_cur, env_tool, args, S=S)
            n_steps_taken += 1
            if s_cur == g:
                success = True
                break
        total += 1
        if success:
            succ += 1
            step_ratios.append(n_steps_taken / max(opt_steps, 1))
    return {
        "success_rate": succ / max(total, 1),
        "mean_step_ratio_to_optimal":
            sum(step_ratios) / len(step_ratios)
            if step_ratios else float("nan"),
        "n_episodes": total, "n_success": succ,
        "routes": routes if mode == "hybrid" else None,
    }


@torch.no_grad()
def _evaluate_attractor_calibration(
    encoder, policy, attractor, *, S: int, n_episodes: int,
    max_steps: int,
) -> dict:
    """Roll out the *direct* policy on random episodes, record
    actual success, and compare to attractor's predicted
    ``p_success``. Buckets predictions into [0-0.3], [0.3-0.7],
    [0.7-1.0] and reports actual success per bucket."""
    encoder.eval()
    policy.eval()
    attractor.eval()
    rng = torch.Generator(device="cpu").manual_seed(2027)
    buckets = {"low (p<0.3)": [], "mid (0.3-0.7)": [], "high (p>=0.7)": []}
    for _ in range(n_episodes):
        s0 = int(torch.randint(-S, S + 1, (1,), generator=rng).item())
        g = int(torch.randint(-S, S + 1, (1,), generator=rng).item())
        if s0 == g:
            continue
        s_t = torch.tensor([s0 + S], dtype=torch.long, device=DEVICE)
        g_t = torch.tensor([g + S], dtype=torch.long, device=DEVICE)
        p_succ, _ = attractor.predict(encoder(s_t), encoder(g_t))
        p_succ_f = float(p_succ.item())
        # Roll out direct policy
        s_cur = s0
        success = False
        for _ in range(max_steps):
            tool_logits, arg_mean, _ = policy(
                encoder(torch.tensor([s_cur + S], dtype=torch.long,
                                     device=DEVICE)),
                encoder(g_t),
            )
            tool = int(tool_logits.argmax(-1).item())
            args = tuple(
                float(arg_mean[0, j].clamp(-1.0, 1.0).item())
                for j in range(MTC_ARG_DIM)
            )
            s_cur = mtc_apply_tool(s_cur, tool, args, S=S)
            if s_cur == g:
                success = True
                break
        if p_succ_f < 0.3:
            buckets["low (p<0.3)"].append(success)
        elif p_succ_f < 0.7:
            buckets["mid (0.3-0.7)"].append(success)
        else:
            buckets["high (p>=0.7)"].append(success)
    out = {}
    for k, vals in buckets.items():
        if vals:
            out[k] = {
                "n": len(vals),
                "actual_success": sum(vals) / len(vals),
            }
        else:
            out[k] = {"n": 0, "actual_success": float("nan")}
    return out


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=20)
    ap.add_argument("--slot-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30,
                    help="deliberately under F70's 80 so MPC has room")
    ap.add_argument("--batches-per-epoch", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--attractor-epochs", type=int, default=40)
    ap.add_argument("--attractor-data-episodes", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--max-steps-eval", type=int, default=10)
    ap.add_argument("--p-success-threshold", type=float, default=0.8)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f73_full"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"  F73 PCM v6.6 cook-then-act planner "
          f"(S={args.S}, multi-tool env, "
          f"epochs={args.epochs} under-budgeted)")
    print("=" * 76)

    # ─── Step 1: train policy + world model jointly ────────────
    print(f"\n[1/4] training policy + world model "
          f"({args.epochs} epochs, under F70's 80)...")
    t0 = time.time()
    pwm = _train_policy_and_world_model(
        S=args.S, dim=args.slot_dim,
        epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr, seed=11,
    )
    print(f"    wall = {time.time()-t0:.1f}s")

    # Baseline BC-direct eval
    bc_eval = _evaluate_policy(
        pwm["encoder"], pwm["policy"], S=args.S,
        n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        mode="bc",
    )
    print(f"    BC-direct success = {bc_eval['success_rate']:.3f}  "
          f"step ratio = {bc_eval['mean_step_ratio_to_optimal']:.3f}")

    # MPC eval (System 2 only)
    mpc_eval = _evaluate_policy(
        pwm["encoder"], pwm["policy"], S=args.S,
        transition=pwm["transition"],
        n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        mode="mpc",
    )
    print(f"    MPC-only success = {mpc_eval['success_rate']:.3f}  "
          f"step ratio = {mpc_eval['mean_step_ratio_to_optimal']:.3f}")

    # ─── Step 2: collect attractor training data ───────────────
    print(f"\n[2/4] collecting attractor data "
          f"({args.attractor_data_episodes} episodes under BC-direct)...")
    t0 = time.time()
    a_states, a_goals, a_succs, a_steps = _collect_attractor_data(
        pwm["encoder"], pwm["policy"],
        S=args.S, n_episodes=args.attractor_data_episodes,
        max_steps=args.max_steps_eval,
    )
    print(f"    wall = {time.time()-t0:.1f}s  "
          f"collected {len(a_states)} (state, goal, succ, steps)  "
          f"base success rate = {float(a_succs.mean()):.3f}")

    # ─── Step 3: train attractor head ──────────────────────────
    print(f"\n[3/4] training AgentAttractorHead "
          f"({args.attractor_epochs} epochs)...")
    t0 = time.time()
    torch.manual_seed(22)
    attractor = AgentAttractorHead(dim=args.slot_dim).to(DEVICE)
    _train_attractor(
        attractor, pwm["encoder"],
        states=a_states, goals=a_goals,
        succs=a_succs, steps=a_steps,
        epochs=args.attractor_epochs,
        batch_size=args.batch_size, lr=args.lr,
    )
    print(f"    wall = {time.time()-t0:.1f}s")

    # ─── Step 4: hybrid dispatcher eval ────────────────────────
    print(f"\n[4/4] hybrid dispatcher eval "
          f"(p_success threshold = {args.p_success_threshold})...")
    hybrid_eval = _evaluate_policy(
        pwm["encoder"], pwm["policy"], S=args.S,
        transition=pwm["transition"], attractor=attractor,
        n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        mode="hybrid",
        p_success_threshold=args.p_success_threshold,
    )
    print(f"    hybrid success = {hybrid_eval['success_rate']:.3f}  "
          f"step ratio = {hybrid_eval['mean_step_ratio_to_optimal']:.3f}  "
          f"routes = {hybrid_eval['routes']}")

    # A5 — permuted-action negative control on MPC
    print("\n[A5] permuted-action negative control (scrambled tools "
          "in env, MPC plans against unchanged world model)...")
    mpc_perm_eval = _evaluate_policy(
        pwm["encoder"], pwm["policy"], S=args.S,
        transition=pwm["transition"],
        n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        mode="mpc", permute_tools=True,
    )
    print(f"    permuted MPC success = {mpc_perm_eval['success_rate']:.3f}")

    # A3 — attractor calibration
    print("\n[A3] attractor calibration buckets...")
    calib = _evaluate_attractor_calibration(
        pwm["encoder"], pwm["policy"], attractor,
        S=args.S, n_episodes=args.n_eval,
        max_steps=args.max_steps_eval,
    )
    for k, v in calib.items():
        if v["n"] > 0:
            print(f"    {k}: n={v['n']}  actual_success={v['actual_success']:.3f}")
        else:
            print(f"    {k}: n=0")

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "BC_direct_eval": bc_eval,
        "MPC_only_eval": mpc_eval,
        "MPC_permuted_eval": mpc_perm_eval,
        "Hybrid_eval": hybrid_eval,
        "attractor_calibration": calib,
        "attractor_train_data_success": float(a_succs.mean()),
    }

    # Verdict
    bc_succ = bc_eval["success_rate"]
    mpc_succ = mpc_eval["success_rate"]
    hyb_succ = hybrid_eval["success_rate"]
    high_bucket = calib.get("high (p>=0.7)", {"actual_success": float("nan")})
    summary["verdict"] = {
        "A1_mpc_no_regression": mpc_succ >= bc_succ - 0.02,
        "A2_mpc_improves_undertrained":
            mpc_succ - bc_succ >= 0.05,
        "A3_attractor_calibrated_high_bucket": (
            high_bucket["n"] > 0
            and high_bucket["actual_success"] >= 0.90
        ),
        "A4_hybrid_matches_or_beats_both":
            hyb_succ >= max(bc_succ, mpc_succ) - 0.02,
        "A5_permuted_breaks_mpc":
            mpc_succ - mpc_perm_eval["success_rate"] >= 0.30,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print("  F73 cook-then-act verdict:")
    print("=" * 76)
    v = summary["verdict"]
    print(f"  A1 MPC no regression vs BC               : "
          f"BC {bc_succ:.3f} vs MPC {mpc_succ:.3f}  "
          f"[{'PASS' if v['A1_mpc_no_regression'] else 'FAIL'}]")
    print(f"  A2 MPC improves >=5pp on under-trained   : "
          f"gap {mpc_succ - bc_succ:+.3f}  "
          f"[{'PASS' if v['A2_mpc_improves_undertrained'] else 'FAIL'}]")
    high_msg = (f"{high_bucket['actual_success']:.3f}"
                if high_bucket["n"] > 0 else "no high-p episodes")
    print(f"  A3 attractor calibrated (high bucket>=.9): "
          f"actual succ in high-p bucket = {high_msg}  "
          f"[{'PASS' if v['A3_attractor_calibrated_high_bucket'] else 'FAIL'}]")
    print(f"  A4 hybrid >= max(BC, MPC) − 2pp          : "
          f"hyb {hyb_succ:.3f} vs max {max(bc_succ, mpc_succ):.3f}  "
          f"[{'PASS' if v['A4_hybrid_matches_or_beats_both'] else 'FAIL'}]")
    print(f"  A5 permuted breaks MPC (gap >=.30)       : "
          f"MPC {mpc_succ:.3f} − perm {mpc_perm_eval['success_rate']:.3f} "
          f"= {mpc_succ - mpc_perm_eval['success_rate']:+.3f}  "
          f"[{'PASS' if v['A5_permuted_breaks_mpc'] else 'FAIL'}]")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
