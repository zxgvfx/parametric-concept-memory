"""F70 — PCM v6.2-followup² multi-arg tool integration.

F66 demonstrated that mixed-arity action spaces (0-arg + 1-arg-
continuous tools) work under the unchanged F62 ``UniversalCombiner``.
F70 stresses that further by adding a genuinely **2-arg
continuous tool** (``LERP(α, β)``) to a 6-tool environment, and
verifying:

1. **multi-arg representation works**: an architecture-level
   change ``arg_dim = 2`` ripples through the policy head's
   ``(mean, log_std)`` per-arg outputs and the transition head's
   sum-of-RoPEs encoding without modifying the combiner;
2. **tool-aware arg masking** lets nullary / unary / binary
   tools coexist in a single BC loss; and
3. the F62 universal-operator invariants U1–U6 still hold,
   modulo U6 becoming informational (the env is *over-expressive*:
   SET and LERP can each 1-step most goals, so BFS-optimal step
   ratio saturates near 1.0).

Environment (``pcm.agent.envs.multi_tool_calc.MultiToolCalcEnv``):

* state ``s ∈ [-S, S]`` integer (S=20 → 41 states),
* 6 tools:
    0 ``SET(arg)``    — ``s := round(arg · S)``     [unary]
    1 ``ADD_K(arg)``  — ``s += round(arg · S/2)``   [unary, smaller range]
    2 ``NEG``         — ``s := -s``                 [nullary]
    3 ``HALVE``       — ``s := s // 2``             [nullary]
    4 ``DOUBLE``      — ``s := clamp(s · 2)``       [nullary]
    5 ``LERP(α, β)``  — ``s := clamp(round(α·s + β·S))`` [**binary**]
* arg_dim = 2 (per-tool arg-vector length).

Six falsifiable invariants (mirror F66 U1–U6):

* **U1** in-domain success ≥ 0.90.
* **U2** sharing has no penalty.
* **U3** every independent transition head reaches **within-1-
  bin** transition_acc ≥ 0.85 on uniformly-sampled
  ``(s, tool, arg, s_next)``. Continuous-arg tools (SET, ADD_K,
  LERP) have inherent bin-boundary slop — ``round(arg · S)``
  is ambiguous within ``±1/(2S)`` of a bin edge — so the
  within-1 metric mirrors F62c / F65 and is the architecturally
  honest reading.
* **U4** frozen-transition transfer ≥ 0.85.
* **U5** permuted-tool-label neg control: honest − permuted ≥ 0.30.
* **U6** multi-step horizon: step ratio ≤ 1.30 (informational
  here — the env is dominated by 1-step solutions via SET/LERP).

Usage::

    python -m experiments.agent_multi_tool_poc \\
        --S 20 --slot-dim 32 --epochs 80 \\
        --out outputs/f70_full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from pcm.agent import (
    MultiArgActionTransitionHead,
    MultiArgPolicyHead,
    SlotStateEncoder,
    multi_arg_bc_loss,
    multi_arg_transition_loss,
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
# Data
# ─────────────────────────────────────────────────────────────────


def _arg_mask_for_tool(tool_id: int) -> tuple[bool, bool]:
    """Per-arg-slot mask for the BC arg-NLL loss. Tools with
    arity k consume args[0..k-1] and ignore args[k..arg_dim-1]."""
    arity = MTC_TOOL_ARITY[tool_id]
    return tuple(i < arity for i in range(MTC_ARG_DIM))


def _sample_bc_batch(
    S: int, B: int, device: str,
    *, goal_range: tuple[int, int] | None = None,
) -> tuple:
    s_list, g_list = [], []
    tool_list, args_list, mask_list = [], [], []
    while len(s_list) < B:
        s = int(torch.randint(-S, S + 1, (1,)).item())
        g = int(torch.randint(-S, S + 1, (1,)).item())
        if s == g:
            continue
        if goal_range is not None:
            lo, hi = goal_range
            if not (lo <= (g - s) <= hi):
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


def _sample_transition_batch(
    S: int, B: int, device: str,
) -> tuple:
    """Sample uniformly random ``(state, tool, args, s_next)``.

    Critical detail: for tools that don't consume an arg slot,
    we **zero out** that arg slot in the sample. This avoids
    feeding the transition head nuisance noise on unused
    args — the env ignores those slots anyway, so without
    zeroing the head sees uniform-random arg vectors uncorrelated
    with the deterministic ``apply_tool`` output, polluting the
    learned function. With zeroing, the head learns "unary tools
    always have arg[1] = 0", which is consistent at both
    training and inference time.
    """
    s = torch.randint(-S, S + 1, (B,))
    tool = torch.randint(0, N_TOOLS, (B,))
    args = torch.empty(B, MTC_ARG_DIM).uniform_(-1.0, 1.0)
    for i in range(B):
        arity = MTC_TOOL_ARITY[int(tool[i].item())]
        for k in range(arity, MTC_ARG_DIM):
            args[i, k] = 0.0
    s_next = torch.tensor([
        mtc_apply_tool(
            int(s[i].item()),
            int(tool[i].item()),
            tuple(float(args[i, k].item()) for k in range(MTC_ARG_DIM)),
            S=S,
        )
        for i in range(B)
    ])
    return (
        (s + S).to(device), tool.to(device),
        args.to(device), (s_next + S).to(device),
    )


# ─────────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────────


def _train(
    encoder: SlotStateEncoder,
    transition: MultiArgActionTransitionHead,
    policy: MultiArgPolicyHead,
    *, S: int, epochs: int, batches_per_epoch: int,
    batch_size: int, lr: float,
    train_transition: bool = True, train_policy: bool = True,
    goal_range: tuple[int, int] | None = None,
    arg_weight: float = 0.5,
) -> dict:
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    if train_policy:
        for p in list(encoder.parameters()) + list(policy.parameters()):
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)
    if train_transition:
        for p in list(transition.parameters()):
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)
        if not train_policy:
            for p in encoder.parameters():
                if id(p) not in seen:
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
                s, g, tool, args, mask = _sample_bc_batch(
                    S, batch_size, DEVICE, goal_range=goal_range,
                )
                bc, _diag = multi_arg_bc_loss(
                    policy, encoder, s, g, tool, args, mask,
                    arg_weight=arg_weight,
                )
                losses.append(bc)
                bc_total += float(bc.item())
            if train_transition:
                ts, ttool, targs, tsn = _sample_transition_batch(
                    S, batch_size, DEVICE,
                )
                tl = multi_arg_transition_loss(
                    transition, encoder, ts, ttool, targs, tsn,
                )
                losses.append(tl)
                trans_total += float(tl.item())
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


def _train_triple(
    *, S: int, dim: int,
    epochs: int, batches_per_epoch: int, batch_size: int, lr: float,
    seed: int, n_freqs: int = 8,
    goal_range: tuple[int, int] | None = None,
    train_transition: bool = True,
) -> dict:
    torch.manual_seed(seed)
    n_states = 2 * S + 1
    encoder = SlotStateEncoder(n_states, dim).to(DEVICE)
    transition = MultiArgActionTransitionHead(
        dim, n_tools=N_TOOLS, arg_dim=MTC_ARG_DIM, n_freqs=n_freqs,
    ).to(DEVICE)
    policy = MultiArgPolicyHead(
        dim, n_tools=N_TOOLS, arg_dim=MTC_ARG_DIM,
    ).to(DEVICE)
    history = _train(
        encoder, transition, policy,
        S=S, epochs=epochs,
        batches_per_epoch=batches_per_epoch,
        batch_size=batch_size, lr=lr,
        goal_range=goal_range,
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
    encoder: SlotStateEncoder, policy: MultiArgPolicyHead, *,
    S: int, n_episodes: int = 200, max_steps: int = 12,
    goal_range: tuple[int, int] | None = None,
    permute_tools: bool = False,
) -> dict:
    encoder.eval()
    policy.eval()
    total = succ = 0
    step_ratios = []
    tool_usage = [0] * N_TOOLS
    rng = torch.Generator(device="cpu").manual_seed(2026)
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
    for _ in range(n_episodes):
        while True:
            s0 = int(torch.randint(-S, S + 1, (1,), generator=rng).item())
            g = int(torch.randint(-S, S + 1, (1,), generator=rng).item())
            if s0 == g:
                continue
            if goal_range is None or (
                goal_range[0] <= (g - s0) <= goal_range[1]
            ):
                break
        opt_steps = mtc_bfs_optimal_steps(s0, g, S)
        s_cur = s0
        success = False
        n_steps_taken = 0
        for _ in range(max_steps):
            s_t = torch.tensor([s_cur + S], dtype=torch.long, device=DEVICE)
            g_t = torch.tensor([g + S], dtype=torch.long, device=DEVICE)
            slot_s = encoder(s_t)
            slot_g = encoder(g_t)
            tool_logits, arg_mean, _ = policy(slot_s, slot_g)
            chosen_tool = int(tool_logits.argmax(-1).item())
            args = tuple(
                float(arg_mean[0, k].clamp(-1.0, 1.0).item())
                for k in range(MTC_ARG_DIM)
            )
            env_tool = perm[chosen_tool]
            tool_usage[chosen_tool] += 1
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
        "tool_usage": dict(zip(MULTI_TOOL_CALC_TOOLS, tool_usage)),
        "tool_perm": perm,
    }


@torch.no_grad()
def _transition_accuracy(
    encoder: SlotStateEncoder,
    transition: MultiArgActionTransitionHead,
    *, S: int, n_samples: int = 5000,
    per_tool: bool = False,
):
    """Returns exact + within-1-bin accuracy.

    Continuous-arg tools (SET, ADD_K, LERP) have intrinsic
    bin-boundary slop — ``round(arg · S)`` is ambiguous within
    ``±1/(2S)`` of a bin edge. Within-1-bin is the natural
    metric for these; exact-match underestimates the true
    architectural correctness (cf F62c / F65)."""
    encoder.eval()
    transition.eval()
    s, tool, args, s_next = _sample_transition_batch(
        S, n_samples, DEVICE,
    )
    slot_pred = transition(encoder(s), tool, args)
    logits = slot_pred @ encoder.all_slots().t()
    pred = logits.argmax(dim=-1)
    diff = (pred - s_next).abs()
    correct_exact = (diff == 0)
    correct_w1 = (diff <= 1)
    overall_exact = float(correct_exact.float().mean().item())
    overall_w1 = float(correct_w1.float().mean().item())
    if not per_tool:
        return overall_exact, overall_w1
    per_tool_acc = {}
    tool_cpu = tool.cpu() if tool.is_cuda else tool
    correct_exact_cpu = correct_exact.cpu()
    correct_w1_cpu = correct_w1.cpu()
    for t in range(N_TOOLS):
        mask = (tool_cpu == t)
        if mask.any():
            per_tool_acc[MULTI_TOOL_CALC_TOOLS[t]] = {
                "exact": float(correct_exact_cpu[mask].float().mean().item()),
                "within1": float(correct_w1_cpu[mask].float().mean().item()),
            }
    return overall_exact, overall_w1, per_tool_acc


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=20)
    ap.add_argument("--slot-dim", type=int, default=32)
    ap.add_argument("--n-freqs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batches-per-epoch", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--n-seeds-u3", type=int, default=3)
    ap.add_argument("--max-steps-eval", type=int, default=10)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("outputs/f70_full"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    S = args.S
    print("=" * 76)
    print(f"  F70 PCM v6.2-followup² multi-arg tool agent "
          f"(S={S}, n_tools={N_TOOLS}, arg_dim={MTC_ARG_DIM})")
    print(f"  tools = {MULTI_TOOL_CALC_TOOLS}")
    print(f"  arity = {MTC_TOOL_ARITY}")
    print("=" * 76)

    pos_range = (1, 2 * S)
    neg_range = (-2 * S, -1)

    print("\n[A] joint training, shared on full goal range...")
    t0 = time.time()
    A = _train_triple(
        S=S, dim=args.slot_dim, epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr,
        n_freqs=args.n_freqs, seed=11,
    )
    print(f"    wall = {time.time()-t0:.1f}s")
    a_eval = _evaluate(
        A["encoder"], A["policy"],
        S=S, n_episodes=args.n_eval, max_steps=args.max_steps_eval,
    )
    print(f"    success = {a_eval['success_rate']:.3f}  "
          f"step ratio = {a_eval['mean_step_ratio_to_optimal']:.3f}")
    print(f"    tool usage (joint A) = {a_eval['tool_usage']}")

    print("\n[B-pos] training pos-only goals...")
    t0 = time.time()
    B_pos = _train_triple(
        S=S, dim=args.slot_dim, epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr,
        n_freqs=args.n_freqs, seed=22,
        goal_range=pos_range,
    )
    b_pos_eval = _evaluate(
        B_pos["encoder"], B_pos["policy"], S=S,
        n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        goal_range=pos_range,
    )
    print(f"    wall = {time.time()-t0:.1f}s  "
          f"pos-only success = {b_pos_eval['success_rate']:.3f}")

    print("\n[B-neg] training neg-only goals...")
    t0 = time.time()
    B_neg = _train_triple(
        S=S, dim=args.slot_dim, epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr,
        n_freqs=args.n_freqs, seed=33,
        goal_range=neg_range,
    )
    b_neg_eval = _evaluate(
        B_neg["encoder"], B_neg["policy"], S=S,
        n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        goal_range=neg_range,
    )
    print(f"    wall = {time.time()-t0:.1f}s  "
          f"neg-only success = {b_neg_eval['success_rate']:.3f}")

    a_pos = _evaluate(
        A["encoder"], A["policy"], S=S,
        n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        goal_range=pos_range,
    )
    a_neg = _evaluate(
        A["encoder"], A["policy"], S=S,
        n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        goal_range=neg_range,
    )
    print(f"    [A] pos success = {a_pos['success_rate']:.3f}  "
          f"neg success = {a_neg['success_rate']:.3f}")

    print(f"\n[C] independent training for U3 ({args.n_seeds_u3} seeds)...")
    indep_runs = []
    ranges = [pos_range, neg_range, pos_range]
    for si in range(args.n_seeds_u3):
        gr = ranges[si % len(ranges)]
        r = _train_triple(
            S=S, dim=args.slot_dim, epochs=args.epochs,
            batches_per_epoch=args.batches_per_epoch,
            batch_size=args.batch_size, lr=args.lr,
            n_freqs=args.n_freqs,
            seed=10_000 * (si + 1) + 7,
            goal_range=gr,
        )
        indep_runs.append({"goal_range": gr, **r})
        print(f"    seed {si} range={gr}: trained")

    u3_trans_accs = []
    for i, run in enumerate(indep_runs):
        acc_e, acc_w1, per_tool = _transition_accuracy(
            run["encoder"], run["transition"], S=S, per_tool=True,
        )
        u3_trans_accs.append({
            "i": i,
            "transition_acc_exact": acc_e,
            "transition_acc_within1": acc_w1,
            "per_tool": per_tool,
            "goal_range": run["goal_range"],
        })
        per_tool_str = " ".join(
            f"{k}=({v['exact']:.2f}/{v['within1']:.2f})"
            for k, v in per_tool.items()
        )
        print(f"    seed {i} range={run['goal_range']}: "
              f"exact={acc_e:.3f} within1={acc_w1:.3f}  "
              f"[{per_tool_str}]")
    u3_min_exact = min(r["transition_acc_exact"] for r in u3_trans_accs)
    u3_mean_exact = sum(r["transition_acc_exact"]
                        for r in u3_trans_accs) / len(u3_trans_accs)
    u3_min_w1 = min(r["transition_acc_within1"] for r in u3_trans_accs)
    u3_mean_w1 = sum(r["transition_acc_within1"]
                     for r in u3_trans_accs) / len(u3_trans_accs)

    # U4
    print("\n[U4] frozen-transition transfer "
          "(B_pos transition -> negative-goal policy)...")
    t0 = time.time()
    torch.manual_seed(999)
    src_transition = B_pos["transition"]
    for p in src_transition.parameters():
        p.requires_grad_(False)
    n_states = 2 * S + 1
    u4_encoder = SlotStateEncoder(n_states, args.slot_dim).to(DEVICE)
    u4_policy = MultiArgPolicyHead(
        args.slot_dim, n_tools=N_TOOLS, arg_dim=MTC_ARG_DIM,
    ).to(DEVICE)
    _train(
        u4_encoder, src_transition, u4_policy,
        S=S, epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr,
        train_transition=False, train_policy=True,
        goal_range=neg_range,
    )
    u4_eval = _evaluate(
        u4_encoder, u4_policy, S=S,
        n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        goal_range=neg_range,
    )
    print(f"    wall = {time.time()-t0:.1f}s  "
          f"transfer success = {u4_eval['success_rate']:.3f}")

    # U5
    print("\n[U5] permuted-tool-label negative control...")
    u5_eval = _evaluate(
        A["encoder"], A["policy"],
        S=S, n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        permute_tools=True,
    )
    print(f"    permuted success = {u5_eval['success_rate']:.3f}  "
          f"perm = {u5_eval['tool_perm']}")

    # U6 informational
    hard_pos = (S, 2 * S)
    hard_neg = (-2 * S, -S)
    u6_pos = _evaluate(
        A["encoder"], A["policy"], S=S,
        n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        goal_range=hard_pos,
    )
    u6_neg = _evaluate(
        A["encoder"], A["policy"], S=S,
        n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        goal_range=hard_neg,
    )
    u6_mean_ratio = (
        u6_pos["mean_step_ratio_to_optimal"]
        + u6_neg["mean_step_ratio_to_optimal"]
    ) / 2.0
    print(f"\n[U6] hard-pos success = {u6_pos['success_rate']:.3f}  "
          f"step ratio = {u6_pos['mean_step_ratio_to_optimal']:.3f}")
    print(f"     hard-neg success = {u6_neg['success_rate']:.3f}  "
          f"step ratio = {u6_neg['mean_step_ratio_to_optimal']:.3f}")

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "tools": list(MULTI_TOOL_CALC_TOOLS),
        "arity": list(MTC_TOOL_ARITY),
        "arg_dim": MTC_ARG_DIM,
        "n_states": n_states,
        "A_joint_full_eval": a_eval,
        "A_pos_eval": a_pos,
        "A_neg_eval": a_neg,
        "B_pos_only_eval": b_pos_eval,
        "B_neg_only_eval": b_neg_eval,
        "C_independent_runs": [
            {"goal_range": list(r["goal_range"])} for r in indep_runs
        ],
        "U3_transition_accs": u3_trans_accs,
        "U3_min_transition_acc_exact": u3_min_exact,
        "U3_mean_transition_acc_exact": u3_mean_exact,
        "U3_min_transition_acc_within1": u3_min_w1,
        "U3_mean_transition_acc_within1": u3_mean_w1,
        "U4_frozen_transition_transfer_eval": u4_eval,
        "U5_permuted_tool_eval": u5_eval,
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
        # U3 within-1-bin: continuous-arg tools have intrinsic
        # bin-boundary slop (cf F62c / F65); within-1 is the
        # honest metric. We require min within-1 ≥ 0.85.
        "U3_pass": u3_min_w1 >= 0.85,
        "U4_pass": u4_eval["success_rate"] >= 0.85,
        "U5_pass": (
            a_eval["success_rate"]
            - u5_eval["success_rate"] >= 0.30
            and u5_eval["success_rate"] <= 0.60
        ),
        # U6 informational only — SET/LERP can 1-step every goal
        # so step ratio is dominated by 1.0. We still grade it
        # but at a permissive threshold.
        "U6_pass": u6_mean_ratio <= 1.30,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print("  F70 v6.2-followup² multi-arg verdict:")
    print("=" * 76)
    v = summary["verdict"]
    print(f"  U1 in-domain (success >=0.90)         : "
          f"{a_eval['success_rate']:.3f}  "
          f"[{'PASS' if v['U1_pass'] else 'FAIL'}]")
    print(f"  U2 sharing has no penalty             : "
          f"pos shared {a_pos['success_rate']:.3f} vs sep "
          f"{b_pos_eval['success_rate']:.3f} | "
          f"neg shared {a_neg['success_rate']:.3f} vs sep "
          f"{b_neg_eval['success_rate']:.3f}  "
          f"[{'PASS' if v['U2_pass'] else 'FAIL'}]")
    print(f"  U3 transition within-1 (>=0.85)       : "
          f"min_w1={u3_min_w1:.3f} mean_w1={u3_mean_w1:.3f}  "
          f"(exact min/mean = {u3_min_exact:.3f}/{u3_mean_exact:.3f})  "
          f"[{'PASS' if v['U3_pass'] else 'FAIL'}]")
    print(f"  U4 frozen-transition transfer (>=0.85): "
          f"{u4_eval['success_rate']:.3f}  "
          f"[{'PASS' if v['U4_pass'] else 'FAIL'}]")
    print(f"  U5 permuted-tool neg ctrl (gap>=.30)  : "
          f"{a_eval['success_rate'] - u5_eval['success_rate']:+.3f}  "
          f"[{'PASS' if v['U5_pass'] else 'FAIL'}]")
    print(f"  U6 step ratio (info, <=1.30)          : "
          f"{u6_mean_ratio:.3f}  "
          f"[{'PASS' if v['U6_pass'] else 'FAIL'}]")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
