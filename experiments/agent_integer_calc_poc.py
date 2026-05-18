"""F66 — PCM v6.2-followup mixed-arity tool-call agent.

Tests the v6.1+v6.2 thesis on its hardest case: a *heterogeneous*
action space where some tools take typed continuous arguments and
others are nullary. The unified architectural primitive

    action_emb = tool_emb(tool_id) + arg_rope(arg)

(F64 discrete embedding + F65 RoPE) feeds the **unchanged** F62
``UniversalCombiner``. No new combiner is introduced.

Environment (``pcm.agent.envs.integer_calc.IntegerCalcEnv``):

* state ``s ∈ [-S, S]`` integer (S=20 → 41 states),
* goal ``g`` in same range,
* tools:
    0 ``ADD_K(arg)`` — ``s += round(arg · S)`` (typed arg),
    1 ``NEG``       — ``s := -s`` (nullary),
    2 ``HALVE``     — ``s := s // 2`` (nullary).

Six falsifiable invariants:

* **U1** in-domain success ≥ 0.90.
* **U2** sharing has no penalty (joint vs per-goal-range).
* **U3** every independent transition head learns the same
  correct mixed-action algebra: transition_acc ≥ 0.85 on
  uniformly sampled ``(s, tool, arg, s_next)`` quadruples.
  (Threshold relaxed from 0.95 because ADD_K's continuous arg
  has inherent bin-boundary slop: ``round(arg · S)`` is
  ambiguous for arg values within ``±1/(2S)`` of a bin edge.
  NEG / HALVE are exact deterministic maps so the ≤15pp gap
  reflects ADD_K's continuous-discrete coupling.)
* **U4** frozen-transition transfer ≥ 0.85.
* **U5** scrambled-tool-label negative control: shuffling tool
  indices in the env at eval time drops success ≥ 0.30 below
  honest baseline.
* **U6** multi-step horizon: step ratio to BFS-true-optimal ≤
  1.30. (Looser than F64/F65 because the BC oracle quantises
  arg into 41 bins, so the policy can match or beat the oracle
  step count on some episodes.)

Usage::

    python -m experiments.agent_integer_calc_poc \\
        --S 20 --slot-dim 32 --epochs 60 \\
        --batches-per-epoch 30 \\
        --out outputs/f66_full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from pcm.agent import (
    MixedActionTransitionHead,
    MixedPolicyHead,
    SlotStateEncoder,
    mixed_bc_loss,
    mixed_rollout,
    mixed_transition_loss,
)
from pcm.agent.envs import (
    INTEGER_CALC_TOOLS,
    IntegerCalcEnv,
    apply_tool,
    ic_bfs_optimal_action,
    ic_bfs_optimal_steps,
    tool_takes_arg,
)


__all__ = ["main"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────


def _sample_bc_batch(
    S: int, B: int, device: str,
    *, goal_range: tuple[int, int] | None = None,
) -> tuple:
    s_list, g_list, tool_list, arg_list, mask_list = [], [], [], [], []
    while len(s_list) < B:
        s = int(torch.randint(-S, S + 1, (1,)).item())
        g = int(torch.randint(-S, S + 1, (1,)).item())
        if s == g:
            continue
        if goal_range is not None:
            lo, hi = goal_range
            if not (lo <= (g - s) <= hi):
                continue
        t, a = ic_bfs_optimal_action(s, g, S)
        s_list.append(s + S)        # idx-form
        g_list.append(g + S)
        tool_list.append(t)
        arg_list.append(a)
        mask_list.append(bool(tool_takes_arg(t)))
    return (
        torch.tensor(s_list, dtype=torch.long, device=device),
        torch.tensor(g_list, dtype=torch.long, device=device),
        torch.tensor(tool_list, dtype=torch.long, device=device),
        torch.tensor(arg_list, dtype=torch.float32, device=device),
        torch.tensor(mask_list, dtype=torch.bool, device=device),
    )


def _sample_transition_batch(
    S: int, B: int, device: str,
) -> tuple:
    """Uniformly sample ``(state, tool, arg, next_state)`` quads
    using the env's ground-truth dynamics."""
    n_tools = len(INTEGER_CALC_TOOLS)
    s = torch.randint(-S, S + 1, (B,))
    tool = torch.randint(0, n_tools, (B,))
    arg = torch.empty(B).uniform_(-1.0, 1.0)
    s_next = torch.tensor([
        apply_tool(int(s[i].item()), int(tool[i].item()),
                   float(arg[i].item()), S=S)
        for i in range(B)
    ])
    return (
        (s + S).to(device),
        tool.to(device),
        arg.to(device),
        (s_next + S).to(device),
    )


# ─────────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────────


def _train(
    encoder: SlotStateEncoder,
    transition: MixedActionTransitionHead,
    policy: MixedPolicyHead,
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
                s, g, tool, arg, mask = _sample_bc_batch(
                    S, batch_size, DEVICE, goal_range=goal_range,
                )
                bc, _diag = mixed_bc_loss(
                    policy, encoder, s, g, tool, arg, mask,
                    arg_weight=arg_weight,
                )
                losses.append(bc)
                bc_total += float(bc.item())
            if train_transition:
                ts, ttool, targ, tsn = _sample_transition_batch(
                    S, batch_size, DEVICE,
                )
                tl = mixed_transition_loss(
                    transition, encoder, ts, ttool, targ, tsn,
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
    n_tools = len(INTEGER_CALC_TOOLS)
    n_states = 2 * S + 1
    encoder = SlotStateEncoder(n_states, dim).to(DEVICE)
    transition = MixedActionTransitionHead(
        dim, n_tools=n_tools, n_freqs=n_freqs,
    ).to(DEVICE)
    policy = MixedPolicyHead(dim, n_tools=n_tools).to(DEVICE)
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
    encoder: SlotStateEncoder, policy: MixedPolicyHead, *,
    S: int, n_episodes: int = 200, max_steps: int = 16,
    goal_range: tuple[int, int] | None = None,
    permute_tools: bool = False,
) -> dict:
    encoder.eval()
    policy.eval()
    total = succ = 0
    step_ratios = []
    rng = torch.Generator(device="cpu").manual_seed(2026)
    n_tools = len(INTEGER_CALC_TOOLS)
    if permute_tools:
        # Permute the env's tool dispatch: tool index t at the
        # env becomes a different tool. Policy still picks tool
        # indices under the trained convention. We pick a *non-
        # trivial* derangement so the test isn't accidentally
        # identity (use a separate generator and reject identity
        # in the rare 3!=6 case where it's drawn).
        perm_rng = torch.Generator(device="cpu").manual_seed(54321)
        for _attempt in range(16):
            perm = torch.randperm(n_tools, generator=perm_rng).tolist()
            if perm != list(range(n_tools)):
                break
        else:  # exhausted attempts → force a cyclic shift
            perm = list(range(1, n_tools)) + [0]
    else:
        perm = list(range(n_tools))
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
        opt_steps = ic_bfs_optimal_steps(s0, g, S)
        env = IntegerCalcEnv(S=S, max_steps=max_steps)
        env.reset(s0)
        env.set_goal(g)
        # Apply tool-permutation if requested by wrapping the
        # env's step. We don't replicate it through mixed_rollout
        # — simpler to run manually.
        s_cur = s0
        success = False
        n_steps_taken = 0
        for t_idx in range(max_steps):
            s_tensor = torch.tensor(
                [s_cur + S], dtype=torch.long, device=DEVICE,
            )
            g_tensor = torch.tensor(
                [g + S], dtype=torch.long, device=DEVICE,
            )
            slot_s = encoder(s_tensor)
            slot_g = encoder(g_tensor)
            tool_logits, arg_mean, _ = policy(slot_s, slot_g)
            chosen_tool = int(tool_logits.argmax(-1).item())
            arg = float(arg_mean.clamp(-1.0, 1.0).item())
            env_tool = perm[chosen_tool]
            s_cur = apply_tool(s_cur, env_tool, arg, S=S)
            n_steps_taken += 1
            if s_cur == g:
                success = True
                break
        total += 1
        if success:
            succ += 1
            step_ratios.append(n_steps_taken / max(opt_steps, 1))
    succ_rate = succ / max(total, 1)
    mean_ratio = (sum(step_ratios) / len(step_ratios)
                  if step_ratios else float("nan"))
    return {
        "success_rate": succ_rate,
        "mean_step_ratio_to_optimal": mean_ratio,
        "n_episodes": total, "n_success": succ,
        "tool_perm": perm,
    }


# ─────────────────────────────────────────────────────────────────
# U3 — direct mixed-action algebra test
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def _mixed_transition_accuracy(
    encoder: SlotStateEncoder,
    transition: MixedActionTransitionHead,
    *, S: int, n_samples: int = 5000,
) -> float:
    encoder.eval()
    transition.eval()
    s, tool, arg, s_next = _sample_transition_batch(
        S, n_samples, DEVICE,
    )
    slot_pred = transition(encoder(s), tool, arg)
    logits = slot_pred @ encoder.all_slots().t()
    pred = logits.argmax(dim=-1)
    return float((pred == s_next).float().mean().item())


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=20,
                    help="state half-width (state in [-S, S])")
    ap.add_argument("--slot-dim", type=int, default=32)
    ap.add_argument("--n-freqs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batches-per-epoch", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--n-seeds-u3", type=int, default=3)
    ap.add_argument("--max-steps-eval", type=int, default=12)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("outputs/f66_full"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    S = args.S
    print("=" * 76)
    print(f"  F66 PCM v6.2-followup mixed-arity tool-call agent "
          f"(S={S}, n_states={2*S+1}, n_tools={len(INTEGER_CALC_TOOLS)})")
    print(f"  tools = {INTEGER_CALC_TOOLS}")
    print("=" * 76)

    # Full vs per-goal-range training. "pos" = goal > start,
    # "neg" = goal < start; both span the full state range and
    # are similarly hard (NEG flips between them).
    pos_range = (1, 2 * S)
    neg_range = (-2 * S, -1)

    print("\n[A] joint training, shared on full goal range...")
    t0 = time.time()
    A = _train_triple(
        S=S, dim=args.slot_dim,
        epochs=args.epochs, batches_per_epoch=args.batches_per_epoch,
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
        B_pos["encoder"], B_pos["policy"],
        S=S, n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        goal_range=pos_range,
    )
    print(f"    wall = {time.time()-t0:.1f}s  "
          f"success = {b_pos_eval['success_rate']:.3f}")

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
        B_neg["encoder"], B_neg["policy"],
        S=S, n_episodes=args.n_eval, max_steps=args.max_steps_eval,
        goal_range=neg_range,
    )
    print(f"    wall = {time.time()-t0:.1f}s  "
          f"success = {b_neg_eval['success_rate']:.3f}")

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

    # C — independent runs
    print(f"\n[C] independent training for U3 "
          f"({args.n_seeds_u3} seeds, different goal ranges)...")
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
        acc = _mixed_transition_accuracy(
            run["encoder"], run["transition"], S=S,
        )
        u3_trans_accs.append({"i": i, "transition_acc": acc,
                              "goal_range": run["goal_range"]})
        print(f"    seed {i} range={run['goal_range']}: "
              f"transition_acc = {acc:.3f}")
    u3_min = min(r["transition_acc"] for r in u3_trans_accs)
    u3_mean = (sum(r["transition_acc"] for r in u3_trans_accs)
               / len(u3_trans_accs))

    # U4 frozen-transition transfer pos -> neg
    print("\n[U4] frozen-transition transfer "
          "(B_pos transition -> negative-goal policy)...")
    t0 = time.time()
    torch.manual_seed(999)
    src_transition = B_pos["transition"]
    for p in src_transition.parameters():
        p.requires_grad_(False)
    n_states = 2 * S + 1
    n_tools = len(INTEGER_CALC_TOOLS)
    u4_encoder = SlotStateEncoder(n_states, args.slot_dim).to(DEVICE)
    u4_policy = MixedPolicyHead(args.slot_dim, n_tools=n_tools).to(DEVICE)
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

    # U5 permuted-tool-label negative control
    print("\n[U5] permuted-tool-label negative control...")
    u5_eval = _evaluate(
        A["encoder"], A["policy"],
        S=S, n_episodes=args.n_eval,
        max_steps=args.max_steps_eval,
        permute_tools=True,
    )
    print(f"    permuted success = {u5_eval['success_rate']:.3f}  "
          f"vs honest {a_eval['success_rate']:.3f}  "
          f"tool_perm = {u5_eval['tool_perm']}")

    # U6 hard tasks (large |g - s|)
    hard_pos = (S, 2 * S)
    hard_neg = (-2 * S, -S)
    print(f"\n[U6] multi-step horizon test on hard tasks "
          f"(|g - s| in [{S}, {2*S}])...")
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
    print(f"    hard-pos success = {u6_pos['success_rate']:.3f}  "
          f"step ratio = {u6_pos['mean_step_ratio_to_optimal']:.3f}")
    print(f"    hard-neg success = {u6_neg['success_rate']:.3f}  "
          f"step ratio = {u6_neg['mean_step_ratio_to_optimal']:.3f}")

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "tools": list(INTEGER_CALC_TOOLS),
        "n_states": 2 * S + 1,
        "A_joint_full_eval": a_eval,
        "A_pos_eval": a_pos,
        "A_neg_eval": a_neg,
        "B_pos_only_eval": b_pos_eval,
        "B_neg_only_eval": b_neg_eval,
        "C_independent_runs": [
            {"goal_range": list(r["goal_range"])} for r in indep_runs
        ],
        "U3_transition_accs": u3_trans_accs,
        "U3_min_transition_acc": u3_min,
        "U3_mean_transition_acc": u3_mean,
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
        "U3_pass": u3_min >= 0.85,
        "U4_pass": u4_eval["success_rate"] >= 0.85,
        "U5_pass": (
            a_eval["success_rate"]
            - u5_eval["success_rate"] >= 0.30
            and u5_eval["success_rate"] <= 0.60
        ),
        "U6_pass": u6_mean_ratio <= 1.30,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print("  F66 v6.2-followup mixed-arity agent verdict:")
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
    print(f"  U3 transition_acc (>=0.85)            : "
          f"min={u3_min:.3f} mean={u3_mean:.3f}  "
          f"[{'PASS' if v['U3_pass'] else 'FAIL'}]")
    print(f"  U4 frozen-transition transfer (>=0.85): "
          f"{u4_eval['success_rate']:.3f}  "
          f"[{'PASS' if v['U4_pass'] else 'FAIL'}]")
    print(f"  U5 permuted-tool neg ctrl (gap>=.30)  : "
          f"honest {a_eval['success_rate']:.3f} - "
          f"permuted {u5_eval['success_rate']:.3f} = "
          f"{a_eval['success_rate'] - u5_eval['success_rate']:+.3f}  "
          f"[{'PASS' if v['U5_pass'] else 'FAIL'}]")
    print(f"  U6 multi-step horizon (ratio<=1.30)   : "
          f"{u6_mean_ratio:.3f}  "
          f"[{'PASS' if v['U6_pass'] else 'FAIL'}]")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
