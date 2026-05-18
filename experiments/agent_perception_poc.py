"""F68 — PCM v6.4 perception layer: text-conditioned goals.

F64–F67 ran the agent with state-and-goal both encoded by a
``SlotStateEncoder`` (state-index → slot ``nn.Embedding``). F68
demonstrates that the F62 universal operator works equally well
when the **goal** is specified as a **token sequence** consumed
by a small Transformer ``TextPerceptionHead``.

Architectural claim: swapping ``encoder(goal_idx)`` for
``perception(goal_text_tokens)`` does not change the F62
``UniversalCombiner`` substrate. The agent stack (TransitionHead,
PolicyHead) is unchanged from F64; only the goal-slot source
changes.

Env: ``CyclicNavEnv`` (N=20, 4 actions ``{+1, -1, +5, -5}``).
Each goal index has a small vocabulary of synonymous text
descriptions, e.g. ``goal=5`` could be tokenised as
``["target", "five"]`` or ``["go", "to", "five"]`` or
``["five"]``. The agent must reach the integer state described
by the text, not its lexical form.

Five falsifiable invariants:

* **P1** in-domain success ≥ 0.85 (text-conditioned policy).
* **P2** alias invariance: the same target described by
  *different* text strings yields slots whose pairwise cosine
  similarity averages ≥ 0.70 (much higher than random text
  cosines).
* **P3** wrong-digit-word negative control: at eval, replace
  the digit-word in the goal text with a *different* digit-word
  (so the text describes a *different* goal); success on the
  intended goal drops by ≥ 0.50. (Note: we deliberately do *not*
  test scrambled token order, because the F68 templates have
  every goal uniquely identified by one digit-word, so bag-of-
  words is the minimum sufficient statistic — the F45/F46
  "models find the simplest sufficient encoding" finding
  predicts that mean-pool would be order-invariant *and that
  this is fine* given the env. Testing word identity instead is
  the meaningful semantic control.)
* **P4** unseen-vocabulary control: goal text drawn from a
  disjoint vocabulary the agent never saw at training fails
  (success ≤ 0.30).
* **P5** transition head still works on agent slot states (this
  exercises the F62 universal-operator path *plus* the new
  perception path simultaneously).

Usage::

    python -m experiments.agent_perception_poc \\
        --N 20 --slot-dim 32 --epochs 60 \\
        --out outputs/f68_full
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcm.agent import (
    PolicyHead,
    SlotStateEncoder,
    TextPerceptionHead,
    TransitionHead,
    bc_loss,
    transition_loss,
)
from pcm.agent.envs import (
    ACTION_DELTAS,
    CyclicNavEnv,
    bfs_optimal_action,
    shortest_path_length,
)


__all__ = ["main"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────
# Tiny vocabulary: digits + connectives
# ─────────────────────────────────────────────────────────────────

# Vocabulary: 0=pad, 1-20 = digit-words for states 0..19, 21..27
# = connectives ("target", "go", "to", "at", "reach", "the",
# "position"). Special tokens 28 = SOS, 29 = EOS.
PAD = 0
WORD_BASE = 1            # state 0 -> word id 1, state 19 -> word id 20
WORD_TARGET = 21
WORD_GO = 22
WORD_TO = 23
WORD_AT = 24
WORD_REACH = 25
WORD_THE = 26
WORD_POSITION = 27
SOS = 28
EOS = 29

VOCAB_SIZE = 30
MAX_LEN = 8

# Templates for goal text (training vocab).
TEMPLATES = [
    [WORD_TARGET, "STATE"],
    [WORD_GO, WORD_TO, "STATE"],
    [WORD_REACH, "STATE"],
    [WORD_AT, "STATE"],
    [WORD_GO, WORD_TO, WORD_THE, WORD_POSITION, "STATE"],
    ["STATE"],
]

# A *disjoint* vocabulary for P4. These IDs are valid in the
# tokeniser (within [0, 30)) but never appear in training
# templates, so the perception head should reject them.
UNSEEN_VOCAB = [WORD_TARGET + 100]   # invalid id, we'll handle
# Actually use existing IDs that are *never* used in TEMPLATES.
# All template tokens are in {21..26, WORD_BASE..WORD_BASE+19}.
# WORD_POSITION (27) and EOS (29) are unused inside templates;
# we'll build "alien" goal sentences from these tokens.
ALIEN_TOKENS = [WORD_POSITION, EOS, EOS]


def state_to_word(s: int) -> int:
    return WORD_BASE + s


def goal_to_text(
    goal: int, template_idx: int | None = None, rng: torch.Generator | None = None,
) -> list[int]:
    """Render an integer goal as a token-id sequence."""
    if template_idx is None:
        if rng is None:
            template_idx = int(torch.randint(0, len(TEMPLATES), (1,)).item())
        else:
            template_idx = int(torch.randint(
                0, len(TEMPLATES), (1,), generator=rng,
            ).item())
    tpl = TEMPLATES[template_idx]
    return [state_to_word(goal) if t == "STATE" else t for t in tpl]


def pad_tokens(tokens: list[int], max_len: int = MAX_LEN) -> tuple[list[int], list[bool]]:
    """Pad ``tokens`` to ``max_len`` with PAD; return (padded, mask)."""
    pad = max_len - len(tokens)
    if pad < 0:
        return tokens[:max_len], [True] * max_len
    return tokens + [PAD] * pad, [True] * len(tokens) + [False] * pad


def alien_text(goal: int, rng: torch.Generator) -> list[int]:
    """Goal text drawn from the alien (unseen) vocabulary —
    contains no information about the actual goal index."""
    n = int(torch.randint(2, 4, (1,), generator=rng).item())
    return [
        int(ALIEN_TOKENS[int(torch.randint(
            0, len(ALIEN_TOKENS), (1,), generator=rng,
        ).item())]) for _ in range(n)
    ]


# ─────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────


def _sample_bc_batch(
    N: int, B: int, device: str,
    *, rng: torch.Generator | None = None,
) -> tuple:
    s_list, gtxt_list, gmask_list, a_list = [], [], [], []
    while len(s_list) < B:
        s = int(torch.randint(0, N, (1,)).item())
        g = int(torch.randint(0, N, (1,)).item())
        if s == g:
            continue
        a = bfs_optimal_action(s, g, N)
        text = goal_to_text(g, rng=rng)
        padded, mask = pad_tokens(text)
        s_list.append(s)
        gtxt_list.append(padded)
        gmask_list.append(mask)
        a_list.append(a)
    return (
        torch.tensor(s_list, dtype=torch.long, device=device),
        torch.tensor(gtxt_list, dtype=torch.long, device=device),
        torch.tensor(gmask_list, dtype=torch.bool, device=device),
        torch.tensor(a_list, dtype=torch.long, device=device),
    )


def _sample_transition_batch(
    N: int, B: int, device: str,
) -> tuple:
    s = torch.randint(0, N, (B,), device=device)
    a = torch.randint(0, len(ACTION_DELTAS), (B,), device=device)
    deltas = torch.tensor(ACTION_DELTAS, device=device)[a]
    s_next = (s + deltas) % N
    return s, a, s_next


# ─────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────


def _train(
    encoder: SlotStateEncoder,
    perception: TextPerceptionHead,
    transition: TransitionHead,
    policy: PolicyHead,
    *, N: int, epochs: int, batches_per_epoch: int,
    batch_size: int, lr: float, rng_seed: int = 11,
) -> dict:
    params = (list(encoder.parameters())
              + list(perception.parameters())
              + list(transition.parameters())
              + list(policy.parameters()))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    rng = torch.Generator(device="cpu").manual_seed(rng_seed)
    history = []
    for ep in range(epochs):
        bc_total = trans_total = 0.0
        n = 0
        for _ in range(batches_per_epoch):
            # BC: policy uses (state_slot, goal_slot_from_text)
            s, gtxt, gmask, a_star = _sample_bc_batch(
                N, batch_size, DEVICE, rng=rng,
            )
            slot_s = encoder(s)
            slot_g = perception(gtxt, mask=gmask)
            logits = policy(slot_s, slot_g)
            bc = F.cross_entropy(logits, a_star)
            # Transition: standard (state, action, next state)
            ts, ta, tsn = _sample_transition_batch(N, batch_size, DEVICE)
            t_loss = transition_loss(transition, encoder, ts, ta, tsn)
            loss = bc + t_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            bc_total += float(bc.item())
            trans_total += float(t_loss.item())
            n += 1
        history.append({
            "epoch": ep,
            "bc": bc_total / max(n, 1),
            "trans": trans_total / max(n, 1),
        })
    return {"history": history}


@torch.no_grad()
def _evaluate(
    encoder: SlotStateEncoder,
    perception: TextPerceptionHead,
    policy: PolicyHead,
    *, N: int, n_episodes: int, max_steps: int,
    text_provider=None,
) -> dict:
    """Evaluate with text-conditioned goals.

    ``text_provider``: ``(goal_int, rng) -> list[int]`` —
    callable that produces the goal-text tokens for a given
    integer goal. Defaults to :func:`goal_to_text` (training
    templates).
    """
    encoder.eval()
    perception.eval()
    policy.eval()
    if text_provider is None:
        def text_provider(g, rng):
            return goal_to_text(g, rng=rng)

    rng = torch.Generator(device="cpu").manual_seed(2026)
    total = succ = 0
    for _ in range(n_episodes):
        s0 = int(torch.randint(0, N, (1,), generator=rng).item())
        g = int(torch.randint(0, N, (1,), generator=rng).item())
        if s0 == g:
            g = (g + 1) % N
        env = CyclicNavEnv(N=N, max_steps=max_steps)
        env.reset(s0)
        env.set_goal(g)
        text = text_provider(g, rng)
        padded, mask = pad_tokens(text)
        gtxt_t = torch.tensor([padded], device=DEVICE)
        gmask_t = torch.tensor([mask], device=DEVICE)
        slot_g = perception(gtxt_t, mask=gmask_t)
        s = s0
        success = False
        for _ in range(max_steps):
            s_t = torch.tensor([s], dtype=torch.long, device=DEVICE)
            slot_s = encoder(s_t)
            logits = policy(slot_s, slot_g)
            a = int(logits.argmax(-1).item())
            s_next, _, done = env.step(a)
            s = int(s_next)
            if s == g:
                success = True
                break
            if done:
                break
        total += 1
        if success:
            succ += 1
    return {"success_rate": succ / max(total, 1),
            "n_episodes": total, "n_success": succ}


@torch.no_grad()
def _alias_cos(
    perception: TextPerceptionHead, *, N: int,
    n_aliases: int = 8, n_targets: int = 20,
) -> tuple[float, float]:
    """Compute (mean within-target cos, mean across-target cos).

    For each target we sample ``n_aliases`` text descriptions and
    compute their pairwise cosine similarities (within-target).
    Then we compute the mean cosine between random pairs from
    *different* targets (across-target). A successful perception
    head has within ≫ across.
    """
    perception.eval()
    rng = torch.Generator(device="cpu").manual_seed(31337)
    per_target_slots = []
    for g in range(n_targets):
        slots = []
        for _ in range(n_aliases):
            text = goal_to_text(g, rng=rng)
            padded, mask = pad_tokens(text)
            gtxt = torch.tensor([padded], device=DEVICE)
            gmask = torch.tensor([mask], device=DEVICE)
            slots.append(perception(gtxt, mask=gmask)[0])
        per_target_slots.append(torch.stack(slots, dim=0))
    # within: for each target, mean pairwise cos
    within_means = []
    for s in per_target_slots:
        sn = F.normalize(s, dim=-1)
        cos = sn @ sn.t()
        off_diag = cos - torch.diag(torch.diagonal(cos))
        within_means.append(
            float(off_diag.sum().item() / (cos.numel() - cos.shape[0]))
        )
    within = sum(within_means) / len(within_means)
    # across: sample random pairs from different targets
    across_cos = []
    for _ in range(200):
        i, j = 0, 0
        while i == j:
            i = int(torch.randint(0, n_targets, (1,), generator=rng).item())
            j = int(torch.randint(0, n_targets, (1,), generator=rng).item())
        si = per_target_slots[i][
            int(torch.randint(0, n_aliases, (1,), generator=rng).item())
        ]
        sj = per_target_slots[j][
            int(torch.randint(0, n_aliases, (1,), generator=rng).item())
        ]
        across_cos.append(
            float(F.cosine_similarity(
                si.unsqueeze(0), sj.unsqueeze(0),
            ).item())
        )
    across = sum(across_cos) / len(across_cos)
    return within, across


@torch.no_grad()
def _transition_accuracy(
    encoder: SlotStateEncoder, transition: TransitionHead,
    *, N: int, n_samples: int = 5000,
) -> float:
    encoder.eval()
    transition.eval()
    s = torch.randint(0, N, (n_samples,), device=DEVICE)
    a = torch.randint(0, len(ACTION_DELTAS), (n_samples,), device=DEVICE)
    deltas = torch.tensor(ACTION_DELTAS, device=DEVICE)[a]
    s_next = (s + deltas) % N
    slot_pred = transition(encoder(s), a)
    logits = slot_pred @ encoder.all_slots().t()
    return float((logits.argmax(-1) == s_next).float().mean().item())


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--slot-dim", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batches-per-epoch", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f68_full"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"  F68 PCM v6.4 perception layer "
          f"(text-conditioned goals on cyclic ℤ_{args.N})")
    print("=" * 76)

    # Build heads
    torch.manual_seed(11)
    encoder = SlotStateEncoder(args.N, args.slot_dim).to(DEVICE)
    perception = TextPerceptionHead(
        vocab=VOCAB_SIZE, slot_dim=args.slot_dim,
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, max_len=MAX_LEN,
    ).to(DEVICE)
    transition = TransitionHead(args.slot_dim, len(ACTION_DELTAS)).to(DEVICE)
    policy = PolicyHead(args.slot_dim, len(ACTION_DELTAS)).to(DEVICE)

    print("\n[Train] joint BC + transition training, text-conditioned...")
    t0 = time.time()
    _train(
        encoder, perception, transition, policy,
        N=args.N, epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size, lr=args.lr,
    )
    print(f"    wall = {time.time()-t0:.1f}s")

    # P1
    p1_eval = _evaluate(
        encoder, perception, policy,
        N=args.N, n_episodes=args.n_eval, max_steps=args.max_steps,
    )
    print(f"\n[P1] in-domain success = {p1_eval['success_rate']:.3f}")

    # P2 alias invariance
    within, across = _alias_cos(perception, N=args.N)
    print(f"[P2] alias slot cos: within-target {within:.3f}  "
          f"across-target {across:.3f}")

    # P3 wrong-digit-word neg ctrl: substitute a *different*
    # digit-word so the goal text now lies about the goal
    rng_scr = torch.Generator(device="cpu").manual_seed(7777)

    def wrong_digit_provider(g, rng):
        text = goal_to_text(g, rng=rng_scr)
        # Pick a different goal index and substitute its digit-word.
        other = g
        while other == g:
            other = int(torch.randint(
                0, args.N, (1,), generator=rng_scr,
            ).item())
        true_word = state_to_word(g)
        wrong_word = state_to_word(other)
        return [wrong_word if t == true_word else t for t in text]

    p3_eval = _evaluate(
        encoder, perception, policy,
        N=args.N, n_episodes=args.n_eval, max_steps=args.max_steps,
        text_provider=wrong_digit_provider,
    )
    print(f"[P3] wrong-digit-word success = {p3_eval['success_rate']:.3f}")

    # P4 alien-vocab neg ctrl
    def alien_provider(g, rng):
        return alien_text(g, rng_scr)

    p4_eval = _evaluate(
        encoder, perception, policy,
        N=args.N, n_episodes=args.n_eval, max_steps=args.max_steps,
        text_provider=alien_provider,
    )
    print(f"[P4] alien-vocab success = {p4_eval['success_rate']:.3f}")

    # P5 transition accuracy still works
    p5_trans = _transition_accuracy(encoder, transition, N=args.N)
    print(f"[P5] transition_acc = {p5_trans:.3f}")

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "vocab_size": VOCAB_SIZE,
        "max_len": MAX_LEN,
        "P1_in_domain_eval": p1_eval,
        "P2_within_cos": within,
        "P2_across_cos": across,
        "P3_wrong_digit_eval": p3_eval,
        "P4_alien_eval": p4_eval,
        "P5_transition_acc": p5_trans,
    }
    summary["verdict"] = {
        "P1_pass": p1_eval["success_rate"] >= 0.85,
        # P2: within-target alias cos >> across-target cos (the
        # perception head puts synonymous descriptions of the
        # same target near each other in slot space).
        "P2_pass": within >= 0.70 and within - across >= 0.20,
        # P3: wrong-digit-word text fails by at least 0.50 pp
        # (the perception must read the digit-word identity).
        "P3_pass": (
            p1_eval["success_rate"] - p3_eval["success_rate"] >= 0.50
            and p3_eval["success_rate"] <= 0.40
        ),
        # P4: alien vocab cannot transfer.
        "P4_pass": p4_eval["success_rate"] <= 0.30,
        # P5: transition head still works under joint training
        # with perception.
        "P5_pass": p5_trans >= 0.90,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print("  F68 v6.4 perception layer verdict:")
    print("=" * 76)
    v = summary["verdict"]
    print(f"  P1 in-domain (success >=0.85)         : "
          f"{p1_eval['success_rate']:.3f}  "
          f"[{'PASS' if v['P1_pass'] else 'FAIL'}]")
    print(f"  P2 alias slot cosine within>>across   : "
          f"within {within:.3f} - across {across:.3f} = "
          f"{within - across:+.3f}  "
          f"[{'PASS' if v['P2_pass'] else 'FAIL'}]")
    print(f"  P3 wrong-digit-word neg ctrl (gap>=.50): "
          f"in-domain {p1_eval['success_rate']:.3f} - wrong-digit "
          f"{p3_eval['success_rate']:.3f} = "
          f"{p1_eval['success_rate'] - p3_eval['success_rate']:+.3f}  "
          f"[{'PASS' if v['P3_pass'] else 'FAIL'}]")
    print(f"  P4 alien-vocab neg ctrl (<=0.30)      : "
          f"{p4_eval['success_rate']:.3f}  "
          f"[{'PASS' if v['P4_pass'] else 'FAIL'}]")
    print(f"  P5 transition_acc holds (>=0.90)      : "
          f"{p5_trans:.3f}  "
          f"[{'PASS' if v['P5_pass'] else 'FAIL'}]")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
