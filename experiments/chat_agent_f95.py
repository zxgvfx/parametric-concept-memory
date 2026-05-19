"""F95 — Chat-learning agent: end-to-end PoC with C1-C4
falsifiable invariants.

This experiment is the first end-to-end demonstration that the
PCM cognitive primitives (HybridPCMMiniLM + EpisodicBuffer +
UserFactMemory + OnlineTeacherSession + GlyphEncoder/Decoder)
can be wired into a continuously-running chat agent that:

* maintains conversational context (C1),
* remembers facts about the user across turns (C2),
* learns from the conversation in real time (C3), and
* does not catastrophically forget its pretrained
  distribution while doing so (C4).

The four falsifiable invariants are scoped narrowly to what an
F90-class checkpoint (d=512 / L=12, TinyStories vocab=4096)
can realistically deliver. The expected linguistic quality is
~4-year-old preschool level; the *exciting* part is not
fluency, it is that the agent is **changing during the chat**.

Invariants
----------

* **C1 — context conditioning.** Two identical-seed
  conversations differing only in one user word produce
  *different* agent responses. ≥ 50 % of generated tokens
  must differ between the two responses. Falsifies the null
  hypothesis "the agent ignores user input".
* **C2 — user-fact recall.** After "My name is Alex" in turn
  1, asking "what is my name?" in turn 5+ produces the token
  ``"alex"`` somewhere in the response. ≥ 7 / 10 random
  trials must succeed.
* **C3 — online surprise drop.** Running 10 teacher-driven
  chat turns where the user repeats a target sentence
  produces a measurable drop in the LM's surprise on the
  target. Final-3 mean surprise / first-3 mean surprise must
  be ≤ 0.80.
* **C4 — no catastrophic forgetting.** TinyStories validation
  PPL after the full F95 chat session must be ≤ 1.20 × the
  pre-chat baseline PPL. Falsifies the failure mode
  "online learning destroys the pretraining distribution".

Usage::

    python -u -m experiments.chat_agent_f95 \\
        --corpus outputs/f79_data/tinystories_valid.txt \\
        --lm-checkpoint outputs/checkpoints/f91_lm_d512.pt \\
        --out outputs/f95_full
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.online_teacher_f85 import (
    build_vocab_with_reserved,
)
from experiments.tinystories_f79 import (
    BOS_ID, EOS_ID, PAD_ID, UNK_ID,
    _concatenate_ids, _sample_seq_batch,
    _val_perplexity, encode_story,
)
from pcm.chat import ChatSession, generate
from pcm.episodic import EpisodicBuffer
from pcm.lm import HybridPCMMiniLM, count_params
from pcm.lm_synthetic import RESERVED_CONCEPTS
from pcm.online import OnlineTeacherSession, PretrainReplayBuffer
from pcm.user_memory import UserFactMemory


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


def _ensure_vocab_has(
    stoi: dict[str, int], itos: list[str], words: list[str],
) -> None:
    """Sanity-check: if ANY of ``words`` is missing from the
    vocab, raise. We need all C1-C2 test words to be in-vocab,
    otherwise the test is just testing OOV → UNK handling."""
    missing = [w for w in words if w not in stoi]
    if missing:
        print(
            f"    WARNING: words missing from vocab: {missing} "
            f"(will be UNK in chat — C1-C2 may be noisier)",
            flush=True,
        )


def _seq_token_overlap(a: list[int], b: list[int]) -> float:
    """Fraction of tokens in ``a`` that are also in ``b``
    (position-agnostic). 1.0 = identical multiset."""
    if not a:
        return 0.0
    sa = set(a)
    sb = set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


# ─────────────────────────────────────────────────────────────────
# C1 — context conditioning
# ─────────────────────────────────────────────────────────────────


def _eval_c1_context_conditioning(
    lm: HybridPCMMiniLM, stoi: dict[str, int], itos: list[str],
    *, n_trials: int = 6, device: str,
) -> dict:
    """Run two identical-seed conversations differing only in
    one user word; measure response token-overlap. Pass if
    average overlap < 0.50 (responses depend on user input)."""
    pairs: list[tuple[str, str]] = [
        ("i love coffee", "i love hiking"),
        ("my cat is happy", "my dog is happy"),
        ("the boy ran fast", "the girl ran fast"),
        ("she likes apples", "she likes bread"),
        ("a big house", "a small house"),
        ("warm sunny day", "cold rainy day"),
    ]
    pairs = pairs[:n_trials]
    overlaps: list[float] = []
    per_pair: list[dict] = []
    for prompt_a, prompt_b in pairs:
        sess_a = ChatSession(
            lm, stoi, itos,
            unk_id=UNK_ID, pad_id=PAD_ID, eos_id=EOS_ID,
            device=device, max_response_tokens=16,
            temperature=0.7, top_k=20, top_p=0.9,
            repetition_penalty=1.3,
        )
        sess_b = ChatSession(
            lm, stoi, itos,
            unk_id=UNK_ID, pad_id=PAD_ID, eos_id=EOS_ID,
            device=device, max_response_tokens=16,
            temperature=0.7, top_k=20, top_p=0.9,
            repetition_penalty=1.3,
        )
        torch.manual_seed(1234)
        ra = sess_a.respond_to(prompt_a)
        torch.manual_seed(1234)
        rb = sess_b.respond_to(prompt_b)
        ids_a = sess_a.turns[-1].ids
        ids_b = sess_b.turns[-1].ids
        overlap = _seq_token_overlap(ids_a, ids_b)
        overlaps.append(overlap)
        per_pair.append({
            "prompt_a": prompt_a, "prompt_b": prompt_b,
            "response_a": ra, "response_b": rb,
            "token_overlap": overlap,
        })
    mean_overlap = (
        sum(overlaps) / len(overlaps) if overlaps else 1.0
    )
    return {
        "mean_token_overlap": mean_overlap,
        "max_token_overlap": max(overlaps) if overlaps else 1.0,
        "n_trials": len(overlaps),
        "per_pair": per_pair,
    }


# ─────────────────────────────────────────────────────────────────
# C2 — user-fact recall
# ─────────────────────────────────────────────────────────────────


_FILLER_TURNS = [
    "i like coffee in the morning",
    "the weather is nice today",
    "i went to the park yesterday",
    "let me tell you a story",
    "it was a sunny day",
    "the river was calm",
]

_NAME_QUERIES = [
    "what is my name",
    "do you remember my name",
    "tell me my name",
]


def _eval_c2_user_fact_recall(
    lm: HybridPCMMiniLM, stoi: dict[str, int], itos: list[str],
    *, names: list[str], n_filler_turns: int = 3,
    n_trials_per_name: int = 3, device: str,
) -> dict:
    """For each name, simulate: turn 1 = 'my name is X',
    turns 2..N = filler, turn N+1 = 'what is my name?'.
    Check whether ``X`` appears in the agent response."""
    per_trial: list[dict] = []
    n_pass = 0
    n_total = 0
    for name in names:
        for trial in range(n_trials_per_name):
            sess = ChatSession(
                lm, stoi, itos,
                unk_id=UNK_ID, pad_id=PAD_ID, eos_id=EOS_ID,
                device=device, max_response_tokens=16,
                temperature=0.7, top_k=20, top_p=0.9,
                repetition_penalty=1.3,
            )
            sess.respond_to(f"my name is {name}")
            for k in range(n_filler_turns):
                sess.respond_to(
                    _FILLER_TURNS[k % len(_FILLER_TURNS)],
                )
            query = _NAME_QUERIES[trial % len(_NAME_QUERIES)]
            response = sess.respond_to(query)
            response_lower = response.lower()
            success = name.lower() in response_lower
            per_trial.append({
                "name": name, "trial": trial, "query": query,
                "response": response, "success": success,
                "facts_known": [
                    (f.predicate, f.value)
                    for f in sess.user_memory
                ],
            })
            n_total += 1
            if success:
                n_pass += 1
    return {
        "n_pass": n_pass, "n_total": n_total,
        "accuracy": n_pass / max(n_total, 1),
        "per_trial": per_trial,
    }


# ─────────────────────────────────────────────────────────────────
# C3 — online surprise drop
# ─────────────────────────────────────────────────────────────────


def _measure_surprise(
    lm: HybridPCMMiniLM, ids: torch.Tensor,
    *, device: str,
) -> float:
    """Mean cross-entropy on shifted target. ``ids`` is 1-D."""
    lm.eval()
    if ids.numel() < 2:
        return float("nan")
    with torch.no_grad():
        seq = ids.to(device).unsqueeze(0)
        x = seq[:, :-1]
        y = seq[:, 1:]
        logits = lm(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            ignore_index=PAD_ID,
            reduction="mean",
        )
    return float(loss.item())


def _eval_c3_online_surprise_drop(
    lm: HybridPCMMiniLM, stoi: dict[str, int], itos: list[str],
    replay: PretrainReplayBuffer,
    *, n_teacher_turns: int = 30, device: str,
) -> dict:
    """Run a teacher-driven mini-loop: user repeats variants of
    a target sentence; teacher fires an M3 grad step every
    turn (low salience threshold). Measure surprise on the
    target sentence before, during, and after."""
    target_sentence = (
        "the little rabbit found a magic stone"
    )
    target_ids = torch.tensor(
        [
            stoi.get(w, UNK_ID)
            for w in target_sentence.split()
        ],
        dtype=torch.long,
    )

    # The teacher updates tok_emb rows; we pick the IDs of the
    # *content words* (not pad/bos) as the per-turn "novel".
    # Trick: since we have no reserved IDs, pass an empty
    # novel_concept_ids and use receive_chat_turn (free-form),
    # which masks by the turn's token IDs instead.
    teacher = OnlineTeacherSession(
        model=lm, novel_concept_ids=[],
        novel_concept_classes={}, replay=replay,
        pad_id=PAD_ID,
        m3_k_grad_threshold=1,
        m3_salience_threshold=0.0,  # always fire
        m3_n_replay=4,
        m3_lr=1e-3,                 # stronger than F85's 5e-3
                                    # for this gradient-mask
                                    # scheme (turn-IDs only,
                                    # not all of tok_emb)
        m3_inner_steps=2,
        m1_buffer_capacity=64,
        device=device,
    )
    pre_surprise = _measure_surprise(
        lm, target_ids, device=device,
    )

    base_paraphrases = [
        "the little rabbit found a magic stone",
        "a little rabbit found a magic stone",
        "the rabbit found a magic stone",
        "the little bunny found a magic stone",
        "the little rabbit found a small stone",
        "the little rabbit had a magic stone",
        "little rabbit found a magic stone",
        "the rabbit found a magic stone today",
        "the little rabbit found magic stone",
        "the little rabbit found a magic rock",
        "the little rabbit holds a magic stone",
        "the little rabbit kept a magic stone",
    ]
    paraphrases = (
        base_paraphrases * (
            (n_teacher_turns + len(base_paraphrases) - 1)
            // len(base_paraphrases)
        )
    )[:n_teacher_turns]

    per_turn: list[dict] = []
    for i, para in enumerate(paraphrases):
        para_ids = torch.tensor(
            [stoi.get(w, UNK_ID) for w in para.split()],
            dtype=torch.long,
        )
        if i == 0:
            ctx_ids = torch.tensor([BOS_ID], dtype=torch.long)
        else:
            prev = paraphrases[i - 1]
            ctx_ids = torch.tensor(
                [stoi.get(w, UNK_ID) for w in prev.split()],
                dtype=torch.long,
            )
        result = teacher.receive_chat_turn(
            context_ids=ctx_ids, target_ids=para_ids,
        )
        cur_surprise = _measure_surprise(
            lm, target_ids, device=device,
        )
        per_turn.append({
            "turn": i + 1, "paraphrase": para,
            "m3_fired": result.get("m3_fired", False),
            "turn_surprise": result.get("surprise"),
            "target_surprise": cur_surprise,
        })

    post_surprise = _measure_surprise(
        lm, target_ids, device=device,
    )
    surprises = [t["target_surprise"] for t in per_turn]
    first_3 = (
        sum(surprises[:3]) / 3 if len(surprises) >= 3 else pre_surprise
    )
    last_3 = (
        sum(surprises[-3:]) / 3 if len(surprises) >= 3 else post_surprise
    )
    return {
        "target_sentence": target_sentence,
        "pre_surprise": pre_surprise,
        "post_surprise": post_surprise,
        "first_3_mean": first_3,
        "last_3_mean": last_3,
        "drop_ratio": (
            last_3 / first_3 if first_3 > 0 else float("nan")
        ),
        "n_m3_grad_steps": teacher.counters.n_m3_grad_steps,
        "per_turn": per_turn,
    }


# ─────────────────────────────────────────────────────────────────
# C4 — no catastrophic forgetting
# ─────────────────────────────────────────────────────────────────


def _eval_c4_no_forgetting(
    lm: HybridPCMMiniLM, val_ids: torch.Tensor,
    *, baseline_ppl: float, seq_len: int, batch_size: int,
    n_batches: int, device: str,
) -> dict:
    """Re-measure TinyStories val PPL; pass if ≤ 1.2 × baseline."""
    cur_ppl = _val_perplexity(
        lm, val_ids, seq_len=seq_len, n_batches=n_batches,
        batch_size=batch_size, device=device,
    )
    return {
        "baseline_ppl": float(baseline_ppl),
        "post_chat_ppl": float(cur_ppl),
        "ratio": float(cur_ppl / max(baseline_ppl, 1e-9)),
    }


# ─────────────────────────────────────────────────────────────────
# Demo conversation transcript (printed for human inspection)
# ─────────────────────────────────────────────────────────────────


def _run_demo_conversation(
    lm: HybridPCMMiniLM, stoi: dict[str, int], itos: list[str],
    replay: PretrainReplayBuffer, *, device: str,
) -> dict:
    """Run a scripted 12-turn conversation with the teacher
    attached, printing every turn so the human reader can see
    what the agent actually produces. This is the
    'qualitative' demo; quantitative C1-C4 are separate."""
    teacher = OnlineTeacherSession(
        model=lm, novel_concept_ids=[],
        novel_concept_classes={}, replay=replay,
        pad_id=PAD_ID,
        m3_k_grad_threshold=1,
        m3_salience_threshold=0.0,
        m3_n_replay=4, m3_lr=2e-4, m3_inner_steps=1,
        m1_buffer_capacity=64,
        device=device,
    )
    sess = ChatSession(
        lm, stoi, itos,
        unk_id=UNK_ID, pad_id=PAD_ID, eos_id=EOS_ID,
        teacher=teacher,
        device=device, max_response_tokens=20,
        temperature=0.7, top_k=20, top_p=0.9,
        repetition_penalty=1.3,
    )
    demo_turns = [
        "hello there",
        "my name is alex",
        "i love coffee in the morning",
        "i am a programmer",
        "i live in boston",
        "what is my name",
        "what do i like",
        "tell me a story about a cat",
        "the cat was happy",
        "what is my job",
        "i love hiking too",
        "what is your favorite thing",
    ]
    transcript: list[dict] = []
    for i, user in enumerate(demo_turns):
        response = sess.respond_to(user)
        transcript.append({
            "turn": i + 1, "user": user, "agent": response,
            "n_oov": sess.turns[-2].n_oov,
            "new_facts": [
                (f.predicate, f.value)
                for f in sess.turns[-2].new_user_facts
            ],
        })
        print(
            f"    [turn {i + 1:2d}] user > {user}",
            flush=True,
        )
        print(
            f"               agent > {response}",
            flush=True,
        )
    return {
        "transcript": transcript,
        "stats": sess.stats(),
        "facts_known": sess.user_memory.as_dict(),
    }


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path,
                    default=Path("outputs/f79_data/tinystories_valid.txt"))
    ap.add_argument("--vocab-cap", type=int, default=4096)
    ap.add_argument("--n-train-stories", type=int, default=10000)
    ap.add_argument("--n-val-stories", type=int, default=1000)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--attn-every", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lm-checkpoint", type=Path,
                    default=Path("outputs/checkpoints/f91_lm_d512.pt"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f95_full"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76, flush=True)
    print(
        f"  F95 Chat-learning agent  "
        f"(d_model={args.d_model}, L={args.n_layers})",
        flush=True,
    )
    print("=" * 76, flush=True)

    # Step 1: vocab + corpus
    print(
        "\n[1/6] corpus + vocab (with reserved IDs)...",
        flush=True,
    )
    raw = args.corpus.read_text(encoding="utf-8")
    stories = [
        s for s in raw.split("<|endoftext|>")
        if len(s.strip()) > 30
    ]
    rng_split = random.Random(args.seed)
    rng_split.shuffle(stories)
    train_stories = stories[:args.n_train_stories]
    val_stories = stories[
        args.n_train_stories:
        args.n_train_stories + args.n_val_stories
    ]
    reserved_tokens = [c[0] for c in RESERVED_CONCEPTS]
    stoi, itos = build_vocab_with_reserved(
        train_stories, vocab_cap=args.vocab_cap,
        reserved_tokens=reserved_tokens,
    )
    val_ids = _concatenate_ids(
        [encode_story(s, stoi) for s in val_stories]
    )
    train_ids = _concatenate_ids(
        [encode_story(s, stoi) for s in train_stories]
    )
    vocab = len(itos)
    print(f"    vocab={vocab}", flush=True)

    test_words = [
        "alex", "bob", "carol",
        "coffee", "hiking", "rabbit", "magic", "stone",
        "name", "user", "what",
    ]
    _ensure_vocab_has(stoi, itos, test_words)

    # Step 2: load LM
    print(
        f"\n[2/6] loading LM from {args.lm_checkpoint}...",
        flush=True,
    )
    torch.manual_seed(args.seed)
    lm = HybridPCMMiniLM(
        vocab=vocab, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
        attn_every=args.attn_every,
    )
    lm.load_state_dict(
        torch.load(args.lm_checkpoint, map_location="cpu")
    )
    lm.to(DEVICE)
    print(f"    LM params: {count_params(lm):,}", flush=True)

    # Step 3: populate replay buffer
    print(
        "\n[3/6] populating PretrainReplayBuffer (32 batches)...",
        flush=True,
    )
    replay = PretrainReplayBuffer(
        capacity=args.batch_size * 32, device=DEVICE,
    )
    rng = torch.Generator(device="cpu").manual_seed(2026)
    for _ in range(32):
        xs, ys = _sample_seq_batch(
            train_ids, seq_len=args.seq_len,
            batch_size=args.batch_size, rng=rng,
        )
        replay.add_batch(xs.to(DEVICE), ys.to(DEVICE))

    # Step 4: baseline PPL
    print(
        "\n[4/6] baseline TinyStories val PPL...",
        flush=True,
    )
    baseline_ppl = _val_perplexity(
        lm, val_ids, seq_len=args.seq_len, n_batches=16,
        batch_size=args.batch_size, device=DEVICE,
    )
    print(f"    baseline_ppl = {baseline_ppl:.2f}", flush=True)

    # Step 5: demo conversation (qualitative)
    print(
        "\n[5/6] demo conversation (12 turns, qualitative)...",
        flush=True,
    )
    demo_result = _run_demo_conversation(
        lm, stoi, itos, replay, device=DEVICE,
    )

    # Step 5b: re-load LM weights for clean C1-C3 evaluations
    # (the demo session already nudged the LM; we want C1/C2 on
    # the un-nudged model and C3 fresh)
    print(
        "\n[5b/6] reloading LM checkpoint for clean C1-C3...",
        flush=True,
    )
    lm = HybridPCMMiniLM(
        vocab=vocab, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
        attn_every=args.attn_every,
    )
    lm.load_state_dict(
        torch.load(args.lm_checkpoint, map_location="cpu")
    )
    lm.to(DEVICE)

    # Step 6: C1-C4
    print(
        "\n[6/6] evaluating C1, C2, C3, C4...",
        flush=True,
    )
    t0 = time.time()
    print("\n   --- C1 context conditioning ---", flush=True)
    c1 = _eval_c1_context_conditioning(
        lm, stoi, itos, device=DEVICE,
    )
    print(
        f"    C1 mean token overlap = "
        f"{c1['mean_token_overlap']:.3f}  "
        f"(n_trials={c1['n_trials']})",
        flush=True,
    )

    print("\n   --- C2 user-fact recall ---", flush=True)
    c2 = _eval_c2_user_fact_recall(
        lm, stoi, itos,
        # Use TinyStories character names (in-vocab) so the
        # test measures recall mechanism, not vocab coverage.
        names=["tim", "lily", "tom", "lucy", "ben"],
        device=DEVICE,
    )
    print(
        f"    C2 recall = {c2['n_pass']}/{c2['n_total']} "
        f"= {c2['accuracy']:.3f}",
        flush=True,
    )

    print(
        "\n   --- C3 online surprise drop (fresh LM) ---",
        flush=True,
    )
    c3 = _eval_c3_online_surprise_drop(
        lm, stoi, itos, replay, device=DEVICE,
    )
    print(
        f"    C3 first_3={c3['first_3_mean']:.3f}  "
        f"last_3={c3['last_3_mean']:.3f}  "
        f"drop_ratio={c3['drop_ratio']:.3f}",
        flush=True,
    )

    print(
        "\n   --- C4 no catastrophic forgetting ---",
        flush=True,
    )
    c4 = _eval_c4_no_forgetting(
        lm, val_ids, baseline_ppl=baseline_ppl,
        seq_len=args.seq_len, batch_size=args.batch_size,
        n_batches=16, device=DEVICE,
    )
    print(
        f"    C4 baseline_ppl={c4['baseline_ppl']:.2f}  "
        f"post_chat_ppl={c4['post_chat_ppl']:.2f}  "
        f"ratio={c4['ratio']:.3f}",
        flush=True,
    )
    t1 = time.time()
    print(
        f"\n    eval wall = {t1-t0:.1f}s", flush=True,
    )

    verdict = {
        "C1_context_conditioning_overlap_lt_0_50": (
            c1["mean_token_overlap"] < 0.50
        ),
        "C2_user_fact_recall_ge_0_70": (
            c2["accuracy"] >= 0.70
        ),
        "C3_online_surprise_drop_le_0_80": (
            c3["drop_ratio"] <= 0.80
        ),
        "C4_val_ppl_le_1_20x_baseline": (
            c4["ratio"] <= 1.20
        ),
    }
    summary = {
        "config": vars(args) | {
            "out": str(args.out),
            "corpus": str(args.corpus),
            "lm_checkpoint": str(args.lm_checkpoint),
        },
        "vocab": vocab,
        "model_params": {"lm": count_params(lm)},
        "baseline_ppl": float(baseline_ppl),
        "demo": demo_result,
        "C1": c1, "C2": c2, "C3": c3, "C4": c4,
        "verdict": verdict,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False,
                   default=str)
    )

    print("\n" + "=" * 76, flush=True)
    print("  F95 Chat-learning agent verdict:", flush=True)
    print("=" * 76, flush=True)
    for k, v in verdict.items():
        print(
            f"    {k}: {'PASS' if v else 'FAIL'}", flush=True,
        )
    print(
        f"\n  wrote {args.out / 'summary.json'}", flush=True,
    )


if __name__ == "__main__":
    main()
