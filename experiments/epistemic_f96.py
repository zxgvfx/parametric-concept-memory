"""F96 — Epistemic agency: hypothesis-test before accept.

F95 built a *gullible* chat-learning agent: whatever the user
says enters the M3 micro-gradient and the user-fact memory
unchallenged. If the user asserts ``1+1=3``, the F95 agent
would store and learn from that.

F96 inserts a verification stage between the user input and
the rest of the chat pipeline. Five invariants:

* **E1 — Arithmetic skepticism.** 10 ``X op Y = Z`` trials
  where ``Z`` is wrong → pushback ≥ 8 / 10.
* **E2 — Arithmetic acceptance.** 10 ``X op Y = Z`` trials
  where ``Z`` is correct → accept ≥ 8 / 10.
* **E3 — Contradiction detection.** 10 paired
  ``a X is a Y`` then ``a X is not a Y`` trials → pushback
  on second utterance ≥ 6 / 10.
* **E4 — Self-report acceptance.** 10 ``my name is X``
  trials → accept ≥ 9 / 10 (self-reports must always
  pass through; the agent must not interrogate its user
  about who they are).
* **E5 — F95 non-regression.** With epistemic enabled,
  the F95 C1-C4 invariants must still pass on inputs that
  contain no contradictions (i.e., the verification stage
  must not silently break the conversational agent).

Usage::

    python -u -m experiments.epistemic_f96 \\
        --corpus outputs/f79_data/tinystories_valid.txt \\
        --lm-checkpoint outputs/checkpoints/f91_lm_d512.pt \\
        --out outputs/f96_full
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from experiments.chat_agent_f95 import (
    _eval_c1_context_conditioning,
    _eval_c2_user_fact_recall,
    _eval_c4_no_forgetting,
    _ensure_vocab_has,
)
from experiments.online_teacher_f85 import (
    build_vocab_with_reserved,
)
from experiments.tinystories_f79 import (
    BOS_ID, EOS_ID, PAD_ID, UNK_ID,
    _concatenate_ids, _sample_seq_batch,
    _val_perplexity, encode_story,
)
from pcm.chat import ChatSession
from pcm.epistemic import EpistemicAgent
from pcm.lm import HybridPCMMiniLM, count_params
from pcm.lm_synthetic import RESERVED_CONCEPTS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────
# E1 — arithmetic skepticism
# ─────────────────────────────────────────────────────────────────


_ARITH_BAD_TRIALS: list[tuple[str, str]] = [
    ("1 + 1 = 3", "2"),
    ("2 + 2 = 5", "4"),
    ("3 + 4 = 8", "7"),
    ("5 - 2 = 4", "3"),
    ("10 - 7 = 4", "3"),
    ("3 * 4 = 11", "12"),
    ("6 * 2 = 13", "12"),
    ("8 / 2 = 5", "4"),
    ("9 / 3 = 4", "3"),
    ("2 plus 3 is 6", "5"),
]


_ARITH_GOOD_TRIALS: list[str] = [
    "1 + 1 = 2",
    "2 + 2 = 4",
    "3 + 4 = 7",
    "5 - 2 = 3",
    "10 - 7 = 3",
    "3 * 4 = 12",
    "6 * 2 = 12",
    "8 / 2 = 4",
    "9 / 3 = 3",
    "2 plus 3 is 5",
]


def _eval_e1_arithmetic_skepticism(
    lm, stoi, itos, *, device: str,
) -> dict:
    """For each wrong arithmetic claim, the agent must push
    back (not silently absorb)."""
    per_trial: list[dict] = []
    n_pushback = 0
    for claim_text, true_value in _ARITH_BAD_TRIALS:
        epi = EpistemicAgent(
            lm=lm, stoi=stoi, use_lm_plausibility=False,
        )
        sess = ChatSession(
            lm, stoi, itos,
            unk_id=UNK_ID, pad_id=PAD_ID, eos_id=EOS_ID,
            epistemic=epi, device=device,
            max_response_tokens=16,
            temperature=0.7, top_k=20, top_p=0.9,
        )
        response = sess.respond_to(claim_text)
        action = sess.turns[-2].epistemic_action
        ok = action == "pushback"
        per_trial.append({
            "claim": claim_text,
            "true_value": true_value,
            "epistemic_action": action,
            "agent_response": response,
            "passed": ok,
        })
        if ok:
            n_pushback += 1
    return {
        "n_trials": len(_ARITH_BAD_TRIALS),
        "n_pushback": n_pushback,
        "accuracy": n_pushback / len(_ARITH_BAD_TRIALS),
        "per_trial": per_trial,
    }


# ─────────────────────────────────────────────────────────────────
# E2 — arithmetic acceptance
# ─────────────────────────────────────────────────────────────────


def _eval_e2_arithmetic_acceptance(
    lm, stoi, itos, *, device: str,
) -> dict:
    """For each correct arithmetic claim, the agent must
    accept (not falsely contradict)."""
    per_trial: list[dict] = []
    n_accept = 0
    for claim_text in _ARITH_GOOD_TRIALS:
        epi = EpistemicAgent(
            lm=lm, stoi=stoi, use_lm_plausibility=False,
        )
        sess = ChatSession(
            lm, stoi, itos,
            unk_id=UNK_ID, pad_id=PAD_ID, eos_id=EOS_ID,
            epistemic=epi, device=device,
            max_response_tokens=16,
            temperature=0.7, top_k=20, top_p=0.9,
        )
        sess.respond_to(claim_text)
        action = sess.turns[-2].epistemic_action
        ok = action == "accept"
        per_trial.append({
            "claim": claim_text,
            "epistemic_action": action,
            "passed": ok,
        })
        if ok:
            n_accept += 1
    return {
        "n_trials": len(_ARITH_GOOD_TRIALS),
        "n_accept": n_accept,
        "accuracy": n_accept / len(_ARITH_GOOD_TRIALS),
        "per_trial": per_trial,
    }


# ─────────────────────────────────────────────────────────────────
# E3 — contradiction detection
# ─────────────────────────────────────────────────────────────────


_CONTRA_PAIRS: list[tuple[str, str]] = [
    ("a cat is an animal", "a cat is not an animal"),
    ("a dog is an animal", "a dog is not an animal"),
    ("a bird is an animal", "a bird is not an animal"),
    ("an apple is a fruit", "an apple is not a fruit"),
    ("a horse is an animal", "a horse is not an animal"),
    ("a car is a vehicle", "a car is not a vehicle"),
    ("a bus is a vehicle", "a bus is not a vehicle"),
    ("a rabbit is an animal", "a rabbit is not an animal"),
    ("a fox is an animal", "a fox is not an animal"),
    ("a cake is a food", "a cake is not a food"),
]


def _eval_e3_contradiction_detection(
    lm, stoi, itos, *, device: str,
) -> dict:
    """Turn 1 establishes 'X is Y'; turn 2 asserts the
    negation. The agent must push back on turn 2."""
    per_trial: list[dict] = []
    n_pushback = 0
    for first, second in _CONTRA_PAIRS:
        epi = EpistemicAgent(
            lm=lm, stoi=stoi, use_lm_plausibility=False,
        )
        sess = ChatSession(
            lm, stoi, itos,
            unk_id=UNK_ID, pad_id=PAD_ID, eos_id=EOS_ID,
            epistemic=epi, device=device,
            max_response_tokens=16,
            temperature=0.7, top_k=20, top_p=0.9,
        )
        sess.respond_to(first)
        sess.respond_to(second)
        action = sess.turns[-2].epistemic_action
        ok = action == "pushback"
        per_trial.append({
            "first": first, "second": second,
            "epistemic_action": action,
            "agent_response": sess.turns[-1].text,
            "passed": ok,
        })
        if ok:
            n_pushback += 1
    return {
        "n_trials": len(_CONTRA_PAIRS),
        "n_pushback": n_pushback,
        "accuracy": n_pushback / len(_CONTRA_PAIRS),
        "per_trial": per_trial,
    }


# ─────────────────────────────────────────────────────────────────
# E4 — self-report acceptance
# ─────────────────────────────────────────────────────────────────


_SELF_REPORTS: list[str] = [
    "my name is alex",
    "my name is tim",
    "my name is lily",
    "my name is tom",
    "i love coffee",
    "i love hiking",
    "i am a programmer",
    "i am a teacher",
    "i live in boston",
    "i am 30 years old",
]


def _eval_e4_self_report_acceptance(
    lm, stoi, itos, *, device: str,
) -> dict:
    """Self-reports must always pass through verification —
    the agent must never interrogate the user about who
    they are."""
    per_trial: list[dict] = []
    n_accept = 0
    for utt in _SELF_REPORTS:
        epi = EpistemicAgent(
            lm=lm, stoi=stoi, use_lm_plausibility=False,
        )
        sess = ChatSession(
            lm, stoi, itos,
            unk_id=UNK_ID, pad_id=PAD_ID, eos_id=EOS_ID,
            epistemic=epi, device=device,
            max_response_tokens=16,
            temperature=0.7, top_k=20, top_p=0.9,
        )
        sess.respond_to(utt)
        action = sess.turns[-2].epistemic_action
        ok = action == "accept"
        per_trial.append({
            "utterance": utt,
            "epistemic_action": action,
            "passed": ok,
        })
        if ok:
            n_accept += 1
    return {
        "n_trials": len(_SELF_REPORTS),
        "n_accept": n_accept,
        "accuracy": n_accept / len(_SELF_REPORTS),
        "per_trial": per_trial,
    }


# ─────────────────────────────────────────────────────────────────
# E5 — F95 non-regression with epistemic enabled
# ─────────────────────────────────────────────────────────────────


def _wrap_eval_with_epistemic(
    eval_fn, lm, stoi, itos, **kwargs,
):
    """Hack: monkey-patch ChatSession constructor to inject
    a fresh EpistemicAgent per session. The C1/C2 evals
    already create their own ChatSessions, so we transparently
    inject the agent."""
    import pcm.chat as chat_mod
    original_init = chat_mod.ChatSession.__init__

    def patched_init(self, *args, **kwd):
        kwd["epistemic"] = EpistemicAgent(
            lm=lm, stoi=stoi, use_lm_plausibility=False,
        )
        original_init(self, *args, **kwd)

    chat_mod.ChatSession.__init__ = patched_init
    try:
        result = eval_fn(lm, stoi, itos, **kwargs)
    finally:
        chat_mod.ChatSession.__init__ = original_init
    return result


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus", type=Path,
        default=Path("outputs/f79_data/tinystories_valid.txt"),
    )
    ap.add_argument("--vocab-cap", type=int, default=4096)
    ap.add_argument("--n-train-stories", type=int, default=10000)
    ap.add_argument("--n-val-stories", type=int, default=1000)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--attn-every", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--lm-checkpoint", type=Path,
        default=Path("outputs/checkpoints/f91_lm_d512.pt"),
    )
    ap.add_argument(
        "--skip-e5", action="store_true", default=False,
        help="skip F95 regression (saves ~2 min)",
    )
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--out", type=Path,
        default=Path("outputs/f96_full"),
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76, flush=True)
    print(
        f"  F96 Epistemic agency  "
        f"(d_model={args.d_model}, L={args.n_layers})",
        flush=True,
    )
    print("=" * 76, flush=True)

    print(
        "\n[1/4] corpus + vocab + LM checkpoint...",
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
    vocab = len(itos)
    print(f"    vocab={vocab}", flush=True)

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

    # E1 - E4
    print("\n[2/4] E1 — arithmetic skepticism...", flush=True)
    t0 = time.time()
    e1 = _eval_e1_arithmetic_skepticism(
        lm, stoi, itos, device=DEVICE,
    )
    print(
        f"    E1 pushback rate = {e1['accuracy']:.3f} "
        f"({e1['n_pushback']}/{e1['n_trials']})",
        flush=True,
    )
    for trial in e1["per_trial"]:
        flag = "✓" if trial["passed"] else "✗"
        print(
            f"      {flag} \"{trial['claim']}\"  -> "
            f"{trial['epistemic_action']}",
            flush=True,
        )

    print("\n[2/4] E2 — arithmetic acceptance...", flush=True)
    e2 = _eval_e2_arithmetic_acceptance(
        lm, stoi, itos, device=DEVICE,
    )
    print(
        f"    E2 accept rate = {e2['accuracy']:.3f} "
        f"({e2['n_accept']}/{e2['n_trials']})",
        flush=True,
    )
    for trial in e2["per_trial"]:
        flag = "✓" if trial["passed"] else "✗"
        print(
            f"      {flag} \"{trial['claim']}\"  -> "
            f"{trial['epistemic_action']}",
            flush=True,
        )

    print(
        "\n[3/4] E3 — contradiction detection...",
        flush=True,
    )
    e3 = _eval_e3_contradiction_detection(
        lm, stoi, itos, device=DEVICE,
    )
    print(
        f"    E3 contradiction-detected rate = "
        f"{e3['accuracy']:.3f} "
        f"({e3['n_pushback']}/{e3['n_trials']})",
        flush=True,
    )
    for trial in e3["per_trial"]:
        flag = "✓" if trial["passed"] else "✗"
        print(
            f"      {flag} \"{trial['first']}\" then "
            f"\"{trial['second']}\"  -> "
            f"{trial['epistemic_action']}",
            flush=True,
        )

    print(
        "\n[3/4] E4 — self-report acceptance...",
        flush=True,
    )
    e4 = _eval_e4_self_report_acceptance(
        lm, stoi, itos, device=DEVICE,
    )
    print(
        f"    E4 accept rate = {e4['accuracy']:.3f} "
        f"({e4['n_accept']}/{e4['n_trials']})",
        flush=True,
    )
    for trial in e4["per_trial"]:
        flag = "✓" if trial["passed"] else "✗"
        print(
            f"      {flag} \"{trial['utterance']}\"  -> "
            f"{trial['epistemic_action']}",
            flush=True,
        )

    # E5 — F95 non-regression
    e5: dict = {"skipped": True}
    if not args.skip_e5:
        print(
            "\n[4/4] E5 — F95 non-regression (C1+C2+C4 "
            "with epistemic enabled)...",
            flush=True,
        )
        baseline_ppl = _val_perplexity(
            lm, val_ids, seq_len=args.seq_len, n_batches=16,
            batch_size=args.batch_size, device=DEVICE,
        )
        print(
            f"    baseline_ppl = {baseline_ppl:.2f}",
            flush=True,
        )
        c1 = _wrap_eval_with_epistemic(
            _eval_c1_context_conditioning,
            lm, stoi, itos, device=DEVICE,
        )
        print(
            f"    C1 overlap = {c1['mean_token_overlap']:.3f}",
            flush=True,
        )
        c2 = _wrap_eval_with_epistemic(
            _eval_c2_user_fact_recall,
            lm, stoi, itos,
            names=["tim", "lily", "tom", "lucy", "ben"],
            device=DEVICE,
        )
        print(
            f"    C2 recall = {c2['accuracy']:.3f} "
            f"({c2['n_pass']}/{c2['n_total']})",
            flush=True,
        )
        c4 = _eval_c4_no_forgetting(
            lm, val_ids, baseline_ppl=baseline_ppl,
            seq_len=args.seq_len, batch_size=args.batch_size,
            n_batches=16, device=DEVICE,
        )
        print(
            f"    C4 ratio = {c4['ratio']:.3f}",
            flush=True,
        )
        e5 = {
            "baseline_ppl": baseline_ppl,
            "C1": c1, "C2": c2, "C4": c4,
            "C1_pass": c1["mean_token_overlap"] < 0.50,
            "C2_pass": c2["accuracy"] >= 0.70,
            "C4_pass": c4["ratio"] <= 1.20,
        }

    t1 = time.time()
    print(
        f"\n    eval wall = {t1-t0:.1f}s", flush=True,
    )

    verdict = {
        "E1_arith_skepticism_ge_0_80": (
            e1["accuracy"] >= 0.80
        ),
        "E2_arith_acceptance_ge_0_80": (
            e2["accuracy"] >= 0.80
        ),
        "E3_contradiction_ge_0_60": (
            e3["accuracy"] >= 0.60
        ),
        "E4_self_report_ge_0_90": (
            e4["accuracy"] >= 0.90
        ),
    }
    if not args.skip_e5:
        verdict["E5_F95_non_regression"] = (
            e5["C1_pass"] and e5["C2_pass"] and e5["C4_pass"]
        )

    summary = {
        "config": vars(args) | {
            "out": str(args.out),
            "corpus": str(args.corpus),
            "lm_checkpoint": str(args.lm_checkpoint),
        },
        "vocab": vocab,
        "model_params": {"lm": count_params(lm)},
        "E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5,
        "verdict": verdict,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False,
                   default=str)
    )

    print("\n" + "=" * 76, flush=True)
    print(
        "  F96 Epistemic agency verdict:", flush=True,
    )
    print("=" * 76, flush=True)
    for k, v in verdict.items():
        print(
            f"    {k}: {'PASS' if v else 'FAIL'}",
            flush=True,
        )
    print(
        f"\n  wrote {args.out / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
