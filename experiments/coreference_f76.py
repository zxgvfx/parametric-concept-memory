"""F76 — Pronoun resolution by hypothesis-and-verify.

Tests the user's mechanism: when sentence 2 contains a pronoun
(he/she/it/...) referring to an entity in sentence 1, the model
disambiguates by:

1. Generating *candidates* of class-compatible nouns from
   sentence 1 (via :func:`pcm.coref.candidates_from_buffer` or
   ``candidates_from_token_history``).
2. For each candidate, *substituting* the pronoun and scoring
   the substituted sentence with the trained LM (perplexity).
3. Picking the lowest-perplexity candidate.

Training data follows the **subject-continuity rule**: the
pronoun in sentence 2 always refers to the *subject* of
sentence 1 ("alice gets carol . she is small ." → she = alice,
not carol). Under this rule:

* class-compatibility alone fails (both candidates are same
  class+gender by construction).
* **recency baseline fails** (it picks the *object* — the most
  recent compatible noun).
* hypothesis-verify with a trained LM should succeed (the LM
  learned the subject-continuity statistics from training).

Five falsifiable invariants:

* **R1** PCM hypothesis-verify accuracy ≥ 0.80 on held-out
  test discourses.
* **R2** class-compatibility (predicted referent matches
  pronoun's class+gender) ≥ 0.99 — never picks ``bob`` for
  ``she``.
* **R3** recency baseline ≤ 0.30 — the anti-recency rule
  defeats the standard coreference heuristic.
* **R4** PCM hypothesis-verify − recency baseline gap ≥ 0.40
  (the architecture wins by a wide margin).
* **R5** GPT and PCM both work; absolute difference ≤ 0.10.
  Confirms hypothesis-verify is *architecture-agnostic* given
  a competent LM.

Usage::

    python -m experiments.coreference_f76 \\
        --n-train 4000 --n-test 500 \\
        --out outputs/f76_full
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

from pcm.coref import (
    candidates_from_token_history,
    resolve_pronoun,
    resolve_pronoun_class_only,
    resolve_pronoun_recency,
)
from pcm.lm import (
    GPTMiniLM,
    PCMMiniLM,
    build_matched_pair,
)
from pcm.lm_synthetic import (
    PRONOUN_CLASSES,
    Tokenizer,
    generate_coreference_discourse,
)


__all__ = ["main"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────


def generate_discourse_dataset(
    n: int, *, rule: str = "subject_continuity",
    seed: int = 0,
) -> list:
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        try:
            d = generate_coreference_discourse(rng, rule=rule)
        except RuntimeError:
            continue
        out.append(d)
    return out


def tokenise_discourses(
    discourses: list, tok: Tokenizer, *, max_len: int = 12,
) -> torch.Tensor:
    ids = [tok.encode(d.tokens, max_len=max_len) for d in discourses]
    return torch.tensor(ids, dtype=torch.long)


def build_training_token_seq(
    discourses: list, tok: Tokenizer,
    *, max_len: int = 12, referent_substitution_rate: float = 0.5,
    seed: int = 0,
) -> torch.Tensor:
    """Build training sequences with optional referent
    substitution augmentation.

    Why: under the subject-continuity rule, the LM's training
    data has the pronoun (``she``) at the position-4 slot. The
    LM never sees the *actual referent* there. So at test time,
    asking the LM "P(alice | prefix) vs P(bob | prefix)" is
    asking about tokens it never saw at that position — both are
    out-of-distribution and the LM has nothing to discriminate
    on.

    Children's input is *not* like this: real language
    sometimes uses the pronoun, sometimes the explicit referent
    ("Alice gave Bob a book. Alice was happy." vs "She was
    happy."). We mirror that by augmenting the training set:
    fraction ``referent_substitution_rate`` of discourses have
    their pronoun replaced with the referent. The model then
    learns to assign probability mass to *both* the pronoun and
    the referent at position 4, but **only the correct referent
    ever appears there** — so P(subject | prefix) >> P(object |
    prefix) becomes learnable.
    """
    rng = random.Random(seed)
    ids_list = []
    for d in discourses:
        tokens = list(d.tokens)
        if rng.random() < referent_substitution_rate:
            # Substitute pronoun with ground-truth referent
            tokens[d.pronoun_position] = d.referent
        ids_list.append(tok.encode(tokens, max_len=max_len))
    return torch.tensor(ids_list, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────
# Training (same as F74 _train_one)
# ─────────────────────────────────────────────────────────────────


def _train_one(
    model: nn.Module, train_seq: torch.Tensor, *,
    epochs: int, batch_size: int, lr: float, pad_id: int,
    rng_seed: int = 0,
) -> None:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    N = train_seq.shape[0]
    rng = torch.Generator(device="cpu").manual_seed(rng_seed)
    for _ in range(epochs):
        idx = torch.randperm(N, generator=rng)
        for i in range(0, N, batch_size):
            j = idx[i:i + batch_size]
            batch = train_seq[j].to(DEVICE)
            x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                ignore_index=pad_id,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()


# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────


def evaluate_resolver(
    discourses: list, tok: Tokenizer,
    *, resolver_name: str, lm=None,
    shuffle_candidates: bool = True, shuffle_seed: int = 0,
) -> dict:
    """Apply one resolver to every discourse and compute accuracy
    and class-compatibility stats.

    Note: ``shuffle_candidates=True`` (default) randomly permutes
    the candidate list before passing to each resolver. This is
    essential to make the ``class_only`` baseline a genuine
    coin-flip (otherwise it benefits from synthetic-data token
    ordering: in our generator, the subject is at position 0 and
    therefore appears first in the candidate list, so naive
    ``class_only`` would trivially achieve 100% under
    subject-continuity rule).

    The shuffle does *not* affect ``recency`` (which uses
    timestamps) or ``hypothesis_verify`` (which scores all
    candidates — order-invariant).
    """
    rng = random.Random(shuffle_seed)
    n = 0
    n_correct = 0
    n_class_ok = 0
    n_no_pred = 0
    confusion = {"subject": 0, "object": 0, "other": 0}
    for d in discourses:
        cands = candidates_from_token_history(
            d.tokens, d.pronoun_position, d.pronoun,
        )
        if shuffle_candidates:
            cands = list(cands)
            rng.shuffle(cands)
        if resolver_name == "class_only":
            r = resolve_pronoun_class_only(
                d.tokens, d.pronoun_position, candidates=cands,
            )
        elif resolver_name == "recency":
            r = resolve_pronoun_recency(
                d.tokens, d.pronoun_position, candidates=cands,
            )
        elif resolver_name == "hypothesis_verify":
            r = resolve_pronoun(
                d.tokens, d.pronoun_position, lm, tok,
                candidates=cands, device=DEVICE,
            )
        else:
            raise ValueError(f"unknown resolver {resolver_name}")
        n += 1
        pred = r.predicted_referent
        if pred is None:
            n_no_pred += 1
        else:
            # Class compatibility check (should be enforced by
            # candidate filtering, but verify)
            from pcm.lm_synthetic import is_compatible_referent
            if is_compatible_referent(d.pronoun, pred):
                n_class_ok += 1
            if pred == d.referent:
                n_correct += 1
            # Track which slot the resolver picked
            if pred == d.s1_subject:
                confusion["subject"] += 1
            elif pred == d.s1_object:
                confusion["object"] += 1
            else:
                confusion["other"] += 1
    return {
        "resolver": resolver_name,
        "n": n,
        "n_correct": n_correct,
        "accuracy": n_correct / max(n, 1),
        "n_class_ok": n_class_ok,
        "class_compat_rate": n_class_ok / max(n, 1),
        "n_no_prediction": n_no_pred,
        "confusion": confusion,
    }


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--rule", type=str, default="subject_continuity",
                    choices=["subject_continuity", "object_continuity"])
    ap.add_argument("--referent-substitution-rate", type=float,
                    default=0.5,
                    help="fraction of training discourses where the "
                         "pronoun is replaced with the ground-truth "
                         "referent (data augmentation; see docstring "
                         "of build_training_token_seq)")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f76_full"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer()
    print("=" * 76)
    print(f"  F76 pronoun resolution by hypothesis-and-verify "
          f"(rule={args.rule})")
    print("=" * 76)

    # Generate discourses
    print(f"\nGenerating {args.n_train} train + {args.n_test} test "
          f"discourses (rule={args.rule})...")
    train_discourses = generate_discourse_dataset(
        args.n_train, rule=args.rule, seed=42,
    )
    test_discourses = generate_discourse_dataset(
        args.n_test, rule=args.rule, seed=99999,
    )
    # Confirm the rule: every discourse should have referent = subject
    n_subj_train = sum(1 for d in train_discourses
                        if d.referent == d.s1_subject)
    n_obj_train = sum(1 for d in train_discourses
                       if d.referent == d.s1_object)
    print(f"  train referent breakdown: "
          f"subject={n_subj_train}, object={n_obj_train}")

    # Show a few examples
    print("  sample training discourses:")
    for d in train_discourses[:5]:
        print(f"    {' '.join(d.tokens)}  "
              f"→ referent={d.referent} (={'subj' if d.referent == d.s1_subject else 'obj'})")

    train_seq = build_training_token_seq(
        train_discourses, tok, max_len=args.max_len,
        referent_substitution_rate=args.referent_substitution_rate,
        seed=12345,
    )
    n_substituted = int(args.n_train * args.referent_substitution_rate)
    print(f"  train data: {args.n_train} discourses, "
          f"~{n_substituted} with pronoun→referent substitution "
          f"(rate={args.referent_substitution_rate})")

    # Build matched pair
    gpt, pcm, diag = build_matched_pair(
        vocab=tok.vocab_size, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
    )
    gpt = gpt.to(DEVICE)
    pcm = pcm.to(DEVICE)
    print(f"\nMatched models: GPT={diag['gpt_params']:,}  "
          f"PCM={diag['pcm_params']:,}")

    # Train both
    print(f"\nTraining GPTMiniLM for {args.epochs} epochs...")
    t0 = time.time()
    _train_one(
        gpt, train_seq, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr,
        pad_id=tok.pad_id, rng_seed=11,
    )
    gpt_wall = time.time() - t0
    print(f"  wall = {gpt_wall:.1f}s")
    print(f"Training PCMMiniLM for {args.epochs} epochs...")
    t0 = time.time()
    _train_one(
        pcm, train_seq, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr,
        pad_id=tok.pad_id, rng_seed=22,
    )
    pcm_wall = time.time() - t0
    print(f"  wall = {pcm_wall:.1f}s")

    # Evaluate three resolvers on each LM
    print(f"\nEvaluating resolvers on {args.n_test} held-out discourses...")
    results: dict[str, dict] = {}
    # class-only (LM-independent)
    results["class_only"] = evaluate_resolver(
        test_discourses, tok, resolver_name="class_only",
    )
    # recency (LM-independent)
    results["recency"] = evaluate_resolver(
        test_discourses, tok, resolver_name="recency",
    )
    # hypothesis-verify with GPT
    results["hypothesis_verify_gpt"] = evaluate_resolver(
        test_discourses, tok, resolver_name="hypothesis_verify", lm=gpt,
    )
    # hypothesis-verify with PCM
    results["hypothesis_verify_pcm"] = evaluate_resolver(
        test_discourses, tok, resolver_name="hypothesis_verify", lm=pcm,
    )

    print("\nResults:")
    for name, r in results.items():
        print(f"  {name:<26s}  accuracy = {r['accuracy']:.3f}  "
              f"class_compat = {r['class_compat_rate']:.3f}  "
              f"confusion = {r['confusion']}")

    # Verdict (graded on the headline PCM resolver)
    pcm_acc = results["hypothesis_verify_pcm"]["accuracy"]
    gpt_acc = results["hypothesis_verify_gpt"]["accuracy"]
    recency_acc = results["recency"]["accuracy"]
    class_compat = results["hypothesis_verify_pcm"]["class_compat_rate"]

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "matched_params": diag,
        "wall_gpt_s": gpt_wall,
        "wall_pcm_s": pcm_wall,
        "results": results,
    }
    summary["verdict"] = {
        "R1_pcm_hypothesis_verify_accuracy_pass":
            pcm_acc >= 0.80,
        "R2_class_compatibility_pass":
            class_compat >= 0.99,
        "R3_recency_baseline_low_pass":
            recency_acc <= 0.30,
        "R4_pcm_beats_recency_pass":
            pcm_acc - recency_acc >= 0.40,
        "R5_gpt_pcm_consistent_pass":
            abs(gpt_acc - pcm_acc) <= 0.10,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print("  F76 verdict:")
    print("=" * 76)
    v = summary["verdict"]
    print(f"  R1 PCM hypothesis-verify ≥ 0.80   : "
          f"{pcm_acc:.3f}  "
          f"[{'PASS' if v['R1_pcm_hypothesis_verify_accuracy_pass'] else 'FAIL'}]")
    print(f"  R2 class compatibility ≥ 0.99     : "
          f"{class_compat:.3f}  "
          f"[{'PASS' if v['R2_class_compatibility_pass'] else 'FAIL'}]")
    print(f"  R3 recency baseline ≤ 0.30        : "
          f"{recency_acc:.3f}  "
          f"[{'PASS' if v['R3_recency_baseline_low_pass'] else 'FAIL'}]")
    print(f"  R4 PCM − recency ≥ 0.40           : "
          f"{pcm_acc - recency_acc:+.3f}  "
          f"[{'PASS' if v['R4_pcm_beats_recency_pass'] else 'FAIL'}]")
    print(f"  R5 |GPT − PCM| ≤ 0.10             : "
          f"|{gpt_acc:.3f} − {pcm_acc:.3f}| = {abs(gpt_acc-pcm_acc):.3f}  "
          f"[{'PASS' if v['R5_gpt_pcm_consistent_pass'] else 'FAIL'}]")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
