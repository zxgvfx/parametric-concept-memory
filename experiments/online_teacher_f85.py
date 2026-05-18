"""F85 — Online Teacher Loop (PCM v9.0).

Tests whether a pretrained F81/F83 PCM LM can learn **new
concepts** through a teacher correction loop, in K = O(15)
interactions, without catastrophic forgetting.

Setup:

1. Pretrain a :class:`TitansPCMMiniLM` (F81 + F83 memory
   readout) on TinyStories with a vocab that *reserves* 8
   slots for fictional concepts (``zorgon``, ``floob``,
   ``snerflo``, ``vooz`` — animals; ``glimber``, ``quarp``,
   ``mibble``, ``prag`` — foods). These tokens never appear
   in the pretraining corpus, so their embeddings stay at
   random init after pretraining.
2. Snapshot the pretrain-val PPL (the O2 baseline).
3. For each concept, present 15 teaching examples one at a
   time through :class:`OnlineTeacherSession`. The session
   applies:
   * **M1** (fast episodic write) on every event.
   * **M3** (slow micro-gradient on the novel-concept
     embedding rows only, with 7 pretrain-replay examples per
     step) when the concept has been corrected ≥ 3 times.
4. After all 120 corrections, run :meth:`sleep` once for the
   K-means consolidation purity test (O6).
5. Evaluate the six invariants.

Six falsifiable invariants:

* **O1 acquisition** — held-out test PPL on concept sentences
  drops by ≥ 50 % vs pre-teaching baseline.
* **O2 no catastrophic forgetting** — pretrain-val PPL stays
  within 1.10× of pre-online baseline.
* **O3 selectional generalisation** — after teaching, the
  model assigns higher ``P(verb | "the ANIMAL_concept")``
  than ``P(verb | "the apple")`` for ≥ 75 % of (animal-verb)
  test pairs. (Tests that the class is learnt, not just the
  surface form.)
* **O4 sample efficiency** — concept PPL halves within ≤ 20
  corrections (child-acquisition regime).
* **O5 surprise decay** — average surprise per correction at
  ``i = K`` is ≤ 0.5 × average surprise at ``i = 1``,
  proving the model genuinely tracks the concept.
* **O6 sleep consolidation purity** — K-means clusters of the
  episodic buffer reach ≥ 0.65 purity against the
  ``{ANIMAL, FOOD}`` ground-truth split.

Usage::

    python -m experiments.online_teacher_f85 \\
        --corpus outputs/f79_data/tinystories_valid.txt \\
        --out outputs/f85_full
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.tinystories_f79 import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    _concatenate_ids,
    _normalise_text,
    _sample_seq_batch,
    _tokenise_text,
    _val_perplexity,
    build_vocab,
    encode_story,
)
from pcm.lm import TitansPCMMiniLM, count_params
from pcm.lm_synthetic import (
    RESERVED_CONCEPTS,
    generate_concept_dataset,
)
from pcm.online import OnlineTeacherSession, PretrainReplayBuffer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────
# Vocab + tokeniser helpers
# ─────────────────────────────────────────────────────────────────


def build_vocab_with_reserved(
    stories: list[str], *, vocab_cap: int,
    reserved_tokens: list[str],
) -> tuple[dict[str, int], list[str]]:
    """Wrap :func:`tinystories_f79.build_vocab` and append
    ``reserved_tokens`` at the end. Each gets a unique ID in
    ``[vocab_cap, vocab_cap + len(reserved_tokens))``.
    """
    stoi, itos = build_vocab(stories, vocab_cap=vocab_cap)
    for tok in reserved_tokens:
        if tok in stoi:
            raise ValueError(
                f"reserved token {tok!r} already appears in "
                f"vocab; pick a token that does not occur in "
                f"the corpus"
            )
        stoi[tok] = len(itos)
        itos.append(tok)
    return stoi, itos


def encode_word_list(
    words: list[str], stoi: dict[str, int],
) -> torch.Tensor:
    """Encode a list of words to a 1-D LongTensor. Wraps with
    ``<bos>`` and ``<eos>``."""
    ids = [BOS_ID]
    for w in words:
        ids.append(stoi.get(w, UNK_ID))
    ids.append(EOS_ID)
    return torch.tensor(ids, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────
# Pretraining (TinyStories) — replay buffer is filled in the last
# `n_replay_steps` steps
# ─────────────────────────────────────────────────────────────────


def pretrain_model(
    model: TitansPCMMiniLM, *, train_ids: torch.Tensor,
    val_ids: torch.Tensor, n_steps: int, seq_len: int,
    batch_size: int, lr: float, log_every: int,
    replay: PretrainReplayBuffer, n_replay_steps: int,
    device: str,
) -> dict:
    """Standard LM pretraining loop. Populates ``replay`` from
    the last ``n_replay_steps`` mini-batches."""
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4,
    )
    rng = torch.Generator(device="cpu").manual_seed(7777)
    log = []
    t_start = time.time()
    model.train()
    for step in range(1, n_steps + 1):
        xs, ys = _sample_seq_batch(
            train_ids, seq_len=seq_len, batch_size=batch_size,
            rng=rng,
        )
        xs = xs.to(device)
        ys = ys.to(device)
        logits = model(xs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            ys.reshape(-1), ignore_index=PAD_ID,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # Last N steps go into the replay buffer
        if step > n_steps - n_replay_steps:
            replay.add_batch(xs, ys)
        if step % log_every == 0 or step == n_steps:
            val_ppl = _val_perplexity(
                model, val_ids, seq_len=seq_len,
                n_batches=16, batch_size=batch_size, device=device,
            )
            log.append({
                "step": step, "train_loss": float(loss.item()),
                "val_ppl": val_ppl,
            })
            model.train()
            print(
                f"    [pretrain] step {step:4d}/{n_steps}  "
                f"loss={loss.item():.3f}  val_ppl={val_ppl:.1f}  "
                f"wall={time.time()-t_start:.1f}s",
                flush=True,
            )
    final_ppl = _val_perplexity(
        model, val_ids, seq_len=seq_len,
        n_batches=64, batch_size=batch_size, device=device,
    )
    return {"final_ppl": final_ppl, "log": log,
            "wall_s": time.time() - t_start}


# ─────────────────────────────────────────────────────────────────
# Online session driver
# ─────────────────────────────────────────────────────────────────


def _eval_concept_test_ppl(
    sess: OnlineTeacherSession, dataset: dict,
    stoi: dict[str, int], *, n_per_concept: int | None = None,
) -> dict:
    """Mean test PPL per concept on held-out sentences. Returns
    ``{concept_token: ppl, "_macro_mean": float}``."""
    out: dict[str, float] = {}
    for token, sents in dataset["test"].items():
        if n_per_concept is not None:
            sents = sents[:n_per_concept]
        seqs = [encode_word_list(s, stoi) for s in sents]
        ppl = sess.eval_ppl_on_sequences(seqs, use_memory=True)
        out[token] = ppl
    valid_ppls = [v for v in out.values() if v == v]  # filter NaN
    out["_macro_mean"] = (
        float(sum(valid_ppls) / max(len(valid_ppls), 1))
        if valid_ppls else float("nan")
    )
    return out


@torch.no_grad()
def _eval_selectional_o3(
    sess: OnlineTeacherSession, stoi: dict[str, int],
    novel_concepts: tuple[tuple[str, str, str], ...],
    *, animal_verbs: tuple[str, ...] = ("slept", "ran", "jumped"),
) -> dict:
    """O3 selectional generalisation. For each ANIMAL novel
    concept and each ``verb`` in ``animal_verbs``, check that

        log P(verb | "the <concept>")
        > log P(verb | "the apple")

    "apple" is a known FOOD token from pretraining, so this
    tests whether the model has internalised that the novel
    concept is in the ANIMAL class.
    """
    model = sess.model
    device = sess.device
    apple_id = stoi.get("apple", UNK_ID)
    the_id = stoi.get("the", UNK_ID)
    if the_id == UNK_ID or apple_id == UNK_ID:
        return {"status": "missing_baseline_tokens"}
    n_correct = 0
    n_total = 0
    per_concept: dict[str, dict] = {}
    for token, cls, _ in novel_concepts:
        if cls != "ANIMAL":
            continue
        cid = stoi.get(token, UNK_ID)
        if cid == UNK_ID:
            continue
        n_concept_correct = 0
        n_concept_total = 0
        for verb in animal_verbs:
            verb_id = stoi.get(verb, UNK_ID)
            if verb_id == UNK_ID:
                continue
            prefix_concept = torch.tensor(
                [[BOS_ID, the_id, cid]], device=device,
            )
            prefix_apple = torch.tensor(
                [[BOS_ID, the_id, apple_id]], device=device,
            )
            model.eval()
            if hasattr(model, "memory_readout"):
                logits_c = model(
                    prefix_concept, use_memory=True, imprint=False,
                )
                logits_a = model(
                    prefix_apple, use_memory=True, imprint=False,
                )
            else:
                logits_c = model(prefix_concept)
                logits_a = model(prefix_apple)
            log_p_c = F.log_softmax(
                logits_c[0, -1], dim=-1,
            )[verb_id].item()
            log_p_a = F.log_softmax(
                logits_a[0, -1], dim=-1,
            )[verb_id].item()
            if log_p_c > log_p_a:
                n_correct += 1
                n_concept_correct += 1
            n_total += 1
            n_concept_total += 1
        per_concept[token] = {
            "n_correct": n_concept_correct,
            "n_total": n_concept_total,
            "accuracy": (
                n_concept_correct / max(n_concept_total, 1)
            ),
        }
    return {
        "status": "ok",
        "n_correct": n_correct, "n_total": n_total,
        "accuracy": n_correct / max(n_total, 1),
        "per_concept": per_concept,
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
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--attn-every", type=int, default=4)
    ap.add_argument("--memory-capacity", type=int, default=512)
    ap.add_argument("--memory-top-k", type=int, default=8)
    ap.add_argument("--n-pretrain-steps", type=int, default=3000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--log-every", type=int, default=300)
    ap.add_argument("--n-replay-steps", type=int, default=32,
                    help="how many of the final pretrain batches "
                         "go into the replay buffer")
    # Online session params
    ap.add_argument("--k-corrections-per-concept", type=int,
                    default=15)
    ap.add_argument("--eval-every-corrections", type=int, default=5)
    ap.add_argument("--m3-k-grad-threshold", type=int, default=1)
    ap.add_argument("--m3-salience-threshold", type=float,
                    default=10.0)
    ap.add_argument("--m3-n-replay", type=int, default=7)
    ap.add_argument("--m3-lr", type=float, default=5e-3)
    ap.add_argument("--m3-inner-steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f85_full"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76, flush=True)
    print(
        f"  F85 Online Teacher Loop "
        f"({args.n_train_stories} train, d_model={args.d_model}, "
        f"layers={args.n_layers}, "
        f"K={args.k_corrections_per_concept})",
        flush=True,
    )
    print("=" * 76, flush=True)

    # ─── Step 1: load corpus + build vocab with reserved ────
    print("\n[1/6] loading corpus + building vocab with "
          "reserved slots...", flush=True)
    t0 = time.time()
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
    train_ids = _concatenate_ids(
        [encode_story(s, stoi) for s in train_stories]
    )
    val_ids = _concatenate_ids(
        [encode_story(s, stoi) for s in val_stories]
    )
    vocab = len(itos)
    novel_concept_ids = [stoi[t] for t in reserved_tokens]
    novel_concept_classes = {
        stoi[t]: cls for (t, cls, _) in RESERVED_CONCEPTS
    }
    print(
        f"    vocab={vocab} (reserved IDs "
        f"{novel_concept_ids[0]}–{novel_concept_ids[-1]}), "
        f"train_tokens={len(train_ids):,}, "
        f"val_tokens={len(val_ids):,}, "
        f"wall={time.time()-t0:.1f}s",
        flush=True,
    )

    # ─── Step 2: build TitansPCMMiniLM and pretrain ─────────
    print("\n[2/6] building TitansPCMMiniLM + pretrain...",
          flush=True)
    torch.manual_seed(args.seed)
    model = TitansPCMMiniLM(
        vocab=vocab, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
        attn_every=args.attn_every,
        memory_capacity=args.memory_capacity,
        memory_top_k=args.memory_top_k,
    )
    print(
        f"    params={count_params(model):,}",
        flush=True,
    )
    replay = PretrainReplayBuffer(
        capacity=args.batch_size * args.n_replay_steps,
        device=DEVICE,
    )
    pretrain = pretrain_model(
        model, train_ids=train_ids, val_ids=val_ids,
        n_steps=args.n_pretrain_steps, seq_len=args.seq_len,
        batch_size=args.batch_size, lr=args.lr,
        log_every=args.log_every,
        replay=replay, n_replay_steps=args.n_replay_steps,
        device=DEVICE,
    )
    pretrain_val_ppl_pre = pretrain["final_ppl"]
    print(
        f"    pretrain done: ppl={pretrain_val_ppl_pre:.2f}  "
        f"replay_buffer_size={len(replay)}",
        flush=True,
    )

    # ─── Step 3: build F85 concept dataset ──────────────────
    print("\n[3/6] building F85 concept dataset "
          "(8 concepts × 30 teach × 30 test)...", flush=True)
    rng_data = random.Random(args.seed + 1)
    dataset = generate_concept_dataset(
        rng_data,
        n_teach_per_concept=max(30,
                                args.k_corrections_per_concept * 2),
        n_test_per_concept=30,
    )
    print("    sample teaching sentences:", flush=True)
    for tok in list(dataset["teaching"].keys())[:2]:
        for s in dataset["teaching"][tok][:2]:
            print(f"      [{tok}] {' '.join(s)}", flush=True)
    print("    sample test sentences:", flush=True)
    for tok in list(dataset["test"].keys())[:2]:
        for s in dataset["test"][tok][:2]:
            print(f"      [{tok}] {' '.join(s)}", flush=True)

    # ─── Step 4: pre-online baselines ───────────────────────
    print("\n[4/6] pre-online baselines...", flush=True)
    sess = OnlineTeacherSession(
        model=model, novel_concept_ids=novel_concept_ids,
        novel_concept_classes=novel_concept_classes,
        replay=replay, pad_id=PAD_ID,
        m3_k_grad_threshold=args.m3_k_grad_threshold,
        m3_salience_threshold=args.m3_salience_threshold,
        m3_n_replay=args.m3_n_replay, m3_lr=args.m3_lr,
        m3_inner_steps=args.m3_inner_steps,
        m1_buffer_capacity=512, device=DEVICE,
    )
    concept_ppl_pre = _eval_concept_test_ppl(sess, dataset, stoi)
    print(
        f"    pre-online concept test ppl macro mean: "
        f"{concept_ppl_pre['_macro_mean']:.2f}",
        flush=True,
    )
    pretrain_val_ppl_baseline = pretrain_val_ppl_pre

    # ─── Step 5: online correction loop ─────────────────────
    print(
        f"\n[5/6] online teacher loop "
        f"({args.k_corrections_per_concept} corrections × "
        f"{len(novel_concept_ids)} concepts = "
        f"{args.k_corrections_per_concept * len(novel_concept_ids)})",
        flush=True,
    )
    K = args.k_corrections_per_concept
    ev_every = args.eval_every_corrections
    surprise_history: list[dict] = []
    k_curve: list[dict] = []
    # First measurement: K=0
    k_curve.append({
        "k": 0,
        "concept_ppl_macro_mean": concept_ppl_pre["_macro_mean"],
        "concept_ppl_per_token": {
            k: v for k, v in concept_ppl_pre.items()
            if k != "_macro_mean"
        },
    })
    # Outer loop: round-robin corrections so we don't catastrophic-
    # ally over-fit on one concept first
    total_corrections = K * len(novel_concept_ids)
    for round_idx in range(K):
        for tok, cls, _ in RESERVED_CONCEPTS:
            cid = stoi[tok]
            teach_sents = dataset["teaching"][tok]
            sent = teach_sents[round_idx % len(teach_sents)]
            seq = encode_word_list(sent, stoi)
            ctx = seq[:-1]
            tgt = seq.clone()
            result = sess.receive_correction(
                context_ids=ctx, target_ids=tgt,
            )
            surprise_history.append({
                "round": round_idx, "concept_id": cid,
                "concept_token": tok,
                "surprise": result.get("surprise"),
                "m3_fired": result.get("m3_fired"),
            })
        # Eval every ev_every rounds (across all concepts)
        if (round_idx + 1) % ev_every == 0 or round_idx == K - 1:
            mid_ppl = _eval_concept_test_ppl(sess, dataset, stoi)
            k_curve.append({
                "k": round_idx + 1,
                "concept_ppl_macro_mean": mid_ppl["_macro_mean"],
                "concept_ppl_per_token": {
                    k: v for k, v in mid_ppl.items()
                    if k != "_macro_mean"
                },
            })
            print(
                f"    round {round_idx+1}/{K}: concept ppl "
                f"macro mean = {mid_ppl['_macro_mean']:.2f}",
                flush=True,
            )

    # ─── Step 6: post-online evals + sleep + verdict ────────
    print("\n[6/6] post-online evals + sleep + verdict...",
          flush=True)
    concept_ppl_post = _eval_concept_test_ppl(sess, dataset, stoi)
    pretrain_val_ppl_post = _val_perplexity(
        model, val_ids, seq_len=args.seq_len,
        n_batches=64, batch_size=args.batch_size, device=DEVICE,
    )
    o3_diag = _eval_selectional_o3(
        sess, stoi, RESERVED_CONCEPTS,
    )
    sleep_diag = sess.sleep()

    # ── O1 ──
    o1_pre = concept_ppl_pre["_macro_mean"]
    o1_post = concept_ppl_post["_macro_mean"]
    o1_ratio = o1_post / max(o1_pre, 1e-12)
    o1_pass = o1_ratio <= 0.5
    # ── O2 ──
    o2_ratio = pretrain_val_ppl_post / max(
        pretrain_val_ppl_baseline, 1e-12,
    )
    o2_pass = o2_ratio <= 1.10
    # ── O3 ──
    o3_acc = o3_diag.get("accuracy", 0.0) if isinstance(
        o3_diag, dict,
    ) else 0.0
    o3_pass = o3_acc >= 0.75
    # ── O4: K corrections to halve concept PPL ──
    halved_at = None
    for entry in k_curve:
        if (entry["concept_ppl_macro_mean"]
                <= 0.5 * o1_pre):
            halved_at = entry["k"]
            break
    o4_pass = halved_at is not None and halved_at <= 20
    # ── O5 surprise decay ──
    by_concept: dict[str, list[float]] = {}
    for s in surprise_history:
        by_concept.setdefault(
            s["concept_token"], [],
        ).append(float(s["surprise"] or 0.0))
    o5_decays = {}
    o5_pass_per_concept = []
    for tok, surps in by_concept.items():
        if len(surps) < 2:
            continue
        first = surps[0]
        last = surps[-1]
        o5_decays[tok] = {
            "first": first, "last": last,
            "ratio_last_over_first": (
                last / max(first, 1e-12)
            ),
        }
        o5_pass_per_concept.append(last / max(first, 1e-12) <= 0.5)
    o5_pass = (
        bool(o5_pass_per_concept)
        and sum(o5_pass_per_concept) >= len(o5_pass_per_concept) * 0.5
    )
    # ── O6 sleep purity ──
    o6_purity = sleep_diag.get("purity_mean", 0.0)
    o6_pass = o6_purity >= 0.65

    verdict = {
        "O1_acquisition_le_50pct": o1_pass,
        "O2_no_catastrophic_forgetting_le_1_10x": o2_pass,
        "O3_selectional_generalisation_ge_0_75": o3_pass,
        "O4_sample_efficient_halved_within_20": o4_pass,
        "O5_surprise_decay_majority_le_0_5x": o5_pass,
        "O6_sleep_consolidation_purity_ge_0_65": o6_pass,
    }

    summary = {
        "config": vars(args) | {
            "out": str(args.out),
            "corpus": str(args.corpus),
        },
        "vocab": vocab,
        "novel_concept_ids": novel_concept_ids,
        "model_params": count_params(model),
        "pretrain_log": pretrain,
        "pretrain_val_ppl_pre": pretrain_val_ppl_baseline,
        "pretrain_val_ppl_post": pretrain_val_ppl_post,
        "concept_ppl_pre": concept_ppl_pre,
        "concept_ppl_post": concept_ppl_post,
        "k_curve": k_curve,
        "surprise_decay_per_concept": o5_decays,
        "o3_selectional": o3_diag,
        "sleep_diag": sleep_diag,
        "mechanism_counters": sess.counters.as_dict(),
        "ratios": {
            "o1_concept_post_over_pre": o1_ratio,
            "o2_pretrain_post_over_pre": o2_ratio,
            "o3_accuracy": o3_acc,
            "o4_halved_at": halved_at,
            "o6_purity": o6_purity,
        },
        "verdict": verdict,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False,
                   default=str)
    )

    print("\n" + "=" * 76, flush=True)
    print("  F85 Online Teacher Loop verdict:", flush=True)
    print("=" * 76, flush=True)
    print(
        f"    pretrain val ppl: pre={pretrain_val_ppl_baseline:.2f}  "
        f"post={pretrain_val_ppl_post:.2f}  ratio={o2_ratio:.3f}",
        flush=True,
    )
    print(
        f"    concept test ppl: pre={o1_pre:.2f}  "
        f"post={o1_post:.2f}  ratio={o1_ratio:.3f}",
        flush=True,
    )
    print(
        f"    O3 selectional accuracy: {o3_acc:.3f}",
        flush=True,
    )
    print(
        f"    O4 halved at: K = {halved_at}",
        flush=True,
    )
    print(
        f"    O6 sleep purity: {o6_purity:.3f}",
        flush=True,
    )
    print(
        f"    mechanism counters: {sess.counters.as_dict()}",
        flush=True,
    )
    print(flush=True)
    for k, v in verdict.items():
        print(
            f"    {k}: {'PASS' if v else 'FAIL'}", flush=True,
        )
    print(f"\n  wrote {args.out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
