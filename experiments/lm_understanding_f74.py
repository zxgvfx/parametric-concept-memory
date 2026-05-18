"""F74 — does PCM **understand** each word, or just predict next?

Direct comparison of a PCM-style LM (F62 ``UniversalCombiner``
instead of self-attention) against a parameter-matched scratch
Transformer LM (GPT-style baseline) on a synthetic language
with **known ground-truth semantics** (word → semantic class is
hard-coded, sentence grammar respects selectional restrictions).

This is the cleanest possible test of the user's design
intuition::

    LLM learns next-token *probability*.
    PCM should learn each word's *meaning*.

Five falsifiable invariants:

* **U1** PCM perplexity within 1.25× of GPT (PCM is not much
  worse at next-token prediction despite no self-attention).
* **U2** *Linear probe* on token embeddings recovers semantic
  class — PCM ≥ GPT by ≥ 5pp.
* **U3** *Selectional-violation* detection: PCM assigns lower
  prob to type-violating sentences than GPT does (PCM violation
  gap ≥ GPT violation gap).
* **U4** *Compositional generalisation*: train with 20% of valid
  ``(verb, object)`` pairs held out; test perplexity on held-
  out pairs. PCM held-out / train ppl ratio ≤ GPT's.
* **U5** *Sample-efficiency curve*: at small data (e.g. 5K
  sentences) PCM probe accuracy ≥ GPT probe accuracy + 5pp.

Total compute: ~10 minutes per scale × 4 scales × 2 models =
~80 min on RTX 3070. Self-contained — no external data.

Usage::

    python -m experiments.lm_understanding_f74 \\
        --scales 5000,20000,80000 \\
        --out outputs/f74_full
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

from pcm.lm import (
    GPTMiniLM,
    PCMMiniLM,
    build_matched_pair,
    count_params,
)
from pcm.lm_synthetic import (
    NOUNS_BY_CLASS,
    VERBS_BY_CLASS,
    Tokenizer,
    all_valid_verb_object_pairs,
    generate_corpus,
    generate_sentence,
    generate_violation_pair,
    noun_class_of,
    verb_class_of,
)


__all__ = ["main"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────
# Data tensorisation
# ─────────────────────────────────────────────────────────────────


def tokenise_corpus(
    tok: Tokenizer, corpus: list[list[str]], *, max_len: int = 12,
) -> torch.Tensor:
    ids = [tok.encode(s, max_len=max_len) for s in corpus]
    return torch.tensor(ids, dtype=torch.long)


def make_xy(seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal LM ``(x = seq[:-1], y = seq[1:])`` shift."""
    return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()


# ─────────────────────────────────────────────────────────────────
# Training
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
            x, y = make_xy(batch)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                ignore_index=pad_id,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def held_out_perplexity(
    model: nn.Module, seq: torch.Tensor, *,
    pad_id: int, batch_size: int = 256,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    for i in range(0, seq.shape[0], batch_size):
        batch = seq[i:i + batch_size].to(DEVICE)
        x, y = make_xy(batch)
        logits = model(x)
        loss_elem = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            ignore_index=pad_id, reduction="sum",
        )
        n = (y != pad_id).sum().item()
        total_loss += float(loss_elem.item())
        total_count += int(n)
    if total_count == 0:
        return float("nan")
    return math.exp(total_loss / total_count)


# ─────────────────────────────────────────────────────────────────
# U2 — Linear probe on noun semantic class
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def linear_probe_acc(
    model: nn.Module, tok: Tokenizer, *,
    classes: tuple[str, ...] = ("ANIMAL", "FOOD", "PERSON", "PLACE"),
    n_train: int | None = None,
    n_folds: int = 5,
) -> dict:
    """Train a multinomial logistic-regression probe on the
    model's token embeddings to predict noun semantic class.

    Uses *k*-fold cross-validation across the 15 nouns per
    class so the result is robust. Returns:

    * ``embedding_probe_acc`` — mean fold accuracy on raw
      token embeddings (input-side).
    * ``hidden_probe_acc`` — mean fold accuracy on **mean-
      pooled hidden states** at the last layer (output-side).
      For each noun ``n``, we form short sentences like
      ``["the", n, "is", "happy", ".", "<eos>"]`` and average
      the model's final-layer hidden state over the *noun
      position only*. This isolates how each model encodes the
      noun in the residual stream after context.
    """
    model.eval()
    emb = model.token_embeddings().cpu()

    nouns: list[tuple[str, int]] = []
    for ci, cls in enumerate(classes):
        for w in NOUNS_BY_CLASS[cls]:
            nouns.append((w, ci))
    rng = random.Random(0)
    rng.shuffle(nouns)

    # Embedding probe
    embeds = torch.stack([emb[tok.stoi[w]] for w, _ in nouns])
    labels = torch.tensor([c for _, c in nouns], dtype=torch.long)

    def _kfold_acc(X: torch.Tensor, y: torch.Tensor) -> float:
        # ``X`` from the frozen LM is detached; the probe itself
        # needs grad enabled. We're inside a no_grad context
        # (parent function decorator), so wrap probe training
        # in an explicit enable_grad block.
        n = X.shape[0]
        fold_size = n // n_folds
        accs = []
        X = X.detach()
        for k in range(n_folds):
            test_lo = k * fold_size
            test_hi = (k + 1) * fold_size if k < n_folds - 1 else n
            test_mask = torch.zeros(n, dtype=torch.bool)
            test_mask[test_lo:test_hi] = True
            Xtr, Xte = X[~test_mask], X[test_mask]
            ytr, yte = y[~test_mask], y[test_mask]
            probe = nn.Linear(X.shape[1], len(classes))
            opt = torch.optim.AdamW(probe.parameters(), lr=5e-2,
                                     weight_decay=1e-3)
            with torch.enable_grad():
                for _ in range(400):
                    logits = probe(Xtr)
                    loss = F.cross_entropy(logits, ytr)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                pred = probe(Xte).argmax(-1)
                accs.append(float((pred == yte).float().mean().item()))
        return sum(accs) / len(accs)

    emb_acc = _kfold_acc(embeds, labels)

    # Hidden-state probe: gather per-noun representation by
    # running short probe sentences through the model.
    hidden_vecs = []
    for w, _ in nouns:
        # Probe sentence: "the {w} is happy . <eos>"
        ids = tok.encode(["the", w, "is", "happy", "."], max_len=12)
        x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        h = model.hidden_states(x)[0]  # (L, D)
        # Noun position is at index 1 ("the" is 0)
        hidden_vecs.append(h[1].cpu())
    hidden = torch.stack(hidden_vecs, dim=0)
    hidden_acc = _kfold_acc(hidden, labels)

    return {
        "embedding_probe_acc": emb_acc,
        "hidden_probe_acc": hidden_acc,
    }


# ─────────────────────────────────────────────────────────────────
# U3 — Selectional violation detection
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def selectional_violation_gap(
    model: nn.Module, tok: Tokenizer, *,
    n_pairs: int = 200, seed: int = 7777, max_len: int = 12,
) -> dict:
    model.eval()
    rng = random.Random(seed)
    gaps = []
    valid_pps = []
    violation_pps = []
    n_succeeded = 0
    for _ in range(n_pairs):
        try:
            valid, vio = generate_violation_pair(rng)
        except RuntimeError:
            continue
        v_ids = tok.encode(valid, max_len=max_len)
        b_ids = tok.encode(vio, max_len=max_len)
        x_v = torch.tensor([v_ids[:-1]], dtype=torch.long, device=DEVICE)
        y_v = torch.tensor([v_ids[1:]], dtype=torch.long, device=DEVICE)
        x_b = torch.tensor([b_ids[:-1]], dtype=torch.long, device=DEVICE)
        y_b = torch.tensor([b_ids[1:]], dtype=torch.long, device=DEVICE)
        logits_v = model(x_v)
        logits_b = model(x_b)
        loss_v = F.cross_entropy(
            logits_v.reshape(-1, logits_v.shape[-1]),
            y_v.reshape(-1), ignore_index=tok.pad_id,
            reduction="sum",
        )
        loss_b = F.cross_entropy(
            logits_b.reshape(-1, logits_b.shape[-1]),
            y_b.reshape(-1), ignore_index=tok.pad_id,
            reduction="sum",
        )
        n_v = int((y_v != tok.pad_id).sum().item())
        n_b = int((y_b != tok.pad_id).sum().item())
        ppl_v = math.exp(float(loss_v.item()) / max(n_v, 1))
        ppl_b = math.exp(float(loss_b.item()) / max(n_b, 1))
        valid_pps.append(ppl_v)
        violation_pps.append(ppl_b)
        gaps.append(ppl_b - ppl_v)
        n_succeeded += 1
    return {
        "mean_valid_ppl": sum(valid_pps) / max(len(valid_pps), 1),
        "mean_violation_ppl": sum(violation_pps) / max(len(violation_pps), 1),
        "mean_violation_minus_valid_ppl":
            sum(gaps) / max(len(gaps), 1),
        "fraction_violation_higher_ppl":
            sum(1 for g in gaps if g > 0) / max(len(gaps), 1),
        "n_pairs": n_succeeded,
    }


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────


def _epochs_for_scale(n_sentences: int) -> int:
    """Heuristic epochs: more for small data so we don't
    undertrain it."""
    if n_sentences <= 5_000:
        return 25
    if n_sentences <= 20_000:
        return 15
    if n_sentences <= 80_000:
        return 8
    return 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=str, default="5000,20000,80000")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-violation-pairs", type=int, default=300)
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="fraction of (V, O) pairs held out for U4")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f74_full"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scales = [int(s) for s in args.scales.split(",")]

    tok = Tokenizer()
    print("=" * 76)
    print(f"  F74 PCM-vs-Transformer word-meaning understanding test")
    print(f"  vocab={tok.vocab_size}, d_model={args.d_model}, "
          f"n_layers={args.n_layers}, scales={scales}")
    print("=" * 76)

    # ─── Build U4 held-out (V, O) pairs ───────────────────────
    all_vo = all_valid_verb_object_pairs()
    rng = random.Random(31337)
    rng.shuffle(all_vo)
    n_holdout = int(args.holdout_frac * len(all_vo))
    holdout_pairs = set(all_vo[:n_holdout])
    print(f"\nU4 holdout: {n_holdout} / {len(all_vo)} (V,O) pairs "
          f"reserved for compositional test")

    # ─── Held-out test corpora ────────────────────────────────
    print(f"\nGenerating held-out test corpus ({args.n_test} sentences)...")
    test_corpus = generate_corpus(args.n_test, seed=99999)
    test_seq = tokenise_corpus(tok, test_corpus, max_len=args.max_len)
    print(f"  test_seq shape: {tuple(test_seq.shape)}")

    print(f"\nGenerating held-out compositional test corpus "
          f"(sentences using ONLY held-out V-O pairs)...")
    # Force-generate sentences that use held-out pairs by
    # picking from the pair list and constructing SVO templates.
    holdout_test_corpus = _generate_holdout_test_corpus(
        list(holdout_pairs), n_per_pair=4, rng_seed=88888,
    )
    holdout_test_seq = tokenise_corpus(
        tok, holdout_test_corpus, max_len=args.max_len,
    )
    print(f"  holdout_test_seq shape: {tuple(holdout_test_seq.shape)}")

    rows: list[dict] = []
    for scale in scales:
        print("\n" + "=" * 76)
        print(f"  Scale: {scale} training sentences")
        print("=" * 76)
        print(f"  Generating training corpus...")
        train_corpus = generate_corpus(
            scale, seed=42 + scale,
            holdout_verb_object_pairs=holdout_pairs,
        )
        train_seq = tokenise_corpus(tok, train_corpus, max_len=args.max_len)
        epochs = _epochs_for_scale(scale)
        print(f"  train tokens: {int((train_seq != tok.pad_id).sum())}  "
              f"epochs: {epochs}")

        # Build matched pair
        gpt, pcm, diag = build_matched_pair(
            vocab=tok.vocab_size,
            d_model=args.d_model, n_layers=args.n_layers,
            n_heads=args.n_heads,
            max_len=max(args.max_len, 32),
        )
        gpt = gpt.to(DEVICE)
        pcm = pcm.to(DEVICE)
        print(f"  matched params: GPT={diag['gpt_params']:,}  "
              f"PCM={diag['pcm_params']:,}  "
              f"(ratio {diag['ratio_pcm_over_gpt']:.3f})")

        # Train both
        t0 = time.time()
        _train_one(
            gpt, train_seq, epochs=epochs,
            batch_size=args.batch_size, lr=args.lr,
            pad_id=tok.pad_id, rng_seed=11,
        )
        gpt_wall = time.time() - t0
        t0 = time.time()
        _train_one(
            pcm, train_seq, epochs=epochs,
            batch_size=args.batch_size, lr=args.lr,
            pad_id=tok.pad_id, rng_seed=22,
        )
        pcm_wall = time.time() - t0
        print(f"  trained GPT in {gpt_wall:.1f}s, PCM in {pcm_wall:.1f}s")

        # U1 — perplexity
        gpt_ppl = held_out_perplexity(gpt, test_seq, pad_id=tok.pad_id)
        pcm_ppl = held_out_perplexity(pcm, test_seq, pad_id=tok.pad_id)
        print(f"  U1 held-out ppl  : GPT {gpt_ppl:.3f}  PCM {pcm_ppl:.3f}  "
              f"(PCM/GPT = {pcm_ppl/gpt_ppl:.3f})")

        # U2 — linear probe
        gpt_probe = linear_probe_acc(gpt, tok)
        pcm_probe = linear_probe_acc(pcm, tok)
        print(f"  U2 embedding probe acc: "
              f"GPT {gpt_probe['embedding_probe_acc']:.3f}  "
              f"PCM {pcm_probe['embedding_probe_acc']:.3f}  "
              f"(gap PCM−GPT = "
              f"{pcm_probe['embedding_probe_acc'] - gpt_probe['embedding_probe_acc']:+.3f})")
        print(f"  U2 hidden    probe acc: "
              f"GPT {gpt_probe['hidden_probe_acc']:.3f}  "
              f"PCM {pcm_probe['hidden_probe_acc']:.3f}  "
              f"(gap PCM−GPT = "
              f"{pcm_probe['hidden_probe_acc'] - gpt_probe['hidden_probe_acc']:+.3f})")

        # U3 — violation gap
        gpt_vio = selectional_violation_gap(
            gpt, tok, n_pairs=args.n_violation_pairs,
            max_len=args.max_len,
        )
        pcm_vio = selectional_violation_gap(
            pcm, tok, n_pairs=args.n_violation_pairs,
            max_len=args.max_len,
        )
        print(f"  U3 violation > valid ppl : "
              f"GPT mean gap {gpt_vio['mean_violation_minus_valid_ppl']:+.3f} "
              f"(frac up {gpt_vio['fraction_violation_higher_ppl']:.3f})  "
              f"PCM mean gap "
              f"{pcm_vio['mean_violation_minus_valid_ppl']:+.3f} "
              f"(frac up {pcm_vio['fraction_violation_higher_ppl']:.3f})")

        # U4 — held-out compositional ppl
        gpt_holdout = held_out_perplexity(
            gpt, holdout_test_seq, pad_id=tok.pad_id,
        )
        pcm_holdout = held_out_perplexity(
            pcm, holdout_test_seq, pad_id=tok.pad_id,
        )
        gpt_ratio = gpt_holdout / max(gpt_ppl, 1e-6)
        pcm_ratio = pcm_holdout / max(pcm_ppl, 1e-6)
        print(f"  U4 holdout/train ppl ratio: "
              f"GPT {gpt_holdout:.3f}/{gpt_ppl:.3f}={gpt_ratio:.3f}  "
              f"PCM {pcm_holdout:.3f}/{pcm_ppl:.3f}={pcm_ratio:.3f}")

        rows.append({
            "scale": scale,
            "epochs": epochs,
            "params": diag,
            "U1_gpt_ppl": gpt_ppl,
            "U1_pcm_ppl": pcm_ppl,
            "U2_gpt_probe": gpt_probe,
            "U2_pcm_probe": pcm_probe,
            "U3_gpt_violation": gpt_vio,
            "U3_pcm_violation": pcm_vio,
            "U4_gpt_holdout_ppl": gpt_holdout,
            "U4_pcm_holdout_ppl": pcm_holdout,
            "U4_gpt_ratio": gpt_ratio,
            "U4_pcm_ratio": pcm_ratio,
            "wall_gpt_s": gpt_wall,
            "wall_pcm_s": pcm_wall,
        })

    # ─── Verdict ─────────────────────────────────────────────
    # Use the *largest* scale as the canonical comparison; report
    # all scales but grade on the largest.
    final = rows[-1]
    smallest = rows[0]

    verdict = {
        # U1: PCM ppl ≤ 1.25 × GPT ppl at largest scale
        "U1_ppl_within_25pct_pass":
            final["U1_pcm_ppl"] <= final["U1_gpt_ppl"] * 1.25,
        # U2: PCM embedding probe ≥ GPT embedding probe + 5pp at largest scale
        "U2_embedding_probe_pass":
            (final["U2_pcm_probe"]["embedding_probe_acc"]
             >= final["U2_gpt_probe"]["embedding_probe_acc"] + 0.05),
        # U3: PCM violation gap ≥ GPT violation gap at largest scale
        "U3_violation_detection_pass":
            (final["U3_pcm_violation"]["mean_violation_minus_valid_ppl"]
             >= final["U3_gpt_violation"]["mean_violation_minus_valid_ppl"]),
        # U4: PCM holdout/train ratio ≤ GPT's at largest scale
        "U4_compositional_pass":
            final["U4_pcm_ratio"] <= final["U4_gpt_ratio"],
        # U5: PCM probe ≥ GPT probe + 5pp at the smallest scale
        "U5_sample_efficiency_pass":
            (smallest["U2_pcm_probe"]["embedding_probe_acc"]
             >= smallest["U2_gpt_probe"]["embedding_probe_acc"] + 0.05),
    }

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "vocab_size": tok.vocab_size,
        "n_holdout_pairs": len(holdout_pairs),
        "rows": rows,
        "verdict": verdict,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print("  F74 verdict (PCM understands vs GPT predicts):")
    print("=" * 76)
    v = verdict
    print(f"  U1 PCM ppl ≤ 1.25× GPT ppl   : "
          f"PCM {final['U1_pcm_ppl']:.3f} vs GPT {final['U1_gpt_ppl']:.3f}  "
          f"[{'PASS' if v['U1_ppl_within_25pct_pass'] else 'FAIL'}]")
    print(f"  U2 PCM emb probe − GPT ≥ 5pp : "
          f"{final['U2_pcm_probe']['embedding_probe_acc'] - final['U2_gpt_probe']['embedding_probe_acc']:+.3f}  "
          f"[{'PASS' if v['U2_embedding_probe_pass'] else 'FAIL'}]")
    print(f"  U3 PCM violation gap ≥ GPT   : "
          f"PCM {final['U3_pcm_violation']['mean_violation_minus_valid_ppl']:+.3f}  "
          f"GPT {final['U3_gpt_violation']['mean_violation_minus_valid_ppl']:+.3f}  "
          f"[{'PASS' if v['U3_violation_detection_pass'] else 'FAIL'}]")
    print(f"  U4 PCM holdout-ratio ≤ GPT   : "
          f"PCM {final['U4_pcm_ratio']:.3f}  GPT {final['U4_gpt_ratio']:.3f}  "
          f"[{'PASS' if v['U4_compositional_pass'] else 'FAIL'}]")
    print(f"  U5 small-data PCM probe ≥ +5pp: "
          f"@{smallest['scale']} gap "
          f"{smallest['U2_pcm_probe']['embedding_probe_acc'] - smallest['U2_gpt_probe']['embedding_probe_acc']:+.3f}  "
          f"[{'PASS' if v['U5_sample_efficiency_pass'] else 'FAIL'}]")
    print(f"\n  wrote {args.out / 'summary.json'}")


def _generate_holdout_test_corpus(
    pairs: list[tuple[str, str]], *, n_per_pair: int = 4,
    rng_seed: int = 0,
) -> list[list[str]]:
    """Generate SVO sentences explicitly using each held-out
    (verb, object) pair so we can measure compositional
    generalisation directly. Each pair gets ``n_per_pair``
    sentence instances with random subjects.
    """
    from pcm.lm_synthetic import _SELECTIONAL
    rng = random.Random(rng_seed)
    out = []
    for v, o in pairs:
        v_cls = verb_class_of(v)
        sel = _SELECTIONAL[v_cls]
        for _ in range(n_per_pair):
            subj_cls = rng.choice(sel["subject_classes"])
            subj = rng.choice(NOUNS_BY_CLASS[subj_cls])
            out.append([subj, v, o, "."])
    return out


if __name__ == "__main__":
    main()
