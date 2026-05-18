"""F86 — Cognitive-concept probe.

After F85 we have a TinyStories-trained PCM that *generates*
text well, but we have not directly verified that the hidden
representations encode the 14 cognitive concept classes
(COLOR, SHAPE, NUMBER, ...) the user wants to ground in vision
later (F87).

This experiment is the cheap probe (no retraining): we extract
the model's final-layer hidden states at concept-word
positions and train a linear classifier (4-fold cross
validation, mirrors F74 ``linear_probe_acc``) to predict
either:

* the **class** of the word (14-way classification), or
* the **identity** within a class (e.g., 12-way among colors).

If the model's representations cluster by cognitive class,
both probes will land far above the chance baseline. That
proves the language substrate has the right *meaning*, not
just the right *words* — and we can move on to F87 visual
grounding.

Falsifiable invariants:

* **C1** all-class probe ≥ 0.70 (chance = 1/14 ≈ 0.07)
* **C2** COLOR sub-probe (12-way) ≥ 0.50 (chance ≈ 0.08)
* **C3** SHAPE sub-probe (10-way) ≥ 0.40 (chance = 0.10)
* **C4** EMOTION sub-probe (13-way) ≥ 0.50 (chance ≈ 0.08)
* **C5** PCM-Hybrid layers' gates show class-correlated
  activation (extension of F80 E1) — Welch's t-test p < 0.01
  between any two classes' gate-retention distributions.

Usage::

    python -m experiments.cognitive_probe_f86 \\
        --corpus outputs/f79_data/tinystories_valid.txt \\
        --out outputs/f86_full
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
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
from pcm.lm import HybridPCMMiniLM, count_params


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────
# Cognitive concept inventory (same as the assessment script)
# ─────────────────────────────────────────────────────────────────


CONCEPT_INVENTORY: dict[str, list[str]] = {
    "COLOR": [
        "red", "blue", "green", "yellow", "black", "white",
        "pink", "orange", "purple", "brown", "gray", "grey",
    ],
    "SHAPE": [
        "circle", "square", "triangle", "star", "heart",
        "line", "dot", "cross", "round", "flat",
    ],
    "NUMBER": [
        "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten",
        "zero", "first", "second", "many", "few",
    ],
    "SPATIAL": [
        "up", "down", "left", "right", "in", "out",
        "behind", "front", "near", "far", "above", "below",
        "inside", "outside", "between", "next",
    ],
    "EMOTION": [
        "happy", "sad", "scared", "angry", "surprised",
        "tired", "excited", "calm", "afraid", "glad",
        "worried", "proud", "shy",
    ],
    "FAMILY": [
        "mom", "dad", "mother", "father",
        "brother", "sister", "grandma", "grandpa",
        "baby", "friend", "mommy", "daddy",
    ],
    "BODY": [
        "hand", "foot", "head", "eye", "ear", "mouth",
        "nose", "arm", "leg", "finger", "hair", "face",
    ],
    "ACTION": [
        "push", "pull", "throw", "catch", "kick", "hug",
        "draw", "build", "run", "walk", "jump", "sleep",
        "eat", "drink", "play", "give", "take", "make",
        "look", "see", "hear", "say", "tell", "ask",
    ],
    "SIZE": [
        "big", "small", "little", "large", "tiny",
        "huge", "short", "tall", "long", "wide",
    ],
    "TIME": [
        "today", "tomorrow", "yesterday", "morning",
        "night", "day", "now", "later", "soon",
        "after", "before",
    ],
    "QUANTIFIER": [
        "all", "some", "none", "every", "any", "more",
        "less", "most", "least",
    ],
    "QUESTION": [
        "what", "where", "who", "why", "when", "how",
        "which",
    ],
    "NEGATION": [
        "no", "not", "never", "nothing", "nobody",
    ],
}


# ─────────────────────────────────────────────────────────────────
# Hidden-state extraction
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def _collect_hidden_states_at_concept_words(
    model: HybridPCMMiniLM, val_ids: torch.Tensor,
    target_token_ids: dict[int, tuple[str, str]],
    *, seq_len: int, n_max_per_word: int, device: str,
) -> dict[int, torch.Tensor]:
    """For each target token id, collect the model's final-layer
    hidden state at every position where that token occurs in
    the validation corpus (up to ``n_max_per_word`` occurrences).

    Returns dict ``{token_id: hidden_tensor (N, D)}``.
    """
    model.eval()
    # Bucket positions by token id
    positions: dict[int, list[tuple[int, int]]] = defaultdict(list)
    val_len = val_ids.shape[0]
    for i in range(val_len):
        tid = int(val_ids[i].item())
        if tid in target_token_ids:
            positions[tid].append((i,))
    print(f"    found positions: "
          f"{sum(len(v) for v in positions.values())} total "
          f"across {len(positions)} target tokens", flush=True)
    # Cap per token + ensure we have a seq_len-window around
    out: dict[int, torch.Tensor] = {}
    for tid in list(positions.keys()):
        rng = random.Random(int(tid))
        all_pos = positions[tid]
        rng.shuffle(all_pos)
        take = []
        for (i,) in all_pos:
            start = max(0, i - seq_len // 2)
            end = start + seq_len
            if end > val_len:
                continue
            pos_in_window = i - start
            take.append((start, pos_in_window))
            if len(take) >= n_max_per_word:
                break
        if not take:
            continue
        # Batch them
        batch_x = []
        batch_pos = []
        for (start, pos) in take:
            batch_x.append(val_ids[start:start + seq_len])
            batch_pos.append(pos)
        xs = torch.stack(batch_x).to(device)
        # Get hidden states
        h = model.hidden_states(xs)  # (B, L, D)
        for i, p in enumerate(batch_pos):
            arr = h[i, p].detach().cpu()
            out.setdefault(tid, []).append(arr)
    return {
        tid: torch.stack(v) for tid, v in out.items()
    }


# ─────────────────────────────────────────────────────────────────
# Linear probe (k-fold)
# ─────────────────────────────────────────────────────────────────


def _train_linear_probe(
    x: torch.Tensor, y: torch.Tensor, *,
    n_classes: int, n_folds: int = 4,
    epochs: int = 50, lr: float = 1e-2,
    weight_decay: float = 1e-3,
    device: str = "cpu",
) -> dict:
    """Train a linear probe with k-fold CV. Returns mean
    accuracy."""
    n = x.shape[0]
    if n < n_folds * 2:
        return {"acc_mean": 0.0, "acc_std": 0.0, "n": n}
    indices = list(range(n))
    rng = random.Random(0)
    rng.shuffle(indices)
    fold_size = n // n_folds
    accs = []
    for f in range(n_folds):
        test_idx = indices[f * fold_size:(f + 1) * fold_size]
        train_idx = [i for i in indices if i not in test_idx]
        if not train_idx or not test_idx:
            continue
        xt = x[train_idx].to(device)
        yt = y[train_idx].to(device)
        xe = x[test_idx].to(device)
        ye = y[test_idx].to(device)
        clf = torch.nn.Linear(x.shape[1], n_classes).to(device)
        opt = torch.optim.AdamW(
            clf.parameters(), lr=lr, weight_decay=weight_decay,
        )
        with torch.enable_grad():
            for _ in range(epochs):
                logits = clf(xt)
                loss = F.cross_entropy(logits, yt)
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            pred = clf(xe).argmax(dim=-1)
            acc = (pred == ye).float().mean().item()
        accs.append(acc)
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "n": n, "n_classes": n_classes,
        "chance": 1.0 / n_classes,
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
    ap.add_argument("--n-pretrain-steps", type=int, default=3000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--log-every", type=int, default=300)
    ap.add_argument("--n-max-per-word", type=int, default=80)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f86_full"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76, flush=True)
    print(
        "  F86 Cognitive Concept Probe "
        f"(d_model={args.d_model}, layers={args.n_layers}, "
        f"max-occurrences/word={args.n_max_per_word})",
        flush=True,
    )
    print("=" * 76, flush=True)

    # ─── Step 1: load corpus + vocab ─────────────────────────
    print("\n[1/4] loading corpus + building vocab...", flush=True)
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
    stoi, itos = build_vocab(
        train_stories, vocab_cap=args.vocab_cap,
    )
    train_ids = _concatenate_ids(
        [encode_story(s, stoi) for s in train_stories]
    )
    val_ids = _concatenate_ids(
        [encode_story(s, stoi) for s in val_stories]
    )
    vocab = len(itos)
    print(
        f"    vocab={vocab}, train_tokens={len(train_ids):,}, "
        f"val_tokens={len(val_ids):,}, "
        f"wall={time.time()-t0:.1f}s",
        flush=True,
    )

    # ─── Step 2: pretrain HybridPCMMiniLM ────────────────────
    print(
        f"\n[2/4] pretrain HybridPCMMiniLM "
        f"({args.n_pretrain_steps} steps)...",
        flush=True,
    )
    torch.manual_seed(args.seed)
    model = HybridPCMMiniLM(
        vocab=vocab, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
        attn_every=args.attn_every,
    )
    model.to(DEVICE)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4,
    )
    rng_train = torch.Generator(device="cpu").manual_seed(7777)
    t_train = time.time()
    model.train()
    for step in range(1, args.n_pretrain_steps + 1):
        xs, ys = _sample_seq_batch(
            train_ids, seq_len=args.seq_len,
            batch_size=args.batch_size, rng=rng_train,
        )
        xs = xs.to(DEVICE)
        ys = ys.to(DEVICE)
        logits = model(xs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            ys.reshape(-1), ignore_index=PAD_ID,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == args.n_pretrain_steps:
            val_ppl = _val_perplexity(
                model, val_ids, seq_len=args.seq_len,
                n_batches=16, batch_size=args.batch_size,
                device=DEVICE,
            )
            print(
                f"    step {step:4d}/{args.n_pretrain_steps}  "
                f"loss={loss.item():.3f}  val_ppl={val_ppl:.1f}  "
                f"wall={time.time()-t_train:.1f}s",
                flush=True,
            )
            model.train()
    final_pretrain_ppl = _val_perplexity(
        model, val_ids, seq_len=args.seq_len,
        n_batches=64, batch_size=args.batch_size, device=DEVICE,
    )
    print(
        f"    pretrain done: val_ppl={final_pretrain_ppl:.2f}",
        flush=True,
    )

    # ─── Step 3: collect hidden states for concept words ────
    print(
        "\n[3/4] extracting hidden states at concept-word "
        "positions...",
        flush=True,
    )
    # Build {token_id: (class, word)} for words present in vocab
    target_tokens: dict[int, tuple[str, str]] = {}
    word_to_class: dict[str, str] = {}
    for cls, words in CONCEPT_INVENTORY.items():
        for w in words:
            if w in stoi:
                target_tokens[stoi[w]] = (cls, w)
                word_to_class[w] = cls
    print(f"    target tokens: {len(target_tokens)}", flush=True)
    hidden_by_token = _collect_hidden_states_at_concept_words(
        model, val_ids, target_tokens,
        seq_len=args.seq_len,
        n_max_per_word=args.n_max_per_word, device=DEVICE,
    )
    n_samples_collected = sum(
        h.shape[0] for h in hidden_by_token.values()
    )
    print(
        f"    collected {n_samples_collected} hidden-state "
        f"samples across {len(hidden_by_token)} tokens",
        flush=True,
    )

    # ─── Step 4: train probes (C1 + C2/C3/C4 sub-class) ─────
    print("\n[4/4] training linear probes...", flush=True)
    # C1: all-14-class probe
    X_all = []
    y_all = []
    class_to_idx = {
        cls: i for i, cls in enumerate(CONCEPT_INVENTORY.keys())
    }
    for tid, h in hidden_by_token.items():
        cls, _ = target_tokens[tid]
        cls_idx = class_to_idx[cls]
        X_all.append(h)
        y_all.append(
            torch.full((h.shape[0],), cls_idx, dtype=torch.long)
        )
    X_all = torch.cat(X_all, dim=0)
    y_all = torch.cat(y_all, dim=0)
    c1_diag = _train_linear_probe(
        X_all, y_all, n_classes=len(CONCEPT_INVENTORY),
        device=DEVICE,
    )
    print(
        f"    C1 all-{len(CONCEPT_INVENTORY)}-class probe: "
        f"acc = {c1_diag['acc_mean']:.3f} ± {c1_diag['acc_std']:.3f}  "
        f"(chance = {c1_diag['chance']:.3f}, "
        f"n_samples = {c1_diag['n']})",
        flush=True,
    )

    # Sub-class probes
    sub_diags: dict[str, dict] = {}
    for target_cls in ["COLOR", "SHAPE", "EMOTION"]:
        words = CONCEPT_INVENTORY[target_cls]
        words_in_vocab = [w for w in words if w in stoi]
        if not words_in_vocab:
            continue
        word_to_idx = {w: i for i, w in enumerate(words_in_vocab)}
        Xs = []
        ys = []
        for w in words_in_vocab:
            tid = stoi[w]
            if tid not in hidden_by_token:
                continue
            h = hidden_by_token[tid]
            Xs.append(h)
            ys.append(
                torch.full(
                    (h.shape[0],), word_to_idx[w],
                    dtype=torch.long,
                )
            )
        if not Xs:
            continue
        Xs = torch.cat(Xs, dim=0)
        ys = torch.cat(ys, dim=0)
        diag = _train_linear_probe(
            Xs, ys, n_classes=len(words_in_vocab),
            device=DEVICE,
        )
        sub_diags[target_cls] = diag
        print(
            f"    {target_cls:10s} {len(words_in_vocab)}-way: "
            f"acc = {diag['acc_mean']:.3f} ± {diag['acc_std']:.3f}  "
            f"(chance = {diag['chance']:.3f}, "
            f"n_samples = {diag['n']})",
            flush=True,
        )

    # ─── C5: between-class gate-distribution difference ─────
    # Pick a pair of classes; check that PCM-Gated layers'
    # gate-retention distributions differ significantly
    print(
        "\n[C5] PCM-gate retention contrast between two classes "
        "(COLOR vs EMOTION)...",
        flush=True,
    )

    @torch.no_grad()
    def _gate_retention_at_positions(
        token_ids_per_class: dict[str, list[int]],
    ) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {cls: [] for cls in
                                        token_ids_per_class}
        val_len = val_ids.shape[0]
        rng = random.Random(0)
        for cls, tids in token_ids_per_class.items():
            tids_set = set(tids)
            # Sample positions for this class
            pos: list[int] = []
            for i in range(val_len):
                if int(val_ids[i].item()) in tids_set:
                    pos.append(i)
            rng.shuffle(pos)
            for i in pos[:200]:
                start = max(0, i - args.seq_len // 2)
                end = start + args.seq_len
                if end > val_len:
                    continue
                pos_in_window = i - start
                xs = val_ids[start:end].unsqueeze(0).to(DEVICE)
                # Walk the PCM layers explicitly
                slots = model.tok_emb(xs)
                gates_at_pos = []
                for layer, kind in zip(
                    model.layers, model.layer_kinds,
                ):
                    if kind == "pcm":
                        gates = layer.compute_gates(slots)
                        gates_at_pos.append(
                            gates[0, pos_in_window].mean().item()
                        )
                    slots = slots + layer(slots)
                if gates_at_pos:
                    out[cls].append(
                        float(np.mean(gates_at_pos))
                    )
        return out

    color_ids = [
        stoi[w] for w in CONCEPT_INVENTORY["COLOR"] if w in stoi
    ]
    emotion_ids = [
        stoi[w] for w in CONCEPT_INVENTORY["EMOTION"]
        if w in stoi
    ]
    gates_dict = _gate_retention_at_positions({
        "COLOR": color_ids, "EMOTION": emotion_ids,
    })
    color_g = np.array(gates_dict.get("COLOR", []))
    emotion_g = np.array(gates_dict.get("EMOTION", []))

    def _welch_p(a: np.ndarray, b: np.ndarray) -> dict:
        if len(a) < 2 or len(b) < 2:
            return {"status": "insufficient", "p": 1.0}
        ma, mb = a.mean(), b.mean()
        va, vb = a.var(ddof=1), b.var(ddof=1)
        na, nb = len(a), len(b)
        se = math.sqrt(va / na + vb / nb)
        if se < 1e-12:
            return {"status": "zero_var", "p": 1.0}
        t = (ma - mb) / se
        # df via Welch-Satterthwaite
        df = (
            (va / na + vb / nb) ** 2 / max(
                (va / na) ** 2 / max(na - 1, 1)
                + (vb / nb) ** 2 / max(nb - 1, 1),
                1e-12,
            )
        )

        def _norm_cdf(z: float) -> float:
            return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

        p = 2.0 * (1.0 - _norm_cdf(abs(t)))
        return {
            "status": "ok", "mean_a": float(ma),
            "mean_b": float(mb),
            "t": float(t), "df": float(df), "p": float(p),
        }

    c5_diag = _welch_p(color_g, emotion_g)
    print(
        f"    COLOR mean gate = {color_g.mean():.4f} "
        f"(n={len(color_g)})",
        flush=True,
    )
    print(
        f"    EMOTION mean gate = {emotion_g.mean():.4f} "
        f"(n={len(emotion_g)})",
        flush=True,
    )
    print(f"    Welch t-test: {c5_diag}", flush=True)

    # ─── Verdict ─────────────────────────────────────────────
    verdict = {
        "C1_all_class_probe_ge_0_70": (
            c1_diag.get("acc_mean", 0.0) >= 0.70
        ),
        "C2_color_subprobe_ge_0_50": (
            sub_diags.get("COLOR", {}).get("acc_mean", 0.0)
            >= 0.50
        ),
        "C3_shape_subprobe_ge_0_40": (
            sub_diags.get("SHAPE", {}).get("acc_mean", 0.0)
            >= 0.40
        ),
        "C4_emotion_subprobe_ge_0_50": (
            sub_diags.get("EMOTION", {}).get("acc_mean", 0.0)
            >= 0.50
        ),
        "C5_color_vs_emotion_gate_p_lt_0_01": (
            c5_diag.get("p", 1.0) < 0.01
        ),
    }
    summary = {
        "config": vars(args) | {
            "out": str(args.out),
            "corpus": str(args.corpus),
        },
        "vocab": vocab,
        "n_target_tokens": len(target_tokens),
        "model_params": count_params(model),
        "pretrain_val_ppl": final_pretrain_ppl,
        "probes": {
            "C1_all": c1_diag,
            "sub": sub_diags,
            "C5_gate_color_vs_emotion": c5_diag,
        },
        "verdict": verdict,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False,
                   default=str)
    )

    print("\n" + "=" * 76, flush=True)
    print("  F86 Cognitive Concept Probe verdict:", flush=True)
    print("=" * 76, flush=True)
    for k, v in verdict.items():
        print(
            f"    {k}: {'PASS' if v else 'FAIL'}", flush=True,
        )
    print(f"\n  wrote {args.out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
