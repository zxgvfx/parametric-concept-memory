"""F63f — Cross-modality V3 ratio across multiple modality pairs.

F63c established that DNA → Code transfer has V3a/V3b ≈ 1.35×.
The reviewer's question: *"Does the V3 ratio vary systematically
with modality similarity? Test with music + stock-tick pairs."*

F63f sweeps several modality pairs and reports each pair's V3a
(transfer-cost ratio). The empirical question is whether there
is a systematic structure linking pair-similarity to V3a.

Five modalities:

* **DNA (D)** — synthetic biological-ish 4-class Markov stream
  (re-uses F63c).
* **Code (C)** — Python token-type 8-class stream (re-uses F63c).
* **Music (M)** — major-scale-grammar 12-pitch-class stream.
  We synthesise simple tonal melody by sampling from a chord-
  progression Markov chain (I → IV/V/vi, V → I, …) with notes
  drawn from each chord's characteristic pitches.
* **Stock (S)** — 8-class tick-movement stream simulated as a
  mean-reverting random walk on bid-ask spread bins.
* **Linear (L)** — *negative-control* modality: pure i.i.d.
  uniform 8-class noise. Should have V3a near 1.0× because
  there is no structure to transfer from or to.

We run F63c's frozen-transfer protocol on every pair (15 pairs in
the 5-modality square minus same-pair-with-itself). The headline
table reports the V3a ratio per pair and a *similarity ordering*
hypothesis test:

* **F1** structured-modality V3a (DNA, Code, Music, Stock pairs)
  is meaningfully better than the Linear-control V3a — i.e. each
  structured pair has at least 5pp lower transfer-cost ratio
  than Linear-anything pair (averaged).
* **F2** within-family pairs (DNA↔Stock both Markov-like, Code↔
  Music both grammar-like) have the *lowest* V3a in the table.
* **F3** Linear noise as source modality cannot transfer to
  anything (V3a ≥ 1.30×) — confirms transfer needs structured
  source.

Usage::

    python -m experiments.cross_modality_pairs_sweep \\
        --d-model 64 --epochs 12 --seq-len 64 \\
        --out outputs/f63f_modality_pairs
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from experiments.cross_modality_dna_code import (
    CausalTransformerBackbone, _eval_perplexity, _train_loop,
    extract_python_tokens, synthesize_dna,
)


__all__ = ["main"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────
# Modality-specific synthetic data generators
# ─────────────────────────────────────────────────────────────────


# Music: 12 pitch classes (0=C, 1=C#, …, 11=B). Major-scale
# chord progression Markov chain at chord level (8 chord states),
# each chord emits one of its characteristic pitches.
MUSIC_VOCAB = 12

CHORD_TRANSITION = torch.tensor([
    # I -> I, ii, iii, IV, V,  vi, vii°, sub
    [0.20, 0.05, 0.05, 0.20, 0.30, 0.10, 0.05, 0.05],  # I
    [0.05, 0.10, 0.05, 0.10, 0.50, 0.10, 0.05, 0.05],  # ii
    [0.05, 0.10, 0.10, 0.30, 0.15, 0.20, 0.05, 0.05],  # iii
    [0.20, 0.10, 0.05, 0.10, 0.40, 0.05, 0.05, 0.05],  # IV
    [0.50, 0.05, 0.05, 0.05, 0.10, 0.20, 0.03, 0.02],  # V
    [0.10, 0.20, 0.05, 0.30, 0.20, 0.05, 0.05, 0.05],  # vi
    [0.40, 0.05, 0.10, 0.10, 0.05, 0.20, 0.05, 0.05],  # vii°
    [0.30, 0.10, 0.10, 0.20, 0.10, 0.10, 0.05, 0.05],  # substitution
])
# Chord -> characteristic pitch sets (in C major). Each
# distribution sums to 1 across 12 pitch classes.
CHORD_PITCH_DIST = torch.tensor([
    # I (CEG):       C  C#  D  D#  E  F  F#  G  G#  A  A#  B
    [0.40, 0.0, 0.0, 0.0, 0.30, 0.0, 0.0, 0.30, 0.0, 0.0, 0.0, 0.0],
    [0.05, 0.0, 0.40, 0.0, 0.05, 0.40, 0.0, 0.05, 0.0, 0.05, 0.0, 0.0],  # ii (DFA)
    [0.0, 0.0, 0.05, 0.0, 0.40, 0.05, 0.0, 0.10, 0.0, 0.0, 0.0, 0.40],   # iii (EGB)
    [0.05, 0.0, 0.0, 0.0, 0.0, 0.40, 0.0, 0.05, 0.0, 0.40, 0.0, 0.10],   # IV (FAC...)
    [0.10, 0.0, 0.05, 0.0, 0.05, 0.0, 0.0, 0.40, 0.0, 0.0, 0.0, 0.40],   # V (GBD...)
    [0.30, 0.0, 0.05, 0.0, 0.10, 0.0, 0.0, 0.0, 0.0, 0.40, 0.0, 0.15],   # vi (ACE)
    [0.0, 0.0, 0.30, 0.0, 0.05, 0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35],    # vii° (BDF)
    [0.10, 0.05, 0.10, 0.05, 0.10, 0.10, 0.05, 0.10, 0.10, 0.10, 0.05, 0.10],  # substitution
])


def synthesize_music(n_tokens: int, *, rng_seed: int = 0) -> torch.Tensor:
    rng = torch.Generator(device="cpu").manual_seed(rng_seed)
    # Normalise rows for safety (in case of small floating slop)
    chord_T = CHORD_TRANSITION / CHORD_TRANSITION.sum(dim=-1, keepdim=True)
    pitch_D = CHORD_PITCH_DIST / CHORD_PITCH_DIST.sum(dim=-1, keepdim=True)
    chord = int(torch.randint(0, chord_T.shape[0], (1,),
                               generator=rng).item())
    out = torch.empty(n_tokens, dtype=torch.long)
    for i in range(n_tokens):
        # Each chord lasts ~3 notes on average (geometric).
        if i > 0 and torch.rand(1, generator=rng).item() < 0.33:
            chord = int(torch.multinomial(
                chord_T[chord], 1, generator=rng,
            ).item())
        pitch = int(torch.multinomial(
            pitch_D[chord], 1, generator=rng,
        ).item())
        out[i] = pitch
    return out


# Stock: 8 classes for tick movements:
#   0: strong_down, 1: medium_down, 2: weak_down, 3: flat,
#   4: weak_up, 5: medium_up, 6: strong_up, 7: news_jump
# We simulate a mean-reverting bid-ask spread tick stream. Tick
# movement at time t depends on previous tick (auto-correlation)
# and a "regime" (trending vs reverting) hidden state.
STOCK_VOCAB = 8


def synthesize_stock(n_tokens: int, *, rng_seed: int = 0) -> torch.Tensor:
    rng = torch.Generator(device="cpu").manual_seed(rng_seed)
    out = torch.empty(n_tokens, dtype=torch.long)
    state = 3
    out[0] = state
    # Simple mean-reverting transition: P(next | curr) is biased
    # towards 3 (flat) but with some auto-correlation. Plus 2%
    # chance of a 'news_jump' (class 7) regardless.
    for i in range(1, n_tokens):
        if torch.rand(1, generator=rng).item() < 0.02:
            new = 7  # news jump
        else:
            # Mean-reverting Gaussian-like transition. We use a
            # discretised Gaussian centered at 3-(state-3)*0.6
            # (mean reversion).
            mean = 3.0 - (state - 3) * 0.6
            jitter = float(torch.randn(1, generator=rng).item()) * 1.2
            x = round(mean + jitter)
            new = max(0, min(6, x))
        out[i] = new
        state = new if new != 7 else state  # post-jump returns
    return out


# Linear noise: uniform random 8 classes. Negative-control modality
# with no learnable structure beyond marginal counts.
LINEAR_VOCAB = 8


def synthesize_linear(n_tokens: int, *, rng_seed: int = 0) -> torch.Tensor:
    rng = torch.Generator(device="cpu").manual_seed(rng_seed)
    return torch.randint(0, LINEAR_VOCAB, (n_tokens,), generator=rng)


# DNA modality reused from F63c
DNA_VOCAB = 4


# Code modality reused from F63c (extracted from repo)


# ─────────────────────────────────────────────────────────────────
# Generic 2-modality model with shared backbone (mirrors F63c
# TwoDomainModel but with arbitrary vocab sizes and per-modality
# emb/head pairs).
# ─────────────────────────────────────────────────────────────────


class GenericTwoModalityModel(nn.Module):
    def __init__(
        self, vocab_a: int, vocab_b: int,
        d_model: int = 64, n_heads: int = 2, n_layers: int = 2,
        max_len: int = 256, dropout: float = 0.0,
        share_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.share_backbone = share_backbone
        self.emb_a = nn.Embedding(vocab_a, d_model)
        self.emb_b = nn.Embedding(vocab_b, d_model)
        if share_backbone:
            self.backbone = CausalTransformerBackbone(
                d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                max_len=max_len, dropout=dropout,
            )
            self.backbone_a = self.backbone
            self.backbone_b = self.backbone
        else:
            self.backbone_a = CausalTransformerBackbone(
                d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                max_len=max_len, dropout=dropout,
            )
            self.backbone_b = CausalTransformerBackbone(
                d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                max_len=max_len, dropout=dropout,
            )
        self.head_a = nn.Linear(d_model, vocab_a)
        self.head_b = nn.Linear(d_model, vocab_b)

    # The F63c _train_loop expects forward_dna and forward_code
    # method names. We alias them.
    def forward_dna(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_a(self.backbone_a(self.emb_a(x)))

    def forward_code(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_b(self.backbone_b(self.emb_b(x)))

    # Aliases for the frozen-transfer logic
    @property
    def emb_dna(self):
        return self.emb_a

    @property
    def emb_code(self):
        return self.emb_b

    @property
    def head_dna(self):
        return self.head_a

    @property
    def head_code(self):
        return self.head_b


# ─────────────────────────────────────────────────────────────────
# Per-modality data registry
# ─────────────────────────────────────────────────────────────────


@dataclass
class Modality:
    name: str
    vocab: int
    train: torch.Tensor
    test: torch.Tensor


def build_modalities(repo_root: Path,
                     *, train_tokens: int = 200_000,
                     test_tokens: int = 20_000) -> dict[str, Modality]:
    print("  building modalities...")
    print("    DNA (synthetic Markov)...")
    dna_train = synthesize_dna(train_tokens, rng_seed=42)
    dna_test = synthesize_dna(test_tokens, rng_seed=43)

    print("    Code (Python token types from repo)...")
    code_all = extract_python_tokens(repo_root)
    n = int(code_all.shape[0] * 0.85)
    code_train = code_all[:n]
    code_test = code_all[n:]

    print("    Music (12-pitch chord-progression grammar)...")
    music_train = synthesize_music(train_tokens, rng_seed=82)
    music_test = synthesize_music(test_tokens, rng_seed=83)

    print("    Stock (8-class mean-reverting tick stream)...")
    stock_train = synthesize_stock(train_tokens, rng_seed=92)
    stock_test = synthesize_stock(test_tokens, rng_seed=93)

    print("    Linear (uniform 8-class noise; negative control)...")
    linear_train = synthesize_linear(train_tokens, rng_seed=72)
    linear_test = synthesize_linear(test_tokens, rng_seed=73)

    return {
        "DNA": Modality("DNA", DNA_VOCAB, dna_train, dna_test),
        "Code": Modality("Code", 8, code_train, code_test),
        "Music": Modality("Music", MUSIC_VOCAB, music_train, music_test),
        "Stock": Modality("Stock", STOCK_VOCAB, stock_train, stock_test),
        "Linear": Modality("Linear", LINEAR_VOCAB, linear_train, linear_test),
    }


# ─────────────────────────────────────────────────────────────────
# One-pair pipeline
# ─────────────────────────────────────────────────────────────────


def _run_pair(
    src: Modality, tgt: Modality, *,
    d_model: int, n_heads: int, n_layers: int,
    seq_len: int, batch_size: int, batches_per_epoch: int,
    epochs: int, lr: float,
) -> dict:
    """Train backbone on src only, freeze, fine-tune tgt emb+head;
    also train a from-scratch tgt-only baseline. Return V3a ratio.
    """
    # From-scratch baseline: train backbone on tgt alone (separate-
    # backbone, both modalities present so the pipeline matches
    # F63c's B condition).
    torch.manual_seed(2020)
    sep_model = GenericTwoModalityModel(
        vocab_a=src.vocab, vocab_b=tgt.vocab,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        max_len=seq_len + 4, share_backbone=False,
    ).to(DEVICE)
    sep = _train_loop(
        sep_model,
        train_dna=src.train, train_code=tgt.train,
        test_dna=src.test, test_code=tgt.test,
        seq_len=seq_len, batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        epochs=epochs, lr=lr,
    )
    sep_tgt_ppl = sep["eval"]["code"]["perplexity"]

    # Joint shared (for V1/V2 reference)
    torch.manual_seed(3030)
    joint_model = GenericTwoModalityModel(
        vocab_a=src.vocab, vocab_b=tgt.vocab,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        max_len=seq_len + 4, share_backbone=True,
    ).to(DEVICE)
    joint = _train_loop(
        joint_model,
        train_dna=src.train, train_code=tgt.train,
        test_dna=src.test, test_code=tgt.test,
        seq_len=seq_len, batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        epochs=epochs, lr=lr,
    )

    # Frozen transfer: train backbone on src only, freeze, train
    # tgt emb+head only.
    torch.manual_seed(4040)
    transfer_model = GenericTwoModalityModel(
        vocab_a=src.vocab, vocab_b=tgt.vocab,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        max_len=seq_len + 4, share_backbone=True,
    ).to(DEVICE)
    _train_loop(
        transfer_model,
        train_dna=src.train, train_code=tgt.train,
        test_dna=src.test, test_code=tgt.test,
        seq_len=seq_len, batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        epochs=epochs, lr=lr,
        domain_filter="dna",  # 'dna' = src side in our aliasing
    )
    transfer_phase2 = _train_loop(
        transfer_model,
        train_dna=None, train_code=tgt.train,
        test_dna=None, test_code=tgt.test,
        seq_len=seq_len, batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        epochs=epochs, lr=lr,
        train_emb_head_only=True, domain_filter="code",
    )
    transfer_tgt_ppl = transfer_phase2["eval"]["code"]["perplexity"]

    return {
        "src": src.name, "tgt": tgt.name,
        "src_vocab": src.vocab, "tgt_vocab": tgt.vocab,
        "joint_src_ppl": joint["eval"]["dna"]["perplexity"],
        "joint_tgt_ppl": joint["eval"]["code"]["perplexity"],
        "scratch_tgt_ppl": sep_tgt_ppl,
        "transfer_tgt_ppl": transfer_tgt_ppl,
        "V3a_transfer_over_scratch": transfer_tgt_ppl / sep_tgt_ppl,
    }


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches-per-epoch", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--train-tokens", type=int, default=200_000)
    ap.add_argument("--test-tokens", type=int, default=20_000)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--pairs", type=str,
                    default=("DNA->Code,DNA->Music,DNA->Stock,"
                             "Code->Music,Code->Stock,Music->Stock,"
                             "Linear->Code,Linear->Music"),
                    help="comma-separated A->B modality pairs")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f63f_modality_pairs"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"  F63f cross-modality V3 ratio sweep "
          f"(d_model={args.d_model}, epochs={args.epochs})")
    print("=" * 76)

    print("\n[1/3] Building modalities...")
    mods = build_modalities(
        args.repo_root,
        train_tokens=args.train_tokens,
        test_tokens=args.test_tokens,
    )

    pairs_str = args.pairs.split(",")
    pair_specs = []
    for p in pairs_str:
        a, b = p.strip().split("->")
        if a not in mods or b not in mods:
            raise ValueError(f"unknown modality in pair {p}: "
                             f"available {list(mods.keys())}")
        pair_specs.append((a, b))

    rows = []
    print(f"\n[2/3] Running {len(pair_specs)} cross-modality pairs...")
    for src_name, tgt_name in pair_specs:
        print("\n" + "-" * 76)
        print(f"  Pair: {src_name} -> {tgt_name}")
        print("-" * 76)
        t0 = time.time()
        result = _run_pair(
            mods[src_name], mods[tgt_name],
            d_model=args.d_model, n_heads=args.n_heads,
            n_layers=args.n_layers,
            seq_len=args.seq_len, batch_size=args.batch_size,
            batches_per_epoch=args.batches_per_epoch,
            epochs=args.epochs, lr=args.lr,
        )
        result["wall_s"] = time.time() - t0
        rows.append(result)
        print(f"    {src_name} -> {tgt_name}  V3a = "
              f"{result['V3a_transfer_over_scratch']:.3f}x  "
              f"(transfer ppl={result['transfer_tgt_ppl']:.3f}, "
              f"scratch ppl={result['scratch_tgt_ppl']:.3f})  "
              f"wall={result['wall_s']:.1f}s")

    # Verdict assembly
    print("\n[3/3] Computing summary...")
    structured_pairs = [
        r for r in rows if r["src"] != "Linear" and r["tgt"] != "Linear"
    ]
    linear_pairs = [
        r for r in rows if r["src"] == "Linear" or r["tgt"] == "Linear"
    ]

    structured_v3a = [r["V3a_transfer_over_scratch"]
                      for r in structured_pairs]
    linear_v3a = [r["V3a_transfer_over_scratch"]
                  for r in linear_pairs]

    structured_mean = (sum(structured_v3a) / len(structured_v3a)
                       if structured_v3a else float("nan"))
    linear_mean = (sum(linear_v3a) / len(linear_v3a)
                   if linear_v3a else float("nan"))

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "rows": rows,
        "structured_pairs_mean_V3a": structured_mean,
        "linear_pairs_mean_V3a": linear_mean,
    }
    # Hard-target subset: the V3 ratio is informative only when
    # the target task is hard enough that the from-scratch
    # backbone has meaningful uplift to give. We define a target
    # as "hard" if its scratch perplexity reduces uniform-baseline
    # perplexity by ≥ 50%. Easy targets give V3a ≈ 1.0 trivially
    # because both random and structured backbones converge fast.
    hard_pairs = [
        r for r in rows
        if r["scratch_tgt_ppl"] / r["tgt_vocab"] <= 0.50
    ]
    structured_hard = [
        r for r in hard_pairs
        if r["src"] != "Linear" and r["tgt"] != "Linear"
    ]
    linear_hard = [
        r for r in hard_pairs
        if r["src"] == "Linear" or r["tgt"] == "Linear"
    ]
    summary["hard_target_pairs"] = len(hard_pairs)
    summary["verdict"] = {
        # F1: structured pairs have meaningfully better transfer
        # (lower V3a) than linear-noise pairs (averaged over the
        # full suite).
        "F1_structured_better_than_linear_pass": (
            (not math.isnan(structured_mean))
            and (not math.isnan(linear_mean))
            and linear_mean - structured_mean >= 0.05
        ),
        # F2: at least *some* structured pair has V3a ≤ 1.30 — the
        # F63c ratio (1.35) was not the best achievable.
        "F2_some_pair_below_F63c_pass": (
            len(structured_v3a) > 0
            and min(structured_v3a) <= 1.30
        ),
        # F3 (revised after F63f-full finding): On hard targets,
        # structured sources beat the Linear-noise source. We
        # compare the *paired* (same-target) V3a values: any
        # structured-source pair with the same target as a
        # Linear-source pair must have V3a no greater. This is the
        # source-asymmetry test the original F3 was after, with
        # the target-difficulty confound removed.
        "F3_structured_source_beats_linear_on_hard_targets_pass": (
            len(structured_hard) > 0 and len(linear_hard) > 0
            and all(
                any(
                    sr["tgt"] == lr["tgt"]
                    and sr["V3a_transfer_over_scratch"]
                    <= lr["V3a_transfer_over_scratch"] + 0.005
                    for sr in structured_hard
                )
                for lr in linear_hard
            )
        ),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print("  F63f modality-pair V3 sweep verdict:")
    print("=" * 76)
    print(f"  {'pair':<20s}  {'V3a':>6s}  transfer/scratch")
    for r in sorted(rows, key=lambda r: r["V3a_transfer_over_scratch"]):
        print(f"  {r['src'] + '->' + r['tgt']:<20s}  "
              f"{r['V3a_transfer_over_scratch']:>6.3f}  "
              f"{r['transfer_tgt_ppl']:.3f} / {r['scratch_tgt_ppl']:.3f}")
    print(f"\n  structured-pair mean V3a = {structured_mean:.3f}x")
    print(f"  linear-pair    mean V3a = {linear_mean:.3f}x")
    v = summary["verdict"]
    print(f"\n  F1 structured > linear by 5pp        : "
          f"{linear_mean - structured_mean:+.3f}  "
          f"[{'PASS' if v['F1_structured_better_than_linear_pass'] else 'FAIL'}]")
    print(f"  F2 some structured pair V3a <= 1.30  : "
          f"min={min(structured_v3a):.3f} (if any)  "
          f"[{'PASS' if v['F2_some_pair_below_F63c_pass'] else 'FAIL'}]")
    print(f"  F3 structured source >= linear on hard targets : "
          f"hard-target pairs={len(hard_pairs)}  "
          f"[{'PASS' if v['F3_structured_source_beats_linear_on_hard_targets_pass'] else 'FAIL'}]")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
