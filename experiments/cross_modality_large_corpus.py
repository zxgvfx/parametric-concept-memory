"""F63d — Real 5M-token Python corpus saturation sweep.

F63c trained the cross-modality DNA + Python backbone on the
project's own ~120K-token Python stream. The reviewer's concern
(paraphrased): *"Code modality is starved — V3a 1.35× is partly an
artefact of Python's training-set being capacity-saturated.
Re-run with a real 5M-token corpus and see whether V3a moves."*

F63d is the sweep that answers it. We extract Python token streams
from progressively larger sources and re-run the F63c pipeline at
each size:

* **S** — this repo only (~100K-200K tokens, F63c's setting).
* **M** — repo + Python standard library top-level (~1M tokens).
* **L** — repo + stdlib + site-packages (~5M tokens).

For each corpus size we report the four F63c invariants
(V1, V2, V3a, V3b, V4) and add **two saturation invariants**:

* **S1** V3a (transfer cost) saturates as Code corpus grows ≥ 5×
  but stays in [0.95, 1.50]. The architectural sharing is real but
  bounded — adding 10× more Code data does not erase the structural
  ratio.
* **S2** From-scratch Code perplexity *drops* as corpus grows (more
  modality capacity) — i.e. ``B_code(L) < B_code(S)``. If it
  doesn't drop, the small-model is already saturated and the
  comparison is uninformative.

The headline question (the user's): does V3a stay around 1.35× as
the Code modality saturates, or does it drift? F63d gives the
falsifiable answer.

Usage::

    python -m experiments.cross_modality_large_corpus \\
        --d-model 64 --epochs 12 --seq-len 64 \\
        --out outputs/f63d_corpus_sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import sysconfig
import time
from pathlib import Path

import torch

from experiments.cross_modality_dna_code import (
    DNA_VOCAB, CODE_VOCAB, TwoDomainModel, _eval_perplexity,
    _train_loop, extract_python_tokens, synthesize_dna,
)


__all__ = ["main"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────
# Corpus assembly
# ─────────────────────────────────────────────────────────────────


def _stdlib_root() -> Path:
    return Path(sysconfig.get_paths()["stdlib"])


def _site_packages_root() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def _filter_tokens(tokens: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if tokens.shape[0] <= max_tokens:
        return tokens
    return tokens[:max_tokens]


def assemble_corpus(
    size: str, repo_root: Path, max_tokens: int | None = None,
) -> torch.Tensor:
    """Concatenate Python token streams from progressively larger
    sources. Returns a 1-D long tensor of token-class IDs.

    ``size`` is one of ``"S"``, ``"M"``, ``"L"``:

    * ``S`` — repo only (F63c default).
    * ``M`` — repo + python stdlib top-level scripts.
    * ``L`` — repo + stdlib top-level + site-packages.
    """
    streams: list[torch.Tensor] = []
    print(f"  [{size}] extracting from repo: {repo_root}")
    streams.append(extract_python_tokens(repo_root))
    print(f"      repo tokens: {streams[-1].shape[0]:,}")

    if size in ("M", "L"):
        # Stdlib top-level only (avoid huge nested packages so we
        # can keep size M reasonably small).
        sl = _stdlib_root()
        print(f"  [{size}] extracting from stdlib top-level: {sl}")
        sl_tokens = []
        for f in sl.glob("*.py"):
            try:
                import tokenize as tknz
                from experiments.cross_modality_dna_code import _classify_token
                with open(f, "rb") as fh:
                    for tok in tknz.tokenize(fh.readline):
                        cls = _classify_token(tok)
                        if cls is not None:
                            sl_tokens.append(cls)
            except Exception:
                pass
        if sl_tokens:
            streams.append(torch.tensor(sl_tokens, dtype=torch.long))
            print(f"      stdlib top-level tokens: {streams[-1].shape[0]:,}")

    if size == "L":
        sp = _site_packages_root()
        print(f"  [{size}] extracting from site-packages: {sp}")
        # Use extract_python_tokens which already filters out
        # __pycache__ / outputs / agent-tools / .venv. Site-packages
        # has its own structure but the same filters apply.
        sp_tokens = extract_python_tokens(sp)
        if sp_tokens.numel() > 0:
            streams.append(sp_tokens)
            print(f"      site-packages tokens: {streams[-1].shape[0]:,}")

    full = torch.cat(streams, dim=0)
    print(f"  [{size}] total tokens: {full.shape[0]:,}")
    if max_tokens is not None and full.shape[0] > max_tokens:
        print(f"      truncating to {max_tokens:,}")
        full = full[:max_tokens]
    return full


# ─────────────────────────────────────────────────────────────────
# Per-corpus pipeline (mirrors cross_modality_dna_code.main but
# parametrised on a pre-assembled Code stream)
# ─────────────────────────────────────────────────────────────────


def _run_one_corpus(
    code_stream: torch.Tensor, dna_train: torch.Tensor,
    dna_test: torch.Tensor,
    *, args, label: str,
) -> dict:
    n_train = int(code_stream.shape[0] * 0.85)
    code_train = code_stream[:n_train]
    code_test = code_stream[n_train:]
    print(f"\n[{label}] code train tokens={code_train.shape[0]:,}, "
          f"test tokens={code_test.shape[0]:,}")

    uniform_dna_perp = DNA_VOCAB
    uniform_code_perp = CODE_VOCAB

    # Condition A — joint shared
    print(f"[{label}/A] joint training, SHARED backbone...")
    t0 = time.time()
    torch.manual_seed(11)
    A_model = TwoDomainModel(
        vocab_dna=DNA_VOCAB, vocab_code=CODE_VOCAB,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.seq_len + 4, share_backbone=True,
    ).to(DEVICE)
    A = _train_loop(
        A_model,
        train_dna=dna_train, train_code=code_train,
        test_dna=dna_test, test_code=code_test,
        seq_len=args.seq_len, batch_size=args.batch_size,
        batches_per_epoch=args.batches_per_epoch,
        epochs=args.epochs, lr=args.lr,
    )
    print(f"      A wall = {time.time()-t0:.1f}s  "
          f"DNA ppl={A['eval']['dna']['perplexity']:.3f}  "
          f"Code ppl={A['eval']['code']['perplexity']:.3f}")

    # Condition B — joint separate
    print(f"[{label}/B] joint training, SEPARATE backbones...")
    t0 = time.time()
    torch.manual_seed(22)
    B_model = TwoDomainModel(
        vocab_dna=DNA_VOCAB, vocab_code=CODE_VOCAB,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.seq_len + 4, share_backbone=False,
    ).to(DEVICE)
    B = _train_loop(
        B_model,
        train_dna=dna_train, train_code=code_train,
        test_dna=dna_test, test_code=code_test,
        seq_len=args.seq_len, batch_size=args.batch_size,
        batches_per_epoch=args.batches_per_epoch,
        epochs=args.epochs, lr=args.lr,
    )
    print(f"      B wall = {time.time()-t0:.1f}s  "
          f"DNA ppl={B['eval']['dna']['perplexity']:.3f}  "
          f"Code ppl={B['eval']['code']['perplexity']:.3f}")

    # Condition C — frozen-backbone transfer
    print(f"[{label}/C] frozen-transfer (DNA train, freeze, "
          f"Code emb+head)...")
    t0 = time.time()
    torch.manual_seed(33)
    C_model = TwoDomainModel(
        vocab_dna=DNA_VOCAB, vocab_code=CODE_VOCAB,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.seq_len + 4, share_backbone=True,
    ).to(DEVICE)
    _train_loop(
        C_model,
        train_dna=dna_train, train_code=code_train,
        test_dna=dna_test, test_code=code_test,
        seq_len=args.seq_len, batch_size=args.batch_size,
        batches_per_epoch=args.batches_per_epoch,
        epochs=args.epochs, lr=args.lr,
        domain_filter="dna",
    )
    Cphase2 = _train_loop(
        C_model,
        train_dna=None, train_code=code_train,
        test_dna=None, test_code=code_test,
        seq_len=args.seq_len, batch_size=args.batch_size,
        batches_per_epoch=args.batches_per_epoch,
        epochs=args.epochs, lr=args.lr,
        train_emb_head_only=True, domain_filter="code",
    )
    print(f"      C wall = {time.time()-t0:.1f}s  "
          f"transfer-Code ppl="
          f"{Cphase2['eval']['code']['perplexity']:.3f}")

    # Condition D — sequence-shuffled negative control
    print(f"[{label}/D] shuffled-position negative control...")
    t0 = time.time()
    torch.manual_seed(44)
    perm = torch.randperm(code_train.shape[0])
    code_train_perm = code_train[perm]
    perm_test = torch.randperm(code_test.shape[0])
    code_test_perm = code_test[perm_test]
    D_model = TwoDomainModel(
        vocab_dna=DNA_VOCAB, vocab_code=CODE_VOCAB,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.seq_len + 4, share_backbone=True,
    ).to(DEVICE)
    _train_loop(
        D_model,
        train_dna=dna_train, train_code=code_train_perm,
        test_dna=dna_test, test_code=code_test_perm,
        seq_len=args.seq_len, batch_size=args.batch_size,
        batches_per_epoch=args.batches_per_epoch,
        epochs=args.epochs, lr=args.lr,
        domain_filter="dna",
    )
    Dphase2 = _train_loop(
        D_model,
        train_dna=None, train_code=code_train_perm,
        test_dna=None, test_code=code_test_perm,
        seq_len=args.seq_len, batch_size=args.batch_size,
        batches_per_epoch=args.batches_per_epoch,
        epochs=args.epochs, lr=args.lr,
        train_emb_head_only=True, domain_filter="code",
    )
    print(f"      D wall = {time.time()-t0:.1f}s  "
          f"shuffled-Code ppl="
          f"{Dphase2['eval']['code']['perplexity']:.3f}")

    a_dna = A["eval"]["dna"]["perplexity"]
    a_code = A["eval"]["code"]["perplexity"]
    b_dna = B["eval"]["dna"]["perplexity"]
    b_code = B["eval"]["code"]["perplexity"]
    c_code = Cphase2["eval"]["code"]["perplexity"]
    d_code = Dphase2["eval"]["code"]["perplexity"]

    return {
        "label": label,
        "code_train_tokens": int(code_train.shape[0]),
        "code_test_tokens": int(code_test.shape[0]),
        "uniform_dna": uniform_dna_perp,
        "uniform_code": uniform_code_perp,
        "A_joint_shared": A["eval"],
        "B_joint_separate": B["eval"],
        "C_frozen_transfer": {"code": Cphase2["eval"]["code"]},
        "D_shuffled_negative": {"code": Dphase2["eval"]["code"]},
        "ratios": {
            "V2_dna": a_dna / b_dna,
            "V2_code": a_code / b_code,
            "V3a_V3b_transfer_over_scratch": c_code / b_code,
            "V4_shuffled_over_transfer": d_code / c_code,
        },
        "verdict": {
            "V1_pass": (a_dna <= uniform_dna_perp * 0.78
                        and a_code <= uniform_code_perp * 0.78),
            "V2_pass": (a_dna <= b_dna * 1.10
                        and a_code <= b_code * 1.10),
            "V3a_pass": c_code <= b_code * 1.50,
            "V3b_pass": c_code >= b_code * 1.05,
            "V4_pass": d_code >= c_code * 1.30,
        },
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
    ap.add_argument("--batches-per-epoch", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dna-train-tokens", type=int, default=200_000)
    ap.add_argument("--dna-test-tokens", type=int, default=20_000)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--corpus-sizes", type=str, default="S,M,L")
    ap.add_argument("--max-code-tokens-l", type=int, default=5_000_000,
                    help="hard cap on the L corpus token count")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f63d_corpus_sweep"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sizes = [s.strip() for s in args.corpus_sizes.split(",")]

    print("=" * 76)
    print(f"  F63d corpus saturation sweep "
          f"(sizes={sizes}, d_model={args.d_model})")
    print("=" * 76)

    print("\n[1/3] Synthesising DNA Markov stream...")
    dna_train = synthesize_dna(args.dna_train_tokens, rng_seed=42)
    dna_test = synthesize_dna(args.dna_test_tokens, rng_seed=43)
    print(f"    DNA train tokens: {dna_train.shape[0]:,}")
    print(f"    DNA test tokens : {dna_test.shape[0]:,}")

    # Pre-assemble each corpus size up-front so the timing for
    # later runs reflects only training, not tokenisation.
    corpora = {}
    print("\n[2/3] Assembling code corpora...")
    for sz in sizes:
        max_tokens = (args.max_code_tokens_l
                      if sz == "L" else None)
        corpora[sz] = assemble_corpus(
            sz, args.repo_root, max_tokens=max_tokens,
        )

    rows = []
    print("\n[3/3] Running cross-modality pipeline at each size...")
    for sz in sizes:
        print("\n" + "=" * 76)
        print(f"  Corpus {sz}: {corpora[sz].shape[0]:,} Python tokens")
        print("=" * 76)
        rows.append(_run_one_corpus(
            corpora[sz], dna_train, dna_test, args=args, label=sz,
        ))

    # Saturation analysis
    by_label = {r["label"]: r for r in rows}
    sat = {}
    if "S" in by_label and "L" in by_label:
        s_b_code = by_label["S"]["B_joint_separate"]["code"]["perplexity"]
        l_b_code = by_label["L"]["B_joint_separate"]["code"]["perplexity"]
        s_v3 = by_label["S"]["ratios"]["V3a_V3b_transfer_over_scratch"]
        l_v3 = by_label["L"]["ratios"]["V3a_V3b_transfer_over_scratch"]
        sat = {
            "S_B_code_ppl": s_b_code, "L_B_code_ppl": l_b_code,
            "S_V3_ratio": s_v3, "L_V3_ratio": l_v3,
            "code_size_ratio_L_over_S":
                by_label["L"]["code_train_tokens"]
                / max(by_label["S"]["code_train_tokens"], 1),
        }
        # S1: V3 ratio stays in [0.95, 1.50] across corpus sizes.
        sat["S1_V3_stable_pass"] = (
            0.95 <= l_v3 <= 1.50 and 0.95 <= s_v3 <= 1.50
        )
        # S2: from-scratch Code perplexity strictly drops as
        # corpus grows (small-model is not saturated at S).
        sat["S2_code_ppl_drops_pass"] = l_b_code < s_b_code

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "rows": rows,
        "saturation": sat,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print("  F63d corpus saturation sweep verdict:")
    print("=" * 76)
    for r in rows:
        v = r["verdict"]
        print(f"  Corpus {r['label']}  ({r['code_train_tokens']:>8,} train tokens)  "
              f"V3a={r['ratios']['V3a_V3b_transfer_over_scratch']:.2f}x  "
              f"V4={r['ratios']['V4_shuffled_over_transfer']:.2f}x  "
              f"V1[{('PASS' if v['V1_pass'] else 'FAIL')}] "
              f"V2[{('PASS' if v['V2_pass'] else 'FAIL')}] "
              f"V3a[{('PASS' if v['V3a_pass'] else 'FAIL')}] "
              f"V3b[{('PASS' if v['V3b_pass'] else 'FAIL')}] "
              f"V4[{('PASS' if v['V4_pass'] else 'FAIL')}]")
    if sat:
        print(f"\n  Saturation summary:")
        print(f"    code_corpus L/S ratio  = "
              f"{sat['code_size_ratio_L_over_S']:.1f}x")
        print(f"    B_code ppl S -> L      = "
              f"{sat['S_B_code_ppl']:.3f} -> {sat['L_B_code_ppl']:.3f}")
        print(f"    V3 ratio  S -> L       = "
              f"{sat['S_V3_ratio']:.3f}x -> {sat['L_V3_ratio']:.3f}x")
        print(f"    S1 V3 stays in [0.95, 1.50]      "
              f"[{'PASS' if sat['S1_V3_stable_pass'] else 'FAIL'}]")
        print(f"    S2 from-scratch ppl drops L<S    "
              f"[{'PASS' if sat['S2_code_ppl_drops_pass'] else 'FAIL'}]")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
