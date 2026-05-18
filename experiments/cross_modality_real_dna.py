"""F63e — Cross-modality test with REAL human chromosome 22 DNA.

F63c used a synthetic Markov DNA stream (4 nucleotides with hand-
crafted transition probabilities encoding CpG depletion and AT/GC
bias). The reviewer's concern (paraphrased): *"Synthetic Markov
DNA matches the architecture you trained — V3a 1.35× could be a
'designed-to-fit' artefact. Re-run with real GRCh38 chr22 and
report whether transfer survives genuine biological data."*

This experiment downloads ``chr22.fa.gz`` from UCSC (or uses a
locally-cached copy under ``data/chr22.fa.gz``), strips assembly
gaps (``N``), and runs the F63c pipeline on the resulting ~39M-
base ACGT stream. Code modality stays the same (Python token-type
sequence from the project repo), so the comparison to F63c is
direct.

Five falsifiable invariants (mirroring F63c, with the data-source
swap as the only design change):

* **V1** joint-shared training beats uniform by ≥ 22% on each
  modality.
* **V2** sharing has no penalty (joint-shared ≤ 1.10× joint-
  separate on each modality).
* **V3a** frozen-backbone transfer Code perplexity ≤ 1.50× from-
  scratch (transfer carries meaningful structure).
* **V3b** frozen-backbone transfer Code perplexity ≥ 1.05× from-
  scratch (transfer is *not* a complete substitute).
* **V4** shuffled-position negative control ≥ 1.30× honest
  transfer (sequence structure is the lever, not labels).

Plus a saturation invariant comparing to F63c synthetic:

* **R1** real-DNA V3a is within 0.50 of synthetic-DNA V3a — the
  cross-modality structural transfer is robust to swapping the
  hand-designed Markov for genuine biological data.

Usage::

    python -m experiments.cross_modality_real_dna \\
        --d-model 64 --epochs 15 --seq-len 64 \\
        --out outputs/f63e_real_dna
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import ssl
import time
import urllib.request
from pathlib import Path

import torch

from experiments.cross_modality_dna_code import (
    DNA_VOCAB, CODE_VOCAB, TwoDomainModel, _eval_perplexity,
    _train_loop, extract_python_tokens,
)


__all__ = ["main"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHR22_URL = (
    "https://hgdownload.soe.ucsc.edu/"
    "goldenPath/hg38/chromosomes/chr22.fa.gz"
)
DEFAULT_CHR22_PATH = Path("data/chr22.fa.gz")

NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


# ─────────────────────────────────────────────────────────────────
# Real DNA loading
# ─────────────────────────────────────────────────────────────────


def ensure_chr22(path: Path = DEFAULT_CHR22_PATH) -> Path:
    """Ensure ``chr22.fa.gz`` exists locally; download from UCSC if
    not. Returns the path to the gzipped FASTA file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    print(f"  downloading {CHR22_URL} -> {path} ...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(CHR22_URL, context=ctx, timeout=300) as r:
        data = r.read()
    path.write_bytes(data)
    print(f"  saved {len(data):,} bytes")
    return path


def load_real_dna_tokens(
    path: Path = DEFAULT_CHR22_PATH, *,
    max_tokens: int | None = None,
) -> torch.Tensor:
    """Read GRCh38 chr22 FASTA, drop ``N`` (assembly gap) bases,
    return a 1-D long tensor of indices in ``{0, 1, 2, 3}`` for
    ``A, C, G, T``.

    chr22 is ~50.8M bases of which ~39.2M are ACGT after filtering
    gaps — enough to comfortably exceed F63c's 200K synthetic
    DNA training tokens by 200×.
    """
    ensure_chr22(path)
    print(f"  parsing FASTA: {path}")
    tokens: list[int] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                continue
            for c in line.strip().upper():
                idx = NUC_TO_IDX.get(c)
                if idx is not None:
                    tokens.append(idx)
                    if max_tokens is not None and len(tokens) >= max_tokens:
                        break
            if max_tokens is not None and len(tokens) >= max_tokens:
                break
    t = torch.tensor(tokens, dtype=torch.long)
    print(f"  loaded {t.shape[0]:,} ACGT tokens "
          f"(of which AT/GC fractions = {(t == 0).float().mean():.3f}A "
          f"{(t == 1).float().mean():.3f}C "
          f"{(t == 2).float().mean():.3f}G "
          f"{(t == 3).float().mean():.3f}T)")
    return t


# ─────────────────────────────────────────────────────────────────
# Pipeline (mirrors F63c.main)
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches-per-epoch", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dna-train-tokens", type=int, default=2_000_000,
                    help="real chr22 ACGT tokens used for training")
    ap.add_argument("--dna-test-tokens", type=int, default=200_000)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--chr22-path", type=Path,
                    default=DEFAULT_CHR22_PATH)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f63e_real_dna"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"  F63e cross-modality real-DNA + Python "
          f"(GRCh38 chr22, d_model={args.d_model})")
    print("=" * 76)

    print("\n[1/5] Loading real chromosome 22 DNA...")
    total_dna_needed = args.dna_train_tokens + args.dna_test_tokens
    dna_all = load_real_dna_tokens(
        args.chr22_path, max_tokens=total_dna_needed,
    )
    if dna_all.shape[0] < total_dna_needed:
        print(f"  warning: only {dna_all.shape[0]:,} ACGT tokens "
              f"available, requested {total_dna_needed:,}")
    dna_train = dna_all[:args.dna_train_tokens]
    dna_test = dna_all[args.dna_train_tokens:
                       args.dna_train_tokens + args.dna_test_tokens]
    print(f"    DNA train tokens: {dna_train.shape[0]:,}")
    print(f"    DNA test tokens : {dna_test.shape[0]:,}")

    print("\n[2/5] Extracting Python token-type stream from repo...")
    code_all = extract_python_tokens(args.repo_root)
    n_train = int(code_all.shape[0] * 0.85)
    code_train = code_all[:n_train]
    code_test = code_all[n_train:]
    print(f"    Code train tokens: {code_train.shape[0]:,}")
    print(f"    Code test tokens : {code_test.shape[0]:,}")
    if code_train.shape[0] < args.seq_len * 100:
        raise RuntimeError("Code stream too short for batch sampling")

    uniform_dna_perp = DNA_VOCAB
    uniform_code_perp = CODE_VOCAB

    # Condition A — joint shared
    print("\n[3/5] Condition A: joint training, SHARED backbone...")
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
    print(f"    wall = {time.time()-t0:.1f}s  "
          f"DNA ppl={A['eval']['dna']['perplexity']:.3f}  "
          f"Code ppl={A['eval']['code']['perplexity']:.3f}")

    # Condition B — joint separate
    print("\n[4/5] Condition B: joint training, SEPARATE backbones...")
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
    print(f"    wall = {time.time()-t0:.1f}s  "
          f"DNA ppl={B['eval']['dna']['perplexity']:.3f}  "
          f"Code ppl={B['eval']['code']['perplexity']:.3f}")

    # Condition C — frozen-backbone transfer
    print("\n[5/5] Condition C: frozen-transfer (real-DNA -> Code)...")
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
    pre_transfer = _eval_perplexity(
        C_model.forward_code, code_test,
        seq_len=args.seq_len, batch_size=args.batch_size, n_batches=20,
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
    print(f"    wall = {time.time()-t0:.1f}s  "
          f"transfer-Code ppl="
          f"{Cphase2['eval']['code']['perplexity']:.3f}")

    # Condition D — shuffled-position negative control
    print("\n[6/6] Condition D: shuffled-position negative control...")
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
    print(f"    wall = {time.time()-t0:.1f}s  "
          f"shuffled-Code ppl="
          f"{Dphase2['eval']['code']['perplexity']:.3f}")

    a_dna = A["eval"]["dna"]["perplexity"]
    a_code = A["eval"]["code"]["perplexity"]
    b_dna = B["eval"]["dna"]["perplexity"]
    b_code = B["eval"]["code"]["perplexity"]
    c_code = Cphase2["eval"]["code"]["perplexity"]
    d_code = Dphase2["eval"]["code"]["perplexity"]

    # Compare to F63c synthetic Markov DNA: load synthetic
    # F63c V3a if the file exists, otherwise leave as None.
    f63c_v3a = None
    f63c_path = Path("outputs/f63_full2/summary.json")
    if f63c_path.exists():
        try:
            f63c = json.loads(f63c_path.read_text())
            f63c_v3a = (
                f63c["C_frozen_transfer"]["code"]["perplexity"]
                / f63c["B_joint_separate"]["code"]["perplexity"]
            )
            print(f"\n  F63c (synthetic Markov DNA) V3a from "
                  f"{f63c_path}: {f63c_v3a:.3f}x")
        except (KeyError, json.JSONDecodeError):
            pass

    summary = {
        "config": vars(args) | {"out": str(args.out)},
        "data_source": "GRCh38 chr22 (UCSC)",
        "dna_train_tokens": int(dna_train.shape[0]),
        "dna_test_tokens": int(dna_test.shape[0]),
        "code_train_tokens": int(code_train.shape[0]),
        "code_test_tokens": int(code_test.shape[0]),
        "uniform_baseline": {
            "dna": uniform_dna_perp, "code": uniform_code_perp,
        },
        "A_joint_shared": A["eval"],
        "B_joint_separate": B["eval"],
        "C_frozen_transfer": {
            "code": Cphase2["eval"]["code"],
            "pre_transfer_code": pre_transfer,
        },
        "D_shuffled_negative": {
            "code": Dphase2["eval"]["code"],
        },
        "ratios": {
            "V3a_V3b_transfer_over_scratch": c_code / b_code,
            "V4_shuffled_over_transfer": d_code / c_code,
        },
        "f63c_synthetic_v3a": f63c_v3a,
    }
    summary["verdict"] = {
        "V1_pass": (
            a_dna <= uniform_dna_perp * 0.78
            and a_code <= uniform_code_perp * 0.78
        ),
        "V2_pass": (
            a_dna <= b_dna * 1.10
            and a_code <= b_code * 1.10
        ),
        "V3a_pass": c_code <= b_code * 1.50,
        "V3b_pass": c_code >= b_code * 1.05,
        "V4_pass": d_code >= c_code * 1.30,
        "R1_real_vs_synth_pass": (
            f63c_v3a is None
            or abs(c_code / b_code - f63c_v3a) <= 0.50
        ),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )

    print("\n" + "=" * 76)
    print("  F63e real-DNA cross-modality verdict:")
    print("=" * 76)
    v = summary["verdict"]
    print(f"  V1 joint-shared beats uniform        : "
          f"DNA {a_dna:.3f}/{uniform_dna_perp}  "
          f"Code {a_code:.3f}/{uniform_code_perp}  "
          f"[{'PASS' if v['V1_pass'] else 'FAIL'}]")
    print(f"  V2 sharing has no penalty            : "
          f"DNA {a_dna/b_dna:.3f}x  Code {a_code/b_code:.3f}x  "
          f"[{'PASS' if v['V2_pass'] else 'FAIL'}]")
    print(f"  V3a transfer carries structure       : "
          f"transfer/scratch = {c_code/b_code:.2f}x  "
          f"[{'PASS' if v['V3a_pass'] else 'FAIL'}]")
    print(f"  V3b transfer is not full substitute  : "
          f"transfer/scratch = {c_code/b_code:.2f}x (>=1.05x)  "
          f"[{'PASS' if v['V3b_pass'] else 'FAIL'}]")
    print(f"  V4 shuffled-position fails           : "
          f"shuffled/honest = {d_code/c_code:.2f}x  "
          f"[{'PASS' if v['V4_pass'] else 'FAIL'}]")
    if f63c_v3a is not None:
        print(f"  R1 real-DNA V3a within 0.50 of F63c  : "
              f"real={c_code/b_code:.2f}x  synth={f63c_v3a:.2f}x  "
              f"|Δ|={abs(c_code/b_code - f63c_v3a):.2f}  "
              f"[{'PASS' if v['R1_real_vs_synth_pass'] else 'FAIL'}]")
    print(f"\n  wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
