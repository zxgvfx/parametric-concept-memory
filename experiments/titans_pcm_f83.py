"""F83 — Titans-style hierarchical memory PCM (Lever C).

F80 added a forget gate (closed 76 % of the F79 GPT gap).
F81 added sparse Gated-Attention "anchor" layers (fully closed
the gap, beats GPT by 4 %). F83 adds a third tier: a
**persistent kNN cache of past hidden states** with a learned
read-out, modelled after Titans' neural-memory module (Behrouz
et al., NeurIPS 2025) and the MLP-Memory / ExplicitLM family.

The architecture has three memory tiers:

1. **Short-term** (within sequence): the sparse Gated-Attention
   layer from F81 — full QKV softmax attention over the
   current 128-token window.
2. **Mid-term** (within sequence, soft): the Gated PCM layers
   from F80 — input-conditioned forget gate.
3. **Long-term** (across batches): new
   :class:`HierarchicalMemoryLayer` — FIFO kNN cache of hidden
   states from past forward passes, retrieved via cosine
   top-K + softmax + per-channel sigmoid gate.

The third tier persists *across batches*. Imprint happens
during training (4 positions per batch, FIFO). Recall is
trainable through ``q_proj`` / ``k_proj`` / ``v_proj`` /
``out_gate``.

Five engineering invariants:

* **T1** init-loss near uniform: ``|loss − ln V| < 1.5``
* **T2** Titans PCM matches Hybrid: ``ppl_titans / ppl_gpt ≤
  1.10`` (the F81 ceiling). With one extra readout layer, we
  expect parity, not improvement on a short-context corpus.
* **T3** memory is being used: ablating ``use_memory=False``
  at inference degrades PPL by ``≥ 1.05×`` (a small but
  measurable contribution at this scale).
* **T4** long-context regime: when evaluated at ``seq_len =
  2 × training seq_len`` (256 tokens, with the buffer
  pre-populated from a warm-up pass), Titans PCM's advantage
  over Hybrid PCM grows by ``≥ 1.05×`` (i.e., memory is
  *more* useful at longer contexts — the Titans claim).
* **T5** recall diagnostics: top-K mass ≥ 0.5 (queries
  retrieve concentrated buffer slots, not uniform soup).

Two emergence carry-overs (light versions):

* **E2-titans** surprisal correlation on gate retention of
  the readout's ``out_gate``: ``|r| ≥ 0.05``.
* **E5-titans** dual-process bimodality on readout output
  gate: ``BIC(GMM-2) < BIC(GMM-1)``.

Usage::

    python -m experiments.titans_pcm_f83 \\
        --corpus outputs/f79_data/tinystories_valid.txt \\
        --out outputs/f83_full
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

from experiments.gated_pcm_f80 import (
    _e2_surprisal_correlation,
    _e5_bimodality,
)
from experiments.tinystories_f79 import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    _concatenate_ids,
    _normalise_text,
    _sample_generation,
    _sample_seq_batch,
    _tokenise_text,
    _val_perplexity,
    build_vocab,
    encode_story,
)
from pcm.lm import (
    HybridPCMMiniLM,
    TitansPCMMiniLM,
    build_matched_pentad,
    count_params,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────
# Train helpers
# ─────────────────────────────────────────────────────────────────


def _train_one_model(
    model, name: str, *, train_ids: torch.Tensor,
    val_ids: torch.Tensor, n_steps: int, seq_len: int,
    batch_size: int, lr: float, log_every: int,
    device: str,
) -> tuple[dict, list[dict]]:
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
                f"    [{name}] step {step:4d}/{n_steps}  "
                f"loss={loss.item():.3f}  val_ppl={val_ppl:.1f}  "
                f"wall={time.time()-t_start:.1f}s",
                flush=True,
            )
    final_ppl = _val_perplexity(
        model, val_ids, seq_len=seq_len,
        n_batches=64, batch_size=batch_size, device=device,
    )
    return {
        "name": name, "final_ppl": final_ppl,
        "wall_s": time.time() - t_start,
    }, log


# ─────────────────────────────────────────────────────────────────
# Eval helpers (Titans-specific)
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def _titans_val_ppl(
    model: TitansPCMMiniLM, val_ids: torch.Tensor, *,
    seq_len: int, n_batches: int, batch_size: int,
    use_memory: bool, device: str,
) -> float:
    """Eval Titans PPL with or without memory readout."""
    model.eval()
    n = val_ids.shape[0] - seq_len - 1
    losses = []
    rng = torch.Generator(device="cpu").manual_seed(2026)
    for _ in range(n_batches):
        idx = torch.randint(0, max(n, 1), (batch_size,),
                            generator=rng)
        xs = torch.stack(
            [val_ids[i:i + seq_len] for i in idx]
        ).to(device)
        ys = torch.stack(
            [val_ids[i + 1:i + seq_len + 1] for i in idx]
        ).to(device)
        logits = model(
            xs, use_memory=use_memory, imprint=False,
        )
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            ys.reshape(-1), ignore_index=PAD_ID,
        )
        losses.append(loss.item())
    return float(math.exp(sum(losses) / len(losses)))


@torch.no_grad()
def _titans_recall_diagnostics(
    model: TitansPCMMiniLM, val_ids: torch.Tensor, *,
    seq_len: int, n_batches: int, batch_size: int,
    device: str,
) -> dict:
    """Average recall diagnostics over a validation sample."""
    model.eval()
    n = val_ids.shape[0] - seq_len - 1
    rng = torch.Generator(device="cpu").manual_seed(99)
    all_diag: list[dict] = []
    for _ in range(n_batches):
        idx = torch.randint(0, max(n, 1), (batch_size,),
                            generator=rng)
        xs = torch.stack(
            [val_ids[i:i + seq_len] for i in idx]
        ).to(device)
        slots = model.tok_emb(xs)
        for layer in model.layers:
            slots = slots + layer(slots)
        d = model.memory_readout.recall_diagnostics(slots)
        all_diag.append(d)
    if not all_diag:
        return {}
    keys = all_diag[0].keys()
    return {
        k: float(np.mean([float(d[k]) for d in all_diag]))
        for k in keys
    }


@torch.no_grad()
def _titans_gate_diagnostics(
    model: TitansPCMMiniLM, val_ids: torch.Tensor, *,
    seq_len: int, n_batches: int, batch_size: int,
    device: str,
) -> dict:
    """Per-position retention values from the readout's
    ``out_gate`` (E2/E5 carry-over)."""
    model.eval()
    n = val_ids.shape[0] - seq_len - 1
    rng = torch.Generator(device="cpu").manual_seed(101)
    tokens_all: list[int] = []
    gate_all: list[float] = []
    sup_all: list[float] = []
    for _ in range(n_batches):
        idx = torch.randint(0, max(n, 1), (batch_size,),
                            generator=rng)
        xs = torch.stack(
            [val_ids[i:i + seq_len] for i in idx]
        ).to(device)
        ys = torch.stack(
            [val_ids[i + 1:i + seq_len + 1] for i in idx]
        ).to(device)
        slots = model.tok_emb(xs)
        for layer in model.layers:
            slots = slots + layer(slots)
        gate = torch.sigmoid(
            model.memory_readout.out_gate(slots)
        )
        gate_mean = gate.mean(dim=-1)
        # Run full forward for surprisal
        logits = model(xs, use_memory=True, imprint=False)
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(
            -1, ys.unsqueeze(-1),
        ).squeeze(-1)
        tokens_all.extend(xs.reshape(-1).cpu().tolist())
        gate_all.extend(gate_mean.reshape(-1).cpu().tolist())
        sup_all.extend(nll.reshape(-1).cpu().tolist())
    return {
        "token_ids": tokens_all,
        "gate_retention": gate_all,
        "surprisal": sup_all,
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
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--attn-every", type=int, default=4)
    ap.add_argument("--memory-capacity", type=int, default=512)
    ap.add_argument("--memory-top-k", type=int, default=8)
    ap.add_argument("--memory-n-per-batch", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=3000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--long-seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--log-every", type=int, default=300)
    ap.add_argument("--n-diag-batches", type=int, default=16)
    ap.add_argument("--match-tol", type=float, default=0.25)
    ap.add_argument("--gen-prompts", nargs="+",
                    default=["once upon a time", "the little"])
    ap.add_argument("--gen-tokens", type=int, default=80)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f83_full"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76, flush=True)
    print(
        f"  F83 Titans-style PCM "
        f"({args.n_train_stories} train, d_model={args.d_model}, "
        f"layers={args.n_layers}, mem_cap={args.memory_capacity}, "
        f"top_k={args.memory_top_k})",
        flush=True,
    )
    print("=" * 76, flush=True)

    # ─── Step 1: load corpus ─────────────────────────────────
    print("\n[1/5] loading + tokenising TinyStories...", flush=True)
    t0 = time.time()
    raw = args.corpus.read_text(encoding="utf-8")
    stories = [
        s for s in raw.split("<|endoftext|>")
        if len(s.strip()) > 30
    ]
    rng_split = random.Random(2026)
    rng_split.shuffle(stories)
    train_stories = stories[:args.n_train_stories]
    val_stories = stories[
        args.n_train_stories:
        args.n_train_stories + args.n_val_stories
    ]
    stoi, itos = build_vocab(train_stories, vocab_cap=args.vocab_cap)
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

    # ─── Step 2: build matched pentad + add Titans ────────────
    print(
        "\n[2/5] building matched pentad + Titans PCM...",
        flush=True,
    )
    torch.manual_seed(0)
    gpt, pcm, pcm_topk, gated, hybrid, diag = build_matched_pentad(
        vocab=vocab, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
        max_len=args.max_len, top_k=args.top_k,
        attn_every=args.attn_every,
        match_tol=args.match_tol,
    )
    # Build Titans PCM with the same combiner_hidden as hybrid
    titans = TitansPCMMiniLM(
        vocab=vocab, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
        combiner_hidden=diag["hybrid_pcm_combiner_hidden"],
        attn_every=args.attn_every,
        memory_capacity=args.memory_capacity,
        memory_top_k=args.memory_top_k,
        memory_n_per_batch_imprint=args.memory_n_per_batch,
    )
    diag["titans_pcm_params"] = count_params(titans)
    diag["titans_memory_capacity"] = args.memory_capacity
    diag["titans_memory_top_k"] = args.memory_top_k
    diag["ratio_titans_over_gpt"] = (
        count_params(titans) / max(diag["gpt_params"], 1)
    )
    print(
        f"    params: gpt={count_params(gpt):,}  "
        f"hybrid={count_params(hybrid):,}  "
        f"titans={count_params(titans):,}",
        flush=True,
    )
    print(f"    diag: {diag}", flush=True)

    # ─── T1 init-loss regression ─────────────────────────────
    print("\n[T1] initial-loss regression test...", flush=True)
    init_losses = {}
    for name, m in (
        ("gpt", gpt), ("hybrid", hybrid), ("titans", titans),
    ):
        m.eval()
        m.to(DEVICE)
        if hasattr(m, "reset_memory"):
            m.reset_memory()
        with torch.no_grad():
            xs_init = torch.randint(
                0, vocab, (4, 64), device=DEVICE,
            )
            if isinstance(m, TitansPCMMiniLM):
                logits = m(xs_init, use_memory=False, imprint=False)
            else:
                logits = m(xs_init)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab),
                xs_init[:, 1:].reshape(-1),
            )
            init_losses[name] = float(loss.item())
    init_uniform = math.log(vocab)
    print(
        f"    ln(V)={init_uniform:.3f}  init losses: {init_losses}",
        flush=True,
    )
    t1_pass = all(
        abs(v - init_uniform) < 1.5 for v in init_losses.values()
    )
    print(f"    T1 init-loss-near-uniform PASS={t1_pass}", flush=True)

    # ─── Step 3: train GPT, Hybrid, Titans ───────────────────
    print(
        f"\n[3/5] training (n_steps={args.n_steps}, "
        f"seq_len={args.seq_len})",
        flush=True,
    )
    summaries: dict = {}
    logs: dict = {}
    for name, model in (
        ("gpt", gpt), ("hybrid", hybrid), ("titans", titans),
    ):
        print(f"\n  ── {name} ──", flush=True)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        if hasattr(model, "reset_memory"):
            model.reset_memory()
        s, lg = _train_one_model(
            model, name=name,
            train_ids=train_ids, val_ids=val_ids,
            n_steps=args.n_steps, seq_len=args.seq_len,
            batch_size=args.batch_size, lr=args.lr,
            log_every=args.log_every, device=DEVICE,
        )
        summaries[name] = s
        logs[name] = lg

    gpt_ppl = summaries["gpt"]["final_ppl"]
    hybrid_ppl = summaries["hybrid"]["final_ppl"]
    titans_ppl = summaries["titans"]["final_ppl"]

    # ─── T3 memory ablation ──────────────────────────────────
    print(
        "\n[T3] ablating Titans memory at inference "
        "(use_memory=False)...",
        flush=True,
    )
    titans_no_mem_ppl = _titans_val_ppl(
        titans, val_ids,
        seq_len=args.seq_len, n_batches=64,
        batch_size=args.batch_size, use_memory=False,
        device=DEVICE,
    )
    t3_ratio = titans_no_mem_ppl / titans_ppl
    print(
        f"    titans ppl: {titans_ppl:.2f}  "
        f"titans.no_memory ppl: {titans_no_mem_ppl:.2f}  "
        f"ratio: {t3_ratio:.3f}",
        flush=True,
    )

    # ─── T4 long-context regime ─────────────────────────────
    print(
        f"\n[T4] long-context: eval at seq_len={args.long_seq_len} "
        f"vs seq_len={args.seq_len}...",
        flush=True,
    )
    # Warm-up: do a few forward passes at long seq to populate
    # the buffer with long-context entries
    titans.train()
    rng_warm = torch.Generator(device="cpu").manual_seed(33)
    n_warmup_batches = 8
    n = val_ids.shape[0] - args.long_seq_len - 1
    for _ in range(n_warmup_batches):
        idx = torch.randint(
            0, max(n, 1), (args.batch_size,), generator=rng_warm,
        )
        xs_warm = torch.stack(
            [val_ids[i:i + args.long_seq_len] for i in idx]
        ).to(DEVICE)
        with torch.no_grad():
            titans(xs_warm, use_memory=True, imprint=True)

    # Eval both at long seq_len
    titans_long_ppl = _titans_val_ppl(
        titans, val_ids, seq_len=args.long_seq_len,
        n_batches=32, batch_size=args.batch_size // 2,
        use_memory=True, device=DEVICE,
    )
    hybrid_long_ppl = _val_perplexity(
        hybrid, val_ids, seq_len=args.long_seq_len,
        n_batches=32, batch_size=args.batch_size // 2,
        device=DEVICE,
    )
    short_advantage = hybrid_ppl / titans_ppl
    long_advantage = hybrid_long_ppl / titans_long_ppl
    t4_growth = long_advantage / max(short_advantage, 1e-12)
    print(
        f"    short(seq={args.seq_len}): hybrid={hybrid_ppl:.2f}, "
        f"titans={titans_ppl:.2f}, "
        f"ratio_h/t={short_advantage:.3f}",
        flush=True,
    )
    print(
        f"    long (seq={args.long_seq_len}): "
        f"hybrid={hybrid_long_ppl:.2f}, "
        f"titans={titans_long_ppl:.2f}, "
        f"ratio_h/t={long_advantage:.3f}",
        flush=True,
    )
    print(
        f"    T4 growth (long_advantage / short_advantage): "
        f"{t4_growth:.3f}",
        flush=True,
    )

    # ─── T5 recall diagnostics ───────────────────────────────
    print(
        "\n[T5] recall diagnostics (top-K mass, score "
        "spread)...",
        flush=True,
    )
    recall_diag = _titans_recall_diagnostics(
        titans, val_ids,
        seq_len=args.seq_len, n_batches=args.n_diag_batches,
        batch_size=args.batch_size, device=DEVICE,
    )
    print(f"    recall diag: {recall_diag}", flush=True)

    # ─── E2 / E5 emergence carry-over ─────────────────────────
    print(
        "\n[E2+E5] surprisal correlation + bimodality on "
        "readout gate...",
        flush=True,
    )
    gate_diag = _titans_gate_diagnostics(
        titans, val_ids, seq_len=args.seq_len,
        n_batches=args.n_diag_batches,
        batch_size=args.batch_size, device=DEVICE,
    )
    e2 = _e2_surprisal_correlation(gate_diag)
    e5 = _e5_bimodality(gate_diag)
    print(f"    E2: {e2}", flush=True)
    print(f"    E5: {e5}", flush=True)

    # ─── Generations ────────────────────────────────────────
    print("\n[4/5] generation samples...", flush=True)
    generations: dict[str, list[dict]] = {}
    for name in ("gpt", "hybrid", "titans"):
        generations[name] = []
    for prompt in args.gen_prompts:
        prompt_ids = [BOS_ID] + [
            stoi.get(w, UNK_ID)
            for w in _tokenise_text(_normalise_text(prompt))
        ]
        for name, model in (
            ("gpt", gpt), ("hybrid", hybrid), ("titans", titans),
        ):
            ids = _sample_generation(
                model, prompt_ids=prompt_ids,
                max_new=args.gen_tokens, seq_len=args.seq_len,
                temperature=0.7, device=DEVICE,
            )
            text = " ".join(
                itos[i] for i in ids
                if i not in (BOS_ID, EOS_ID, PAD_ID)
            )
            generations[name].append({"prompt": prompt, "text": text})
            print(f"  [{name}] '{prompt}' →", flush=True)
            print(f"    {text[:220]}", flush=True)

    # ─── Verdict ─────────────────────────────────────────────
    t2_ratio = titans_ppl / gpt_ppl
    verdict = {
        "T1_init_loss_near_uniform": t1_pass,
        "T2_titans_within_1_10x_of_gpt": t2_ratio <= 1.10,
        "T3_memory_contributes_ablation_degrades_ge_1_05x": (
            t3_ratio >= 1.05
        ),
        "T4_long_context_advantage_grows_ge_1_05x": (
            t4_growth >= 1.05
        ),
        "T5_top_k_mass_concentrated_ge_0_50": (
            recall_diag.get("mean_topk_mass", 0.0) >= 0.50
        ),
        "E2_titans_surprisal_correlation_abs_r_ge_0_05": (
            abs(e2.get("pearson_r", 0.0)) >= 0.05
        ),
        "E5_titans_bimodal_delta_bic_negative": (
            e5.get("delta_bic_gmm2_minus_gmm1", 1e9) < 0.0
            if isinstance(e5, dict)
            and "delta_bic_gmm2_minus_gmm1" in e5
            else False
        ),
    }

    summary = {
        "config": vars(args) | {
            "out": str(args.out),
            "corpus": str(args.corpus),
        },
        "vocab": vocab,
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
        "model_params": {
            "gpt": count_params(gpt),
            "hybrid": count_params(hybrid),
            "titans": count_params(titans),
        },
        "matched_diag": diag,
        "init_losses": init_losses,
        "ln_V": init_uniform,
        "final_ppl": {
            "gpt": gpt_ppl, "hybrid": hybrid_ppl,
            "titans": titans_ppl,
            "titans_no_memory": titans_no_mem_ppl,
            "titans_long": titans_long_ppl,
            "hybrid_long": hybrid_long_ppl,
            "uniform_baseline": float(vocab),
        },
        "gap_ratios": {
            "titans_over_gpt": t2_ratio,
            "hybrid_over_gpt": hybrid_ppl / gpt_ppl,
            "no_memory_over_titans": t3_ratio,
            "short_hybrid_over_titans": short_advantage,
            "long_hybrid_over_titans": long_advantage,
            "long_over_short_advantage_growth": t4_growth,
        },
        "training_log": logs,
        "generations": generations,
        "recall_diagnostics": recall_diag,
        "emergence": {"E2": e2, "E5": e5},
        "verdict": verdict,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False,
                   default=str)
    )

    print("\n" + "=" * 76, flush=True)
    print("  F83 Titans PCM verdict:", flush=True)
    print("=" * 76, flush=True)
    print(f"    uniform-vocab baseline ppl: {float(vocab):.0f}", flush=True)
    print(f"    gpt        final ppl: {gpt_ppl:.2f}", flush=True)
    print(f"    hybrid     final ppl: {hybrid_ppl:.2f}", flush=True)
    print(f"    titans     final ppl: {titans_ppl:.2f}", flush=True)
    print(
        f"    titans.no_memory      ppl: {titans_no_mem_ppl:.2f}",
        flush=True,
    )
    print(
        f"    titans@long_seq={args.long_seq_len}: "
        f"{titans_long_ppl:.2f}, hybrid@long: "
        f"{hybrid_long_ppl:.2f}",
        flush=True,
    )
    print(
        f"\n    gap (titans/gpt): {t2_ratio:.3f}  (F81 was 0.96)",
        flush=True,
    )
    print(
        f"    T3 no-memory/titans: {t3_ratio:.3f}  "
        f"(target ≥ 1.05)",
        flush=True,
    )
    print(
        f"    T4 long-context advantage growth: {t4_growth:.3f}  "
        f"(target ≥ 1.05)",
        flush=True,
    )
    print(
        f"    T5 recall top-K mass: "
        f"{recall_diag.get('mean_topk_mass', 0.0):.3f}  "
        f"(target ≥ 0.50)",
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
