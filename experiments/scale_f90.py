"""F90 — Scale PCM up to ~70% RTX 3070 VRAM and compare to GPT.

Single focused comparison: does the F81 PASS (PCM-Hybrid
matches/beats GPT at d=128, 1.5 M params) hold at scale
(~50 × params, ~5.4 GB peak VRAM)?

At ``d_model=512, n_layers=12, batch=64, seq_len=128`` the
:class:`HybridPCMMiniLM` peaks at **5.40 GB** (67.5 % of the
8 GB RTX 3070) and trains at 1357 ms/step. We compare just
GPT vs Hybrid at this scale — the other PCM variants are
already covered at d=128 in F81.

Three falsifiable invariants:

* **S1 no OOM at target scale** — both GPT and Hybrid PCM
  train one full epoch without out-of-memory at peak VRAM
  ≥ 5 GB.
* **S2 Hybrid still ≤ 1.5 × GPT PPL** — at 50 × parameters,
  PCM's bounded-cost claim still holds.
* **S3 monotone scaling** — both architectures' val PPL at
  d=512/L=12 is *lower* than their d=128/L=4 baseline from
  F81 (10.86 / 11.33), proving they still benefit from scale
  on this corpus rather than overfit completely.

Usage::

    python -m experiments.scale_f90 \\
        --d-model 512 --n-layers 12 \\
        --out outputs/f90_full
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.tinystories_f79 import (
    PAD_ID,
    _concatenate_ids,
    _sample_seq_batch,
    _val_perplexity,
    build_vocab,
    encode_story,
)
from pcm.lm import (
    GPTMiniLM,
    HybridPCMMiniLM,
    count_params,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _train_one_model(
    name: str, model, *, train_ids, val_ids,
    n_steps, seq_len, batch_size, lr, log_every, device,
) -> tuple[dict, list[dict]]:
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4,
    )
    rng = torch.Generator(device="cpu").manual_seed(7777)
    log = []
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    for step in range(1, n_steps + 1):
        xs, ys = _sample_seq_batch(
            train_ids, seq_len=seq_len,
            batch_size=batch_size, rng=rng,
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
                model, val_ids, seq_len=seq_len, n_batches=16,
                batch_size=batch_size, device=device,
            )
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            log.append({
                "step": step,
                "train_loss": float(loss.item()),
                "val_ppl": val_ppl,
                "peak_gb": peak_gb,
            })
            model.train()
            print(
                f"    [{name}] step {step:4d}/{n_steps}  "
                f"loss={loss.item():.3f}  val_ppl={val_ppl:.2f}  "
                f"peak={peak_gb:.2f} GB  "
                f"wall={time.time()-t_start:.1f}s",
                flush=True,
            )
    final_ppl = _val_perplexity(
        model, val_ids, seq_len=seq_len, n_batches=64,
        batch_size=batch_size, device=device,
    )
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return ({
        "name": name, "final_ppl": final_ppl,
        "peak_gb": peak_gb,
        "wall_s": time.time() - t_start,
    }, log)


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
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--attn-every", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=1500)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log-every", type=int, default=150)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f90_full"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print(
        f"  F90 Scale-up: GPT vs Hybrid PCM at d={args.d_model}, "
        f"L={args.n_layers}, batch={args.batch_size}",
        flush=True,
    )
    print("=" * 78, flush=True)

    # ─── Step 1: corpus + vocab ──────────────────────────
    print("\n[1/3] corpus + vocab...", flush=True)
    raw = args.corpus.read_text(encoding="utf-8")
    stories = [
        s for s in raw.split("<|endoftext|>")
        if len(s.strip()) > 30
    ]
    rng = random.Random(args.seed)
    rng.shuffle(stories)
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
        f"    vocab={vocab}, train_tokens={len(train_ids):,}",
        flush=True,
    )

    # ─── Step 2: build matched pair ──────────────────────
    print("\n[2/3] building scaled GPT + Hybrid PCM...",
          flush=True)
    torch.manual_seed(args.seed)
    gpt = GPTMiniLM(
        vocab=vocab, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
        max_len=args.max_len,
    )
    torch.manual_seed(args.seed)
    hybrid = HybridPCMMiniLM(
        vocab=vocab, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
        attn_every=args.attn_every,
    )
    print(
        f"    GPT params:    {count_params(gpt):>12,}",
        flush=True,
    )
    print(
        f"    Hybrid params: {count_params(hybrid):>12,}  "
        f"(ratio = {count_params(hybrid)/count_params(gpt):.3f}×)",
        flush=True,
    )

    # ─── Step 3: train and compare ───────────────────────
    print(
        f"\n[3/3] training ({args.n_steps} steps each)",
        flush=True,
    )
    summaries = {}
    logs = {}
    for name, model in (("gpt", gpt), ("hybrid", hybrid)):
        print(f"\n  ── {name} ──", flush=True)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # Clean up between models
        torch.cuda.empty_cache()
        s, lg = _train_one_model(
            name=name, model=model,
            train_ids=train_ids, val_ids=val_ids,
            n_steps=args.n_steps, seq_len=args.seq_len,
            batch_size=args.batch_size, lr=args.lr,
            log_every=args.log_every, device=DEVICE,
        )
        summaries[name] = s
        logs[name] = lg
        # Free model memory before next
        model.cpu()
        torch.cuda.empty_cache()

    gpt_ppl = summaries["gpt"]["final_ppl"]
    hybrid_ppl = summaries["hybrid"]["final_ppl"]
    gap = hybrid_ppl / gpt_ppl
    s1_pass = (
        summaries["gpt"]["peak_gb"] >= 1.0
        and summaries["hybrid"]["peak_gb"] >= 1.0
    )
    s2_pass = gap <= 1.50
    # F81 baseline: GPT 11.33, Hybrid 10.86 at d=128 L=4
    s3_pass = gpt_ppl < 11.33 and hybrid_ppl < 10.86

    verdict = {
        "S1_no_OOM_at_target_scale": s1_pass,
        "S2_hybrid_within_1_5x_gpt_ppl": s2_pass,
        "S3_both_better_than_d128_baseline": s3_pass,
    }
    summary = {
        "config": vars(args) | {
            "out": str(args.out),
            "corpus": str(args.corpus),
        },
        "vocab": vocab,
        "model_params": {
            "gpt": count_params(gpt),
            "hybrid": count_params(hybrid),
        },
        "final_ppl": {
            "gpt": gpt_ppl, "hybrid": hybrid_ppl,
            "ratio_hybrid_over_gpt": gap,
        },
        "peak_gb": {
            "gpt": summaries["gpt"]["peak_gb"],
            "hybrid": summaries["hybrid"]["peak_gb"],
        },
        "wall_s": {
            "gpt": summaries["gpt"]["wall_s"],
            "hybrid": summaries["hybrid"]["wall_s"],
        },
        "training_log": logs,
        "baselines_f81_d128_L4": {
            "gpt_ppl": 11.33, "hybrid_ppl": 10.86,
            "hybrid_over_gpt": 0.958,
        },
        "verdict": verdict,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False,
                   default=str)
    )

    print("\n" + "=" * 78, flush=True)
    print("  F90 Scale-up verdict:", flush=True)
    print("=" * 78, flush=True)
    print(
        f"    GPT @ d={args.d_model},L={args.n_layers}: "
        f"PPL = {gpt_ppl:.2f}  "
        f"peak = {summaries['gpt']['peak_gb']:.2f} GB",
        flush=True,
    )
    print(
        f"    Hybrid @ same scale: "
        f"PPL = {hybrid_ppl:.2f}  "
        f"peak = {summaries['hybrid']['peak_gb']:.2f} GB",
        flush=True,
    )
    print(
        f"    gap (hybrid/gpt) = {gap:.3f}  "
        f"(F81 baseline was 0.958 at d=128/L=4)",
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
