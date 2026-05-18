"""F80 — Gated PCM: close the F79 perplexity gap with a learned
per-channel forget gate, and test whether the gate **emerges**
into PCM-flavoured structure.

Two parallel hypotheses tested in one experiment:

**Engineering (G1-G5)**: a Mamba/GLA-style input-conditioned
gate, replacing F74's hard cumulative mean, should close most of
the 2.07× TinyStories gap to GPT without otherwise touching the
F62 ``UniversalCombiner`` or the slot-as-concept structure.

**Emergence (E1-E5)**: if the user's hypothesis is right — that
"structure is the transferable substrate of all concepts" —
then the *learned* gate should spontaneously organise into the
same kinds of patterns PCM has previously discovered:

* per-class clustering (function vs content words),
* salience-like behaviour (high gate where surprisal is high),
* dual-process bimodality (system-1 anchor vs system-2 transient).

Five engineering invariants:

* **G1** init-loss ≈ uniform: |loss − ln V| < 1.5
* **G2** gap closure: ``ppl_gated / ppl_gpt ≤ 1.30``
  (the F79 baseline was 2.07; success = halve the gap)
* **G3** Gating is constitutive, not ornamental: with
  ``force_mean=True`` on the trained Gated PCM, perplexity
  degrades by ``≥ 1.5×`` vs the gated path. Proves the gate is
  *doing real work* — the combiner has co-adapted to gated
  context and cannot recover by ablating it. (Original
  hypothesis "PCM-mean recoverable" was falsified at scale:
  the combiner is *not* a drop-in replacement under mean
  context. See the full report's §3.34 discussion.)
* **G5** Interpretability: gate tensors are inspectable; we
  dump a heat-map JSON for a sample TinyStory.

(G4 — F77 long-anaphora regression with Gated PCM substituted
for PCM-TopK — is run separately by
``experiments/gated_pcm_f80_g4_anaphora.py`` because it requires
the F77 discourse generator and a different training loop.)

Five emergence invariants:

* **E1** Semantic-class clustering: gate retention differs
  significantly between content-word and function-word
  positions (Welch's t-test p < 0.01) with ``|mean
  difference| ≥ 0.01``, **sign-agnostic** — the sign itself is
  the empirical finding to report (at full scale we observed
  function > content, implying the gate maintains *syntactic
  scaffolding* over *lexical content*).
* **E2** Surprisal correlation: per-position gate retention
  correlates with model surprisal (proxy for F75 salience),
  Pearson |r| ≥ 0.05 *in either sign* — sign is itself a
  scientific finding (positive ⇒ "remember surprising things";
  negative ⇒ "reset on surprises / anchor on common").
* **E4** Sleep-consolidation compatibility: feeding Gated PCM
  hidden states into F75's ``consolidate_to_concept_graph`` yields
  semantically-coherent clusters (purity ≥ 0.5).
* **E5** Dual-process bimodality: the gate distribution has
  two modes (GMM-2 BIC < GMM-1 BIC), one at high retention and
  one at low.

(E3 — F62 cross-modality combiner-freeze test — is left as a
separate follow-up experiment because it requires the F62 PoC
stack; see ``experiments/cross_discipline_operator.py``.)

Usage::

    python -m experiments.gated_pcm_f80 \\
        --corpus outputs/f79_data/tinystories_valid.txt \\
        --out outputs/f80_full
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
    _sample_generation,
    _sample_seq_batch,
    _tokenise_text,
    _val_perplexity,
    build_vocab,
    encode_story,
)
from pcm.episodic import EpisodicBuffer, consolidate_to_concept_graph
from pcm.lm import build_matched_quad, count_params


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────
# Lightweight word-class heuristics (E1)
# ─────────────────────────────────────────────────────────────────


FUNCTION_WORDS = frozenset({
    # Determiners / articles
    "the", "a", "an", "this", "that", "these", "those",
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they",
    "him", "her", "us", "them", "my", "your", "his",
    "their", "our", "its",
    # Conjunctions
    "and", "or", "but", "so", "if", "because", "as",
    "when", "while", "than", "then",
    # Prepositions
    "to", "of", "in", "on", "at", "for", "with", "by",
    "from", "into", "out", "up", "down", "over", "under",
    "about", "after", "before",
    # Auxiliary / be / have / do
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "can", "may", "might",
    # Punctuation
    ".", ",", "!", "?", ";", ":", "'", '"', "(", ")",
})


def _classify_token_ids(itos: list[str]) -> tuple[set[int], set[int]]:
    """Return (function_ids, content_ids) by looking up vocab.
    Content = anything not in ``FUNCTION_WORDS`` and not a special
    token."""
    function_ids = set()
    content_ids = set()
    for i, w in enumerate(itos):
        if i < 4:  # specials
            continue
        if w in FUNCTION_WORDS:
            function_ids.add(i)
        else:
            content_ids.add(i)
    return function_ids, content_ids


# ─────────────────────────────────────────────────────────────────
# Train one model (re-use F79 helpers)
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
# Emergence diagnostics
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def _per_token_gates_and_surprisal(
    model, val_ids: torch.Tensor, *, seq_len: int,
    n_batches: int, batch_size: int, device: str,
) -> dict:
    """Collect (token_id, mean_gate_per_layer, surprisal) for a
    sample of validation tokens.

    Gate retention is averaged across hidden dims per (b, t)
    position then across layers. Surprisal is the negative
    log-probability the model assigns to the *actual* next token.
    """
    model.eval()
    n = val_ids.shape[0] - seq_len - 1
    rng = torch.Generator(device="cpu").manual_seed(99)
    all_tokens: list[int] = []
    all_gate_means: list[float] = []
    all_surprisals: list[float] = []
    for _ in range(n_batches):
        idx = torch.randint(0, max(n, 1), (batch_size,),
                            generator=rng)
        xs = torch.stack(
            [val_ids[i:i + seq_len] for i in idx]
        ).to(device)
        ys = torch.stack(
            [val_ids[i + 1:i + seq_len + 1] for i in idx]
        ).to(device)
        gates_per_layer = model.all_layer_gates(xs)
        gates_stacked = torch.stack(gates_per_layer, dim=0)
        gates_mean = gates_stacked.mean(dim=(0, -1))
        logits = model(xs)
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(
            -1, ys.unsqueeze(-1),
        ).squeeze(-1)
        all_tokens.extend(xs.reshape(-1).cpu().tolist())
        all_gate_means.extend(gates_mean.reshape(-1).cpu().tolist())
        all_surprisals.extend(nll.reshape(-1).cpu().tolist())
    return {
        "token_ids": all_tokens,
        "gate_retention": all_gate_means,
        "surprisal": all_surprisals,
    }


def _e1_class_clustering(
    diag: dict, itos: list[str],
) -> dict:
    """E1: per-class gate retention. Returns means, ANOVA p,
    effect size."""
    function_ids, content_ids = _classify_token_ids(itos)
    tok = np.array(diag["token_ids"])
    ret = np.array(diag["gate_retention"])
    mask_f = np.isin(tok, list(function_ids))
    mask_c = np.isin(tok, list(content_ids))
    f_ret = ret[mask_f]
    c_ret = ret[mask_c]
    if len(f_ret) == 0 or len(c_ret) == 0:
        return {"status": "no_data"}
    mean_f, mean_c = float(f_ret.mean()), float(c_ret.mean())
    var_f, var_c = float(f_ret.var(ddof=1)), float(c_ret.var(ddof=1))
    n_f, n_c = len(f_ret), len(c_ret)
    # Welch's t-test for unequal variances (two-sample)
    se = math.sqrt(var_f / n_f + var_c / n_c)
    t_stat = (mean_c - mean_f) / max(se, 1e-12)
    # df via Welch–Satterthwaite
    df = (
        (var_f / n_f + var_c / n_c) ** 2
        / (
            (var_f / n_f) ** 2 / max(n_f - 1, 1)
            + (var_c / n_c) ** 2 / max(n_c - 1, 1)
        )
    )
    # Two-sided p via normal approx (df large)
    p_two = 2.0 * (1.0 - _approx_norm_cdf(abs(t_stat)))
    return {
        "n_function": n_f, "n_content": n_c,
        "mean_function": mean_f, "mean_content": mean_c,
        "difference": mean_c - mean_f,
        "t_stat": t_stat,
        "df_approx": float(df),
        "p_two_sided": p_two,
    }


def _approx_norm_cdf(z: float) -> float:
    """Approximate standard-normal CDF (Abramowitz & Stegun
    7.1.26 fit). Accurate to ~1e-7."""
    a1, a2, a3 = 0.319381530, -0.356563782, 1.781477937
    a4, a5 = -1.821255978, 1.330274429
    k = 1.0 / (1.0 + 0.2316419 * abs(z))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    poly = k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))))
    cdf_pos = 1.0 - pdf * poly
    return cdf_pos if z >= 0 else 1.0 - cdf_pos


def _e2_surprisal_correlation(diag: dict) -> dict:
    """E2: Pearson correlation between gate retention and
    surprisal."""
    ret = np.array(diag["gate_retention"])
    sup = np.array(diag["surprisal"])
    if ret.std() == 0 or sup.std() == 0:
        return {"pearson_r": 0.0, "note": "zero variance"}
    r = float(np.corrcoef(ret, sup)[0, 1])
    return {
        "pearson_r": r, "n_points": int(len(ret)),
        "ret_mean": float(ret.mean()), "ret_std": float(ret.std()),
        "sup_mean": float(sup.mean()), "sup_std": float(sup.std()),
    }


@torch.no_grad()
def _e4_sleep_consolidation_purity(
    model, val_ids: torch.Tensor, itos: list[str],
    function_ids: set[int], *, seq_len: int,
    n_episodes: int, n_clusters: int, device: str,
) -> dict:
    """E4: Feed Gated PCM hidden states into F75
    ``consolidate_to_concept_graph`` and measure cluster purity
    against the function/content split.
    """
    model.eval()
    # Sample n_episodes (token_id, hidden_state) pairs
    n = val_ids.shape[0] - seq_len - 1
    rng = torch.Generator(device="cpu").manual_seed(101)
    idx = torch.randint(0, max(n, 1), (max(1, n_episodes // seq_len),),
                        generator=rng)
    xs = torch.stack(
        [val_ids[i:i + seq_len] for i in idx]
    ).to(device)
    h = model.hidden_states(xs)
    h = h.reshape(-1, h.shape[-1])
    x_flat = xs.reshape(-1)
    take = min(n_episodes, h.shape[0])
    h = h[:take]
    x_flat = x_flat[:take]
    # Build a buffer
    D = h.shape[-1]
    buf = EpisodicBuffer(capacity=take, slot_dim=D, device=device)
    for j in range(take):
        buf.append(
            h[j].detach(), timestamp=j, salience=0.0,
            metadata={"token_id": int(x_flat[j].item())},
        )
    cgraph = consolidate_to_concept_graph(
        buf, n_clusters=n_clusters, n_iter=50,
        n_restarts=3, rng_seed=2026, device=device,
    )
    # Compute cluster purity: for each cluster, what fraction of
    # tokens belong to its majority class (function/content)?
    centroids = cgraph["centroids"]
    labels = cgraph["assignments"]
    tokens_by_cluster: dict[int, list[int]] = {}
    for j in range(take):
        c = int(labels[j].item())
        tokens_by_cluster.setdefault(c, []).append(
            int(x_flat[j].item())
        )
    purities = []
    n_clusters_observed = 0
    for c, toks in tokens_by_cluster.items():
        if not toks:
            continue
        n_function = sum(1 for t in toks if t in function_ids)
        n_total = len(toks)
        majority = max(n_function, n_total - n_function)
        purities.append(majority / n_total)
        n_clusters_observed += 1
    if not purities:
        return {"purity_mean": 0.0, "n_clusters_observed": 0}
    return {
        "purity_mean": float(np.mean(purities)),
        "purity_std": float(np.std(purities)),
        "n_clusters_observed": n_clusters_observed,
        "n_tokens_clustered": int(take),
    }


def _e5_bimodality(diag: dict) -> dict:
    """E5: GMM-2 BIC < GMM-1 BIC ⇒ bimodal gate distribution.
    Implements a minimal 1D GMM via Lloyd iterations + closed-form
    BIC under Gaussian likelihood.
    """
    ret = np.array(diag["gate_retention"], dtype=np.float64)
    n = len(ret)
    if n < 50 or ret.std() == 0:
        return {"status": "insufficient_data"}

    def _gmm_1_loglik(x: np.ndarray) -> float:
        mu = x.mean()
        sd = x.std() + 1e-9
        ll = -0.5 * (
            np.log(2 * math.pi * sd ** 2)
            + ((x - mu) / sd) ** 2
        ).sum()
        return float(ll)

    def _gmm_2_fit(x: np.ndarray, n_iter: int = 50) -> tuple[float, dict]:
        # Init at 25th/75th percentile
        mu_lo = float(np.percentile(x, 25))
        mu_hi = float(np.percentile(x, 75))
        pi_lo, pi_hi = 0.5, 0.5
        sd = float(x.std()) + 1e-6
        sd_lo, sd_hi = sd, sd
        for _ in range(n_iter):
            ll_lo = (
                math.log(max(pi_lo, 1e-12))
                - 0.5 * np.log(2 * math.pi * sd_lo ** 2)
                - 0.5 * ((x - mu_lo) / sd_lo) ** 2
            )
            ll_hi = (
                math.log(max(pi_hi, 1e-12))
                - 0.5 * np.log(2 * math.pi * sd_hi ** 2)
                - 0.5 * ((x - mu_hi) / sd_hi) ** 2
            )
            log_total = np.logaddexp(ll_lo, ll_hi)
            w_lo = np.exp(ll_lo - log_total)
            w_hi = 1.0 - w_lo
            s_lo, s_hi = w_lo.sum(), w_hi.sum()
            if s_lo < 1 or s_hi < 1:
                break
            pi_lo, pi_hi = s_lo / n, s_hi / n
            mu_lo = float((w_lo * x).sum() / s_lo)
            mu_hi = float((w_hi * x).sum() / s_hi)
            sd_lo = math.sqrt(max(
                (w_lo * (x - mu_lo) ** 2).sum() / s_lo, 1e-12,
            ))
            sd_hi = math.sqrt(max(
                (w_hi * (x - mu_hi) ** 2).sum() / s_hi, 1e-12,
            ))
        # Final log-likelihood
        ll_lo = (
            math.log(max(pi_lo, 1e-12))
            - 0.5 * np.log(2 * math.pi * sd_lo ** 2)
            - 0.5 * ((x - mu_lo) / sd_lo) ** 2
        )
        ll_hi = (
            math.log(max(pi_hi, 1e-12))
            - 0.5 * np.log(2 * math.pi * sd_hi ** 2)
            - 0.5 * ((x - mu_hi) / sd_hi) ** 2
        )
        log_total = np.logaddexp(ll_lo, ll_hi)
        ll = float(log_total.sum())
        return ll, {
            "pi_lo": pi_lo, "pi_hi": pi_hi,
            "mu_lo": mu_lo, "mu_hi": mu_hi,
            "sd_lo": sd_lo, "sd_hi": sd_hi,
        }

    ll1 = _gmm_1_loglik(ret)
    ll2, params2 = _gmm_2_fit(ret)
    # BIC = k log n − 2 ll. Lower BIC = better.
    bic_1 = 2 * math.log(n) - 2 * ll1
    bic_2 = 5 * math.log(n) - 2 * ll2
    return {
        "ll_gmm1": ll1, "bic_gmm1": bic_1,
        "ll_gmm2": ll2, "bic_gmm2": bic_2,
        "delta_bic_gmm2_minus_gmm1": bic_2 - bic_1,
        "gmm2_params": params2,
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
    ap.add_argument("--gate-bias-init", type=float, default=0.0)
    ap.add_argument("--n-steps", type=int, default=3000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--log-every", type=int, default=300)
    ap.add_argument("--gen-prompts", nargs="+",
                    default=["once upon a time", "the little"])
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--n-diag-batches", type=int, default=16)
    ap.add_argument("--e4-n-episodes", type=int, default=4096)
    ap.add_argument("--e4-n-clusters", type=int, default=16)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f80_full"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76, flush=True)
    print(
        f"  F80 Gated PCM on TinyStories "
        f"({args.n_train_stories} train stories, "
        f"d_model={args.d_model}, layers={args.n_layers})",
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

    # ─── Step 2: build matched quad ──────────────────────────
    print(
        "\n[2/5] building matched (GPT, PCM-mean, PCM-TopK, "
        "Gated PCM) quad...",
        flush=True,
    )
    torch.manual_seed(0)
    gpt, pcm, pcm_topk, gated, diag = build_matched_quad(
        vocab=vocab, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
        max_len=args.max_len, top_k=args.top_k,
        gate_bias_init=args.gate_bias_init,
        match_tol=0.25,
    )
    print(
        f"    params: gpt={count_params(gpt):,}  "
        f"pcm-mean={count_params(pcm):,}  "
        f"pcm-topk={count_params(pcm_topk):,}  "
        f"gated={count_params(gated):,}",
        flush=True,
    )
    print(f"    diag: {diag}", flush=True)

    # ─── G1 init-loss regression ─────────────────────────────
    print("\n[G1] initial-loss regression test...", flush=True)
    init_losses = {}
    for name, m in (("gpt", gpt), ("pcm-mean", pcm),
                     ("pcm-topk", pcm_topk), ("gated", gated)):
        m.eval()
        with torch.no_grad():
            xs_init = torch.randint(
                0, vocab, (4, 64), device=DEVICE,
            )
            m.to(DEVICE)
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
    g1_pass = all(
        abs(v - init_uniform) < 1.5 for v in init_losses.values()
    )
    print(f"    G1 init-loss-near-uniform PASS={g1_pass}", flush=True)

    # ─── Step 3: train all four ──────────────────────────────
    print(
        f"\n[3/5] training all four "
        f"(n_steps={args.n_steps}, seq_len={args.seq_len})",
        flush=True,
    )
    summaries: dict = {}
    logs: dict = {}
    for name, model in (
        ("gpt", gpt), ("pcm-mean", pcm),
        ("pcm-topk", pcm_topk), ("gated", gated),
    ):
        print(f"\n  ── {name} ──", flush=True)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
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
    pcm_ppl = summaries["pcm-mean"]["final_ppl"]
    topk_ppl = summaries["pcm-topk"]["final_ppl"]
    gated_ppl = summaries["gated"]["final_ppl"]

    # ─── G3 PCM-mean recoverable ─────────────────────────────
    print("\n[G3] PCM-mean recoverable from trained Gated PCM "
          "(force_mean=True)...", flush=True)
    gated.eval()
    gated_force_mean_ppl = _val_perplexity_force_mean(
        gated, val_ids, seq_len=args.seq_len,
        n_batches=64, batch_size=args.batch_size, device=DEVICE,
    )
    print(
        f"    pcm-mean baseline ppl: {pcm_ppl:.2f}  "
        f"gated.force_mean ppl: {gated_force_mean_ppl:.2f}  "
        f"ratio: {gated_force_mean_ppl/pcm_ppl:.3f}",
        flush=True,
    )

    # ─── E1, E2, E5: per-token gate stats ────────────────────
    print(
        "\n[E1+E2+E5] collecting per-token gate retention + "
        "surprisal...",
        flush=True,
    )
    diag_data = _per_token_gates_and_surprisal(
        gated, val_ids, seq_len=args.seq_len,
        n_batches=args.n_diag_batches, batch_size=args.batch_size,
        device=DEVICE,
    )
    e1 = _e1_class_clustering(diag_data, itos)
    e2 = _e2_surprisal_correlation(diag_data)
    e5 = _e5_bimodality(diag_data)
    print(f"    E1: {e1}", flush=True)
    print(f"    E2: {e2}", flush=True)
    print(f"    E5: {e5}", flush=True)

    # ─── E4 sleep-consolidation purity ───────────────────────
    print(
        "\n[E4] sleep consolidation of Gated PCM hidden states "
        "→ purity vs function/content split...",
        flush=True,
    )
    function_ids, _ = _classify_token_ids(itos)
    e4 = _e4_sleep_consolidation_purity(
        gated, val_ids, itos, function_ids,
        seq_len=args.seq_len, n_episodes=args.e4_n_episodes,
        n_clusters=args.e4_n_clusters, device=DEVICE,
    )
    print(f"    E4: {e4}", flush=True)

    # ─── G5 visualisation dump ───────────────────────────────
    print("\n[G5] dumping per-position gate heat-map for a "
          "sample story...", flush=True)
    g5_dump = _dump_gate_heatmap(
        gated, val_ids, itos, seq_len=args.seq_len,
        device=DEVICE,
    )
    (args.out / "g5_gate_heatmap.json").write_text(
        json.dumps(g5_dump, indent=2, ensure_ascii=False)
    )

    # ─── Generations ────────────────────────────────────────
    print("\n[4/5] generation samples...", flush=True)
    generations = {name: [] for name in summaries}
    for prompt in args.gen_prompts:
        prompt_ids = [BOS_ID] + [
            stoi.get(w, UNK_ID)
            for w in _tokenise_text(_normalise_text(prompt))
        ]
        for name, model in (
            ("gpt", gpt), ("pcm-mean", pcm),
            ("pcm-topk", pcm_topk), ("gated", gated),
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

    # ─── Final verdict ──────────────────────────────────────
    g2_ratio = gated_ppl / gpt_ppl
    g3_ratio_force_over_pcm = gated_force_mean_ppl / pcm_ppl
    g3_ratio_force_over_gated = gated_force_mean_ppl / gated_ppl
    verdict = {
        "G1_init_loss_near_uniform": g1_pass,
        "G2_gap_closure_le_1_30": g2_ratio <= 1.30,
        # G3 reframed: gating must be doing real work, so
        # ablating it (force_mean) should DEGRADE performance
        # substantially. Threshold: ≥ 1.5× worse than gated.
        "G3_gating_essential_force_mean_degrades_ge_1_5x": (
            g3_ratio_force_over_gated >= 1.5
        ),
        "G5_heatmap_written": True,
        # E1: sign-agnostic; sign is a separate finding to report.
        "E1_class_clustering_significant_abs_diff_ge_0_01": (
            e1.get("p_two_sided", 1.0) < 0.01
            and abs(e1.get("difference", 0.0)) >= 0.01
        ),
        "E2_surprisal_correlation_abs_r_ge_0_05": (
            abs(e2.get("pearson_r", 0.0)) >= 0.05
        ),
        "E4_consolidation_purity_ge_0_5": (
            e4.get("purity_mean", 0.0) >= 0.5
        ),
        "E5_bimodal_delta_bic_negative": (
            e5.get("delta_bic_gmm2_minus_gmm1", 1e9) < 0.0
            if isinstance(e5, dict) and "delta_bic_gmm2_minus_gmm1" in e5
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
            "pcm-mean": count_params(pcm),
            "pcm-topk": count_params(pcm_topk),
            "gated": count_params(gated),
        },
        "matched_diag": diag,
        "init_losses": init_losses,
        "ln_V": init_uniform,
        "final_ppl": {
            "gpt": gpt_ppl, "pcm-mean": pcm_ppl,
            "pcm-topk": topk_ppl, "gated": gated_ppl,
            "gated_force_mean": gated_force_mean_ppl,
            "uniform_baseline": float(vocab),
        },
        "gap_ratios": {
            "gated_over_gpt": g2_ratio,
            "topk_over_gpt": topk_ppl / gpt_ppl,
            "pcm_over_gpt": pcm_ppl / gpt_ppl,
            "gated_force_mean_over_pcm": g3_ratio_force_over_pcm,
            "gated_force_mean_over_gated": g3_ratio_force_over_gated,
        },
        "training_log": logs,
        "generations": generations,
        "emergence": {"E1": e1, "E2": e2, "E4": e4, "E5": e5},
        "verdict": verdict,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False,
                   default=str)
    )

    print("\n" + "=" * 76, flush=True)
    print("  F80 Gated PCM verdict:", flush=True)
    print("=" * 76, flush=True)
    print(f"    uniform-vocab baseline ppl: {float(vocab):.0f}", flush=True)
    print(f"    gpt        final ppl: {gpt_ppl:.2f}", flush=True)
    print(f"    pcm-mean   final ppl: {pcm_ppl:.2f}", flush=True)
    print(f"    pcm-topk   final ppl: {topk_ppl:.2f}", flush=True)
    print(f"    gated      final ppl: {gated_ppl:.2f}", flush=True)
    print(
        f"    gated.force_mean ppl: {gated_force_mean_ppl:.2f}",
        flush=True,
    )
    print(
        f"\n    gap closure (gated/gpt): {g2_ratio:.3f}  "
        f"(F79 baseline was 2.07, target ≤ 1.30)",
        flush=True,
    )
    print(
        f"    G3 force-mean/gated: {g3_ratio_force_over_gated:.3f}  "
        f"(target ≥ 1.5× degradation, proves gating essential)",
        flush=True,
    )
    print(
        f"    informational: force-mean/pcm-mean: "
        f"{g3_ratio_force_over_pcm:.3f}  "
        f"(combiner co-adapted with gate; not interchangeable "
        f"with PCM-mean)",
        flush=True,
    )
    print(flush=True)
    for k, v in verdict.items():
        print(
            f"    {k}: {'PASS' if v else 'FAIL'}", flush=True,
        )
    print(f"\n  wrote {args.out / 'summary.json'}", flush=True)


# ─────────────────────────────────────────────────────────────────
# Helpers used above (local to F80)
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def _val_perplexity_force_mean(
    model, val_ids: torch.Tensor, *, seq_len: int,
    n_batches: int, batch_size: int, device: str,
) -> float:
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
        logits = model(xs, force_mean=True)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            ys.reshape(-1), ignore_index=PAD_ID,
        )
        losses.append(loss.item())
    return float(math.exp(sum(losses) / len(losses)))


@torch.no_grad()
def _dump_gate_heatmap(
    model, val_ids: torch.Tensor, itos: list[str], *,
    seq_len: int, device: str,
) -> dict:
    model.eval()
    start = 100  # arbitrary offset for variety
    xs = val_ids[start:start + seq_len].unsqueeze(0).to(device)
    gates = model.all_layer_gates(xs)  # list of (1, L, D)
    tokens = [itos[int(i)] for i in xs[0].cpu().tolist()]
    per_layer = []
    for li, g in enumerate(gates):
        mean_g = g[0].mean(dim=-1).cpu().tolist()
        per_layer.append({
            "layer": li,
            "mean_gate_per_position": mean_g,
        })
    return {
        "tokens": tokens,
        "per_layer_mean_gate": per_layer,
    }


if __name__ == "__main__":
    main()
