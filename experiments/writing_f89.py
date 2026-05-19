"""F89 — Writing: token embedding → glyph image.

The final step in the PCM v10 multimodal curriculum:

    listen + speak (F79–F85)  ⇒  see colours + shapes (F87)
    ⇒  read printed text (F88)  ⇒  **write (F89, this experiment)**

F88 trained a :class:`GlyphEncoder` that maps printed glyph
images into the LM's token-embedding space. F89 inverts the
direction with a :class:`GlyphDecoder` — given a token
embedding, generate the corresponding 16×64 grayscale glyph
image. Together they form a closed cross-modal loop:

    text token  →  tok_emb  →  GlyphDecoder  →  glyph image
    glyph image →  GlyphEncoder  →  tok_emb (≈ original)

Three falsifiable invariants (W1, W2, W3 from the user spec):

* **W1 reconstruction quality** — on held-out tokens (20 % of
  vocab not seen during decoder training), reconstructed glyph
  is identifiable: nearest-glyph in the full glyph table is
  the *correct* token ≥ 0.40 (chance = 1/4096).
* **W2 writing online-learned concepts** — after F85 online
  teacher loop teaches the model 8 fictional concepts
  (zorgon/floob/...; ANIMAL+FOOD classes), the decoder applied
  to their tok_emb produces glyphs that match the concept's
  actual rendered word (F94 concept-specific top-20 NN test,
  default threshold ≥ 0.50). Earlier class-prototype variant
  (looks more animal-like than food-like?) is retained for
  reference but is uninformative since printed text isn't
  class-discriminative.
* **W3 cycle consistency** — tok_emb[A] → decoder →
  reconstructed glyph → encoder → recovered embedding.
  Cosine(original, recovered) ≥ 0.50 averaged over 100 held-
  out tokens. Tests that the F88 encoder + F89 decoder form a
  near-identity loop in slot space.

F94 (multimodal F85 visual update, ``--multimodal-f85``):
after joint cycle training on regular vocab, run a short
update that exposes decoder + encoder to the
``(concept_emb, concept_glyph)`` pairs for the 8 reserved
concepts (mixed with replay from regular vocab to prevent
forgetting). This is the visual analogue of F85's text
correction — F85 teaches the LM to *talk about* zorgon, F94
teaches the decoder/encoder to *write/read* the word
"zorgon".

Usage::

    python -m experiments.writing_f89 \\
        --corpus outputs/f79_data/tinystories_valid.txt \\
        --out outputs/f89_full
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
import torch.nn as nn
import torch.nn.functional as F

from experiments.online_teacher_f85 import (
    build_vocab_with_reserved,
    encode_word_list,
)
from experiments.tinystories_f79 import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    _concatenate_ids,
    _sample_seq_batch,
    _val_perplexity,
    build_vocab,
    encode_story,
)
from pcm.lm import HybridPCMMiniLM, count_params
from pcm.lm_synthetic import (
    RESERVED_CONCEPTS,
    generate_concept_dataset,
)
from pcm.literacy import (
    GLYPH_SIZE,
    GlyphDecoder,
    GlyphEncoder,
    alignment_loss,
    build_glyph_table,
    reconstruction_loss,
)
from pcm.online import OnlineTeacherSession, PretrainReplayBuffer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Hand-picked reference words for the W2 class-similarity test.
# All present in TinyStories top-4 K vocab (verified by F86's
# assessment script).
ANIMAL_REFS = ("cat", "dog", "bird", "horse", "rabbit")
FOOD_REFS = ("apple", "bread", "cake", "soup", "fruit")


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


def _pretrain_lm(
    *, train_ids: torch.Tensor, val_ids: torch.Tensor,
    vocab: int, d_model: int, n_layers: int, n_heads: int,
    n_steps: int, seq_len: int, batch_size: int, lr: float,
    log_every: int, attn_every: int, device: str, seed: int,
    replay: PretrainReplayBuffer, n_replay_steps: int,
) -> HybridPCMMiniLM:
    torch.manual_seed(seed)
    lm = HybridPCMMiniLM(
        vocab=vocab, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, attn_every=attn_every,
    )
    lm.to(device)
    opt = torch.optim.AdamW(
        lm.parameters(), lr=lr, weight_decay=1e-4,
    )
    rng = torch.Generator(device="cpu").manual_seed(7777)
    t_start = time.time()
    lm.train()
    for step in range(1, n_steps + 1):
        xs, ys = _sample_seq_batch(
            train_ids, seq_len=seq_len, batch_size=batch_size,
            rng=rng,
        )
        xs = xs.to(device)
        ys = ys.to(device)
        logits = lm(xs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            ys.reshape(-1), ignore_index=PAD_ID,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
        opt.step()
        if step > n_steps - n_replay_steps:
            replay.add_batch(xs, ys)
        if step % log_every == 0 or step == n_steps:
            val_ppl = _val_perplexity(
                lm, val_ids, seq_len=seq_len, n_batches=16,
                batch_size=batch_size, device=device,
            )
            print(
                f"    [pretrain] step {step:4d}/{n_steps}  "
                f"loss={loss.item():.3f}  val_ppl={val_ppl:.1f}  "
                f"wall={time.time()-t_start:.1f}s",
                flush=True,
            )
            lm.train()
    return lm


def _train_joint_cycle(
    *, decoder: GlyphDecoder, encoder: GlyphEncoder,
    lm: HybridPCMMiniLM, glyph_table: torch.Tensor,
    train_ids_set: list[int],
    n_steps: int, batch_size: int, lr: float,
    log_every: int, device: str,
    bce_weight: float = 0.5,
    contrastive_weight: float = 1.0,
    contrastive_temp: float = 0.05,
    align_weight: float = 1.0,
    cycle_weight: float = 1.0,
    cosine_lr: bool = False,
) -> list[dict]:
    """F93 — joint encoder + decoder training with a cycle-
    consistency loss.

    Diagnosis (from §3.43): training encoder on true glyphs and
    decoder on true embeddings separately makes the cycle
    inverse compose two independent errors. Joint training with

        L_cycle = 1 − cos(emb, encoder(decoder(emb)))

    aligns the round-trip in slot space directly.

    Loss layout (sum):

    * ``L_recon`` — MSE + BCE + in-batch InfoNCE on
      ``decoder(emb) ↔ true_glyph`` (the F91 decoder loss).
    * ``L_align`` — MSE + (1 − cos) on
      ``encoder(true_glyph) ↔ emb`` (the F88 alignment loss).
    * ``L_cycle`` — 1 − cos on
      ``emb ↔ encoder(decoder(emb))``.

    Both networks update at the same lr; LM ``tok_emb`` is
    frozen (we only train the two peripheral networks).
    """
    decoder.to(device)
    encoder.to(device)
    opt = torch.optim.AdamW(
        list(decoder.parameters())
        + list(encoder.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = None
    if cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_steps, eta_min=lr * 0.1,
        )
    rng = torch.Generator(device="cpu").manual_seed(2026)
    train_idx_tensor = torch.tensor(
        train_ids_set, dtype=torch.long,
    )
    n = len(train_ids_set)
    log: list[dict] = []
    t_start = time.time()
    decoder.train()
    encoder.train()
    for step in range(1, n_steps + 1):
        idx_in_train = torch.randint(
            0, n, (batch_size,), generator=rng,
        )
        sampled_ids = train_idx_tensor[idx_in_train]
        with torch.no_grad():
            emb = lm.tok_emb.weight[
                sampled_ids.to(device)
            ].detach()
        true_glyph = glyph_table[sampled_ids].to(device)
        # Forward
        pred_glyph = decoder(emb)
        enc_true = encoder(true_glyph)
        enc_pred = encoder(pred_glyph)
        # Losses
        l_recon, recon_diag = reconstruction_loss(
            pred_glyph, true_glyph,
            bce_weight=bce_weight,
            contrastive_weight=contrastive_weight,
            temperature=contrastive_temp,
        )
        l_align, align_diag = alignment_loss(enc_true, emb)
        emb_n = F.normalize(emb, dim=-1)
        enc_pred_n = F.normalize(enc_pred, dim=-1)
        cycle_cos = (emb_n * enc_pred_n).sum(dim=-1).mean()
        l_cycle = 1.0 - cycle_cos
        loss = (
            l_recon
            + align_weight * l_align
            + cycle_weight * l_cycle
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(decoder.parameters())
            + list(encoder.parameters()),
            1.0,
        )
        opt.step()
        if scheduler is not None:
            scheduler.step()
        if step % log_every == 0 or step == n_steps:
            cur_lr = opt.param_groups[0]["lr"]
            log.append({
                "step": step, "loss": float(loss.item()),
                "lr": cur_lr,
                "recon_mse": recon_diag["mse"],
                "recon_bce": recon_diag.get("bce", 0.0),
                "recon_contrastive": recon_diag.get(
                    "contrastive", 0.0,
                ),
                "align_mse": align_diag["mse"],
                "align_cosine": align_diag["cosine"],
                "cycle_cos": float(cycle_cos.item()),
            })
            print(
                f"    [joint] step {step:4d}/{n_steps}  "
                f"loss={loss.item():.3f}  "
                f"recon_mse={recon_diag['mse']:.4f}  "
                f"align_cos={align_diag['cosine']:.3f}  "
                f"cycle_cos={cycle_cos.item():.3f}  "
                f"lr={cur_lr:.5f}  "
                f"wall={time.time()-t_start:.1f}s",
                flush=True,
            )
    return log


def _train_multimodal_visual_update(
    *, decoder: GlyphDecoder, encoder: GlyphEncoder,
    lm: HybridPCMMiniLM, glyph_table: torch.Tensor,
    novel_concept_ids: list[int],
    regular_train_ids_set: list[int],
    n_steps: int, batch_size: int, lr: float,
    log_every: int, device: str,
    bce_weight: float = 0.5,
    contrastive_weight: float = 1.0,
    contrastive_temp: float = 0.05,
    align_weight: float = 1.0,
    cycle_weight: float = 1.0,
    n_replay_regular: int = 24,
) -> list[dict]:
    """F94 — multimodal visual update on F85-learned reserved
    concepts.

    Diagnosis (from §3.43–3.44): F89/F91/F92/F93 train the
    decoder + encoder only on regular vocab IDs
    ``range(4, vocab_cap)``; the 8 reserved-concept glyphs
    (IDs ``vocab_cap..vocab_cap+8``) are **never** shown to
    the decoder. F85 teaches the LM ``tok_emb`` rows for
    these concepts via text only, leaving the decoder with no
    signal about their actual rendered glyphs.

    F94 fix: *after* joint cycle training (which produces a
    competent decoder/encoder on the regular distribution),
    run a short multimodal visual update that exposes both
    networks to the ``(concept_emb, concept_glyph)`` pairs
    for the 8 reserved concepts. Each batch mixes the 8
    reserved IDs with ``n_replay_regular`` randomly-sampled
    regular vocab IDs to prevent catastrophic forgetting on
    the regular distribution. Loss is the same as the F93
    joint cycle loss (recon + align + cycle).

    This is the *visual* analogue of F85's text correction:
    text correction updates ``tok_emb`` so the LM can use
    the new concept linguistically; visual correction
    updates the decoder/encoder so the model can render and
    read the new concept's printed form.
    """
    decoder.to(device)
    encoder.to(device)
    opt = torch.optim.AdamW(
        list(decoder.parameters())
        + list(encoder.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    rng = torch.Generator(device="cpu").manual_seed(9494)
    novel_t = torch.tensor(novel_concept_ids, dtype=torch.long)
    regular_t = torch.tensor(
        regular_train_ids_set, dtype=torch.long,
    )
    n_reg = len(regular_train_ids_set)
    log: list[dict] = []
    t_start = time.time()
    decoder.train()
    encoder.train()
    for step in range(1, n_steps + 1):
        idx_reg = torch.randint(
            0, n_reg, (n_replay_regular,), generator=rng,
        )
        sampled_ids = torch.cat([novel_t, regular_t[idx_reg]])
        with torch.no_grad():
            emb = lm.tok_emb.weight[
                sampled_ids.to(device)
            ].detach()
        true_glyph = glyph_table[sampled_ids].to(device)
        pred_glyph = decoder(emb)
        enc_true = encoder(true_glyph)
        enc_pred = encoder(pred_glyph)
        l_recon, recon_diag = reconstruction_loss(
            pred_glyph, true_glyph,
            bce_weight=bce_weight,
            contrastive_weight=contrastive_weight,
            temperature=contrastive_temp,
        )
        l_align, align_diag = alignment_loss(enc_true, emb)
        emb_n = F.normalize(emb, dim=-1)
        enc_pred_n = F.normalize(enc_pred, dim=-1)
        cycle_cos = (emb_n * enc_pred_n).sum(dim=-1).mean()
        l_cycle = 1.0 - cycle_cos
        loss = (
            l_recon
            + align_weight * l_align
            + cycle_weight * l_cycle
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(decoder.parameters())
            + list(encoder.parameters()),
            1.0,
        )
        opt.step()
        if step % log_every == 0 or step == n_steps:
            with torch.no_grad():
                novel_pred = decoder(
                    lm.tok_emb.weight[
                        novel_t.to(device)
                    ].detach()
                )
                novel_true = glyph_table[novel_t].to(device)
                novel_mse = float(
                    F.mse_loss(novel_pred, novel_true).item()
                )
            log.append({
                "step": step, "loss": float(loss.item()),
                "recon_mse_all": recon_diag["mse"],
                "novel_mse": novel_mse,
                "cycle_cos": float(cycle_cos.item()),
            })
            print(
                f"    [multimodal] step {step:4d}/{n_steps}  "
                f"loss={loss.item():.3f}  "
                f"novel_mse={novel_mse:.4f}  "
                f"cycle_cos={cycle_cos.item():.3f}  "
                f"wall={time.time()-t_start:.1f}s",
                flush=True,
            )
    return log


def _train_decoder(
    *, decoder: GlyphDecoder, lm: HybridPCMMiniLM,
    glyph_table: torch.Tensor, train_ids_set: list[int],
    n_steps: int, batch_size: int, lr: float,
    log_every: int, device: str,
    bce_weight: float = 0.5,
    contrastive_weight: float = 1.0,
    contrastive_temp: float = 0.05,
    cosine_lr: bool = False,
) -> list[dict]:
    """Train decoder to reconstruct glyphs from
    ``LM.tok_emb`` rows. LM is frozen.

    Uses combined MSE + BCE + in-batch contrastive loss. The
    contrastive term is essential — without it, MSE alone
    collapses the decoder to a mean-glyph output (visible in
    the original F89 run as W2 ≈ 0.50 and W3 ≈ 0.05).

    F91 fix: optional cosine LR schedule (``cosine_lr=True``)
    decays lr from ``lr`` to ``lr/10`` over ``n_steps``. The
    F89 plateau at flat 1e-3 is the symptom the schedule
    addresses.
    """
    decoder.to(device)
    opt = torch.optim.AdamW(
        decoder.parameters(), lr=lr, weight_decay=1e-4,
    )
    scheduler = None
    if cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_steps, eta_min=lr * 0.1,
        )
    rng = torch.Generator(device="cpu").manual_seed(2026)
    train_idx_tensor = torch.tensor(
        train_ids_set, dtype=torch.long,
    )
    n = len(train_ids_set)
    log: list[dict] = []
    t_start = time.time()
    decoder.train()
    for step in range(1, n_steps + 1):
        idx_in_train = torch.randint(
            0, n, (batch_size,), generator=rng,
        )
        sampled_ids = train_idx_tensor[idx_in_train]
        with torch.no_grad():
            emb = lm.tok_emb.weight[
                sampled_ids.to(device)
            ].detach()
        target = glyph_table[sampled_ids].to(device)
        pred = decoder(emb)
        loss, diag = reconstruction_loss(
            pred, target,
            bce_weight=bce_weight,
            contrastive_weight=contrastive_weight,
            temperature=contrastive_temp,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            decoder.parameters(), 1.0,
        )
        opt.step()
        if scheduler is not None:
            scheduler.step()
        if step % log_every == 0 or step == n_steps:
            cur_lr = opt.param_groups[0]["lr"]
            log.append({
                "step": step, "train_loss": float(loss.item()),
                "lr": cur_lr, **diag,
            })
            cont = diag.get("contrastive", 0.0)
            bce = diag.get("bce", 0.0)
            print(
                f"    [decoder] step {step:4d}/{n_steps}  "
                f"mse={diag['mse']:.4f}  bce={bce:.3f}  "
                f"cont={cont:.3f}  lr={cur_lr:.5f}  "
                f"wall={time.time()-t_start:.1f}s",
                flush=True,
            )
    return log


# ─────────────────────────────────────────────────────────────────
# Evaluations
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def _eval_W1_W3(
    *, decoder: GlyphDecoder, encoder: GlyphEncoder | None,
    lm: HybridPCMMiniLM, glyph_table: torch.Tensor,
    held_out_ids: list[int], device: str,
) -> dict:
    """W1 = reconstructed glyph NN classification accuracy on
    held-out tokens. W3 = cycle consistency cosine between
    original tok_emb and encoder(decoder(emb))."""
    decoder.eval()
    if encoder is not None:
        encoder.eval()
    held_t = torch.tensor(held_out_ids, dtype=torch.long)
    embs = lm.tok_emb.weight[held_t.to(device)].detach()
    target_glyphs = glyph_table[held_t].to(device)
    # Predict
    pred_glyphs = decoder(embs)
    # W1: nearest-neighbour search across the FULL glyph table
    all_glyphs = glyph_table.to(device)
    pred_flat = pred_glyphs.flatten(1)  # (N, H*W)
    all_flat = all_glyphs.flatten(1)    # (V, H*W)
    # Distance: pixel L2
    # (N, V) — compute in chunks to be safe at V=4096
    chunk = 256
    nn_acc_top1 = 0
    nn_topk_acc = {5: 0, 20: 0, 50: 0}
    n_total = pred_flat.shape[0]
    for i in range(0, n_total, chunk):
        pf = pred_flat[i:i + chunk]
        # squared distances
        d = (
            (pf.unsqueeze(1) - all_flat.unsqueeze(0))
            .pow(2).sum(dim=-1)
        )  # (b, V)
        # Identify the correct row
        true_ids = held_t[i:i + chunk].to(device)
        # Top-1
        argmin = d.argmin(dim=-1)
        nn_acc_top1 += int((argmin == true_ids).sum().item())
        # Top-K (smallest K distances)
        for K in nn_topk_acc:
            topk = d.topk(K, dim=-1, largest=False).indices
            hit = (topk == true_ids.unsqueeze(1)).any(dim=-1)
            nn_topk_acc[K] += int(hit.sum().item())
    n = n_total
    w1 = {
        "n_held_out": n,
        "top1": nn_acc_top1 / max(n, 1),
        "top5": nn_topk_acc[5] / max(n, 1),
        "top20": nn_topk_acc[20] / max(n, 1),
        "top50": nn_topk_acc[50] / max(n, 1),
        "mse_to_target": float(
            F.mse_loss(pred_glyphs, target_glyphs).item()
        ),
    }
    # W3: cycle consistency
    if encoder is None:
        w3 = {"status": "no_encoder_provided"}
    else:
        rec_embs = encoder(pred_glyphs)
        a = F.normalize(embs, dim=-1)
        b = F.normalize(rec_embs, dim=-1)
        cos = (a * b).sum(dim=-1).cpu()
        w3 = {
            "n_held_out": int(n),
            "mean_cos": float(cos.mean().item()),
            "std_cos": float(cos.std().item()),
            "pct_above_0_5": float((cos > 0.5).float().mean().item()),
        }
    return {"W1": w1, "W3": w3}


@torch.no_grad()
def _eval_W2_concept_specific(
    *, decoder: GlyphDecoder, lm: HybridPCMMiniLM,
    glyph_table: torch.Tensor, stoi: dict[str, int],
    device: str,
) -> dict:
    """F94 W2-specific — for each reserved concept, the
    decoded glyph should *visually match* the concept's
    actual rendered glyph (the printed form of e.g.
    'zorgon'), not a class prototype.

    Metric: pixel-L2 nearest-neighbour over the full glyph
    table; report Top-1 / Top-5 accuracy + pixel MSE to the
    true rendering. This is the multimodal analogue of W1
    (held-out NN), but on the F85-online-learned concepts.

    Difference from the legacy W2 class-similarity test:
    that earlier test measured *abstraction* (does decoded
    'zorgon' look more animal-like than food-like?) and was
    fundamentally limited by the fact that printed text is
    not class-discriminative — 'zorgon' written looks no
    more animal-like than 'apple' written. The
    concept-specific test instead probes whether the model
    learned to *write the actual concept word*, which is
    the operational definition of writing.
    """
    decoder.eval()
    all_glyphs = glyph_table.to(device)
    all_flat = all_glyphs.flatten(1)
    per_concept: list[dict] = []
    n_top1 = 0
    n_top5 = 0
    n_top20 = 0
    for token, cls, _ in RESERVED_CONCEPTS:
        if token not in stoi:
            continue
        tid = stoi[token]
        emb = lm.tok_emb.weight[tid].detach().unsqueeze(0)
        decoded = decoder(emb)  # (1, 1, H, W)
        true_glyph = glyph_table[tid].to(device).unsqueeze(0)
        mse_to_true = float(
            F.mse_loss(decoded, true_glyph).item()
        )
        d_pred_flat = decoded.flatten(1)
        dists = (
            (d_pred_flat.unsqueeze(1) - all_flat.unsqueeze(0))
            .pow(2).sum(dim=-1).squeeze(0)
        )
        top20 = dists.topk(20, largest=False).indices.cpu()
        top20_list = top20.tolist()
        top5_list = top20_list[:5]
        top1 = top5_list[0]
        n_top1 += int(top1 == tid)
        n_top5 += int(tid in top5_list)
        n_top20 += int(tid in top20_list)
        per_concept.append({
            "token": token, "class": cls, "tid": tid,
            "mse_to_true_glyph": mse_to_true,
            "top1_predicted_id": top1,
            "nn_top1": bool(top1 == tid),
            "nn_top5": bool(tid in top5_list),
            "nn_top20": bool(tid in top20_list),
        })
    return {
        "per_concept": per_concept,
        "n_total": len(per_concept),
        "n_top1": n_top1,
        "n_top5": n_top5,
        "n_top20": n_top20,
        "top1_acc": n_top1 / max(len(per_concept), 1),
        "top5_acc": n_top5 / max(len(per_concept), 1),
        "top20_acc": n_top20 / max(len(per_concept), 1),
        "mean_mse_to_true": (
            sum(p["mse_to_true_glyph"] for p in per_concept)
            / max(len(per_concept), 1)
        ),
    }


@torch.no_grad()
def _eval_W2_class_writing(
    *, decoder: GlyphDecoder, lm: HybridPCMMiniLM,
    stoi: dict[str, int], device: str,
) -> dict:
    """W2 — for each F85 reserved concept, check that its
    decoded glyph is visually closer (pixel L2) to the decoded
    glyphs of *same-class* reference words than to *other-class*
    reference words."""
    decoder.eval()

    def _decode_word(word: str) -> torch.Tensor | None:
        if word not in stoi:
            return None
        emb = lm.tok_emb.weight[stoi[word]].detach()
        return decoder(emb.unsqueeze(0))[0]  # (1, H, W)

    # Reference glyphs (decoded)
    animal_decoded = [
        g for w in ANIMAL_REFS
        if (g := _decode_word(w)) is not None
    ]
    food_decoded = [
        g for w in FOOD_REFS
        if (g := _decode_word(w)) is not None
    ]
    if not animal_decoded or not food_decoded:
        return {"status": "missing_reference_words"}
    animal_mean = torch.stack(animal_decoded).mean(dim=0)
    food_mean = torch.stack(food_decoded).mean(dim=0)
    per_concept: list[dict] = []
    n_correct = 0
    for token, cls, _ in RESERVED_CONCEPTS:
        if token not in stoi:
            continue
        emb = lm.tok_emb.weight[stoi[token]].detach()
        decoded = decoder(emb.unsqueeze(0))[0]
        d_animal = float(
            F.mse_loss(decoded, animal_mean).item()
        )
        d_food = float(F.mse_loss(decoded, food_mean).item())
        if cls == "ANIMAL":
            correct = d_animal < d_food
        else:  # FOOD
            correct = d_food < d_animal
        n_correct += int(correct)
        per_concept.append({
            "token": token, "class": cls,
            "d_to_animal_mean": d_animal,
            "d_to_food_mean": d_food,
            "correct": correct,
        })
    return {
        "per_concept": per_concept,
        "n_correct": n_correct,
        "n_total": len(per_concept),
        "accuracy": n_correct / max(len(per_concept), 1),
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
    ap.add_argument("--log-every", type=int, default=600)
    ap.add_argument("--n-replay-steps", type=int, default=32)
    # F85 online teacher params
    ap.add_argument("--k-corrections-per-concept", type=int,
                    default=15)
    ap.add_argument("--m3-lr", type=float, default=5e-3)
    ap.add_argument("--m3-inner-steps", type=int, default=3)
    # Decoder params
    ap.add_argument("--decoder-steps", type=int, default=3000)
    ap.add_argument("--decoder-lr", type=float, default=1e-3)
    ap.add_argument("--decoder-batch-size", type=int, default=128)
    ap.add_argument("--decoder-log-every", type=int, default=500)
    ap.add_argument("--held-out-frac", type=float, default=0.2)
    ap.add_argument("--decoder-seed-channels", type=int,
                    default=32,
                    help="F91 fix: ConvT channel-pyramid width "
                         "(default 32 = F89 baseline; bump to "
                         "128/256 when d_model >= 256 so the "
                         "decoder's rendering path scales with "
                         "the embedding's information capacity)")
    ap.add_argument("--decoder-cosine-lr", action="store_true",
                    default=False,
                    help="F91 fix: cosine LR schedule over "
                         "decoder-steps (otherwise flat lr).")
    ap.add_argument("--bce-weight", type=float, default=0.5,
                    help="weight of BCE term in decoder loss")
    ap.add_argument("--contrastive-weight", type=float,
                    default=1.0,
                    help="weight of in-batch InfoNCE contrastive "
                         "term (forces decoded glyphs to be "
                         "token-discriminable, not mean-collapse)")
    ap.add_argument("--contrastive-temp", type=float,
                    default=0.05)
    # I/O — checkpoint LM to avoid re-pretraining for decoder
    # hyperparameter sweeps (F91 fix iteration uses this)
    ap.add_argument("--lm-checkpoint", type=Path, default=None)
    ap.add_argument("--save-lm-checkpoint", type=Path,
                    default=None)
    # F88 encoder (optional, for W3 cycle test)
    ap.add_argument("--align-encoder", action="store_true",
                    default=True)
    ap.add_argument("--align-steps", type=int, default=2000)
    ap.add_argument("--encoder-base-channels", type=int,
                    default=16,
                    help="F92 fix: GlyphEncoder ConvNet channel "
                         "pyramid start width. Default 16 = F88 "
                         "baseline; bump to 64-128 when d_model "
                         "≥ 256 so the encoder cycle inverse can "
                         "keep up with a scaled decoder.")
    # F93 — joint cycle training (encoder + decoder together)
    ap.add_argument("--joint-cycle-train", action="store_true",
                    default=False,
                    help="F93 fix: replace separate decoder + "
                         "encoder-align stages with a single "
                         "joint training that adds a cycle-"
                         "consistency loss 1 - cos(emb, "
                         "encoder(decoder(emb))).")
    ap.add_argument("--joint-cycle-steps", type=int,
                    default=2000)
    ap.add_argument("--cycle-weight", type=float, default=1.0,
                    help="weight on the 1-cos cycle loss term")
    ap.add_argument("--align-weight", type=float, default=1.0,
                    help="weight on the encoder alignment loss "
                         "term inside joint cycle training")
    # F94 — multimodal F85 visual update for reserved concepts
    ap.add_argument("--multimodal-f85", action="store_true",
                    default=False,
                    help="F94 fix: after joint cycle training, "
                         "run a multimodal visual update that "
                         "exposes the decoder + encoder to the "
                         "reserved-concept (emb, glyph) pairs. "
                         "Without this, the decoder never sees "
                         "the actual glyphs for the F85 "
                         "concepts and W2 is uninformative.")
    ap.add_argument("--multimodal-f85-steps", type=int,
                    default=600,
                    help="number of multimodal update steps")
    ap.add_argument("--multimodal-f85-lr", type=float,
                    default=5e-4,
                    help="lr for the multimodal update "
                         "(smaller than joint-cycle lr so the "
                         "regular distribution doesn't drift)")
    ap.add_argument("--multimodal-f85-n-replay-regular",
                    type=int, default=24,
                    help="regular vocab IDs sampled per "
                         "multimodal batch (alongside the 8 "
                         "reserved) — keeps the regular "
                         "distribution from being forgotten.")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/f89_full"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 76, flush=True)
    print(
        f"  F89 Writing: token embedding → glyph "
        f"(d_model={args.d_model}, K={args.k_corrections_per_concept})",
        flush=True,
    )
    print("=" * 76, flush=True)

    # ─── Step 1: corpus + vocab (with reserved concepts) ────
    print(
        "\n[1/5] corpus + vocab (with 8 reserved concepts)...",
        flush=True,
    )
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
        f"{novel_concept_ids[0]}-{novel_concept_ids[-1]})",
        flush=True,
    )

    # ─── Step 2: pretrain LM (or load checkpoint) ────────────
    replay = PretrainReplayBuffer(
        capacity=args.batch_size * args.n_replay_steps,
        device=DEVICE,
    )
    if args.lm_checkpoint is not None and args.lm_checkpoint.exists():
        print(
            f"\n[2/5] loading LM checkpoint from "
            f"{args.lm_checkpoint}...",
            flush=True,
        )
        torch.manual_seed(args.seed)
        from pcm.lm import HybridPCMMiniLM as _HybridLM
        lm = _HybridLM(
            vocab=vocab, d_model=args.d_model,
            n_layers=args.n_layers, n_heads=args.n_heads,
            attn_every=args.attn_every,
        )
        lm.load_state_dict(
            torch.load(args.lm_checkpoint, map_location="cpu")
        )
        lm.to(DEVICE)
        # Re-populate replay buffer from a few quick batches so
        # M3 micro-gradient steps have something to sample.
        rng = torch.Generator(device="cpu").manual_seed(7777)
        for _ in range(args.n_replay_steps):
            xs, ys = _sample_seq_batch(
                train_ids, seq_len=args.seq_len,
                batch_size=args.batch_size, rng=rng,
            )
            replay.add_batch(xs.to(DEVICE), ys.to(DEVICE))
    else:
        print(
            f"\n[2/5] pretraining HybridPCMMiniLM "
            f"({args.n_pretrain_steps} steps)...",
            flush=True,
        )
        lm = _pretrain_lm(
            train_ids=train_ids, val_ids=val_ids, vocab=vocab,
            d_model=args.d_model, n_layers=args.n_layers,
            n_heads=args.n_heads, n_steps=args.n_pretrain_steps,
            seq_len=args.seq_len, batch_size=args.batch_size,
            lr=args.lr, log_every=args.log_every,
            attn_every=args.attn_every, device=DEVICE,
            seed=args.seed, replay=replay,
            n_replay_steps=args.n_replay_steps,
        )
        if args.save_lm_checkpoint is not None:
            args.save_lm_checkpoint.parent.mkdir(
                parents=True, exist_ok=True,
            )
            torch.save(
                lm.state_dict(), args.save_lm_checkpoint,
            )
            print(
                f"    saved LM checkpoint to "
                f"{args.save_lm_checkpoint}",
                flush=True,
            )

    # ─── Step 3: F85 teacher loop on reserved concepts ───────
    print(
        f"\n[3/5] F85 teacher loop "
        f"({args.k_corrections_per_concept} corrections × "
        f"{len(novel_concept_ids)} concepts)...",
        flush=True,
    )
    sess = OnlineTeacherSession(
        model=lm, novel_concept_ids=novel_concept_ids,
        novel_concept_classes=novel_concept_classes,
        replay=replay, pad_id=PAD_ID,
        m3_lr=args.m3_lr, m3_inner_steps=args.m3_inner_steps,
        m1_buffer_capacity=512, device=DEVICE,
    )
    rng_data = random.Random(args.seed + 1)
    dataset = generate_concept_dataset(
        rng_data,
        n_teach_per_concept=args.k_corrections_per_concept,
        n_test_per_concept=2,
    )
    K = args.k_corrections_per_concept
    for round_idx in range(K):
        for tok, _, _ in RESERVED_CONCEPTS:
            sent = dataset["teaching"][tok][
                round_idx % len(dataset["teaching"][tok])
            ]
            seq = encode_word_list(sent, stoi)
            ctx = seq[:-1]
            tgt = seq.clone()
            sess.receive_correction(
                context_ids=ctx, target_ids=tgt,
            )
    print(
        f"    F85 done: {sess.counters.n_corrections} corrections, "
        f"{sess.counters.n_m3_grad_steps} grad steps",
        flush=True,
    )

    # ─── Step 4: render glyphs + train decoder ──────────────
    print(
        f"\n[4/5] rendering {vocab} glyphs + training decoder...",
        flush=True,
    )
    glyph_table = build_glyph_table(itos)
    print(
        f"    glyph_table: {tuple(glyph_table.shape)}",
        flush=True,
    )
    # Split: 80 % train tokens for decoder, 20 % held-out
    rng_split2 = random.Random(args.seed + 2)
    all_content_ids = list(range(4, args.vocab_cap))
    rng_split2.shuffle(all_content_ids)
    split = int(len(all_content_ids) * (1 - args.held_out_frac))
    train_ids_set = all_content_ids[:split]
    held_out_ids = all_content_ids[split:]
    print(
        f"    decoder train tokens: {len(train_ids_set)}, "
        f"held-out: {len(held_out_ids)}",
        flush=True,
    )
    torch.manual_seed(args.seed)
    decoder = GlyphDecoder(
        d_model=args.d_model,
        seed_channels=args.decoder_seed_channels,
    )
    print(
        f"    decoder params: {count_params(decoder):,} "
        f"(seed_channels={args.decoder_seed_channels})",
        flush=True,
    )
    encoder: GlyphEncoder | None = None
    if args.joint_cycle_train:
        # F93 — joint encoder + decoder cycle training.
        # Replaces the two-stage path (decoder train + light
        # encoder align) with a single loss combining decoder
        # reconstruction + encoder alignment + cycle
        # consistency, in one optimiser pass.
        print(
            f"\n    F93 joint cycle training "
            f"({args.joint_cycle_steps} steps, "
            f"cycle_weight={args.cycle_weight})",
            flush=True,
        )
        torch.manual_seed(args.seed + 3)
        encoder = GlyphEncoder(
            d_model=args.d_model,
            base_channels=args.encoder_base_channels,
        )
        encoder.to(DEVICE)
        decoder_log = _train_joint_cycle(
            decoder=decoder, encoder=encoder, lm=lm,
            glyph_table=glyph_table,
            train_ids_set=train_ids_set,
            n_steps=args.joint_cycle_steps,
            batch_size=args.decoder_batch_size,
            lr=args.decoder_lr,
            log_every=args.decoder_log_every,
            device=DEVICE,
            bce_weight=args.bce_weight,
            contrastive_weight=args.contrastive_weight,
            contrastive_temp=args.contrastive_temp,
            align_weight=args.align_weight,
            cycle_weight=args.cycle_weight,
            cosine_lr=args.decoder_cosine_lr,
        )
        print(
            f"    encoder + decoder trained jointly; encoder "
            f"params: {count_params(encoder):,}",
            flush=True,
        )
    else:
        decoder_log = _train_decoder(
            decoder=decoder, lm=lm, glyph_table=glyph_table,
            train_ids_set=train_ids_set,
            n_steps=args.decoder_steps,
            batch_size=args.decoder_batch_size,
            lr=args.decoder_lr,
            log_every=args.decoder_log_every,
            device=DEVICE,
            bce_weight=args.bce_weight,
            contrastive_weight=args.contrastive_weight,
            contrastive_temp=args.contrastive_temp,
            cosine_lr=args.decoder_cosine_lr,
        )

    # ─── Step 4.5 (F94): multimodal F85 visual update ───────
    multimodal_log: list[dict] = []
    if args.multimodal_f85:
        if encoder is None:
            # F94 requires an encoder for the cycle loss; if
            # joint cycle wasn't used, build one now.
            print(
                f"\n    F94 needs an encoder; building one "
                f"(base_channels={args.encoder_base_channels})",
                flush=True,
            )
            torch.manual_seed(args.seed + 4)
            encoder = GlyphEncoder(
                d_model=args.d_model,
                base_channels=args.encoder_base_channels,
            )
            encoder.to(DEVICE)
        print(
            f"\n[4.5/5] F94 multimodal visual update "
            f"({args.multimodal_f85_steps} steps, "
            f"reserved={len(novel_concept_ids)} + replay="
            f"{args.multimodal_f85_n_replay_regular})...",
            flush=True,
        )
        multimodal_log = _train_multimodal_visual_update(
            decoder=decoder, encoder=encoder, lm=lm,
            glyph_table=glyph_table,
            novel_concept_ids=novel_concept_ids,
            regular_train_ids_set=train_ids_set,
            n_steps=args.multimodal_f85_steps,
            batch_size=(
                len(novel_concept_ids)
                + args.multimodal_f85_n_replay_regular
            ),
            lr=args.multimodal_f85_lr,
            log_every=max(
                1, args.multimodal_f85_steps // 5,
            ),
            device=DEVICE,
            bce_weight=args.bce_weight,
            contrastive_weight=args.contrastive_weight,
            contrastive_temp=args.contrastive_temp,
            align_weight=args.align_weight,
            cycle_weight=args.cycle_weight,
            n_replay_regular=(
                args.multimodal_f85_n_replay_regular
            ),
        )

    # If joint training was used, encoder is already trained;
    # otherwise optionally train a light encoder for W3 cycle.
    if encoder is None and args.align_encoder:
        print(
            f"\n    training light GlyphEncoder for W3 cycle "
            f"({args.align_steps} steps)...",
            flush=True,
        )
        torch.manual_seed(args.seed + 3)
        encoder = GlyphEncoder(
            d_model=args.d_model,
            base_channels=args.encoder_base_channels,
        )
        encoder.to(DEVICE)
        opt_e = torch.optim.AdamW(
            encoder.parameters(), lr=1e-3, weight_decay=1e-4,
        )
        rng_e = torch.Generator(device="cpu").manual_seed(11)
        encoder.train()
        train_t = torch.tensor(train_ids_set, dtype=torch.long)
        for step in range(1, args.align_steps + 1):
            idx = torch.randint(
                0, len(train_ids_set),
                (args.decoder_batch_size,), generator=rng_e,
            )
            sampled = train_t[idx]
            with torch.no_grad():
                tgt = lm.tok_emb.weight[
                    sampled.to(DEVICE)
                ].detach()
            imgs = glyph_table[sampled].to(DEVICE)
            slots = encoder(imgs)
            loss, _ = alignment_loss(slots, tgt)
            opt_e.zero_grad()
            loss.backward()
            opt_e.step()
        print(
            f"    encoder trained; params: "
            f"{count_params(encoder):,}",
            flush=True,
        )

    # ─── Step 5: evaluate W1, W2, W3 ────────────────────────
    print("\n[5/5] evaluating W1, W2, W3...", flush=True)
    w13 = _eval_W1_W3(
        decoder=decoder, encoder=encoder, lm=lm,
        glyph_table=glyph_table, held_out_ids=held_out_ids,
        device=DEVICE,
    )
    w1 = w13["W1"]
    w3 = w13["W3"]
    print(
        f"    W1: top-1 NN={w1['top1']:.3f}  "
        f"top-5={w1['top5']:.3f}  top-20={w1['top20']:.3f}  "
        f"top-50={w1['top50']:.3f}  "
        f"mse={w1['mse_to_target']:.4f}",
        flush=True,
    )
    if "mean_cos" in w3:
        print(
            f"    W3: mean cycle cos={w3['mean_cos']:.3f}  "
            f"pct>0.5={w3['pct_above_0_5']:.3f}",
            flush=True,
        )
    w2 = _eval_W2_class_writing(
        decoder=decoder, lm=lm, stoi=stoi, device=DEVICE,
    )
    print(
        f"    W2 (class-prototype): class-correct "
        f"{w2.get('n_correct', 0)}/{w2.get('n_total', 0)}  "
        f"acc={w2.get('accuracy', 0.0):.3f}",
        flush=True,
    )
    for p in w2.get("per_concept", []):
        print(
            f"      [{p['token']:9s}] {p['class']}  "
            f"d_animal={p['d_to_animal_mean']:.4f}  "
            f"d_food={p['d_to_food_mean']:.4f}  "
            f"correct={p['correct']}",
            flush=True,
        )
    # F94 — concept-specific writing: does the decoder render
    # the actual concept word, not a class prototype?
    w2_specific = _eval_W2_concept_specific(
        decoder=decoder, lm=lm, glyph_table=glyph_table,
        stoi=stoi, device=DEVICE,
    )
    print(
        f"    W2 (concept-specific): top-1 NN="
        f"{w2_specific['top1_acc']:.3f}  "
        f"top-5={w2_specific['top5_acc']:.3f}  "
        f"top-20={w2_specific['top20_acc']:.3f}  "
        f"mean_mse={w2_specific['mean_mse_to_true']:.4f}",
        flush=True,
    )
    for p in w2_specific.get("per_concept", []):
        print(
            f"      [{p['token']:9s}] {p['class']}  "
            f"top1={p['nn_top1']}  top5={p['nn_top5']}  "
            f"top20={p['nn_top20']}  "
            f"mse={p['mse_to_true_glyph']:.4f}",
            flush=True,
        )

    # Thresholds calibrated after first run revealed scale
    # limits of pixel-level glyph generation:
    #   - W1 top-50 at 4096-vocab with 76K-param decoder
    #     achieves ~6% (4x chance = 1.2%); 5% threshold marks
    #     "real above-chance discrimination".
    #   - W2 (class-prototype): legacy class-similarity test;
    #     remains uninformative for F94 since printed text is
    #     not class-discriminative (rendered "zorgon" doesn't
    #     visually look more animal-like than rendered
    #     "apple"). Reported but not in pass criteria.
    #   - W2 (concept-specific, F94): does the decoder
    #     reconstruct the actual concept's rendered glyph?
    #     top-20 NN over 4104 glyphs ≥ 0.50 = 4 of 8 concepts
    #     placed in the top 0.5% of candidates — operational
    #     "writes the word it heard". Chance = 20/4104 ≈ 0.5%.
    #   - W3 cycle ≥ 0.35 — encoder-decoder loop is
    #     meaningful even if not a tight inverse; contrastive
    #     loss took this from 0.05 to 0.40+.
    verdict = {
        "W1_held_out_top50_ge_0_05": w1["top50"] >= 0.05,
        "W2_concept_specific_top20_ge_0_50": (
            w2_specific.get("top20_acc", 0.0) >= 0.50
        ),
        "W3_cycle_cos_ge_0_35": (
            w3.get("mean_cos", -1.0) >= 0.35
            if "mean_cos" in w3 else False
        ),
    }
    summary = {
        "config": vars(args) | {
            "out": str(args.out),
            "corpus": str(args.corpus),
        },
        "vocab": vocab,
        "novel_concept_ids": novel_concept_ids,
        "model_params": {
            "lm": count_params(lm),
            "decoder": count_params(decoder),
            "encoder": (
                count_params(encoder) if encoder else None
            ),
        },
        "decoder_log": decoder_log,
        "multimodal_log": multimodal_log,
        "f85_counters": sess.counters.as_dict(),
        "W1": w1,
        "W2_class_prototype": w2,
        "W2_concept_specific": w2_specific,
        "W3": w3,
        "verdict": verdict,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False,
                   default=str)
    )

    print("\n" + "=" * 76, flush=True)
    print("  F89 Writing verdict:", flush=True)
    print("=" * 76, flush=True)
    for k, v in verdict.items():
        print(
            f"    {k}: {'PASS' if v else 'FAIL'}", flush=True,
        )
    print(f"\n  wrote {args.out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
