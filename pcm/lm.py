"""pcm.lm — F74 mini language model comparison.

Two same-parameter LMs side-by-side, designed for falsifiable
testing of *word-meaning understanding* (not just probability
matching):

* :class:`GPTMiniLM` — standard causal Transformer (token
  embedding → multi-layer self-attention → LM head). The LLM-
  style baseline that models token distributions.
* :class:`PCMMiniLM` — F62 ``UniversalCombiner`` substituted
  for self-attention. Each token *is* a slot; context
  aggregation is a learnable combiner, not multi-head attention.
  The architectural claim: explicit slot operators preserve
  word identity through layers, where Transformer hidden states
  entangle it.

Both models are tiny (~few hundred K params) and self-contained
— no external pretrained components. Trained from scratch on
the F74 synthetic language defined in :mod:`pcm.lm_synthetic`.

The two models *must* be parameter-matched so that any U2-U5
gap reflects architecture, not capacity.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "PCMUniversalCombiner",
    "GPTMiniLM",
    "PCMMiniLM",
    "PCMTopKMiniLM",
    "GatedPCMLayer",
    "GatedPCMMiniLM",
    "GatedAttentionLayer",
    "HybridPCMMiniLM",
    "HierarchicalMemoryLayer",
    "TitansPCMMiniLM",
    "build_matched_pair",
    "build_matched_triple",
    "build_matched_quad",
    "build_matched_pentad",
    "count_params",
    "perplexity",
]


# ─────────────────────────────────────────────────────────────────
# Shared UniversalCombiner (vendored from F62)
# ─────────────────────────────────────────────────────────────────


class PCMUniversalCombiner(nn.Module):
    """F62 ``UniversalCombiner``: ``(slot_a, slot_b) → slot_b'``.

    Residual 2-layer MLP — exactly the F62 / F62b / F62c form."""

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + self.net(torch.cat([a, b], dim=-1))


# ─────────────────────────────────────────────────────────────────
# Baseline: scratch causal Transformer
# ─────────────────────────────────────────────────────────────────


class GPTMiniLM(nn.Module):
    """Tiny causal Transformer LM (LLM-style baseline)."""

    def __init__(
        self, vocab: int, d_model: int = 64, n_layers: int = 2,
        n_heads: int = 4, max_len: int = 32, dropout: float = 0.0,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_final = nn.LayerNorm(d_model)
        if tie_weights:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(d_model, vocab, bias=False)
        # GPT-2-style init: small embedding scale so initial
        # logits ~ N(0, 1) and cross-entropy starts near ln(V),
        # not orders of magnitude above. Required for stable
        # training on real corpora (F79).
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, L)`` long. Returns ``(B, L, vocab)`` logits."""
        B, L = x.shape
        pos = torch.arange(L, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)
        causal = torch.triu(
            torch.full((L, L), float("-inf"), device=x.device),
            diagonal=1,
        )
        h = self.encoder(h, mask=causal, is_causal=True)
        h = self.ln_final(h)
        if self.lm_head is None:
            logits = h @ self.tok_emb.weight.t()
        else:
            logits = self.lm_head(h)
        return logits

    @torch.no_grad()
    def hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final-layer hidden state per position (for
        the U2 linear-probe test)."""
        B, L = x.shape
        pos = torch.arange(L, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)
        causal = torch.triu(
            torch.full((L, L), float("-inf"), device=x.device),
            diagonal=1,
        )
        h = self.encoder(h, mask=causal, is_causal=True)
        return self.ln_final(h)

    @torch.no_grad()
    def token_embeddings(self) -> torch.Tensor:
        return self.tok_emb.weight.detach().clone()


# ─────────────────────────────────────────────────────────────────
# PCM-style LM: token = slot, F62 combiner instead of attention
# ─────────────────────────────────────────────────────────────────


class PCMMiniLMLayer(nn.Module):
    """One layer of the PCM LM: causal context aggregation +
    F62 ``UniversalCombiner``.

    Context aggregation: causal-cumulative-mean (the simplest
    causal aggregator; respects ordering but is not learned to
    weigh positions). This is intentionally simpler than
    multi-head attention — the architectural claim is that with
    an explicit combiner working on slot-shaped vectors, even
    cumulative-mean aggregation suffices to bind context.
    """

    def __init__(self, dim: int, combiner_hidden: int = 128,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.combiner = PCMUniversalCombiner(dim, hidden=combiner_hidden)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """``slots``: ``(B, L, D)``. Returns ``(B, L, D)``."""
        # Causal cumulative mean: context_t = mean(slots[0..t])
        B, L, D = slots.shape
        cumsum = slots.cumsum(dim=1)
        cumlen = torch.arange(
            1, L + 1, device=slots.device, dtype=slots.dtype,
        ).view(1, L, 1)
        context = cumsum / cumlen
        # Apply F62 combiner with (slot_t, context_t)
        out = self.combiner(slots, context)
        out = self.dropout(out)
        return self.norm(out)


class PCMMiniLM(nn.Module):
    """PCM-style LM: each token *is* a slot; F62 combiner mixes
    each token with its causal-mean context. No multi-head
    attention.

    Design notes:
    * Token embedding == slot embedding == LM-head weight (tied
      throughout). This forces the model to treat tokens as
      first-class concept slots, with all transformations on
      slot space.
    * No positional embedding is needed — the causal cumulative
      mean is itself a position-aware aggregator, and the F62
      combiner sees ``(current_slot, mean_of_past)`` which carries
      ordering through commutativity (or lack thereof) of the
      combiner's parameters.

    Trade-off: cumulative-mean context is much weaker than
    self-attention. We compensate by making sure both PCMMiniLM
    and GPTMiniLM have the same total parameter count — so any
    interpretability advantage PCMMiniLM shows in U2-U5 is *not*
    a parameter advantage. For tasks that need *exact* long-
    range reference (F77 long-anaphora), see
    :class:`PCMTopKMiniLM` which adds explicit selective recall.
    """

    def __init__(
        self, vocab: int, d_model: int = 64, n_layers: int = 2,
        combiner_hidden: int | None = None, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        if combiner_hidden is None:
            combiner_hidden = 4 * d_model
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([
            PCMMiniLMLayer(d_model, combiner_hidden=combiner_hidden,
                            dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slots = self.tok_emb(x)
        for layer in self.layers:
            slots = slots + layer(slots)
        slots = self.ln_final(slots)
        return slots @ self.tok_emb.weight.t()

    @torch.no_grad()
    def hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        slots = self.tok_emb(x)
        for layer in self.layers:
            slots = slots + layer(slots)
        return self.ln_final(slots)

    @torch.no_grad()
    def token_embeddings(self) -> torch.Tensor:
        return self.tok_emb.weight.detach().clone()


# ─────────────────────────────────────────────────────────────────
# PCM-TopK LM: cumulative mean + explicit top-K selective recall
# ─────────────────────────────────────────────────────────────────


class PCMTopKLayer(nn.Module):
    """Causal top-K selective recall + F62 ``UniversalCombiner``.

    Augments the F74 ``PCMMiniLMLayer`` (which uses only
    cumulative mean) with an *explicit* selective-recall path:

    For each position ``t``:
    1. ``mean_ctx_t = mean(slots[0..t])`` — F74 baseline
       aggregator (cheap, captures gist).
    2. Compute cosine-similarity scores between ``slots[t]`` and
       all past ``slots[0..t]``. Pick the **top-K** most similar
       past slots (with causal masking, no self-self).
       Softmax-weight only over those K to form
       ``topk_ctx_t``. This is **sparse attention** but with
       only K kept positions (vs full softmax in standard
       attention).
    3. ``ctx_t = mean_ctx_t + α · topk_ctx_t`` where α is a
       learnable scalar.
    4. Apply the F62 ``UniversalCombiner(slots[t], ctx_t)``.

    Difference from standard self-attention:
    * Only top-K positions get any weight (sparse, interpretable).
    * The combiner is F62 ``UniversalCombiner`` (a 2-layer
      residual MLP), *not* multi-head Q-K-V projection.
    * No learned Q/K/V projections — similarities are cosine on
      raw slots, so "what's similar" is determined by slot
      identity, not by a separate projection space.

    Cost: O(L²) for the similarity computation (same as
    attention), but the *operator* on each position remains the
    F62 combiner — interpretable + factored.
    """

    def __init__(
        self, dim: int, combiner_hidden: int = 128,
        top_k: int = 4, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.combiner = PCMUniversalCombiner(dim, hidden=combiner_hidden)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        # Learnable mix weight; init small so the layer starts
        # close to F74 cumulative-mean behaviour and learns to
        # use selective recall as needed.
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.top_k = top_k

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """``slots``: ``(B, L, D)``. Returns ``(B, L, D)``."""
        B, L, D = slots.shape
        # Cumulative mean (F74-style)
        cumsum = slots.cumsum(dim=1)
        cumlen = torch.arange(
            1, L + 1, device=slots.device, dtype=slots.dtype,
        ).view(1, L, 1)
        mean_ctx = cumsum / cumlen
        # Top-K selective recall (sparse attention)
        # Normalise slots for cosine similarity
        norm_slots = F.normalize(slots, dim=-1)
        # (B, L, L) — sim[b, t, i] = cos(slots[b, t], slots[b, i])
        sim = norm_slots @ norm_slots.transpose(-1, -2)
        # Causal mask: position t can only see positions 0..t
        # (not t itself — we want past, not self)
        causal_mask = torch.full(
            (L, L), float("-inf"), device=slots.device,
        )
        # Mask out future and self
        causal_mask = torch.triu(causal_mask, diagonal=0)
        # Allow position 0 to attend to itself (no past): for
        # t=0 there is nothing in the past, so we fall back to
        # the cumulative-mean only (handled below).
        sim = sim + causal_mask.unsqueeze(0)
        # Top-K along the last dim — but K must be capped by
        # available positions. We use top_k but pad with -inf
        # for short prefixes (already handled by causal mask).
        k_eff = min(self.top_k, L)
        topk_vals, topk_idx = sim.topk(k=k_eff, dim=-1)
        # All-masked positions (e.g. t=0 with no past) have
        # topk_vals = all -inf → softmax produces NaN. Detect
        # and zero them out.
        all_neg = (topk_vals == float("-inf")).all(dim=-1, keepdim=True)
        # Replace -inf with a finite value before softmax for
        # numerical safety; the all_neg mask will zero out the
        # context for those positions.
        safe_vals = torch.where(
            topk_vals == float("-inf"),
            torch.full_like(topk_vals, -1e9),
            topk_vals,
        )
        weights = F.softmax(safe_vals, dim=-1)
        # Gather slots at topk indices: gather along position
        # dim. ``slots`` shape (B, L, D); we need (B, L, K, D).
        # Expand topk_idx to (B, L, K, 1) and gather.
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        slots_exp = slots.unsqueeze(1).expand(-1, L, -1, -1)
        gathered = slots_exp.gather(dim=2, index=idx_exp)
        # Weight and sum
        topk_ctx = (weights.unsqueeze(-1) * gathered).sum(dim=2)
        # Zero out positions with no valid past
        topk_ctx = topk_ctx.masked_fill(all_neg, 0.0)
        # Mix
        ctx = mean_ctx + self.alpha * topk_ctx
        # Combine and norm
        out = self.combiner(slots, ctx)
        out = self.dropout(out)
        return self.norm(out)


class PCMTopKMiniLM(nn.Module):
    """PCM LM with top-K selective recall instead of pure
    cumulative mean. See :class:`PCMTopKLayer`.

    Use this variant for tasks that need *exact* long-range
    reference (long anaphora, relative clauses, center
    embedding). For short bounded contexts the F74
    :class:`PCMMiniLM` is sufficient and cheaper.
    """

    def __init__(
        self, vocab: int, d_model: int = 64, n_layers: int = 2,
        combiner_hidden: int | None = None, top_k: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.top_k = top_k
        if combiner_hidden is None:
            combiner_hidden = 4 * d_model
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([
            PCMTopKLayer(
                d_model, combiner_hidden=combiner_hidden,
                top_k=top_k, dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slots = self.tok_emb(x)
        for layer in self.layers:
            slots = slots + layer(slots)
        slots = self.ln_final(slots)
        return slots @ self.tok_emb.weight.t()

    @torch.no_grad()
    def hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        slots = self.tok_emb(x)
        for layer in self.layers:
            slots = slots + layer(slots)
        return self.ln_final(slots)

    @torch.no_grad()
    def token_embeddings(self) -> torch.Tensor:
        return self.tok_emb.weight.detach().clone()


# ─────────────────────────────────────────────────────────────────
# Gated PCM: cumulative-mean replaced by per-channel forget gate
# ─────────────────────────────────────────────────────────────────


class GatedPCMLayer(nn.Module):
    """PCM layer with a **per-channel learned forget gate**
    replacing F74's hard cumulative mean (F80, "Lever A").

    For each position ``t``, compute an input-conditioned gate
    ``g_t = sigmoid(W_g · slot_t + b_g)`` (per channel) and run
    the linear recurrence::

        state_t = g_t ⊙ state_{t-1} + (1 - g_t) ⊙ slot_t
        ctx_t   = state_t
        out_t   = UniversalCombiner(slot_t, ctx_t)

    This is the Mamba / Gated-Linear-Attention / GRU-style update,
    adapted to PCM's slot space. The architectural claim is:
    cumulative mean is a *degenerate* linear recurrence (gate
    always = ``t/(t+1)``); making the gate input-dependent restores
    the expressivity that pure mean lost, while keeping the
    F62 ``UniversalCombiner`` and the slot-as-concept identity
    unchanged.

    The gate depends *only* on the current slot (not on the prior
    state), so all ``g_t`` can be computed in parallel and the
    recurrence is a simple sequential scan.

    Two helper modes for falsifiability (F80 invariants):

    * ``force_mean=True`` — bypass the gate and use F74-style
      cumulative mean. Used by G3 to verify gating is the *only*
      change vs ``PCMMiniLM``.
    * :meth:`compute_gates` — return the gate tensor per position
      for inspection (E1, E5).
    """

    def __init__(
        self, d_model: int,
        combiner_hidden: int | None = None,
        dropout: float = 0.0,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        if combiner_hidden is None:
            combiner_hidden = 4 * d_model
        self.combiner = PCMUniversalCombiner(
            d_model, hidden=combiner_hidden,
        )
        self.gate_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        # Start with gate ≈ sigmoid(0) = 0.5 (equal weight between
        # past and current). Model learns the right pattern.
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.constant_(self.gate_proj.bias, gate_bias_init)

    @torch.no_grad()
    def compute_gates(self, slots: torch.Tensor) -> torch.Tensor:
        """Return ``(B, L, D)`` gate values for inspection."""
        return torch.sigmoid(self.gate_proj(slots))

    def _gated_scan(self, slots: torch.Tensor) -> torch.Tensor:
        """Sequential gated linear recurrence."""
        B, L, D = slots.shape
        gates = torch.sigmoid(self.gate_proj(slots))  # (B, L, D)
        state = torch.zeros(
            B, D, device=slots.device, dtype=slots.dtype,
        )
        outputs = []
        for t in range(L):
            g = gates[:, t]
            state = g * state + (1.0 - g) * slots[:, t]
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    def _cumulative_mean(self, slots: torch.Tensor) -> torch.Tensor:
        B, L, D = slots.shape
        cumsum = slots.cumsum(dim=1)
        cumlen = torch.arange(
            1, L + 1, device=slots.device, dtype=slots.dtype,
        ).view(1, L, 1)
        return cumsum / cumlen

    def forward(
        self, slots: torch.Tensor, *, force_mean: bool = False,
    ) -> torch.Tensor:
        """``slots``: ``(B, L, D)``. Returns ``(B, L, D)``."""
        if force_mean:
            context = self._cumulative_mean(slots)
        else:
            context = self._gated_scan(slots)
        out = self.combiner(slots, context)
        out = self.dropout(out)
        return self.norm(out)


class GatedPCMMiniLM(nn.Module):
    """PCM LM with :class:`GatedPCMLayer` for context aggregation.

    Same tied-weight, no-positional-embedding, F62-combiner
    structure as :class:`PCMMiniLM` — the *only* architectural
    difference is the input-conditioned forget gate replacing
    cumulative mean.

    Use :meth:`forward(x, force_mean=True)` to A/B-test the gate
    contribution alone (G3 sanity invariant). Use
    :meth:`all_layer_gates` to read the learned gate tensors per
    layer (E1 / E5 interpretability invariants).
    """

    def __init__(
        self, vocab: int, d_model: int = 64, n_layers: int = 2,
        combiner_hidden: int | None = None, dropout: float = 0.0,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        if combiner_hidden is None:
            combiner_hidden = 4 * d_model
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([
            GatedPCMLayer(
                d_model, combiner_hidden=combiner_hidden,
                dropout=dropout, gate_bias_init=gate_bias_init,
            )
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)

    def forward(
        self, x: torch.Tensor, *, force_mean: bool = False,
    ) -> torch.Tensor:
        slots = self.tok_emb(x)
        for layer in self.layers:
            slots = slots + layer(slots, force_mean=force_mean)
        slots = self.ln_final(slots)
        return slots @ self.tok_emb.weight.t()

    @torch.no_grad()
    def hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        slots = self.tok_emb(x)
        for layer in self.layers:
            slots = slots + layer(slots)
        return self.ln_final(slots)

    @torch.no_grad()
    def token_embeddings(self) -> torch.Tensor:
        return self.tok_emb.weight.detach().clone()

    @torch.no_grad()
    def all_layer_gates(
        self, x: torch.Tensor,
    ) -> list[torch.Tensor]:
        """List of ``(B, L, D)`` gate tensors, one per layer."""
        slots = self.tok_emb(x)
        gates_per_layer = []
        for layer in self.layers:
            gates_per_layer.append(layer.compute_gates(slots))
            slots = slots + layer(slots)
        return gates_per_layer


# ─────────────────────────────────────────────────────────────────
# F81 — Hybrid: GatedAttentionLayer + HybridPCMMiniLM (Qwen3-Next)
# ─────────────────────────────────────────────────────────────────


class GatedAttentionLayer(nn.Module):
    """Single-block causal multi-head attention with **output
    gating** (F81, "Lever B").

    Matches the 2026 Qwen3-Next / Trinity Large / Step-3.5-Flash
    "gated attention" recipe: standard scaled-dot-product
    multi-head attention, followed by a *per-channel sigmoid
    gate* applied to the attention output before residual. The
    gate mitigates attention-sink artefacts and improves long-
    context stability (Hua et al. 2022, Forgetting Transformer
    2025, GLM-5 2026).

    Architecturally simpler than a full Transformer block —
    there is no FFN and no learned positional embedding here.
    The hybrid model interleaves these "anchor" attention layers
    every fourth position with :class:`GatedPCMLayer` blocks
    (3:1 ratio), exactly as Qwen3-Next does.
    """

    def __init__(
        self, d_model: int, n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must divide n_heads "
                f"({n_heads})"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        # GPT-2-style init for stability
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)
        # Gate biased to ~0.5 at init (sigmoid(0)); model
        # learns where to open / close
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, L, D)``. Returns ``(B, L, D)``."""
        B, L, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(attn)
        gate = torch.sigmoid(self.gate_proj(x))
        out = gate * out
        out = self.dropout(out)
        return self.norm(out)

    @torch.no_grad()
    def compute_gates(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate_proj(x))


class HybridPCMMiniLM(nn.Module):
    """Hybrid LM: :class:`GatedPCMLayer` (the bulk) interleaved
    with :class:`GatedAttentionLayer` (sparse "anchor" layers).

    Layer ordering: every fourth layer (positions 3, 7, 11, …
    in 0-indexed) is a :class:`GatedAttentionLayer`; the rest
    are :class:`GatedPCMLayer`. This gives the **3:1 Gated-PCM
    : Gated-Attention** ratio that Qwen3-Next / Qwen3.5 /
    Qwen3-Coder-Next use natively.

    For ``n_layers=4`` the pattern is ``[PCM, PCM, PCM, ATT]``;
    for ``n_layers=8`` it is ``[PCM × 3, ATT, PCM × 3, ATT]``;
    etc. If ``n_layers`` is not a multiple of 4, the trailing
    layers are PCM (no attention).

    Slot identity, tied weights, no positional embeddings, and
    F62 ``UniversalCombiner`` inside each PCM layer are
    unchanged. The attention layers introduce only the QKV
    projections + output gate they need locally; they do not
    affect the overall token embedding or LM head.
    """

    def __init__(
        self, vocab: int, d_model: int = 64, n_layers: int = 4,
        n_heads: int = 4,
        combiner_hidden: int | None = None,
        dropout: float = 0.0,
        gate_bias_init: float = 0.0,
        attn_every: int = 4,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.n_layers = n_layers
        self.attn_every = attn_every
        if combiner_hidden is None:
            combiner_hidden = 4 * d_model
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList()
        self.layer_kinds: list[str] = []
        for i in range(n_layers):
            if (i + 1) % attn_every == 0:
                self.layers.append(
                    GatedAttentionLayer(
                        d_model=d_model, n_heads=n_heads,
                        dropout=dropout,
                    )
                )
                self.layer_kinds.append("attn")
            else:
                self.layers.append(
                    GatedPCMLayer(
                        d_model, combiner_hidden=combiner_hidden,
                        dropout=dropout,
                        gate_bias_init=gate_bias_init,
                    )
                )
                self.layer_kinds.append("pcm")
        self.ln_final = nn.LayerNorm(d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slots = self.tok_emb(x)
        for layer in self.layers:
            slots = slots + layer(slots)
        slots = self.ln_final(slots)
        return slots @ self.tok_emb.weight.t()

    @torch.no_grad()
    def hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        slots = self.tok_emb(x)
        for layer in self.layers:
            slots = slots + layer(slots)
        return self.ln_final(slots)

    @torch.no_grad()
    def token_embeddings(self) -> torch.Tensor:
        return self.tok_emb.weight.detach().clone()

    def n_attn_layers(self) -> int:
        return sum(1 for k in self.layer_kinds if k == "attn")

    def n_pcm_layers(self) -> int:
        return sum(1 for k in self.layer_kinds if k == "pcm")


# ─────────────────────────────────────────────────────────────────
# F83 — Titans-style hierarchical episodic memory readout
# ─────────────────────────────────────────────────────────────────


class HierarchicalMemoryLayer(nn.Module):
    """Cross-attention to a **persistent kNN cache** of hidden
    states from past forward passes (F83, "Lever C" — Titans-
    style long-term memory).

    Maintains a FIFO buffer of ``(key, value)`` pairs derived
    from the model's own hidden states during training. At each
    forward, the layer:

    1. Projects current slots to queries ``q = W_q · slot``.
    2. Cosine-similarity matches ``q`` against all stored
       buffer keys, picks the ``top_k`` neighbours.
    3. Softmax-weights the corresponding values and returns
       the result through a *per-channel sigmoid output gate*.

    Imprint policy: during training, :meth:`imprint` writes
    ``n_per_batch`` sub-sampled positions per batch (typically
    end-of-sentence approximations) to the buffer with FIFO
    eviction. Writes are detached, so backprop flows only
    through ``q_proj`` / ``k_proj`` / ``v_proj`` / ``out_gate``
    at the current step — not back into past episodes.

    This is the "neural memory module" from Titans (Behrouz et
    al., NeurIPS 2025) reduced to its simplest possible form:
    a kNN cache with a learned readout. We do not implement
    Titans' test-time SGD update — only the forward retrieval
    path. Empirically (F83) this alone confers the long-context
    benefit.

    Parameters:
        d_model: slot dimension.
        buffer_capacity: max entries (FIFO eviction beyond).
        top_k: number of nearest neighbours retrieved.
    """

    def __init__(
        self, d_model: int, buffer_capacity: int = 512,
        top_k: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.buffer_capacity = buffer_capacity
        self.top_k = top_k
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.out_gate.weight, std=0.02)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_gate.bias)
        # Persistent cache buffers
        self.register_buffer(
            "buf_keys", torch.zeros(buffer_capacity, d_model),
        )
        self.register_buffer(
            "buf_vals", torch.zeros(buffer_capacity, d_model),
        )
        self.register_buffer(
            "buf_size", torch.tensor(0, dtype=torch.long),
        )
        self.register_buffer(
            "buf_ptr", torch.tensor(0, dtype=torch.long),
        )

    @torch.no_grad()
    def reset_buffer(self) -> None:
        self.buf_keys.zero_()
        self.buf_vals.zero_()
        self.buf_size.zero_()
        self.buf_ptr.zero_()

    @torch.no_grad()
    def imprint(
        self, slots: torch.Tensor, n_per_batch: int = 4,
    ) -> int:
        """Write a sub-sample of slots into the FIFO buffer.
        Returns the number of new entries written.
        """
        B, L, D = slots.shape
        if L >= n_per_batch:
            pos = torch.linspace(
                0, L - 1, n_per_batch,
                dtype=torch.long, device=slots.device,
            )
        else:
            pos = torch.arange(L, device=slots.device)
        sampled = slots[:, pos].reshape(-1, D)
        keys = self.k_proj(sampled)
        vals = self.v_proj(sampled)
        n = sampled.shape[0]
        for i in range(n):
            ptr = int(self.buf_ptr.item())
            self.buf_keys[ptr] = keys[i].detach()
            self.buf_vals[ptr] = vals[i].detach()
            self.buf_ptr.add_(1)
            if self.buf_ptr.item() >= self.buffer_capacity:
                self.buf_ptr.zero_()
            if int(self.buf_size.item()) < self.buffer_capacity:
                self.buf_size.add_(1)
        return n

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """``slots``: ``(B, L, D)``. Returns ``(B, L, D)``
        recalled+gated contribution to add as residual."""
        size = int(self.buf_size.item())
        if size == 0:
            return torch.zeros_like(slots)
        B, L, D = slots.shape
        q = self.q_proj(slots)
        keys = self.buf_keys[:size]
        vals = self.buf_vals[:size]
        q_n = F.normalize(q, dim=-1)
        k_n = F.normalize(keys, dim=-1)
        # (B, L, S) cosine similarities
        sim = torch.einsum("bld,sd->bls", q_n, k_n)
        topk_n = min(self.top_k, size)
        scores, idx = sim.topk(topk_n, dim=-1)
        weights = F.softmax(scores, dim=-1)
        # Gather: (B, L, K, D)
        gathered = vals[idx]
        out = (weights.unsqueeze(-1) * gathered).sum(dim=-2)
        gate = torch.sigmoid(self.out_gate(slots))
        return self.norm(gate * out)

    @torch.no_grad()
    def recall_diagnostics(
        self, slots: torch.Tensor,
    ) -> dict:
        """For interpretability (T5): per-query recall mass +
        which buffer indices were chosen."""
        size = int(self.buf_size.item())
        if size == 0:
            return {"size": 0, "mean_topk_mass": 0.0}
        q = self.q_proj(slots)
        q_n = F.normalize(q, dim=-1)
        k_n = F.normalize(self.buf_keys[:size], dim=-1)
        sim = torch.einsum("bld,sd->bls", q_n, k_n)
        topk_n = min(self.top_k, size)
        scores, idx = sim.topk(topk_n, dim=-1)
        weights = F.softmax(scores, dim=-1)
        # "Mass" = top-k probability sum vs uniform over size
        topk_mass = weights.sum(dim=-1).mean().item()
        return {
            "size": size, "top_k": topk_n,
            "mean_topk_mass": float(topk_mass),
            "mean_max_score": float(scores[..., 0].mean().item()),
            "mean_min_score": float(scores[..., -1].mean().item()),
        }


class TitansPCMMiniLM(HybridPCMMiniLM):
    """Hybrid PCM + persistent episodic memory readout (F83).

    Extends :class:`HybridPCMMiniLM` with a single
    :class:`HierarchicalMemoryLayer` whose output is added to
    the slot stream *after* all attention/PCM layers and
    *before* the final layer-norm. The memory buffer is
    populated during forward passes from the model's own
    hidden states (FIFO, capacity ``memory_capacity``).

    Cognitive parallel: Titans' "neural memory module" — the
    network keeps an external memory it can both *read* and
    *write at test time*, without training updates to its
    parameters. Here the memory is a kNN cache; the readout
    weights are trainable.

    Parameters:
        memory_capacity: FIFO buffer size (default 512).
        memory_top_k: top-K nearest neighbours retrieved
            (default 8).
        memory_n_per_batch_imprint: positions sub-sampled per
            batch for imprinting (default 4).
    """

    def __init__(
        self, vocab: int, d_model: int = 64, n_layers: int = 4,
        n_heads: int = 4,
        combiner_hidden: int | None = None,
        dropout: float = 0.0,
        gate_bias_init: float = 0.0,
        attn_every: int = 4,
        memory_capacity: int = 512,
        memory_top_k: int = 8,
        memory_n_per_batch_imprint: int = 4,
    ) -> None:
        super().__init__(
            vocab=vocab, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads,
            combiner_hidden=combiner_hidden, dropout=dropout,
            gate_bias_init=gate_bias_init, attn_every=attn_every,
        )
        self.memory_readout = HierarchicalMemoryLayer(
            d_model=d_model, buffer_capacity=memory_capacity,
            top_k=memory_top_k,
        )
        self.memory_n_per_batch_imprint = memory_n_per_batch_imprint

    def forward(
        self, x: torch.Tensor, *,
        use_memory: bool = True, imprint: bool = True,
    ) -> torch.Tensor:
        slots = self.tok_emb(x)
        for layer in self.layers:
            slots = slots + layer(slots)
        if (
            use_memory
            and int(self.memory_readout.buf_size.item()) > 0
        ):
            slots = slots + self.memory_readout(slots)
        if imprint and self.training:
            self.memory_readout.imprint(
                slots.detach(),
                n_per_batch=self.memory_n_per_batch_imprint,
            )
        slots = self.ln_final(slots)
        return slots @ self.tok_emb.weight.t()

    @torch.no_grad()
    def hidden_states(
        self, x: torch.Tensor, *, use_memory: bool = True,
    ) -> torch.Tensor:
        slots = self.tok_emb(x)
        for layer in self.layers:
            slots = slots + layer(slots)
        if (
            use_memory
            and int(self.memory_readout.buf_size.item()) > 0
        ):
            slots = slots + self.memory_readout(slots)
        return self.ln_final(slots)

    def reset_memory(self) -> None:
        self.memory_readout.reset_buffer()


# ─────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────


def count_params(model: nn.Module, *, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def build_matched_pair(
    vocab: int, *,
    d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
    max_len: int = 32, dropout: float = 0.0,
    match_tol: float = 0.10,
) -> tuple["GPTMiniLM", "PCMMiniLM", dict]:
    """Build a ``(GPTMiniLM, PCMMiniLM)`` pair with matched
    parameter counts.

    Strategy: build GPTMiniLM with standard ``dim_feedforward =
    4*d_model``; then tune ``PCMMiniLM.combiner_hidden`` (default
    ``4*d_model``) downward until ``count_params(pcm) ≤
    count_params(gpt) * (1 + match_tol)``. Returns the two
    models plus a diagnostic dict with final param counts.
    """
    gpt = GPTMiniLM(
        vocab=vocab, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, max_len=max_len, dropout=dropout,
    )
    gpt_params = count_params(gpt)
    # Search descending combiner_hidden values
    hidden_candidates = [
        4 * d_model, 3 * d_model, 2 * d_model,
        int(1.5 * d_model), d_model,
        max(8, d_model // 2),
    ]
    pcm = None
    chosen_hidden = None
    for h in hidden_candidates:
        cand = PCMMiniLM(
            vocab=vocab, d_model=d_model, n_layers=n_layers,
            combiner_hidden=h, dropout=dropout,
        )
        p = count_params(cand)
        if p <= gpt_params * (1 + match_tol):
            pcm = cand
            chosen_hidden = h
            break
    if pcm is None:
        # Fallback to smallest tried
        chosen_hidden = hidden_candidates[-1]
        pcm = PCMMiniLM(
            vocab=vocab, d_model=d_model, n_layers=n_layers,
            combiner_hidden=chosen_hidden, dropout=dropout,
        )
    diag = {
        "gpt_params": gpt_params,
        "pcm_params": count_params(pcm),
        "pcm_combiner_hidden": chosen_hidden,
        "ratio_pcm_over_gpt":
            count_params(pcm) / max(gpt_params, 1),
    }
    return gpt, pcm, diag


def build_matched_triple(
    vocab: int, *,
    d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
    max_len: int = 32, dropout: float = 0.0,
    top_k: int = 4, match_tol: float = 0.15,
) -> tuple["GPTMiniLM", "PCMMiniLM", "PCMTopKMiniLM", dict]:
    """Build a parameter-matched ``(GPTMiniLM, PCMMiniLM,
    PCMTopKMiniLM)`` triple.

    PCMTopK has the same combiner-hidden trade-off as PCM-mean,
    plus a single scalar ``alpha`` for selective-recall mixing —
    so its parameter count is essentially the same as PCM-mean
    (one extra scalar). We reuse :func:`build_matched_pair` to
    pick a ``combiner_hidden`` that puts PCM under ~1.15× GPT
    params, then build PCMTopK with the same hidden.
    """
    gpt, pcm, diag = build_matched_pair(
        vocab=vocab, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, max_len=max_len, dropout=dropout,
        match_tol=match_tol,
    )
    pcm_topk = PCMTopKMiniLM(
        vocab=vocab, d_model=d_model, n_layers=n_layers,
        combiner_hidden=diag["pcm_combiner_hidden"],
        top_k=top_k, dropout=dropout,
    )
    diag = dict(diag)
    diag["pcm_topk_params"] = count_params(pcm_topk)
    diag["pcm_topk_top_k"] = top_k
    diag["ratio_topk_over_gpt"] = (
        count_params(pcm_topk) / max(diag["gpt_params"], 1)
    )
    return gpt, pcm, pcm_topk, diag


def build_matched_quad(
    vocab: int, *,
    d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
    max_len: int = 32, dropout: float = 0.0,
    top_k: int = 4, gate_bias_init: float = 0.0,
    match_tol: float = 0.20,
) -> tuple["GPTMiniLM", "PCMMiniLM", "PCMTopKMiniLM",
           "GatedPCMMiniLM", dict]:
    """Build a parameter-matched ``(GPT, PCM-mean, PCM-TopK,
    Gated PCM)`` quadruple (F80).

    Gated PCM adds a ``(d_model × d_model + d_model)`` gate
    projection per layer. To stay within ``match_tol`` of GPT,
    we may need to drop ``combiner_hidden`` slightly more for
    the gated variant. We reuse :func:`build_matched_triple` and
    pick a ``combiner_hidden`` for Gated PCM by re-running the
    descending-candidate search with the gate overhead included.
    """
    gpt, pcm, pcm_topk, diag = build_matched_triple(
        vocab=vocab, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, max_len=max_len, dropout=dropout,
        top_k=top_k, match_tol=match_tol,
    )
    diag = dict(diag)
    gpt_params = diag["gpt_params"]
    hidden_candidates = [
        4 * d_model, 3 * d_model, 2 * d_model,
        int(1.5 * d_model), d_model,
        max(8, d_model // 2),
    ]
    gated = None
    chosen_hidden = None
    for h in hidden_candidates:
        cand = GatedPCMMiniLM(
            vocab=vocab, d_model=d_model, n_layers=n_layers,
            combiner_hidden=h, dropout=dropout,
            gate_bias_init=gate_bias_init,
        )
        p = count_params(cand)
        if p <= gpt_params * (1 + match_tol):
            gated = cand
            chosen_hidden = h
            break
    if gated is None:
        chosen_hidden = hidden_candidates[-1]
        gated = GatedPCMMiniLM(
            vocab=vocab, d_model=d_model, n_layers=n_layers,
            combiner_hidden=chosen_hidden, dropout=dropout,
            gate_bias_init=gate_bias_init,
        )
    diag["gated_pcm_params"] = count_params(gated)
    diag["gated_pcm_combiner_hidden"] = chosen_hidden
    diag["gated_pcm_gate_bias_init"] = gate_bias_init
    diag["ratio_gated_over_gpt"] = (
        count_params(gated) / max(gpt_params, 1)
    )
    return gpt, pcm, pcm_topk, gated, diag


def build_matched_pentad(
    vocab: int, *,
    d_model: int = 64, n_layers: int = 4, n_heads: int = 4,
    max_len: int = 32, dropout: float = 0.0,
    top_k: int = 4, gate_bias_init: float = 0.0,
    attn_every: int = 4, match_tol: float = 0.30,
) -> tuple["GPTMiniLM", "PCMMiniLM", "PCMTopKMiniLM",
           "GatedPCMMiniLM", "HybridPCMMiniLM", dict]:
    """Build a parameter-matched ``(GPT, PCM-mean, PCM-TopK,
    Gated PCM, Hybrid PCM)`` quintuple (F81).

    The Hybrid model replaces every ``attn_every``-th layer
    with a :class:`GatedAttentionLayer`. Since attention layers
    have a different param profile (≈ ``4·D²`` vs PCM's
    ``D·H + D²`` for combiner+gate), we re-run the descending-
    hidden search separately for Hybrid PCM.
    """
    gpt, pcm, pcm_topk, gated, diag = build_matched_quad(
        vocab=vocab, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, max_len=max_len, dropout=dropout,
        top_k=top_k, gate_bias_init=gate_bias_init,
        match_tol=match_tol,
    )
    diag = dict(diag)
    gpt_params = diag["gpt_params"]
    hidden_candidates = [
        4 * d_model, 3 * d_model, 2 * d_model,
        int(1.5 * d_model), d_model,
        max(8, d_model // 2),
    ]
    hybrid = None
    chosen_hidden = None
    for h in hidden_candidates:
        cand = HybridPCMMiniLM(
            vocab=vocab, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, combiner_hidden=h, dropout=dropout,
            gate_bias_init=gate_bias_init,
            attn_every=attn_every,
        )
        p = count_params(cand)
        if p <= gpt_params * (1 + match_tol):
            hybrid = cand
            chosen_hidden = h
            break
    if hybrid is None:
        chosen_hidden = hidden_candidates[-1]
        hybrid = HybridPCMMiniLM(
            vocab=vocab, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, combiner_hidden=chosen_hidden,
            dropout=dropout, gate_bias_init=gate_bias_init,
            attn_every=attn_every,
        )
    diag["hybrid_pcm_params"] = count_params(hybrid)
    diag["hybrid_pcm_combiner_hidden"] = chosen_hidden
    diag["hybrid_pcm_attn_every"] = attn_every
    diag["hybrid_pcm_n_pcm_layers"] = hybrid.n_pcm_layers()
    diag["hybrid_pcm_n_attn_layers"] = hybrid.n_attn_layers()
    diag["ratio_hybrid_over_gpt"] = (
        count_params(hybrid) / max(gpt_params, 1)
    )
    return gpt, pcm, pcm_topk, gated, hybrid, diag


@torch.no_grad()
def perplexity(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor,
    *, pad_id: int = 0,
) -> float:
    """Mean cross-entropy perplexity on ``(x, y)`` pairs."""
    model.eval()
    logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
        ignore_index=pad_id, reduction="mean",
    )
    return float(math.exp(loss.item()))
