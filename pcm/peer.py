"""pcm.peer — Tier C: PEER-style product-key concept router (micro version).

Implements a concept-aware variant of the **Mixture of a Million Experts**
(Lample et al., DeepMind 2024, arXiv 2407.04153) / **Memory Layers at
Scale** (Berges et al., Meta 2024, arXiv 2412.09764) lookup primitive,
adapted to PCM's slot-bank semantics.

Two modes coexist on the same dense pool:

* **Symbolic mode** (default; preserves all paper §4-§6 / §B claims):
  for a known ``concept_id``, ``cg.collapse_batch`` does the standard
  ``cid_to_slot`` deterministic lookup. PEER is invisible.

* **Discovery mode** (opt-in): for an *unknown* perceptual query
  ``q`` of dimension ``d_query``, ``ProductKeyRouter.route(q)`` returns
  the top-K ``(slot_idx, weight)`` pairs sub-linearly (O(2 sqrt(N))).
  This unlocks the §3.6 "Pipeline A" auto-discovery path that the
  paper deferred.

The 8GB micro version targets ``num_experts = 16_384`` (PEER's "small"
config); a full 1M-experts run requires ~24GB and is not enabled here.
The product-key construction follows PEER §3.1 verbatim:

    keys ∈ R^{2 × sqrt(N) × (d/2)}        # two halves, sqrt(N) sub-keys each
    q = head(input) ∈ R^d                  # split into (q1, q2) ∈ R^{d/2}
    s_x = q1 · K1.T  ∈ R^{sqrt(N)}         # sub-scores on first half
    s_y = q2 · K2.T  ∈ R^{sqrt(N)}         # sub-scores on second half
    top-K (s_x ⊕ s_y) → 2 sqrt(N) candidates pre-merge → final top-K

Attribution under discovery mode generalises ``consumed_by`` to a soft
distribution: instead of a single ``slot_idx``, we record the top-K
slots and their softmax weights. ``cg._consumed_by_by_slot`` still gets
the union of selected slots so Tier-A's dict-lookup attribution
question keeps working.

This module is **opt-in**: nothing in the default ConceptGraph touches
PEER unless a ``ProductKeyRouter`` is explicitly constructed and used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as _cfg

if TYPE_CHECKING:
    from .concept_graph import ConceptGraph


@dataclass
class PEERConfig:
    num_experts: int = _cfg.PEER_NUM_EXPERTS_8GB
    top_k: int = _cfg.PEER_TOP_K
    num_heads: int = _cfg.PEER_NUM_HEADS
    query_dim: int = 128

    def __post_init__(self) -> None:
        n = self.num_experts
        s = int(round(math.sqrt(n)))
        if s * s != n:
            raise ValueError(
                f"num_experts must be a perfect square (got {n}; closest: "
                f"{s*s} or {(s+1)*(s+1)})"
            )
        if self.query_dim % 2 != 0:
            raise ValueError("query_dim must be even (split into halves)")
        if self.query_dim % self.num_heads != 0:
            raise ValueError(f"query_dim ({self.query_dim}) must be divisible "
                              f"by num_heads ({self.num_heads})")


class ProductKeyRouter(nn.Module):
    """Sub-linear top-K router over up to ``num_experts`` slots.

    The module owns:

    - ``q_proj`` (``Linear(input_dim, num_heads * query_dim)``):
      projects each input vector to a multi-head query.
    - ``keys_x`` / ``keys_y`` (``Embedding(sqrt(N), query_dim/2)`` each):
      the product key bank. Total parameter count is
      ``2 * sqrt(N) * d/2 = sqrt(N) * d``, vs ``N * d`` for a flat key
      bank — sub-linear by design.

    ``route(x)`` returns ``(top_slots, top_weights)`` of shape
    ``(B, num_heads, top_k)``. Down-stream code may either:

    1. Fuse via ``F.embedding`` over ``cg.bundle_pool[facet]`` weighted
       by ``top_weights`` (PEER's ``y = sum_k w_k * v_k`` recipe).
    2. Treat ``top_slots[:, 0]`` as the discovered slot and bind a fresh
       ``concept_id`` to it (Tier-C "discovery mode").
    """

    def __init__(
        self,
        input_dim: int,
        cfg: PEERConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or PEERConfig(query_dim=input_dim)
        if self.cfg.query_dim != input_dim:
            self.q_proj = nn.Linear(input_dim, self.cfg.num_heads * self.cfg.query_dim)
        else:
            self.q_proj = nn.Linear(input_dim, self.cfg.num_heads * input_dim)

        n_sqrt = int(round(math.sqrt(self.cfg.num_experts)))
        d_half = self.cfg.query_dim // 2
        # Two halves of the product key (PEER §3.1).
        self.keys_x = nn.Parameter(torch.randn(self.cfg.num_heads, n_sqrt, d_half) * 0.02)
        self.keys_y = nn.Parameter(torch.randn(self.cfg.num_heads, n_sqrt, d_half) * 0.02)
        self.n_sqrt = n_sqrt

    @property
    def num_experts(self) -> int:
        return self.cfg.num_experts

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(top_slots, top_weights)`` for input ``x`` of shape ``(B, D)``.

        Outputs:
            top_slots:   (B, num_heads, top_k) long tensor of slot indices in [0, N).
            top_weights: (B, num_heads, top_k) float tensor of softmax weights.
        """
        B = x.shape[0]
        h = self.cfg.num_heads
        d = self.cfg.query_dim
        q = self.q_proj(x).view(B, h, d)
        q1, q2 = q[..., : d // 2], q[..., d // 2:]

        # sub-scores per half, per head: (B, h, n_sqrt)
        s_x = torch.einsum("bhd,hnd->bhn", q1, self.keys_x)
        s_y = torch.einsum("bhd,hnd->bhn", q2, self.keys_y)

        # top-K per half, then expand to outer product
        K = self.cfg.top_k
        topx_v, topx_i = s_x.topk(K, dim=-1)   # (B, h, K)
        topy_v, topy_i = s_y.topk(K, dim=-1)

        # outer-sum scores over the top-K × top-K square: (B, h, K, K)
        scores = topx_v.unsqueeze(-1) + topy_v.unsqueeze(-2)
        slots = topx_i.unsqueeze(-1) * self.n_sqrt + topy_i.unsqueeze(-2)
        # flatten and pick the global top-K
        scores_flat = scores.flatten(-2, -1)   # (B, h, K*K)
        slots_flat = slots.flatten(-2, -1)
        final_v, final_i = scores_flat.topk(K, dim=-1)
        top_slots = slots_flat.gather(-1, final_i)  # (B, h, K)
        top_weights = F.softmax(final_v, dim=-1)
        return top_slots, top_weights

    def gather_values(
        self,
        cg: "ConceptGraph",
        facet: str,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Run the router and gather expert values from the dense pool.

        Returns ``(B, value_dim)`` weighted-sum of the top-K rows from
        ``cg.bundle_pool[facet]``. Multiple heads are mean-pooled at the
        end (matching PEER §3 "shared-output" variant).
        """
        if facet not in cg.bundle_pool:
            raise KeyError(
                f"facet {facet!r} has no pool. Allocate it via collapse_batch first."
            )
        pool = cg.bundle_pool[facet]
        # PEER assumes num_experts <= pool capacity. We cap at min(N, capacity).
        N_eff = min(self.cfg.num_experts, pool.shape[0])
        top_slots, top_weights = self.forward(x)
        # clamp out-of-range slots (in case capacity < num_experts; rare)
        top_slots = torch.clamp(top_slots, 0, N_eff - 1)
        # gather values: (B, h, K, D)
        flat_slots = top_slots.reshape(-1)  # (B*h*K,)
        values = F.embedding(flat_slots, pool)
        values = values.view(*top_slots.shape, -1)  # (B, h, K, D)
        weighted = (values * top_weights.unsqueeze(-1)).sum(dim=-2)  # (B, h, D)
        return weighted.mean(dim=1)  # (B, D)


def soft_consumed_by_log(
    cg: "ConceptGraph",
    facet: str,
    caller: str,
    top_slots: torch.Tensor,
    top_weights: torch.Tensor,
    threshold: float = 0.05,
    tick: int = 0,
) -> int:
    """Tier-C attribution under discovery mode.

    Walks the (B, h, K) router output and writes ``caller`` into
    ``cg._consumed_by_by_slot`` for every slot that received weight
    above ``threshold`` in any batch position. Returns the count of
    distinct slots logged. Soft-attribution generalises the Tier-A
    ``consumed_by`` registry: it answers "for which slots did the router
    *strongly* select this caller" rather than "for which slots did
    this caller deterministically declare a binding".

    This satisfies the design promise that PCM keeps a recoverable
    architectural attribution even under sub-linear routing.
    """
    mask = top_weights >= threshold
    sel_slots = top_slots[mask].view(-1).tolist()
    seen: set[int] = set()
    for slot in sel_slots:
        slot = int(slot)
        if slot in seen:
            continue
        seen.add(slot)
        cg._consumed_by_by_slot.setdefault(slot, {}).setdefault(facet, set()).add(caller)
        cg._collapse_history_by_slot.setdefault(slot, {}).setdefault(facet, []).append(
            (caller, int(tick))
        )
        cg._active_facets_by_slot.setdefault(slot, set()).add(facet)
    return len(seen)


__all__ = ["PEERConfig", "ProductKeyRouter", "soft_consumed_by_log"]
