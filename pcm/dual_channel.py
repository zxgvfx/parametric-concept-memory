"""pcm.dual_channel — PCM v2 dual-channel concept encoding.

See ``docs/PCM_V2_DUAL_CHANNEL_DESIGN.md`` for the full design rationale
and falsifiability contract (V1 / V2 / V3 invariants).

Each concept is encoded as a **pair** of facets:

* **slot facet** (``f"{base}_slot"``) — supervised + Tier-G k-means
  clustered. Holds the cluster identity (the "positional address").
* **attr facet** (``f"{base}_attr"``) — trained by contrastive
  InfoNCE (positives = same slot) + arithmetic-consistency
  ``(attr_a + attr_c − attr_b ≈ attr_d)``. Excluded from sleep
  clustering.

Backwards compatibility: every existing v1 caller that uses
``cg.collapse_batch`` on a single facet keeps working unchanged. v2 is
opt-in by calling :func:`register_dual_channel_facet`.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

if False:  # TYPE_CHECKING
    from .concept_graph import ConceptGraph


__all__ = [
    "SLOT_FACET_TEMPLATE",
    "ATTR_FACET_TEMPLATE",
    "register_dual_channel_facet",
    "is_dual_channel_facet",
    "iter_dual_channel_pairs",
    "collapse_dual_channel",
    "info_nce_loss",
    "arithmetic_consistency_loss",
    "successor_consistency_loss",
    "spread_regularizer",
    "pair_attention_logits",
]


SLOT_FACET_TEMPLATE = "{base}_slot"
ATTR_FACET_TEMPLATE = "{base}_attr"


# ---------------------------------------------------------------------------
# Facet registration
# ---------------------------------------------------------------------------


def register_dual_channel_facet(
    cg: "ConceptGraph", base: str,
    *, slot_dim: int, attr_dim: int,
) -> tuple[str, str]:
    """Mark ``cg`` as carrying a dual-channel facet pair anchored at
    ``base``. Returns ``(slot_facet_name, attr_facet_name)``.

    The returned facet names are not yet allocated in
    ``cg.bundle_pool``; that happens lazily on the first
    :func:`collapse_dual_channel` call (the same as v1's lazy
    facet allocation in ``ConceptGraph._ensure_facet``).

    The function records the pairing on the graph so that:

    * Tier-G's :func:`pcm.sleep.run_sleep_pass` can opt to skip the
      attr facet (``cg._attr_facets_excluded_from_sleep``);
    * downstream heads can discover paired facets via
      :func:`iter_dual_channel_pairs`.
    """
    sf = SLOT_FACET_TEMPLATE.format(base=str(base))
    af = ATTR_FACET_TEMPLATE.format(base=str(base))

    pairs = getattr(cg, "_dual_channel_pairs", None)
    if pairs is None:
        pairs = {}
        cg._dual_channel_pairs = pairs
    pairs[str(base)] = (sf, af, int(slot_dim), int(attr_dim))

    excluded = getattr(cg, "_attr_facets_excluded_from_sleep", None)
    if excluded is None:
        excluded = set()
        cg._attr_facets_excluded_from_sleep = excluded
    excluded.add(af)

    return sf, af


def is_dual_channel_facet(cg: "ConceptGraph", facet: str) -> tuple[bool, str | None, str | None]:
    """Return ``(is_paired, base, role)`` where role ∈ {"slot", "attr"}.

    If ``facet`` is not part of any pair, returns ``(False, None, None)``.
    """
    pairs = getattr(cg, "_dual_channel_pairs", {}) or {}
    for base, info in pairs.items():
        sf, af = info[0], info[1]
        if facet == sf:
            return True, base, "slot"
        if facet == af:
            return True, base, "attr"
    return False, None, None


def iter_dual_channel_pairs(cg: "ConceptGraph") -> Iterable[tuple[str, str, str, int, int]]:
    """Yield ``(base, slot_facet, attr_facet, slot_dim, attr_dim)``."""
    pairs = getattr(cg, "_dual_channel_pairs", {}) or {}
    for base, info in pairs.items():
        sf, af, sdim, adim = info
        yield (base, sf, af, sdim, adim)


# ---------------------------------------------------------------------------
# Collapse helper
# ---------------------------------------------------------------------------


def collapse_dual_channel(
    cg: "ConceptGraph", *,
    caller: str, base_facet: str, concept_ids: list[str],
    slot_shape, attr_shape,
    tick: int = 0, device=None, init: str = "normal_small",
    attr_init: str | None = None,
    normalize_attr: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in dual-channel replacement for ``cg.collapse_batch``.

    Returns ``(slot_rows, attr_rows)``, each shaped ``(B, *shape)``.

    Both calls go through the existing :meth:`ConceptGraph.collapse_batch`,
    so attribution / consumed_by tracking is preserved automatically.

    Args:
        attr_init: optional override for attr-facet init mode. When
            ``None`` (default), uses ``init`` for both facets. Use
            ``"normal"`` (unit-variance Gaussian) on the attr facet to
            avoid the v2-MVP trivial collapse `attr ≈ 0` that satisfies
            :func:`arithmetic_consistency_loss` with zero gradient.
        normalize_attr: when ``True``, ``F.normalize(attr_rows, dim=-1)``
            is applied before returning, projecting attribute rows onto
            the unit sphere. Combined with
            :func:`arithmetic_consistency_loss` this makes the
            arithmetic constraint live on the sphere (cosine analogy
            in the word2vec sense), preventing the trivial all-zero
            solution. Recommended for v2 MVP.
    """
    sf = SLOT_FACET_TEMPLATE.format(base=str(base_facet))
    af = ATTR_FACET_TEMPLATE.format(base=str(base_facet))
    slot_rows = cg.collapse_batch(
        caller=caller, facet=sf, concept_ids=concept_ids,
        shape=slot_shape, tick=tick, device=device, init=init,
    )
    attr_rows = cg.collapse_batch(
        caller=caller, facet=af, concept_ids=concept_ids,
        shape=attr_shape, tick=tick, device=device,
        init=attr_init if attr_init is not None else init,
    )
    if normalize_attr:
        attr_rows = F.normalize(attr_rows, dim=-1, eps=1e-8)
    return slot_rows, attr_rows


def successor_consistency_loss(
    attr_table: torch.Tensor,
    successor_idx: list[tuple[int, int]] | None = None,
    *,
    target_step_norm: float = 1.0,
    norm_penalty_weight: float = 1.0,
) -> torch.Tensor:
    """Enforce a shared, **non-trivial** "step" across consecutive
    members of an ordinal facet:

        attr_{i+1} − attr_i ≈ shared_shift   ∀ i  (variance term)
        ‖shared_shift‖ ≈ target_step_norm           (norm term)

    The variance term alone admits a degenerate minimum at
    ``attr_i ≡ const`` (all diffs zero, var = 0). The norm term
    forbids that minimum by penalising shared_shift norm below
    ``target_step_norm``. Together they pick out a **linear
    monotone** embedding of the ordinal index, which is the
    structure that makes vector analogy work
    (verified empirically: a synthetic ``attr_i = i·v`` table
    achieves V2_top1 = 1.0; the same table after L2-normalisation
    drops to 0.10, confirming that the norm direction matters).

    Args:
        attr_table: ``(N, D_attr)`` table; rows are assumed
            sorted by their ordinal index unless
            ``successor_idx`` is supplied.
        successor_idx: optional explicit ``[(i, j), ...]`` pairs.
        target_step_norm: target ‖attr_{i+1} − attr_i‖. Default 1.0.
        norm_penalty_weight: relative weight of the hinge norm
            penalty vs the variance term.

    Returns the combined loss as a scalar tensor.
    """
    if attr_table.shape[0] < 2:
        return attr_table.new_zeros(())
    if successor_idx is None:
        successor_idx = [(i, i + 1) for i in range(attr_table.shape[0] - 1)]
    if not successor_idx:
        return attr_table.new_zeros(())
    diffs = torch.stack([
        attr_table[j] - attr_table[i] for (i, j) in successor_idx
    ])
    mean_diff = diffs.mean(dim=0, keepdim=True)
    var = (diffs - mean_diff).flatten(start_dim=1).pow(2).sum(dim=-1).mean()
    mean_norm = mean_diff.flatten().norm()
    norm_pen = F.relu(target_step_norm - mean_norm).pow(2)
    return var + norm_penalty_weight * norm_pen


def spread_regularizer(
    attr_rows: torch.Tensor,
    *,
    target_min_dist: float = 0.5,
) -> torch.Tensor:
    """Hinge-style penalty for under-separated attribute rows.

    For every pair of rows whose pairwise cosine similarity is above
    ``1 − target_min_dist`` (i.e. cosine distance below
    ``target_min_dist``), adds a hinge penalty equal to the deficit.
    Encourages the attribute rows to spread across the unit sphere
    so :func:`arithmetic_consistency_loss` cannot escape into the
    trivial ``attr_i ≈ 0 ∀i`` minimum.
    """
    if attr_rows.shape[0] < 2:
        return attr_rows.new_zeros(())
    norm = F.normalize(attr_rows.flatten(start_dim=1), dim=-1, eps=1e-8)
    sim = norm @ norm.t()
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    cos_dist = 1.0 - sim
    cos_dist = cos_dist.masked_fill(eye, target_min_dist + 1.0)
    deficit = F.relu(target_min_dist - cos_dist)
    return deficit.mean()


# ---------------------------------------------------------------------------
# Losses for the attribute channel
# ---------------------------------------------------------------------------


def info_nce_loss(
    attr_rows: torch.Tensor,
    slot_labels: torch.Tensor,
    *,
    temperature: float = 0.5,
) -> torch.Tensor:
    """InfoNCE contrastive loss with positives = same slot label.

    Args:
        attr_rows: ``(B, D_attr)`` continuous attribute embeddings.
        slot_labels: ``(B,)`` integer slot ids.
        temperature: softmax temperature.

    Returns scalar loss; ``0`` when no batch element has another batch
    element of the same slot (degenerate batch).
    """
    if attr_rows.shape[0] < 2:
        return attr_rows.new_zeros(())
    norm = F.normalize(attr_rows.flatten(start_dim=1), dim=-1)
    sim = norm @ norm.t() / float(temperature)
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(eye, -1e9)

    pos_mask = (
        slot_labels.unsqueeze(0) == slot_labels.unsqueeze(1)
    ) & ~eye
    n_pos = pos_mask.sum(dim=-1).float()
    valid = n_pos > 0
    if not valid.any():
        return attr_rows.new_zeros(())

    log_prob = sim - sim.logsumexp(dim=-1, keepdim=True)
    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=-1)
    pos_log_prob = pos_log_prob[valid] / n_pos[valid]
    return -pos_log_prob.mean()


def arithmetic_consistency_loss(
    attr_rows: torch.Tensor,
    idx_a: torch.Tensor,
    idx_b: torch.Tensor,
    idx_c: torch.Tensor,
    idx_d: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Squared-error penalty enforcing
    ``attr_a + attr_c − attr_b ≈ attr_d`` over a batch of analogy
    quadruples.

    Args:
        attr_rows: ``(N, D_attr)`` table of attribute vectors.
        idx_a, idx_b, idx_c, idx_d: index tensors of identical
            shape ``(B,)`` selecting rows from ``attr_rows``.
        reduction: ``"mean"`` (default) or ``"sum"``.

    Returns the L2 penalty as a scalar tensor.
    """
    a = attr_rows[idx_a]
    b = attr_rows[idx_b]
    c = attr_rows[idx_c]
    d = attr_rows[idx_d]
    diff = (a + c - b - d).flatten(start_dim=1)
    sq = diff.pow(2).sum(dim=-1)
    if reduction == "sum":
        return sq.sum()
    return sq.mean()


# ---------------------------------------------------------------------------
# Pair attention head primitive (for V3 / mixed_OOD ceiling)
# ---------------------------------------------------------------------------


def pair_attention_logits(
    slot_a: torch.Tensor,
    slot_b: torch.Tensor,
    *,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    weight_v: torch.Tensor,
    out_proj: torch.Tensor,
) -> torch.Tensor:
    """Pairwise attention over (slot_a, slot_b) — minimal building block
    used by :class:`DualChannelMoveHead`.

    The motivation (see PCM_V2_DUAL_CHANNEL_DESIGN §2 / V3) is that
    MLP ``fc1`` over a concatenation of two bundle rows learns a
    *joint* function whose decision boundary is bounded by the
    train-time joint distribution. Replacing the joint MLP with a
    ``query=slot_a / key=value=slot_b`` attention reduces the
    coupling: each test-time pair only needs slot_a and slot_b's
    *individual* representations to be in-distribution, not their
    joint.

    Returns ``(B, D_out)`` attention output. Batched, no head split,
    no positional encoding (those layers are added by callers if
    relevant for the domain).
    """
    q = slot_a @ weight_q  # (B, D)
    k = slot_b @ weight_k  # (B, D)
    v = slot_b @ weight_v  # (B, D)
    scale = 1.0 / float(q.shape[-1]) ** 0.5
    score = (q * k).sum(dim=-1, keepdim=True) * scale  # (B, 1)
    attn = torch.sigmoid(score)
    out = attn * v
    return out @ out_proj
