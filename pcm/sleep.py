"""pcm.sleep — Tier G: offline sleep abstraction pass.

NREM-style codebook compression + interleaved replay + cookable
abstract-relation subgraphs. See
``docs/PCM_TIER_G_SLEEP_ABSTRACTION.md`` for the design rationale,
literature mapping and the full G1–G6 falsifiable contract.

Design notes:

- **Default off**: importing ``pcm.sleep`` is a no-op for forward
  outputs; you must call :func:`attach_sleep` to switch on the
  metadata, and :func:`run_sleep_pass` to actually perform the
  offline pass. This keeps Tier-A/B/C/D bit-identity (G1).
- **Opt-in helper module**: mirrors :mod:`pcm.gate` (Tier-B) and
  :mod:`pcm.peer` (Tier-C). No mixin into ``ConceptGraph``.
- **Reused cookable kind**: abstract-relation subgraphs are
  registered with ``kind="parametric_muscle_subgraph"`` (Tier-D's
  cookable kind) so :class:`pcm.GraphEvaluator` can cook them
  unmodified. Sub-typing is via
  ``metadata.constants["kind_hint"] = "abstract_relation"``.
- **Pool memory safety (G2)**: K-means runs under
  :func:`torch.no_grad`; centroids are written to *new* slots only,
  residuals to a *new* facet (``"<f>_residual"``). Pre-existing rows
  are never mutated by phases A–D.

Public API::

    SleepConfig, FacetSleepReport, SleepReport
    attach_sleep, run_sleep_pass, sleep_status
    register_prototype, register_relation, iter_abstract_relations
    make_replay_source_from_buffer
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable

import torch
import torch.nn.functional as F

from .graph_eval import PARAMETRIC_KIND

if TYPE_CHECKING:
    from .concept_graph import ConceptGraph


# ---------------------------------------------------------------------------
# Constants — naming conventions for Tier-G nodes / facets.
# ---------------------------------------------------------------------------

PROTO_KIND = "abstract_prototype"
"""ConceptNode.kind for cluster-centroid data nodes (non-cookable)."""

RELATION_KIND_HINT = "abstract_relation"
"""metadata.constants['kind_hint'] value distinguishing Tier-G cook
subgraphs from Tier-D head subgraphs."""

SLEEP_CALLER = "sleep"
"""Caller name stamped into ``consumed_by`` and ``collapse_history``
during the sleep pass."""

PROTO_CID_TEMPLATE = "concept:cluster:{facet}:{k}"
RELATION_CID_TEMPLATE = "concept:rel:{facet}:{k}:{member}"
RESIDUAL_FACET_TEMPLATE = "{facet}_residual"


__all__ = [
    "PROTO_KIND",
    "RELATION_KIND_HINT",
    "SLEEP_CALLER",
    "SleepConfig",
    "FacetSleepReport",
    "SleepReport",
    "attach_sleep",
    "run_sleep_pass",
    "sleep_status",
    "register_prototype",
    "register_relation",
    "iter_abstract_relations",
    "make_replay_source_from_buffer",
    "build_member_to_anchor_index",
    "collapse_via_abstract",
    "collapse_with_optional_abstract",
    "materialize_effective_bundle_state",
]


# ---------------------------------------------------------------------------
# Configuration + reports.
# ---------------------------------------------------------------------------


@dataclass
class SleepConfig:
    """Configuration for a single :func:`run_sleep_pass` invocation.

    Defaults are tuned to be safe ("does nothing harmful") rather than
    aggressive: ``replay_steps=0`` means phase E is skipped, so a fresh
    config trivially satisfies G2 (pool memory safety) without needing
    any optimizer.
    """

    k_clusters: int | str = "auto"
    """Number of clusters per facet. ``"auto"`` → ``max(2, round(sqrt(N)/2))``
    where ``N`` is the number of active slots on that facet."""

    kmeans_iters: int = 32
    kmeans_tol: float = 1e-5
    distance: str = "cosine"
    """Distance for assignment: ``"cosine"`` (recommended; matches PAPER
    geometry metrics) or ``"l2"``."""
    init: str = "kmeans++"
    """Centroid init: ``"kmeans++"`` or ``"uniform"``."""

    replay_steps: int = 0
    """Number of interleaved-replay gradient steps in phase E. 0 disables."""
    replay_batch_size: int = 64

    seed: int = 0
    abstract_scope: str = "ABSTRACT"
    relation_mode: str = "add"
    """Combination mode for the abstract-relation cook subgraph; one of
    ``"add"`` / ``"mul"`` / ``"concat"``."""

    residual_init: str = "zero"
    """Init mode passed to ``_ensure_facet`` for the residual pool. We
    use ``"zero"`` (and then immediately overwrite with the actual
    residuals) so freshly registered concepts that *don't* go through
    the sleep pass yet get a literal zero residual = identity behaviour."""

    anchor_ema: float = 1.0
    """S1: blend factor for anchor updates on a re-cluster pass.

    * ``1.0`` (default) → legacy behaviour: anchor row is hard-overwritten
      with the new centroid (matches Tier-G v1 semantics; required for
      G4 'centroid is mean of members' bit-identity).
    * ``0 < anchor_ema < 1`` → Online Codebook / VectorQuantizeEMA style
      slow update: ``anchor.data = (1 - α) * old_anchor + α * new_centroid``
      where ``α = anchor_ema``. Reduces topology-hop variance on
      ``force_recluster=True`` runs (see PCM_TIER_G_SLEEP_ABSTRACTION
      §"failure modes" for the variance numbers).

    The first time a prototype slot is created (no prior anchor row to
    blend with), the EMA path falls back to direct write so a fresh
    sleep pass is still well-defined.
    """

    assignment: str = "hard"
    """S2: cluster assignment for member→anchor routing.

    * ``"hard"`` (default) → argmin distance, single anchor per member.
    * ``"soft"`` → softmax(-d / soft_tau) over all anchors in the facet.
      The residual stored on disk is the same hard-assignment residual
      (preserves G5 cook reconstruction); ``soft`` only changes the
      runtime read path used by :func:`collapse_via_abstract` when the
      caller passes ``assignment="soft"``.
    """

    soft_tau: float = 0.5
    """Temperature for ``assignment="soft"``; lower → harder, higher →
    smoother mixture. Annealing is left to the caller."""


@dataclass
class FacetSleepReport:
    facet: str
    n_active: int
    k_clusters: int
    silhouette: float
    centroid_norms: list[float]
    residual_rms: list[float]
    member_counts: list[int]


@dataclass
class SleepReport:
    facets: list[FacetSleepReport] = field(default_factory=list)
    replay_loss_curve: list[float] = field(default_factory=list)
    skipped_facets: list[str] = field(default_factory=list)
    tick: int = 0

    def to_dict(self) -> dict:
        """JSON-serialisable summary."""
        return {
            "tick": int(self.tick),
            "skipped_facets": list(self.skipped_facets),
            "replay_loss_curve": [float(x) for x in self.replay_loss_curve],
            "facets": [
                {
                    "facet": r.facet,
                    "n_active": r.n_active,
                    "k_clusters": r.k_clusters,
                    "silhouette": r.silhouette,
                    "centroid_norms": r.centroid_norms,
                    "residual_rms": r.residual_rms,
                    "member_counts": r.member_counts,
                }
                for r in self.facets
            ],
        }


# ---------------------------------------------------------------------------
# Public surface — opt-in attach + status query.
# ---------------------------------------------------------------------------


def attach_sleep(
    cg: "ConceptGraph", *, facets: Iterable[str] | None = None
) -> None:
    """Mark sleep machinery as enabled on ``cg``.

    Idempotent. After this call, :func:`run_sleep_pass` may be called
    one or more times. Forward outputs over Tier-A/B/C/D paths remain
    bit-identical (G1) until ``run_sleep_pass`` actually fires.

    Args:
        cg: target graph.
        facets: optional allow-list. If provided, only listed facets
            are eligible for clustering on subsequent
            :func:`run_sleep_pass` calls. If ``None``, every facet
            seen by the time of the call is eligible (the default
            "wake up and dream about everything" behaviour).
    """
    if not getattr(cg, "sleep_enabled", False):
        cg.sleep_enabled = True
        cg._sleep_facets_allowlist = (
            set(str(f) for f in facets) if facets is not None else None
        )
        cg._sleep_last_tick = -1
        return
    if facets is not None:
        existing = getattr(cg, "_sleep_facets_allowlist", None)
        new = set(str(f) for f in facets)
        cg._sleep_facets_allowlist = (
            new if existing is None else (existing | new)
        )


def sleep_status(cg: "ConceptGraph") -> dict:
    """Return a JSON-serialisable summary of sleep state on ``cg``.

    Keys: ``enabled``, ``facets_allowlist``, ``n_prototypes``,
    ``n_relations``, ``residual_facets``, ``last_tick``.
    """
    if not getattr(cg, "sleep_enabled", False):
        return {"enabled": False}
    proto_count = sum(
        1
        for n in cg.concepts.values()
        if getattr(n, "kind", None) == PROTO_KIND
    )
    rel_nodes = list(iter_abstract_relations(cg))
    residual_facets = sorted(
        f for f in cg.bundle_pool if f.endswith("_residual")
    )
    allowlist = getattr(cg, "_sleep_facets_allowlist", None)
    return {
        "enabled": True,
        "facets_allowlist": (sorted(allowlist) if allowlist is not None
                             else None),
        "n_prototypes": int(proto_count),
        "n_relations": int(len(rel_nodes)),
        "residual_facets": residual_facets,
        "last_tick": int(getattr(cg, "_sleep_last_tick", -1)),
    }


def iter_abstract_relations(cg: "ConceptGraph"):
    """Iterate over Tier-G abstract-relation cook nodes (kind_hint match)."""
    for nid, node in cg.concepts.items():
        if getattr(node, "kind", None) != PARAMETRIC_KIND:
            continue
        meta = getattr(node, "metadata", None) or {}
        consts = meta.get("constants") or {}
        if consts.get("kind_hint") == RELATION_KIND_HINT:
            yield node


def build_member_to_anchor_index(
    cg: "ConceptGraph",
) -> dict[tuple[str, str], str]:
    """Index ``(facet, member_id) → anchor_id`` over registered relations.

    Each ``register_relation`` call records the (facet, member, anchor)
    mapping in ``metadata.constants``; this helper materialises the
    inverse lookup once so batched abstract reads in :func:`collapse_via_abstract`
    don't need to re-scan ``cg.concepts``.
    """
    out: dict[tuple[str, str], str] = {}
    for rel_node in iter_abstract_relations(cg):
        consts = (rel_node.metadata or {}).get("constants", {}) or {}
        facet = consts.get("anchor_facet")
        member_id = consts.get("member_id")
        anchor_id = consts.get("anchor_id")
        if not (facet and member_id and anchor_id):
            continue
        out[(str(facet), str(member_id))] = str(anchor_id)
    return out


def collapse_via_abstract(
    cg: "ConceptGraph",
    *,
    caller: str,
    facet: str,
    concept_ids: list[str],
    shape,
    tick: int = 0,
    init: str = "normal_small",
    device=None,
    fallback: str = "direct",
    index: dict[tuple[str, str], str] | None = None,
    assignment: str = "hard",
    soft_tau: float = 0.5,
) -> torch.Tensor:
    """Read each member row as ``anchor_centroid + member_residual``.

    Drop-in replacement for :meth:`ConceptGraph.collapse_batch` that
    routes the read through the Tier-G abstract relations registered by
    :func:`run_sleep_pass`. The math is::

        out[i] = bundle_pool[facet][anchor_slot_for(member_i)]
               + bundle_pool[<facet>_residual][member_slot_i]

    Right after a sleep pass this is **bit-identical** to the direct
    read (because residual = row - anchor by construction); see G7 in
    ``docs/PCM_TIER_G_SLEEP_ABSTRACTION.md``. Once training resumes,
    autograd reroutes the gradient through anchors (shared across
    cluster members ⇒ inter-member coupling) and residuals (per-member
    idiosyncrasy), which is the entire point of the abstraction
    pathway.

    Args:
        cg: target graph (must have :func:`attach_sleep` called and at
            least one prior :func:`run_sleep_pass` for this facet).
        caller: attribution string, stamped into ``consumed_by``
            entries on every visited slot (anchor, residual, and the
            fallback direct row when applicable).
        facet: original facet name (NOT the residual facet).
        concept_ids: member concept ids; one row returned per id.
        shape: row shape, e.g. ``(bias_dim,)``; passed through to
            :meth:`_ensure_facet` for facet allocation parity.
        tick: monotonic tick stamped into attribution.
        init: facet-init mode for any auto-grown facets.
        device: target device.
        fallback: ``"direct"`` (default) or ``"raise"``. Controls
            behaviour for member ids that have no registered abstract
            relation on this facet (e.g. concepts created after the
            last sleep pass): ``"direct"`` falls back to
            :meth:`collapse_batch`-equivalent rows for those positions.
        index: optional pre-built ``(facet, member_id) → anchor_id``
            mapping returned by :func:`build_member_to_anchor_index`.
            Pass it in if you call ``collapse_via_abstract`` many
            times per training step to amortise the scan cost.

    Returns:
        Tensor of shape ``(len(concept_ids), *shape)`` on ``device``.
    """
    if not concept_ids:
        return torch.empty((0, *tuple(int(s) for s in shape)), device=device)

    shape_t = tuple(int(s) for s in shape)
    res_facet = RESIDUAL_FACET_TEMPLATE.format(facet=str(facet))
    cg._ensure_facet(facet, shape_t, device, init)
    if res_facet not in cg.bundle_pool:
        raise RuntimeError(
            f"collapse_via_abstract: residual facet {res_facet!r} missing — "
            "did you call pcm.sleep.run_sleep_pass(cg, ...) for this facet?"
        )
    if index is None:
        index = build_member_to_anchor_index(cg)

    pool = cg.bundle_pool[facet]
    res_pool = cg.bundle_pool[res_facet]
    pool_device = pool.device

    anchor_slots: list[int] = []
    member_slots_for_residual: list[int] = []
    is_abstract: list[bool] = []
    for cid in concept_ids:
        if cid not in cg.cid_to_slot:
            raise KeyError(f"concept {cid!r} not in ConceptGraph")
        member_slot = cg.cid_to_slot[cid]
        anchor_id = index.get((str(facet), str(cid)))
        if anchor_id is None or anchor_id not in cg.cid_to_slot:
            if fallback == "raise":
                raise KeyError(
                    f"no abstract relation for ({facet!r}, {cid!r}); "
                    "set fallback='direct' to silently use the direct row"
                )
            cg._init_slot_if_unset(facet, member_slot, init)
            anchor_slots.append(member_slot)
            member_slots_for_residual.append(member_slot)
            is_abstract.append(False)
        else:
            anchor_slots.append(cg.cid_to_slot[anchor_id])
            member_slots_for_residual.append(member_slot)
            is_abstract.append(True)

    a_t = torch.as_tensor(anchor_slots, dtype=torch.long, device=pool_device)
    r_t = torch.as_tensor(
        member_slots_for_residual, dtype=torch.long, device=res_pool.device
    )
    residual_rows = F.embedding(r_t, res_pool)

    if assignment == "soft":
        # S2: mixture-of-anchors routing. Use the (frozen) original
        # member row in bundle_pool[facet][member_slot] as the query;
        # this is exactly anchor + residual at sleep snapshot time so
        # the softmax is well-anchored to the hard assignment but
        # gives every other anchor in the facet a non-zero weight,
        # which lets gradient flow into all anchors (dense-backprop
        # / Default-MoE style; cures expert starvation under heavy
        # use_abstract training).
        anchor_id_set = sorted(set(index.values())) if index else []
        if not anchor_id_set:
            anchor_rows = F.embedding(a_t, pool)
            out = anchor_rows + residual_rows
        else:
            all_anchor_slots = torch.as_tensor(
                [cg.cid_to_slot[aid] for aid in anchor_id_set],
                dtype=torch.long, device=pool_device,
            )
            all_anchors = F.embedding(all_anchor_slots, pool)  # (K, D)
            member_q_t = torch.as_tensor(
                [cg.cid_to_slot[cid] for cid in concept_ids],
                dtype=torch.long, device=pool_device,
            )
            member_q = F.embedding(member_q_t, pool).detach()
            mq = F.normalize(member_q, dim=-1, eps=1e-12)
            an = F.normalize(all_anchors, dim=-1, eps=1e-12)
            d = 1.0 - mq @ an.t()  # (B, K) cosine distance
            tau = max(float(soft_tau), 1e-6)
            weights = F.softmax(-d / tau, dim=-1)  # (B, K)
            soft_anchor = weights @ all_anchors  # (B, D)
            if is_abstract and not all(is_abstract):
                # For un-registered members, soft mixture is undefined
                # (no anchor in cluster) → fall back to direct read.
                hard_anchor = F.embedding(a_t, pool)
                mask = torch.tensor(
                    is_abstract, dtype=residual_rows.dtype,
                    device=residual_rows.device,
                ).view(-1, *([1] * (residual_rows.ndim - 1)))
                out = mask * (soft_anchor + residual_rows) + (1 - mask) * hard_anchor
            else:
                out = soft_anchor + residual_rows
    else:
        anchor_rows = F.embedding(a_t, pool)
        if is_abstract and not all(is_abstract):
            mask = torch.tensor(
                is_abstract, dtype=residual_rows.dtype,
                device=residual_rows.device,
            ).view(-1, *([1] * (residual_rows.ndim - 1)))
            out = anchor_rows + residual_rows * mask
        else:
            out = anchor_rows + residual_rows

    visited: set[int] = set()
    for slot in anchor_slots:
        if slot in visited:
            continue
        visited.add(slot)
        cg._record_attribution(slot, facet, caller, int(tick))
    res_visited: set[int] = set()
    for slot, abstract in zip(member_slots_for_residual, is_abstract):
        if not abstract or slot in res_visited:
            continue
        res_visited.add(slot)
        cg._record_attribution(slot, res_facet, caller, int(tick))

    if device is not None:
        out = out.to(device)
    return out


# ---------------------------------------------------------------------------
# K-means primitives.
# ---------------------------------------------------------------------------


def _pairwise_distances(
    rows: torch.Tensor, centroids: torch.Tensor, distance: str
) -> torch.Tensor:
    """``(N, K)`` distance matrix in the requested metric."""
    if distance == "cosine":
        rn = F.normalize(rows, dim=-1, eps=1e-12)
        cn = F.normalize(centroids, dim=-1, eps=1e-12)
        return 1.0 - rn @ cn.t()
    if distance == "l2":
        return torch.cdist(rows, centroids).pow(2)
    raise ValueError(f"unknown distance {distance!r}; expected cosine|l2")


def _kmeans_init_pp(
    rows: torch.Tensor, k: int, *, distance: str, seed: int
) -> torch.Tensor:
    """K-means++ seeding (Arthur & Vassilvitskii 2007)."""
    n, d = rows.shape
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    centroids = torch.empty(k, d, device=rows.device, dtype=rows.dtype)
    idx0 = int(torch.randint(0, n, (1,), generator=g).item())
    centroids[0] = rows[idx0]
    for c in range(1, k):
        d2 = (
            _pairwise_distances(rows, centroids[:c], distance).min(dim=1).values
        )
        d2 = d2.clamp(min=0.0)
        if float(d2.sum().item()) <= 0.0:
            idx = int(torch.randint(0, n, (1,), generator=g).item())
        else:
            probs = (d2 / d2.sum()).detach().cpu()
            idx = int(torch.multinomial(probs, 1, generator=g).item())
        centroids[c] = rows[idx]
    return centroids


def _kmeans(
    rows: torch.Tensor, k: int, *, cfg: SleepConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Lloyd's algorithm. Returns ``(centroids (K,D), assignments (N,))``.

    Centroids are computed as the **raw mean** of assigned members so
    invariant G4 (``centroid == mean(members)``) holds bit-for-bit.
    Cosine clustering is achieved by using cosine *distance* during
    assignment; the centroid update remains an unnormalised mean.
    """
    n = rows.shape[0]
    if k > n:
        raise ValueError(f"k={k} > n_active={n}; pick fewer clusters")
    if cfg.init == "kmeans++":
        centroids = _kmeans_init_pp(
            rows, k, distance=cfg.distance, seed=cfg.seed
        )
    elif cfg.init == "uniform":
        g = torch.Generator(device="cpu")
        g.manual_seed(cfg.seed)
        idx = torch.randperm(n, generator=g)[:k]
        centroids = rows[idx].clone()
    else:
        raise ValueError(f"unknown init {cfg.init!r}; expected kmeans++|uniform")

    prev = centroids.clone()
    for _it in range(cfg.kmeans_iters):
        d_mat = _pairwise_distances(rows, centroids, cfg.distance)
        assignments = d_mat.argmin(dim=1)
        new_centroids = centroids.clone()
        for c in range(k):
            mask = assignments == c
            if mask.any():
                new_centroids[c] = rows[mask].mean(dim=0)
        delta = float((new_centroids - prev).abs().max().item())
        centroids = new_centroids
        prev = centroids.clone()
        if delta < cfg.kmeans_tol:
            break
    d_mat = _pairwise_distances(rows, centroids, cfg.distance)
    assignments = d_mat.argmin(dim=1)
    return centroids, assignments


def _silhouette_score(
    rows: torch.Tensor, assignments: torch.Tensor, distance: str
) -> float:
    """Mean silhouette over all rows; returns 0.0 when undefined."""
    n = rows.shape[0]
    k = int(assignments.max().item()) + 1 if rows.numel() > 0 else 0
    if k < 2 or n < 3:
        return 0.0
    if distance == "cosine":
        rn = F.normalize(rows, dim=-1, eps=1e-12)
        d_mat = 1.0 - rn @ rn.t()
    else:
        d_mat = torch.cdist(rows, rows)
    sils = torch.zeros(n)
    for i in range(n):
        own = int(assignments[i].item())
        own_mask = (assignments == own)
        own_mask = own_mask.clone()
        own_mask[i] = False
        a = float(d_mat[i, own_mask].mean().item()) if own_mask.any() else 0.0
        b_vals = []
        for c in range(k):
            if c == own:
                continue
            m = assignments == c
            if m.any():
                b_vals.append(float(d_mat[i, m].mean().item()))
        b = min(b_vals) if b_vals else 0.0
        denom = max(a, b)
        if denom > 0:
            sils[i] = (b - a) / denom
    return float(sils.mean().item())


# ---------------------------------------------------------------------------
# Active-slot helpers.
# ---------------------------------------------------------------------------


def _active_slots_for_facet(cg: "ConceptGraph", facet: str) -> list[int]:
    """Slots that have ever been collapsed on ``facet`` (excluding sleep itself).

    Tier-G clusters the *trained* member rows, not its own prototypes.
    We exclude any slot whose *only* attribution on ``facet`` is the
    sleep caller, which keeps idempotency (G6) trivially true.
    """
    out: list[int] = []
    for slot, facets in cg._active_facets_by_slot.items():
        if facet not in facets:
            continue
        cb = cg._consumed_by_by_slot.get(slot, {}).get(facet, set())
        if not cb:
            continue
        if cb == {SLEEP_CALLER}:
            # Pure-sleep slot (a prototype). Skip.
            continue
        out.append(slot)
    return sorted(out)


def _resolve_k(n_active: int, requested: int | str) -> int:
    if isinstance(requested, int):
        if requested < 2:
            return 2
        return min(requested, max(2, n_active))
    if requested == "auto":
        if n_active <= 2:
            return n_active
        return max(2, int(round(math.sqrt(n_active) / 2)))
    raise ValueError(f"k_clusters must be int or 'auto', got {requested!r}")


# ---------------------------------------------------------------------------
# Node registration helpers (phase C).
# ---------------------------------------------------------------------------


def register_prototype(
    cg: "ConceptGraph",
    *,
    facet: str,
    k: int,
    centroid: torch.Tensor,
    tick: int = 0,
    scope: str = "ABSTRACT",
    anchor_ema: float = 1.0,
) -> str:
    """Register an ``abstract_prototype`` node holding ``centroid``.

    The node is allocated a fresh slot in ``cg.bundle_pool[facet]``
    (auto-grown if needed). Idempotent on re-call: an existing
    prototype's anchor row is updated either by hard overwrite
    (``anchor_ema=1.0``, legacy default) or by Online Codebook /
    VectorQuantizeEMA style blend
    (``new_anchor = (1-α) * old_anchor + α * centroid``,
    ``α = anchor_ema``).

    Returns the prototype concept id.
    """
    proto_id = PROTO_CID_TEMPLATE.format(facet=str(facet), k=int(k))
    centroid_d = centroid.detach()
    if proto_id in cg.concepts:
        slot = cg.cid_to_slot[proto_id]
        target_device = cg.bundle_pool[facet].device
        c_on_dev = centroid_d.to(target_device)
        with torch.no_grad():
            if anchor_ema >= 1.0:
                cg.bundle_pool[facet].data[slot] = c_on_dev
            else:
                old = cg.bundle_pool[facet].data[slot]
                cg.bundle_pool[facet].data[slot] = (
                    (1.0 - anchor_ema) * old + anchor_ema * c_on_dev
                )
        cg._record_attribution(slot, facet, SLEEP_CALLER, tick)
        return proto_id

    node = cg.register_concept(
        node_id=proto_id,
        label=f"PROTO_{facet}_{k}",
        scope=scope,
        provenance=f"sleep:tick={tick}:facet={facet}:k={k}",
        tick=tick,
    )
    node.kind = PROTO_KIND
    slot = cg.cid_to_slot[proto_id]
    shape = tuple(int(s) for s in centroid.shape)
    cg._ensure_facet(facet, shape, centroid.device, "zero")
    with torch.no_grad():
        cg.bundle_pool[facet].data[slot] = centroid_d.to(
            cg.bundle_pool[facet].device
        )
    cg._initialized_rows.add((facet, slot))
    cg._record_attribution(slot, facet, SLEEP_CALLER, tick)
    return proto_id


def register_relation(
    cg: "ConceptGraph",
    *,
    anchor_id: str,
    member_id: str,
    facet: str,
    mode: str = "add",
    tick: int = 0,
    scope: str = "ABSTRACT",
) -> str:
    """Register a cookable abstract-relation subgraph.

    The subgraph reads (anchor row on ``facet``) and (member residual
    on ``facet + "_residual"``), then combines them via
    ``concept.relation_apply(mode)``.

    Returns the relation concept id (idempotent on re-call).
    """
    if anchor_id not in cg.concepts:
        raise KeyError(f"anchor_id {anchor_id!r} not in graph")
    if member_id not in cg.concepts:
        raise KeyError(f"member_id {member_id!r} not in graph")
    rel_id = RELATION_CID_TEMPLATE.format(
        facet=str(facet),
        k=anchor_id.split(":")[-1],
        member=member_id.split(":")[-1],
    )
    if rel_id in cg.concepts:
        return rel_id
    res_facet = RESIDUAL_FACET_TEMPLATE.format(facet=str(facet))
    cg.register_muscle_subgraph(
        node_id=rel_id,
        inputs=[],
        constants={
            "kind_hint": RELATION_KIND_HINT,
            "anchor_facet": str(facet),
            "anchor_id": anchor_id,
            "residual_facet": res_facet,
            "member_id": member_id,
            "mode": str(mode),
        },
        nodes=[
            {
                "id": "anchor",
                "op": "concept.codebook_lookup",
                "args": [str(facet), anchor_id],
            },
            {
                "id": "residual",
                "op": "muscle.collapse_facet",
                "args": [res_facet, [member_id]],
            },
            {
                "id": "out",
                "op": "concept.relation_apply",
                "args": ["@anchor", "@residual", str(mode)],
            },
        ],
        output="out",
        provenance=f"sleep:tick={tick}:rel={rel_id}",
        scope=scope,
        tick=tick,
    )
    return rel_id


def collapse_with_optional_abstract(
    cg: "ConceptGraph",
    *,
    caller: str,
    facet: str,
    concept_ids: list[str],
    shape,
    tick: int = 0,
    init: str = "normal_small",
    device=None,
    use_abstract: bool = False,
    assignment: str = "hard",
    soft_tau: float = 0.5,
) -> torch.Tensor:
    """Drop-in replacement for :meth:`ConceptGraph.collapse_batch` that
    routes through :func:`collapse_via_abstract` when:

    * ``use_abstract=True``, AND
    * ``cg`` has had :func:`attach_sleep` called, AND
    * the residual facet ``<facet>_residual`` exists on ``cg`` (i.e. at
      least one :func:`run_sleep_pass` has been completed for this
      facet).

    Otherwise it transparently falls back to ``cg.collapse_batch(...)``.
    Designed so that every muscle head in the codebase can be upgraded
    to abstract reads with a single one-line replacement of its
    ``cg.collapse_batch(...)`` call.

    The function is a thin dispatcher; it adds no mutable state to
    ``cg`` and is safe to call from inside ``forward``.
    """
    if use_abstract and getattr(cg, "sleep_enabled", False):
        res_facet = RESIDUAL_FACET_TEMPLATE.format(facet=str(facet))
        if res_facet in cg.bundle_pool:
            return collapse_via_abstract(
                cg,
                caller=caller,
                facet=facet,
                concept_ids=concept_ids,
                shape=shape,
                tick=tick,
                init=init,
                device=device,
                assignment=assignment,
                soft_tau=soft_tau,
            )
    return cg.collapse_batch(
        caller=caller,
        facet=facet,
        concept_ids=concept_ids,
        shape=shape,
        tick=tick,
        init=init,
        device=device,
    )


def materialize_effective_bundle_state(
    cg: "ConceptGraph",
    bundle_state: dict[str, dict[str, torch.Tensor]],
    *,
    facets: Iterable[str],
    use_abstract: bool,
    assignment: str = "hard",
    soft_tau: float = 0.5,
) -> dict[str, dict[str, torch.Tensor]]:
    """Post-process a host ``bundle_state`` to replace each facet row
    with the **effective** row the head saw under ``use_abstract``.

    For host training loops that compute downstream geometry metrics
    (ρ, cos similarity, MDS …) directly from ``cg.concepts[cid].bundle.
    state_dict()``, this is the bit-perfect equivalent of what
    :func:`collapse_with_optional_abstract` returns at forward time.

    No-op (identity copy) when ``use_abstract`` is ``False`` or no
    abstract relations have been registered for any of ``facets``.

    Args:
        cg: graph after a sleep pass.
        bundle_state: ``{cid: {param_key: tensor}}`` mapping; typically
            built by the host as
            ``{cid: dict(c.bundle.state_dict()) for cid, c in cg.concepts.items()}``.
            The original tensors are kept as fallback for concepts that
            have no abstract relation registered.
        facets: which facets to rewrite; pass the same list you handed
            to :func:`attach_sleep` / :func:`run_sleep_pass`.
        use_abstract / assignment / soft_tau: forwarded to
            :func:`collapse_via_abstract` per facet.

    Returns:
        A new dict; tensors for unaffected entries are shared with the
        input.
    """
    if not use_abstract or not getattr(cg, "sleep_enabled", False):
        return bundle_state
    out = {cid: dict(rows) for cid, rows in bundle_state.items()}
    index = build_member_to_anchor_index(cg)
    if not index:
        return out
    for facet in facets:
        f = str(facet)
        res_facet = RESIDUAL_FACET_TEMPLATE.format(facet=f)
        if res_facet not in cg.bundle_pool:
            continue
        member_cids = [
            cid for cid, anchor in index.items()
            if cid[0] == f and cid[1] in cg.cid_to_slot
        ]
        if not member_cids:
            continue
        ids = [cid[1] for cid in member_cids]
        shape = cg.bundle_pool[f].shape[1:]
        rows = collapse_via_abstract(
            cg, caller="materialize_effective_bundle_state",
            facet=f, concept_ids=ids,
            shape=tuple(int(s) for s in shape),
            tick=0, init=cg._facet_default_init.get(f, "normal_small"),
            device=cg.bundle_pool[f].device,
            index=index,
            assignment=assignment, soft_tau=soft_tau,
        ).detach().cpu()
        key = f"params.{f}"
        for i, cid in enumerate(ids):
            if cid not in out:
                out[cid] = {}
            out[cid][key] = rows[i].clone()
    return out


# ---------------------------------------------------------------------------
# Replay-source helper for host training loops.
# ---------------------------------------------------------------------------

ReplayItem = tuple[str, Callable[[], torch.Tensor], torch.Tensor]
"""(caller, forward_fn, target) — the forward_fn must close over its own
inputs so the replay loop can call it without arguments."""


def make_replay_source_from_buffer(
    buffer: list[ReplayItem],
    *,
    batch_size: int,
    seed: int = 0,
) -> Callable[[], list[ReplayItem]]:
    """Build a ``replay_source`` callable that samples ``batch_size`` items
    from ``buffer`` per call. Sampling is deterministic given ``seed``.
    """
    state = {"step": 0}

    def _source() -> list[ReplayItem]:
        if not buffer:
            return []
        n = len(buffer)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) + int(state["step"]))
        state["step"] += 1
        size = min(int(batch_size), n)
        idx = torch.randperm(n, generator=g)[:size].tolist()
        return [buffer[i] for i in idx]

    return _source


# ---------------------------------------------------------------------------
# The main entry point: run_sleep_pass.
# ---------------------------------------------------------------------------


def _phase_a_snapshot(
    cg: "ConceptGraph", facet: str
) -> tuple[list[int], torch.Tensor]:
    """Phase A — snapshot active rows; runs under ``no_grad``."""
    active = _active_slots_for_facet(cg, facet)
    if not active:
        return [], torch.empty(0)
    pool = cg.bundle_pool[facet]
    with torch.no_grad():
        rows = pool.data[active].clone()
    return active, rows


def _phase_c_register(
    cg: "ConceptGraph",
    *,
    facet: str,
    active_slots: list[int],
    rows: torch.Tensor,
    centroids: torch.Tensor,
    assignments: torch.Tensor,
    cfg: SleepConfig,
    tick: int,
) -> tuple[list[str], list[str], list[float], list[float], list[int]]:
    """Phase C — register prototype + residual + relation nodes."""
    k = centroids.shape[0]
    res_facet = RESIDUAL_FACET_TEMPLATE.format(facet=facet)
    proto_ids: list[str] = []
    rel_ids: list[str] = []
    centroid_norms: list[float] = []
    residual_rms: list[float] = []
    member_counts: list[int] = []

    cg._ensure_facet(
        res_facet,
        tuple(int(s) for s in rows.shape[1:]),
        rows.device,
        cfg.residual_init,
    )

    for c in range(k):
        proto_id = register_prototype(
            cg,
            facet=facet,
            k=c,
            centroid=centroids[c],
            tick=tick,
            scope=cfg.abstract_scope,
            anchor_ema=cfg.anchor_ema,
        )
        proto_ids.append(proto_id)
        centroid_norms.append(float(centroids[c].norm().item()))
        member_counts.append(int((assignments == c).sum().item()))

    proto_slots = [cg.cid_to_slot[pid] for pid in proto_ids]
    effective_anchors = cg.bundle_pool[facet].data[proto_slots].detach()

    for i, slot in enumerate(active_slots):
        cluster = int(assignments[i].item())
        member_cid = cg.slot_to_cid.get(slot)
        if member_cid is None:
            continue
        with torch.no_grad():
            residual = (rows[i] - effective_anchors[cluster]).detach().to(
                cg.bundle_pool[res_facet].device
            )
            cg.bundle_pool[res_facet].data[slot] = residual
        cg._initialized_rows.add((res_facet, slot))
        cg._record_attribution(slot, res_facet, SLEEP_CALLER, tick)
        residual_rms.append(float(residual.pow(2).mean().sqrt().item()))
        rel_id = register_relation(
            cg,
            anchor_id=proto_ids[cluster],
            member_id=member_cid,
            facet=facet,
            mode=cfg.relation_mode,
            tick=tick,
            scope=cfg.abstract_scope,
        )
        rel_ids.append(rel_id)
    return proto_ids, rel_ids, centroid_norms, residual_rms, member_counts


def _phase_e_replay(
    cg: "ConceptGraph",
    *,
    optimizer: "torch.optim.Optimizer",
    replay_source: Callable[[], list[ReplayItem]],
    cfg: SleepConfig,
    protected_slots_by_facet: dict[str, set[int]],
    residual_facets: set[str],
) -> list[float]:
    """Phase E — interleaved replay with hard gradient mask on
    abstract slots / residual facets.

    Returns the per-step loss curve.
    """
    losses: list[float] = []
    if cfg.replay_steps <= 0:
        return losses
    if optimizer is None:
        raise ValueError(
            "run_sleep_pass: replay_steps > 0 requires an optimizer to step "
            "the original muscles. Pass optimizer=... or set replay_steps=0."
        )
    for _step in range(cfg.replay_steps):
        items = replay_source()
        if not items:
            break
        terms: list[torch.Tensor] = []
        for _caller, forward_fn, target in items:
            out = forward_fn()
            if target.dtype.is_floating_point:
                terms.append(F.mse_loss(out, target))
            else:
                terms.append(F.cross_entropy(out, target))
        if not terms:
            break
        loss = torch.stack(terms).mean()
        optimizer.zero_grad(set_to_none=False)
        loss.backward()
        # Hard mask: zero grads on (a) prototype slots in original facets,
        # (b) every slot in residual facets. This protects abstract
        # state from drift and keeps G2 / G3 valid.
        with torch.no_grad():
            for f, slots in protected_slots_by_facet.items():
                pool = cg.bundle_pool.get(f)
                if pool is None or pool.grad is None:
                    continue
                if not slots:
                    continue
                idx = torch.tensor(sorted(slots), dtype=torch.long,
                                   device=pool.device)
                pool.grad.index_fill_(0, idx, 0.0)
            for f in residual_facets:
                pool = cg.bundle_pool.get(f)
                if pool is None or pool.grad is None:
                    continue
                pool.grad.zero_()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    return losses


def run_sleep_pass(
    cg: "ConceptGraph",
    optimizer: "torch.optim.Optimizer | None" = None,
    *,
    facets: Iterable[str] | None = None,
    config: SleepConfig | None = None,
    replay_source: Callable[[], list[ReplayItem]] | None = None,
    tick: int = 0,
    force_recluster: bool = False,
) -> SleepReport:
    """Run one offline sleep pass.

    Args:
        cg: target graph; must have :func:`attach_sleep` called first
            (raises otherwise).
        optimizer: required iff ``config.replay_steps > 0``. Used for
            phase E (interleaved replay). Tier-G never registers new
            parameters into the optimizer; the new abstract slots ride
            on the existing ``bundle_pool`` parameters that the
            optimizer already covers.
        facets: explicit facet list. ``None`` → use the allow-list set
            by :func:`attach_sleep` if any, else every facet that has
            at least one non-sleep attribution.
        config: :class:`SleepConfig`; defaults to a safe no-replay
            config when omitted.
        replay_source: zero-arg callable returning a list of
            ``(caller, forward_fn, target)`` for phase E. Required iff
            ``config.replay_steps > 0``.
        tick: monotonically increasing tick stamped into attribution.
        force_recluster: if ``True``, re-cluster even when residual
            facet already exists with sleep attribution; otherwise the
            pass is idempotent and skips facets that already have a
            valid clustering (G6).

    Returns:
        :class:`SleepReport` with per-facet stats and the replay loss
        curve. ``SleepReport.skipped_facets`` lists facets that were
        not clustered (already sleep-cached, or had < 2 active slots).
    """
    if not getattr(cg, "sleep_enabled", False):
        raise RuntimeError(
            "run_sleep_pass: pcm.sleep.attach_sleep(cg) must be called first"
        )
    cfg = config if config is not None else SleepConfig()
    report = SleepReport(tick=int(tick))

    # Resolve facet list.
    if facets is not None:
        candidate_facets = list(dict.fromkeys(str(f) for f in facets))
    else:
        allowlist = getattr(cg, "_sleep_facets_allowlist", None)
        if allowlist:
            candidate_facets = sorted(allowlist)
        else:
            candidate_facets = sorted(
                f for f in cg.bundle_pool if not f.endswith("_residual")
            )

    protected_slots_by_facet: dict[str, set[int]] = {}
    residual_facets: set[str] = set()

    for facet in candidate_facets:
        if facet not in cg.bundle_pool:
            report.skipped_facets.append(facet)
            continue
        active, rows = _phase_a_snapshot(cg, facet)
        if len(active) < 2:
            report.skipped_facets.append(facet)
            continue
        res_facet = RESIDUAL_FACET_TEMPLATE.format(facet=facet)
        already_clustered = (
            res_facet in cg.bundle_pool
            and any(
                SLEEP_CALLER in cg._consumed_by_by_slot.get(s, {}).get(
                    res_facet, set()
                )
                for s in active
            )
        )
        if already_clustered and not force_recluster:
            report.skipped_facets.append(facet)
            # Still need to mark protected slots so phase E doesn't drift them.
            existing_proto_slots: set[int] = set()
            for nid, node in cg.concepts.items():
                if getattr(node, "kind", None) != PROTO_KIND:
                    continue
                meta_consts = (node.metadata or {}).get(
                    "constants", {}) if hasattr(node, "metadata") else {}
                # Prototypes don't carry constants; identify by id template.
                if nid.startswith(f"concept:cluster:{facet}:"):
                    existing_proto_slots.add(cg.cid_to_slot[nid])
            protected_slots_by_facet[facet] = existing_proto_slots
            residual_facets.add(res_facet)
            continue

        k = _resolve_k(len(active), cfg.k_clusters)
        centroids, assignments = _kmeans(rows, k, cfg=cfg)
        sil = _silhouette_score(rows, assignments, cfg.distance)

        proto_ids, _rel_ids, c_norms, r_rms, m_counts = _phase_c_register(
            cg,
            facet=facet,
            active_slots=active,
            rows=rows,
            centroids=centroids,
            assignments=assignments,
            cfg=cfg,
            tick=tick,
        )
        report.facets.append(
            FacetSleepReport(
                facet=facet,
                n_active=len(active),
                k_clusters=k,
                silhouette=sil,
                centroid_norms=c_norms,
                residual_rms=r_rms,
                member_counts=m_counts,
            )
        )
        protected_slots_by_facet[facet] = {
            cg.cid_to_slot[pid] for pid in proto_ids
        }
        residual_facets.add(res_facet)

    if cfg.replay_steps > 0 and replay_source is not None:
        report.replay_loss_curve = _phase_e_replay(
            cg,
            optimizer=optimizer,
            replay_source=replay_source,
            cfg=cfg,
            protected_slots_by_facet=protected_slots_by_facet,
            residual_facets=residual_facets,
        )

    cg._sleep_last_tick = int(tick)
    return report
