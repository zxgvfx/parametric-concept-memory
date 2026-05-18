"""pcm.episodic — F75 short-term + long-term episodic memory.

PCM until F74 only stores *abstract / distilled* memory: concept
slots in ``ConceptGraph``, learned operators in ``ParamBundle``,
sleep distillation cooks → RPE cache. There is no *time-indexed*
memory — no "what happened just now in this episode".

F75 closes that gap with a Complementary Learning Systems
(McClelland, McNaughton & O'Reilly 1995) architecture:

* :class:`EpisodicBuffer` — FIFO ring buffer of recent
  ``(slot, timestamp, salience, metadata)`` tuples. Models the
  short-term hippocampal trace.
* :class:`LongTermEpisodicTrace` — sparse, salience-gated
  persistent traces. A high-``|reward|`` or high-surprise event
  is *imprinted* once and persists indefinitely; routine events
  are forgotten. Models the "I was bitten by a snake at age 10
  and still fear snakes at 60" phenomenon.
* :func:`consolidate_to_concept_graph` — sleep-style: cluster
  episodes by slot similarity, write cluster centroids as new
  ``ConceptGraph`` slots. Reuses the existing F53/F59 distillation
  pipeline with the *source* swapped from cook history to
  episodic buffer.

All three components are pure data structures (no torch.nn);
they hold detached tensors and are device-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


__all__ = [
    "EpisodeRecord",
    "EpisodicBuffer",
    "LongTermEpisodicTrace",
    "consolidate_to_concept_graph",
]


# ─────────────────────────────────────────────────────────────────
# Episode record dataclass
# ─────────────────────────────────────────────────────────────────


@dataclass
class EpisodeRecord:
    """One episode entry."""

    slot: torch.Tensor                  # detached, shape (D,)
    timestamp: int
    salience: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────
# Short-term episodic buffer (FIFO, capacity-limited)
# ─────────────────────────────────────────────────────────────────


class EpisodicBuffer:
    """Bounded FIFO of recent episodes.

    The hippocampal-analogue: fast write, capacity-limited,
    time-indexed. When the buffer fills, the oldest episode is
    overwritten (FIFO eviction). Each episode carries a slot
    vector + integer timestamp + optional salience score +
    optional metadata dict.
    """

    def __init__(self, capacity: int, slot_dim: int,
                 device: str = "cpu") -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if slot_dim <= 0:
            raise ValueError(f"slot_dim must be positive, got {slot_dim}")
        self.capacity = capacity
        self.slot_dim = slot_dim
        self.device = device
        self._slots = torch.zeros(capacity, slot_dim, device=device)
        self._timestamps = torch.full(
            (capacity,), -1, dtype=torch.long, device=device,
        )
        self._saliences = torch.zeros(capacity, device=device)
        self._metadata: list[dict[str, Any]] = [{} for _ in range(capacity)]
        self._next_idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size >= self.capacity

    def append(
        self, slot: torch.Tensor, timestamp: int,
        salience: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert one episode at the current ring position
        (overwriting the oldest if full)."""
        if slot.shape != (self.slot_dim,):
            raise ValueError(
                f"slot shape {tuple(slot.shape)} != ({self.slot_dim},)"
            )
        idx = self._next_idx
        self._slots[idx] = slot.detach().to(self.device)
        self._timestamps[idx] = int(timestamp)
        self._saliences[idx] = float(salience)
        self._metadata[idx] = dict(metadata) if metadata else {}
        self._next_idx = (self._next_idx + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def _valid_slice(self) -> tuple[torch.Tensor, torch.Tensor,
                                     torch.Tensor, list[int]]:
        """Return (slots, timestamps, saliences, indices) for
        currently-occupied slots only.

        Order: insertion order (so most recent is last). Useful
        when callers want to scan chronologically.
        """
        if not self.is_full:
            idx = list(range(self._size))
        else:
            # Insertion order = (next_idx) is the oldest, wrap around
            idx = (list(range(self._next_idx, self.capacity))
                   + list(range(0, self._next_idx)))
        idx_t = torch.tensor(idx, dtype=torch.long, device=self.device)
        return (
            self._slots[idx_t],
            self._timestamps[idx_t],
            self._saliences[idx_t],
            idx,
        )

    def records(self) -> list[EpisodeRecord]:
        """All currently-stored episodes in insertion order
        (oldest first, most recent last)."""
        slots, timestamps, saliences, idx = self._valid_slice()
        return [
            EpisodeRecord(
                slot=slots[i].clone(),
                timestamp=int(timestamps[i].item()),
                salience=float(saliences[i].item()),
                metadata=dict(self._metadata[idx[i]]),
            )
            for i in range(len(idx))
        ]

    # ─── recall APIs ──────────────────────────────────────────────

    def recall_by_similarity(
        self, query_slot: torch.Tensor, k: int = 5,
    ) -> list[EpisodeRecord]:
        """Top-``k`` episodes by *cosine* similarity to ``query_slot``."""
        if self._size == 0:
            return []
        slots, timestamps, saliences, idx = self._valid_slice()
        q = query_slot.detach().to(self.device).view(1, -1)
        sims = F.cosine_similarity(q, slots, dim=-1)
        k = min(k, self._size)
        top = torch.topk(sims, k=k)
        out: list[EpisodeRecord] = []
        for j in top.indices.tolist():
            out.append(EpisodeRecord(
                slot=slots[j].clone(),
                timestamp=int(timestamps[j].item()),
                salience=float(saliences[j].item()),
                metadata=dict(self._metadata[idx[j]]),
            ))
        return out

    def recall_time_range(
        self, t_start: int, t_end: int,
    ) -> list[EpisodeRecord]:
        """All episodes with ``t_start ≤ timestamp ≤ t_end``,
        sorted by timestamp ascending."""
        if self._size == 0:
            return []
        slots, timestamps, saliences, idx = self._valid_slice()
        mask = (timestamps >= t_start) & (timestamps <= t_end)
        out: list[EpisodeRecord] = []
        for j in mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            out.append(EpisodeRecord(
                slot=slots[j].clone(),
                timestamp=int(timestamps[j].item()),
                salience=float(saliences[j].item()),
                metadata=dict(self._metadata[idx[j]]),
            ))
        out.sort(key=lambda r: r.timestamp)
        return out

    def recall_most_recent(self, k: int = 5) -> list[EpisodeRecord]:
        """The ``k`` most recently appended episodes (newest
        last)."""
        if self._size == 0:
            return []
        slots, timestamps, saliences, idx = self._valid_slice()
        k = min(k, self._size)
        out: list[EpisodeRecord] = []
        # Insertion order has most recent at the *end*
        for j in range(self._size - k, self._size):
            out.append(EpisodeRecord(
                slot=slots[j].clone(),
                timestamp=int(timestamps[j].item()),
                salience=float(saliences[j].item()),
                metadata=dict(self._metadata[idx[j]]),
            ))
        return out

    def recall_most_salient(self, k: int = 5) -> list[EpisodeRecord]:
        """Top-``k`` episodes by salience score."""
        if self._size == 0:
            return []
        slots, timestamps, saliences, idx = self._valid_slice()
        k = min(k, self._size)
        top = torch.topk(saliences, k=k)
        out: list[EpisodeRecord] = []
        for j in top.indices.tolist():
            out.append(EpisodeRecord(
                slot=slots[j].clone(),
                timestamp=int(timestamps[j].item()),
                salience=float(saliences[j].item()),
                metadata=dict(self._metadata[idx[j]]),
            ))
        return out


# ─────────────────────────────────────────────────────────────────
# Long-term episodic trace (salience-gated, sparse)
# ─────────────────────────────────────────────────────────────────


class LongTermEpisodicTrace:
    """Sparse, salience-gated, persistent episodic memory.

    Differences from :class:`EpisodicBuffer`:

    * **Salience-gated writes**: only events with ``salience ≥
      threshold`` are imprinted. Routine events are *not* stored.
    * **No FIFO eviction below capacity**: persistent until
      explicitly cleared. When at capacity, the *lowest-salience*
      stored episode is evicted (not the oldest).
    * **Biased recall**: similarity-based recall scores can be
      weighted by the stored episode's salience, so emotionally-
      strong memories have priority over similar-but-bland ones.

    Models the "I was bitten by a snake at age 10 → still afraid
    of snakes at 60" phenomenon: single rare high-|reward| events
    persist across orders-of-magnitude more bland routine events.
    """

    def __init__(
        self, capacity: int, slot_dim: int,
        salience_threshold: float = 1.0,
        device: str = "cpu",
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if salience_threshold < 0:
            raise ValueError(
                f"threshold must be >= 0, got {salience_threshold}"
            )
        self.capacity = capacity
        self.slot_dim = slot_dim
        self.salience_threshold = float(salience_threshold)
        self.device = device
        self._slots = torch.zeros(capacity, slot_dim, device=device)
        self._timestamps = torch.full(
            (capacity,), -1, dtype=torch.long, device=device,
        )
        self._saliences = torch.full(
            (capacity,), float("-inf"), device=device,
        )
        self._metadata: list[dict[str, Any]] = [{} for _ in range(capacity)]
        self._size = 0
        self._writes_attempted = 0
        self._writes_committed = 0

    def __len__(self) -> int:
        return self._size

    @property
    def writes_attempted(self) -> int:
        return self._writes_attempted

    @property
    def writes_committed(self) -> int:
        return self._writes_committed

    def maybe_imprint(
        self, slot: torch.Tensor, timestamp: int,
        salience: float, metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Try to write one episode. Returns True iff it was
        actually imprinted.

        Rules:
        * ``salience < salience_threshold`` → reject.
        * Has free slot → write to first empty slot.
        * Full → if salience > min stored salience, evict the
          lowest-salience episode and replace; else reject.
        """
        self._writes_attempted += 1
        s = float(salience)
        if s < self.salience_threshold:
            return False
        slot = slot.detach().to(self.device)
        if slot.shape != (self.slot_dim,):
            raise ValueError(
                f"slot shape {tuple(slot.shape)} != ({self.slot_dim},)"
            )
        if self._size < self.capacity:
            idx = self._size
            self._size += 1
        else:
            # Evict lowest-salience
            min_idx = int(self._saliences.argmin().item())
            if self._saliences[min_idx].item() >= s:
                return False
            idx = min_idx
        self._slots[idx] = slot
        self._timestamps[idx] = int(timestamp)
        self._saliences[idx] = s
        self._metadata[idx] = dict(metadata) if metadata else {}
        self._writes_committed += 1
        return True

    def _valid_slice(self):
        idx = list(range(self._size))
        if not idx:
            return None, None, None, idx
        idx_t = torch.tensor(idx, dtype=torch.long, device=self.device)
        return (
            self._slots[idx_t],
            self._timestamps[idx_t],
            self._saliences[idx_t],
            idx,
        )

    def biased_recall(
        self, query_slot: torch.Tensor, *,
        k: int = 5, salience_weight: float = 0.5,
    ) -> list[EpisodeRecord]:
        """Top-``k`` recall using
        ``score = similarity + salience_weight · normalised_salience``.

        Salience normalisation is min-max within currently-stored
        episodes, so the scale of ``salience_weight`` doesn't
        depend on the absolute salience range.
        """
        if self._size == 0:
            return []
        slots, timestamps, saliences, idx = self._valid_slice()
        q = query_slot.detach().to(self.device).view(1, -1)
        sims = F.cosine_similarity(q, slots, dim=-1)
        if saliences.numel() > 1:
            s_min = saliences.min()
            s_max = saliences.max()
            denom = (s_max - s_min).clamp(min=1e-8)
            sal_norm = (saliences - s_min) / denom
        else:
            sal_norm = torch.zeros_like(saliences)
        score = sims + salience_weight * sal_norm
        k = min(k, self._size)
        top = torch.topk(score, k=k)
        out: list[EpisodeRecord] = []
        for j in top.indices.tolist():
            out.append(EpisodeRecord(
                slot=slots[j].clone(),
                timestamp=int(timestamps[j].item()),
                salience=float(saliences[j].item()),
                metadata=dict(self._metadata[idx[j]]),
            ))
        return out

    def records(self) -> list[EpisodeRecord]:
        slots, timestamps, saliences, idx = self._valid_slice()
        if not idx:
            return []
        return [
            EpisodeRecord(
                slot=slots[i].clone(),
                timestamp=int(timestamps[i].item()),
                salience=float(saliences[i].item()),
                metadata=dict(self._metadata[idx[i]]),
            )
            for i in range(len(idx))
        ]


# ─────────────────────────────────────────────────────────────────
# Sleep-style consolidation: episodic → semantic
# ─────────────────────────────────────────────────────────────────


def consolidate_to_concept_graph(
    buffer: EpisodicBuffer,
    *,
    n_clusters: int,
    n_iter: int = 25,
    rng_seed: int = 0,
    n_restarts: int = 5,
    device: str = "cpu",
) -> dict:
    """Cluster episodes by slot similarity (k-means) and return
    the cluster centroids as new concept-slot candidates.

    This is the sleep-style distillation step that converts
    short-term episodic traces into abstract semantic slots.
    Reuses the existing F53/F59 distillation pattern with the
    source swapped from cook history to ``buffer``.

    Returns a dict with:
    * ``centroids`` ``(n_clusters, slot_dim)`` — candidate
      ConceptGraph slot vectors.
    * ``assignments`` ``(n_episodes,)`` — per-episode cluster index.
    * ``inertia`` — sum of squared distances from points to
      their centroid (lower = tighter clusters).
    * ``n_episodes`` — total episodes consolidated.

    The caller decides how to write centroids into a real
    ``ConceptGraph`` — F75 just returns the centroids so the test
    can verify "consolidated centroids match ground-truth pattern".
    """
    if len(buffer) == 0:
        raise ValueError("buffer is empty; nothing to consolidate")
    slots, _, _, _ = buffer._valid_slice()
    n = slots.shape[0]
    n_clusters = max(1, min(n_clusters, n))

    def _one_run(seed: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        # k-means++ init. The generator's device must match the
        # tensors fed to ``torch.multinomial`` — otherwise PyTorch
        # raises "Expected a 'cuda' device type for generator but
        # found 'cpu'" when slots live on GPU.
        gen_device = slots.device.type
        g = torch.Generator(device=gen_device).manual_seed(seed)
        first_idx = int(torch.randint(
            0, n, (1,), generator=g, device=slots.device,
        ).item())
        centroids = slots[first_idx:first_idx + 1].clone()
        for _ in range(1, n_clusters):
            dists = torch.cdist(slots, centroids).min(dim=1).values
            probs = dists / dists.sum().clamp(min=1e-12)
            next_idx = int(torch.multinomial(
                probs, 1, generator=g,
            ).item())
            centroids = torch.cat(
                [centroids, slots[next_idx:next_idx + 1]], dim=0,
            )
        # Lloyd iteration
        for _ in range(n_iter):
            d = torch.cdist(slots, centroids)
            assignments = d.argmin(dim=1)
            new_centroids = centroids.clone()
            for c in range(n_clusters):
                mask = (assignments == c)
                if mask.any():
                    new_centroids[c] = slots[mask].mean(dim=0)
            if torch.allclose(new_centroids, centroids, atol=1e-6):
                centroids = new_centroids
                break
            centroids = new_centroids
        d = torch.cdist(slots, centroids)
        assignments = d.argmin(dim=1)
        inertia = float(
            d.gather(1, assignments.unsqueeze(-1)).pow(2).sum().item()
        )
        return centroids, assignments, inertia

    # Multiple restarts; keep the lowest-inertia solution.
    # k-means is non-convex and prone to local minima; restart
    # is the standard cure (sklearn's KMeans defaults to 10).
    best = None
    for r in range(max(1, n_restarts)):
        centroids, assignments, inertia = _one_run(
            rng_seed + r * 7919,  # different seeds per restart
        )
        if best is None or inertia < best[2]:
            best = (centroids, assignments, inertia)
    centroids, assignments, inertia = best
    return {
        "centroids": centroids,
        "assignments": assignments,
        "inertia": inertia,
        "n_episodes": int(n),
        "n_restarts": n_restarts,
    }
