"""pcm.online — F85 online teacher loop.

Implements the cognitive paradigm requested in the F85 design
doc: a pretrained F81 / F83 PCM LM is taught new concepts
**interactively** through teacher corrections, using three
update mechanisms at three different time scales:

* **M1 — Fast episodic write** (every event). Hidden-state
  snapshot + target token are stored in an
  :class:`~pcm.episodic.EpisodicBuffer` and, if the model is a
  :class:`~pcm.lm.TitansPCMMiniLM`, also imprinted to its
  :class:`HierarchicalMemoryLayer` so future forward passes can
  recall the corrected episode.

* **M2 — Mid-term sleep consolidation** (periodic). Runs
  K-means over the accumulated episodic buffer keyed by
  ``concept_id``. Reports cluster purity against the
  ground-truth {ANIMAL, FOOD} split (the F85 O6 invariant). At
  v9.0 this is *diagnostic only* — does not mutate parameters.

* **M3 — Slow micro-gradient** (conditional). One AdamW step at
  ``lr=1e-3`` on the **embedding rows of novel concepts only**
  (a selective-gradient mask). Each step pairs the correction
  with ``n_replay=7`` random pretraining examples from
  :class:`PretrainReplayBuffer` to prevent catastrophic
  forgetting (F85 O2).

The session class :class:`OnlineTeacherSession` is the public
entry-point. It wraps a pretrained model, a pretrain replay
buffer, and the three update mechanisms. The host script (e.g.
``experiments/online_teacher_f85.py``) drives the loop by
repeatedly calling :meth:`receive_correction` and periodically
:meth:`sleep`.

Cognitive parallels:

* M1 = hippocampal episodic write (Tulving-style "I was told").
* M2 = NREM-sleep replay & cortical consolidation.
* M3 = slow cortical re-tuning of lexical representations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .episodic import EpisodicBuffer, consolidate_to_concept_graph


__all__ = [
    "MechanismCounters",
    "PretrainReplayBuffer",
    "OnlineTeacherSession",
]


# ─────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────


@dataclass
class MechanismCounters:
    """Counts how often each update mechanism fires (F85 O5)."""

    n_corrections: int = 0
    n_m1_imprints: int = 0
    n_m1_titans_imprints: int = 0
    n_m3_grad_steps: int = 0
    n_sleeps: int = 0
    correction_counts: dict[int, int] = field(default_factory=dict)
    grad_step_counts: dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_corrections": self.n_corrections,
            "n_m1_imprints": self.n_m1_imprints,
            "n_m1_titans_imprints": self.n_m1_titans_imprints,
            "n_m3_grad_steps": self.n_m3_grad_steps,
            "n_sleeps": self.n_sleeps,
            "correction_counts": dict(self.correction_counts),
            "grad_step_counts": dict(self.grad_step_counts),
        }


# ─────────────────────────────────────────────────────────────────
# PretrainReplayBuffer
# ─────────────────────────────────────────────────────────────────


class PretrainReplayBuffer:
    """FIFO buffer of pretraining ``(input_ids, target_ids)``
    pairs. Sampled during M3 micro-gradient steps to prevent
    catastrophic forgetting of pretrained vocabulary.

    Stores the *last* ``capacity`` examples observed during the
    final phase of pretraining. Each entry is a pair of 1-D
    ``LongTensor`` of equal length (``seq_len``).
    """

    def __init__(
        self, capacity: int = 256, device: str = "cpu",
    ) -> None:
        self.capacity = capacity
        self.device = device
        self._x: list[torch.Tensor] = []
        self._y: list[torch.Tensor] = []
        self._rng = torch.Generator(device="cpu").manual_seed(2026)

    def __len__(self) -> int:
        return len(self._x)

    def add_batch(
        self, xs: torch.Tensor, ys: torch.Tensor,
    ) -> None:
        """Append every example in a ``(B, L)`` batch."""
        B = xs.shape[0]
        for i in range(B):
            self._x.append(xs[i].detach().cpu())
            self._y.append(ys[i].detach().cpu())
        while len(self._x) > self.capacity:
            self._x.pop(0)
            self._y.pop(0)

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``n`` random examples (with replacement)."""
        if len(self._x) == 0:
            raise RuntimeError(
                "PretrainReplayBuffer is empty; call add_batch "
                "during pretraining."
            )
        idx = torch.randint(
            0, len(self._x), (n,), generator=self._rng,
        ).tolist()
        xs = torch.stack([self._x[i] for i in idx]).to(self.device)
        ys = torch.stack([self._y[i] for i in idx]).to(self.device)
        return xs, ys


# ─────────────────────────────────────────────────────────────────
# OnlineTeacherSession
# ─────────────────────────────────────────────────────────────────


class OnlineTeacherSession:
    """Interactive teacher → model loop with three update
    mechanisms (M1 fast / M2 sleep / M3 slow).

    The model is assumed to be **pretrained** on regular text and
    its ``tok_emb`` to have ``vocab_size`` rows. ``novel_concept_ids``
    are the vocab IDs that were *reserved* during vocab build and
    never seen during pretraining. Only these rows of ``tok_emb``
    are touched by the M3 selective-gradient step.

    Args:
        model: a :class:`HybridPCMMiniLM` / :class:`TitansPCMMiniLM`
            instance. Other PCM LMs work too if they expose
            ``tok_emb``, ``hidden_states``, and forward.
        novel_concept_ids: ids reserved for online learning.
        novel_concept_classes: dict mapping id → ground-truth class
            (e.g., ``"ANIMAL"`` / ``"FOOD"``) for O6 cluster purity.
        replay: a populated :class:`PretrainReplayBuffer`.
        pad_id: the padding token (default 0).
        m3_k_grad_threshold: minimum correction count for that
            concept before M3 fires (default 3).
        m3_salience_threshold: surprise level above which M3 fires
            even on the first correction (default 10.0 nats).
        m3_n_replay: number of replay examples per grad step
            (default 7).
        m3_lr: AdamW lr for M3 (default 1e-3).
        m1_buffer_capacity: capacity of the internal F75 episodic
            buffer (default 512).
        device: ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self, model: nn.Module, novel_concept_ids: Iterable[int],
        novel_concept_classes: dict[int, str],
        replay: PretrainReplayBuffer, *,
        pad_id: int = 0,
        m3_k_grad_threshold: int = 1,
        m3_salience_threshold: float = 10.0,
        m3_n_replay: int = 7,
        m3_lr: float = 5e-3,
        m3_inner_steps: int = 3,
        m1_buffer_capacity: int = 512,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.novel_ids = list(novel_concept_ids)
        self.novel_ids_set = set(self.novel_ids)
        self.concept_classes = dict(novel_concept_classes)
        self.replay = replay
        self.pad_id = pad_id
        self.m3_k_grad_threshold = m3_k_grad_threshold
        self.m3_salience_threshold = m3_salience_threshold
        self.m3_n_replay = m3_n_replay
        self.m3_inner_steps = m3_inner_steps
        self.device = device
        # Episodic buffer (F75) — stores hidden states + metadata
        d_model = model.d_model
        self.episodic = EpisodicBuffer(
            capacity=m1_buffer_capacity,
            slot_dim=d_model, device=device,
        )
        # AdamW optimiser on a small set of parameters
        params = [model.tok_emb.weight]
        if hasattr(model, "ln_final"):
            params.extend(list(model.ln_final.parameters()))
        self.opt = torch.optim.AdamW(
            params, lr=m3_lr, weight_decay=0.0,
        )
        self.counters = MechanismCounters()
        self.counters.correction_counts = {
            int(cid): 0 for cid in self.novel_ids
        }
        self.counters.grad_step_counts = {
            int(cid): 0 for cid in self.novel_ids
        }
        self.t = 0

    # ── M1: episodic write (always fires) ───────────────────────

    @torch.no_grad()
    def _m1_imprint(
        self, context_ids: torch.Tensor,
        target_ids: torch.Tensor,
        concept_id: int, surprise: float,
    ) -> None:
        """Fast: write hidden state + metadata to episodic
        buffer, and (if Titans) to the model's memory readout."""
        self.model.eval()
        if context_ids.dim() == 1:
            context_ids = context_ids.unsqueeze(0)
        hidden = self.model.hidden_states(context_ids.to(self.device))
        last_slot = hidden[0, -1]
        self.episodic.append(
            slot=last_slot,
            timestamp=self.t,
            salience=float(surprise),
            metadata={
                "concept_id": int(concept_id),
                "context_ids": context_ids[0].cpu().tolist(),
                "target_ids": target_ids.cpu().tolist(),
            },
        )
        self.counters.n_m1_imprints += 1
        if hasattr(self.model, "memory_readout"):
            self.model.memory_readout.imprint(
                hidden, n_per_batch=1,
            )
            self.counters.n_m1_titans_imprints += 1

    # ── M3: micro-gradient (conditional) ────────────────────────

    def _m3_micro_gradient_step(
        self, correction_x: torch.Tensor,
        correction_y: torch.Tensor,
    ) -> float:
        """Slow: ``m3_inner_steps`` AdamW steps on (correction +
        replay) with gradient masked to novel-concept embedding
        rows only. Returns the post-step training loss.
        """
        last_loss = 0.0
        for _ in range(self.m3_inner_steps):
            last_loss = self._m3_single_step(
                correction_x, correction_y,
            )
        return last_loss

    def _m3_single_step(
        self, correction_x: torch.Tensor,
        correction_y: torch.Tensor,
    ) -> float:
        """One AdamW step."""
        self.model.train()
        # Build replay batch
        replay_x, replay_y = self.replay.sample(self.m3_n_replay)
        # Match seq lengths via padding to the longest
        cx = correction_x.to(self.device)
        cy = correction_y.to(self.device)
        if cx.dim() == 1:
            cx = cx.unsqueeze(0)
            cy = cy.unsqueeze(0)
        target_len = max(cx.shape[1], replay_x.shape[1])

        def _pad_right(t: torch.Tensor, length: int) -> torch.Tensor:
            B, L = t.shape
            if L == length:
                return t
            pad = torch.full(
                (B, length - L), fill_value=self.pad_id,
                dtype=t.dtype, device=t.device,
            )
            return torch.cat([t, pad], dim=1)

        cx_p = _pad_right(cx, target_len)
        cy_p = _pad_right(cy, target_len)
        rx_p = _pad_right(replay_x, target_len)
        ry_p = _pad_right(replay_y, target_len)
        xs = torch.cat([cx_p, rx_p], dim=0)
        ys = torch.cat([cy_p, ry_p], dim=0)

        # If the model is a TitansPCMMiniLM we don't want imprint
        # to fire here (that's M1's job)
        if hasattr(self.model, "memory_readout"):
            logits = self.model(
                xs, use_memory=True, imprint=False,
            )
        else:
            logits = self.model(xs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            ys.reshape(-1), ignore_index=self.pad_id,
        )
        self.opt.zero_grad()
        loss.backward()
        # Mask tok_emb gradient: keep only novel-concept rows.
        with torch.no_grad():
            grad = self.model.tok_emb.weight.grad
            if grad is not None:
                mask = torch.zeros_like(grad)
                for cid in self.novel_ids:
                    mask[cid] = 1.0
                grad.mul_(mask)
        torch.nn.utils.clip_grad_norm_(
            [self.model.tok_emb.weight], 1.0,
        )
        self.opt.step()
        return float(loss.item())

    # ── M2: sleep consolidation (periodic diagnostic) ──────────

    def sleep(self) -> dict:
        """Mid: run K-means over the episodic buffer; report
        cluster purity against {ANIMAL, FOOD}. This is a *purely
        diagnostic* step at v9.0 — does NOT mutate parameters.
        """
        self.counters.n_sleeps += 1
        if len(self.episodic) == 0:
            return {"status": "empty"}
        # Unique classes seen so far
        recs = self.episodic.records()
        n_classes = max(
            2,
            len({
                self.concept_classes.get(
                    r.metadata["concept_id"], "?",
                )
                for r in recs
            }),
        )
        cgraph = consolidate_to_concept_graph(
            self.episodic,
            n_clusters=n_classes, n_iter=50,
            n_restarts=5, rng_seed=2026,
            device=self.device,
        )
        labels = cgraph["assignments"].tolist()
        # Compute cluster purity against ground-truth classes
        cluster_tokens: dict[int, list[str]] = {}
        for i, r in enumerate(recs):
            c = int(labels[i])
            cls = self.concept_classes.get(
                r.metadata["concept_id"], "?",
            )
            cluster_tokens.setdefault(c, []).append(cls)
        purities = []
        for c, classes in cluster_tokens.items():
            if not classes:
                continue
            counts: dict[str, int] = {}
            for cls in classes:
                counts[cls] = counts.get(cls, 0) + 1
            purities.append(
                max(counts.values()) / len(classes)
            )
        return {
            "status": "ok",
            "n_episodes": len(recs),
            "n_clusters": n_classes,
            "purity_mean": (
                float(sum(purities) / len(purities))
                if purities else 0.0
            ),
            "purity_per_cluster": [
                float(p) for p in purities
            ],
        }

    # ── Main entrypoint ─────────────────────────────────────────

    def receive_correction(
        self, context_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> dict:
        """One teacher-correction event.

        Args:
            context_ids: the prompt prefix the model saw (1-D
                LongTensor of token ids).
            target_ids: the target/correct full sequence
                including the novel concept (1-D LongTensor).

        Returns a diagnostic dict.
        """
        self.t += 1
        self.counters.n_corrections += 1
        # Identify which novel concept this correction is about
        novel_in_target = [
            int(t.item()) for t in target_ids
            if int(t.item()) in self.novel_ids_set
        ]
        if not novel_in_target:
            return {
                "status": "no_novel_concept",
                "skipped": True,
            }
        concept_id = novel_in_target[0]
        self.counters.correction_counts[concept_id] = (
            self.counters.correction_counts.get(concept_id, 0) + 1
        )
        # Compute surprise on the full target sequence (teacher
        # forcing: the model sees the target as input and we
        # measure loss on the shifted target)
        with torch.no_grad():
            t_in = target_ids[:-1].to(self.device).unsqueeze(0)
            t_out = target_ids[1:].to(self.device).unsqueeze(0)
            if hasattr(self.model, "memory_readout"):
                self.model.eval()
                logits = self.model(
                    t_in, use_memory=True, imprint=False,
                )
            else:
                self.model.eval()
                logits = self.model(t_in)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                t_out.reshape(-1), ignore_index=self.pad_id,
                reduction="mean",
            )
            surprise = float(loss.item())
        # M1 — always
        self._m1_imprint(
            context_ids=t_in[0], target_ids=t_out[0],
            concept_id=concept_id, surprise=surprise,
        )
        # M3 — conditional
        n_corr = self.counters.correction_counts[concept_id]
        do_grad = (
            n_corr >= self.m3_k_grad_threshold
            or surprise > self.m3_salience_threshold
        )
        grad_loss: float | None = None
        if do_grad:
            grad_loss = self._m3_micro_gradient_step(
                correction_x=t_in, correction_y=t_out,
            )
            self.counters.n_m3_grad_steps += 1
            self.counters.grad_step_counts[concept_id] = (
                self.counters.grad_step_counts.get(concept_id, 0)
                + 1
            )
        return {
            "status": "applied",
            "concept_id": concept_id,
            "n_corrections_for_concept": n_corr,
            "surprise": surprise,
            "m1_fired": True,
            "m3_fired": do_grad,
            "m3_loss_after": grad_loss,
        }

    # ── F95: free-form chat correction ─────────────────────────

    def _m3_single_step_chat(
        self, correction_x: torch.Tensor,
        correction_y: torch.Tensor,
    ) -> float:
        """Variant of :meth:`_m3_single_step` for free-form chat.

        Differences from the F85 single step:

        * Gradient is masked to the *union* of ``tok_emb`` rows
          that appear in either the context or the target —
          i.e., we only adjust embeddings of words actually
          relevant to this turn, not every row in the vocabulary.
          This is a much weaker constraint than F85's
          ``novel_ids`` mask but still much stronger than
          unmasked update.
        * Replay is used as in F85 to anchor the regular
          distribution.
        """
        self.model.train()
        replay_x, replay_y = self.replay.sample(self.m3_n_replay)
        cx = correction_x.to(self.device)
        cy = correction_y.to(self.device)
        if cx.dim() == 1:
            cx = cx.unsqueeze(0)
            cy = cy.unsqueeze(0)
        target_len = max(cx.shape[1], replay_x.shape[1])

        def _pad_right(t: torch.Tensor, length: int) -> torch.Tensor:
            B, L = t.shape
            if L == length:
                return t
            pad = torch.full(
                (B, length - L), fill_value=self.pad_id,
                dtype=t.dtype, device=t.device,
            )
            return torch.cat([t, pad], dim=1)

        cx_p = _pad_right(cx, target_len)
        cy_p = _pad_right(cy, target_len)
        rx_p = _pad_right(replay_x, target_len)
        ry_p = _pad_right(replay_y, target_len)
        xs = torch.cat([cx_p, rx_p], dim=0)
        ys = torch.cat([cy_p, ry_p], dim=0)
        if hasattr(self.model, "memory_readout"):
            logits = self.model(
                xs, use_memory=True, imprint=False,
            )
        else:
            logits = self.model(xs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            ys.reshape(-1), ignore_index=self.pad_id,
        )
        self.opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            grad = self.model.tok_emb.weight.grad
            if grad is not None:
                mask = torch.zeros_like(grad)
                turn_ids = set(cx[0].cpu().tolist()) | set(
                    cy[0].cpu().tolist()
                )
                turn_ids.discard(self.pad_id)
                for tid in turn_ids:
                    if 0 <= tid < mask.shape[0]:
                        mask[tid] = 1.0
                grad.mul_(mask)
        torch.nn.utils.clip_grad_norm_(
            [self.model.tok_emb.weight], 1.0,
        )
        self.opt.step()
        return float(loss.item())

    def receive_chat_turn(
        self, context_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *, run_grad: bool = True,
        salience_threshold: float | None = None,
    ) -> dict:
        """F95 free-form chat correction.

        Unlike :meth:`receive_correction`, no
        ``novel_concept_id`` is required. The user's response
        becomes a target completion that the model is nudged
        toward — *gently*, with replay anchoring the
        pretraining distribution.

        Args:
            context_ids: 1-D ``LongTensor`` of token ids the
                agent saw before generating its response.
            target_ids: 1-D ``LongTensor`` of the user's
                actual response (the "correction").
            run_grad: if ``False``, only the M1 episodic write
                + surprise measurement happen; no
                gradient step.
            salience_threshold: optional override for the
                session's :attr:`m3_salience_threshold`; the
                grad step runs iff
                ``surprise > salience_threshold`` (default
                = session value).

        Returns a diagnostic dict.
        """
        self.t += 1
        self.counters.n_corrections += 1
        if context_ids.numel() == 0 or target_ids.numel() == 0:
            return {
                "status": "empty_turn",
                "skipped": True,
            }
        thresh = (
            salience_threshold
            if salience_threshold is not None
            else self.m3_salience_threshold
        )

        # Teacher-forcing surprise = cross-entropy of the
        # user response given a (context, target[:-1]) prefix
        # → predicting target. We pad target_ids on the left
        # by context_ids so the model has full conditioning.
        with torch.no_grad():
            ctx = context_ids.to(self.device)
            tgt = target_ids.to(self.device)
            if ctx.dim() == 1:
                ctx = ctx.unsqueeze(0)
            if tgt.dim() == 1:
                tgt = tgt.unsqueeze(0)
            full = torch.cat([ctx, tgt], dim=1)
            t_in = full[:, :-1]
            t_out = full[:, 1:]
            self.model.eval()
            if hasattr(self.model, "memory_readout"):
                logits = self.model(
                    t_in, use_memory=True, imprint=False,
                )
            else:
                logits = self.model(t_in)
            mask = torch.zeros_like(t_out, dtype=torch.bool)
            mask[:, ctx.shape[1] - 1:] = True
            flat_logits = logits.reshape(
                -1, logits.shape[-1],
            )
            flat_targets = t_out.reshape(-1)
            flat_mask = mask.reshape(-1)
            if flat_mask.sum() == 0:
                surprise = 0.0
            else:
                loss = F.cross_entropy(
                    flat_logits[flat_mask],
                    flat_targets[flat_mask],
                    ignore_index=self.pad_id,
                    reduction="mean",
                )
                surprise = float(loss.item())

        # M1 episodic write — every chat turn writes.
        self._m1_imprint(
            context_ids=t_in[0], target_ids=t_out[0],
            concept_id=-1,  # no novel-concept tag in chat
            surprise=surprise,
        )

        # M3 conditional on surprise (no concept-count gate
        # for chat — every novel-distribution turn triggers).
        do_grad = run_grad and surprise > thresh
        grad_loss: float | None = None
        if do_grad and len(self.replay) > 0:
            x_in = full[:, :-1]
            y_in = full[:, 1:]
            for _ in range(self.m3_inner_steps):
                grad_loss = self._m3_single_step_chat(
                    x_in, y_in,
                )
            self.counters.n_m3_grad_steps += 1

        return {
            "status": "applied",
            "surprise": surprise,
            "m1_fired": True,
            "m3_fired": do_grad,
            "m3_loss_after": grad_loss,
        }

    # ── Evaluation helpers ─────────────────────────────────────

    @torch.no_grad()
    def eval_ppl_on_sequences(
        self, sequences: list[torch.Tensor],
        *, use_memory: bool = True,
    ) -> float:
        """PPL over a list of token sequences (each 1-D
        LongTensor). Uses teacher forcing on shifted targets;
        ignores PAD."""
        self.model.eval()
        losses: list[float] = []
        for seq in sequences:
            seq = seq.to(self.device)
            if seq.dim() == 1:
                seq = seq.unsqueeze(0)
            xs = seq[:, :-1]
            ys = seq[:, 1:]
            if xs.shape[1] == 0:
                continue
            if hasattr(self.model, "memory_readout"):
                logits = self.model(
                    xs, use_memory=use_memory, imprint=False,
                )
            else:
                logits = self.model(xs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                ys.reshape(-1),
                ignore_index=self.pad_id,
                reduction="mean",
            )
            losses.append(float(loss.item()))
        if not losses:
            return float("nan")
        return float(math.exp(sum(losses) / len(losses)))
