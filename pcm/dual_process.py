"""pcm.dual_process — PCM v3 dual-process number architecture.

Implements the System-1 (RPE retrieval) / System-2 (procedural
sequencing) split documented in
``docs/PCM_V3_DUAL_PROCESS_DESIGN.md``.

Public API:

* :class:`SuccessorHead` — single-input head that predicts a small
  signed step from a slot row. Trained only on adjacent pairs
  (|Δ| = 1) but can be applied iteratively for arbitrary |Δ|.
* :class:`IterativeDiffCook` — pure-function PCM cook that applies
  ``SuccessorHead`` repeatedly until reaching a target, returning
  step count + diagnostics.
* :func:`route_diff` — System-1 / System-2 dispatcher.

The module is opt-in: importing it has no effect on existing v1 or
v2 callers. Use it when:

* the task answer depends on a **displacement** between two
  concepts (number difference, hue rotation, spatial step), AND
* the test distribution contains displacements **outside** the
  range the v2 :class:`pcm.dual_channel.RelativePositionEmbedding`
  was trained on, AND
* the displacement decomposes into a composition of unit steps
  (i.e. the underlying space is locally connected).

For non-pair tasks or tasks where the answer is not displacement-
shaped (e.g. classification of cell *identity*), v2 heads remain
the right tool.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "SuccessorHead",
    "IterativeDiffCook",
    "DiffCookReport",
    "route_diff",
]


# ---------------------------------------------------------------------------
# SuccessorHead — System-2's atomic step predictor.
# ---------------------------------------------------------------------------


class SuccessorHead(nn.Module):
    """Predicts a small signed step toward a target from a (current,
    target) pair of slot rows.

    Critically, the head is **conditional** on the target — without
    target information the model has no way to know which direction
    is "forward". This matches the dual-process literature: the
    procedural System-2 sequence iteratively asks "where am I now,
    where do I want to be" at every step, not "what comes next" in
    isolation.

    Trained on pairs ``(a, b)`` with target sign ``sign(b - a)`` ∈
    ``{-1, 0, +1}`` (default ``max_step = 1``). During training the
    head only ever sees |Δ| ≤ ``max_step`` — the easiest possible
    arithmetic supervision. At inference time
    :class:`IterativeDiffCook` applies the head repeatedly to bridge
    arbitrary |Δ|, and the head only needs to predict the **sign**
    of the remaining gap correctly. Magnitude generalisation comes
    for free from the iteration count.

    Args:
        slot_dim: dim of the slot facet rows (matches v2 dual-channel
            ``slot_dim``).
        max_step: largest absolute step the head can predict per
            call. Default 1: 3-class output ``{-1, 0, +1}``. Set to
            2 for 5-class ``{-2, -1, 0, +1, +2}`` if you want fewer
            iterations on large gaps (at the cost of needing |Δ| ≤
            2 supervision instead of |Δ| = 1).
        hidden: MLP hidden width.
    """

    def __init__(
        self,
        slot_dim: int,
        *,
        max_step: int = 1,
        hidden: int = 64,
        attr_dim: int = 0,
    ) -> None:
        super().__init__()
        if max_step < 1:
            raise ValueError(f"max_step must be >= 1, got {max_step}")
        self.slot_dim = int(slot_dim)
        self.attr_dim = int(attr_dim)
        self.max_step = int(max_step)
        self.n_classes = 2 * self.max_step + 1
        # Concatenated (current, target) slot rows + optional
        # (current, target) attr rows. The attr channel is the
        # one trained by ``pcm.dual_channel.successor_consistency_loss``
        # to be monotone — when present, sign(b - a) is linearly
        # separable from (attr_a, attr_b), unlocking E1 ≥ 0.99
        # even on a small training budget.
        in_dim = 2 * slot_dim + 2 * self.attr_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, self.n_classes)

    def forward(
        self,
        slot_a: torch.Tensor, slot_b: torch.Tensor,
        attr_a: torch.Tensor | None = None,
        attr_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns ``(B, 2*max_step+1)`` logits over signed step
        classes given ``(current, target)`` slot rows and (optionally)
        attr rows."""
        parts = [slot_a, slot_b]
        if self.attr_dim > 0:
            if attr_a is None or attr_b is None:
                raise ValueError(
                    "SuccessorHead with attr_dim>0 requires attr_a, attr_b"
                )
            parts += [attr_a, attr_b]
        x = torch.cat(parts, dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def predict_step(
        self,
        slot_a: torch.Tensor, slot_b: torch.Tensor,
        attr_a: torch.Tensor | None = None,
        attr_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns ``(B,)`` integer predicted steps in
        ``[-max_step, max_step]``."""
        cls = self.forward(slot_a, slot_b, attr_a, attr_b).argmax(dim=-1)
        return cls - self.max_step


# ---------------------------------------------------------------------------
# IterativeDiffCook — System-2's procedural sequencing.
# ---------------------------------------------------------------------------


@dataclass
class DiffCookReport:
    """Diagnostics returned by :class:`IterativeDiffCook`. Useful for
    E5 (RT-by-Δ scaling) and debugging stuck iterations."""

    n_iters: int
    converged: bool
    final_distance: int
    wall_seconds: float
    step_history: list[int] = field(default_factory=list)


class IterativeDiffCook:
    """Pure-function PCM cook: repeatedly apply :class:`SuccessorHead`
    starting from ``a`` until reaching ``b``; return step count.

    The body is a Python loop with a hard iteration cap. The
    successor head's output is interpreted as a signed step at each
    iteration; the loop terminates when (a) the running cursor
    matches the target, (b) the predicted step is 0, or (c) the
    cap is reached.

    The output is a single tensor giving the total displacement
    estimate ``(b - a)``. By construction, ``|output| ≤ n_iters
    * max_step``.

    Args:
        successor_head: trained :class:`SuccessorHead`.
        max_iters: hard upper bound on iteration count. Default 200
            covers |Δ| ≤ 200 / max_step; raise for longer chains.
        identity_lookup: callable mapping integer cursor → slot row.
            For 1-d ordinal domains, this is typically a closure
            over ``cg.bundle_pool[slot_facet]`` indexed by the
            cursor value.

    Example:

    .. code-block:: python

        def lookup(idx: int) -> torch.Tensor:
            cid = f"concept:num:{idx}"
            slot, _ = collapse_dual_channel(
                cg, caller="cook", base_facet="arith",
                concept_ids=[cid],
                slot_shape=(SLOT_DIM,), attr_shape=(ATTR_DIM,),
            )
            return slot[0]

        cook = IterativeDiffCook(succ_head, identity_lookup=lookup)
        diff_pred, report = cook(start_idx=3, target_idx=27)
        assert report.converged
        assert diff_pred == 24
    """

    def __init__(
        self,
        successor_head: SuccessorHead,
        identity_lookup: Callable[[int], torch.Tensor]
        | Callable[[int], tuple[torch.Tensor, torch.Tensor]],
        *,
        max_iters: int = 200,
        cursor_min: int | None = None,
        cursor_max: int | None = None,
        with_attr: bool = False,
    ) -> None:
        self.successor_head = successor_head
        self.identity_lookup = identity_lookup
        self.max_iters = int(max_iters)
        self.cursor_min = cursor_min
        self.cursor_max = cursor_max
        # When True, identity_lookup is expected to return
        # (slot, attr) tuple; the cook routes both into the
        # successor head.
        self.with_attr = with_attr

    def __call__(
        self,
        start_idx: int,
        target_idx: int,
    ) -> tuple[int, DiffCookReport]:
        """Iteratively step from ``start_idx`` toward ``target_idx``;
        return ``(estimated_diff, report)``.

        ``estimated_diff`` is the *predicted* ``target - start``,
        accumulated by summing per-step predictions. It is independent
        of whether the cursor reaches the target — a stuck iteration
        will report ``converged = False`` but still return a
        diff estimate based on the steps it managed to take.
        """
        t0 = time.time()
        cursor = int(start_idx)
        target = int(target_idx)
        accumulated = 0
        step_history: list[int] = []
        converged = False

        with torch.no_grad():
            if self.with_attr:
                tslot, tattr = self.identity_lookup(target)
                target_slot = tslot.unsqueeze(0)
                target_attr = tattr.unsqueeze(0)
            else:
                target_slot = self.identity_lookup(target).unsqueeze(0)
                target_attr = None
            for _ in range(self.max_iters):
                if cursor == target:
                    converged = True
                    break

                if self.with_attr:
                    cslot, cattr = self.identity_lookup(cursor)
                    cur_slot = cslot.unsqueeze(0)
                    cur_attr = cattr.unsqueeze(0)
                    step = int(
                        self.successor_head.predict_step(
                            cur_slot, target_slot, cur_attr, target_attr,
                        ).item()
                    )
                else:
                    cur_slot = self.identity_lookup(cursor).unsqueeze(0)
                    step = int(
                        self.successor_head.predict_step(
                            cur_slot, target_slot,
                        ).item()
                    )
                step_history.append(step)
                if step == 0:
                    # Head says "stay put" but we haven't reached target;
                    # break to avoid an infinite loop. Caller decides
                    # what to do with non-converged reports.
                    break

                cursor += step
                accumulated += step

                if (
                    self.cursor_min is not None and cursor < self.cursor_min
                ) or (
                    self.cursor_max is not None and cursor > self.cursor_max
                ):
                    break

        report = DiffCookReport(
            n_iters=len(step_history),
            converged=converged,
            final_distance=target - cursor,
            wall_seconds=time.time() - t0,
            step_history=step_history,
        )
        return accumulated, report


# ---------------------------------------------------------------------------
# route_diff — System-1 / System-2 dispatcher.
# ---------------------------------------------------------------------------


def route_diff(
    a_idx: int,
    b_idx: int,
    *,
    rpe_predict: Callable[[int, int], int] | None,
    cook: IterativeDiffCook | None,
    train_max_abs_delta: int,
    coarse_delta: int | None = None,
) -> tuple[int, str]:
    """Dispatch a displacement query to System 1 (RPE retrieval) or
    System 2 (cook procedural sequencing).

    Args:
        a_idx, b_idx: integer endpoints of the displacement query.
        rpe_predict: callable ``(a, b) → predicted_diff`` running the
            v2 RPE head. Pass ``None`` to disable System 1 (forces
            cook).
        cook: :class:`IterativeDiffCook` instance. Pass ``None`` to
            disable System 2 (forces RPE; will fail OOD).
        train_max_abs_delta: largest ``|Δ|`` the RPE table was
            trained on. Cheaper than infering this at runtime.
        coarse_delta: pre-computed ``b_idx - a_idx`` if available;
            saves an inference call in the routing decision.

    Returns ``(predicted_diff, route_taken)`` where
    ``route_taken ∈ {"rpe", "cook"}``.

    Routing rule (deliberately simple, falsifiability E5):
        * if the rough estimate ``|Δ̃| ≤ train_max_abs_delta`` →
          System 1 (RPE);
        * else → System 2 (cook).

    The rough estimate is computed from ``coarse_delta`` if supplied,
    or from ``b_idx - a_idx`` directly. In a fully-emergent setting
    (no oracle access to the integer indices) the rough estimate
    would come from a learned ``sign + magnitude`` head; we leave
    that as a v3 follow-up (E5 §9.2).
    """
    if coarse_delta is None:
        coarse_delta = b_idx - a_idx

    in_range = abs(coarse_delta) <= train_max_abs_delta

    if in_range and rpe_predict is not None:
        return int(rpe_predict(a_idx, b_idx)), "rpe"
    if cook is not None:
        diff, _ = cook(a_idx, b_idx)
        return int(diff), "cook"
    if rpe_predict is not None:
        # Cook missing, fall back to RPE even out of range.
        return int(rpe_predict(a_idx, b_idx)), "rpe"
    raise ValueError("route_diff requires at least one of rpe_predict / cook")
