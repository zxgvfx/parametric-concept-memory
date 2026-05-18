"""pcm.coref — F76 pronoun resolution via hypothesis-and-verify.

Combines F75 episodic buffer (candidate set) + F74 LM (verifier).

Algorithm::

    resolve_pronoun(tokens, pronoun_pos, lm, tokenizer, buffer):
        1. Filter buffer.recall_most_recent() by pronoun class
           + gender compatibility → candidate list.
        2. For each candidate c:
              substituted = tokens[:pronoun_pos] + [c]
                          + tokens[pronoun_pos+1:]
              score_c = -LM_perplexity(substituted)
        3. Return argmax_c score_c.

This implements the user's "假设验证然后确定到底是指的是哪个"
mechanism: each candidate is a *hypothesis*, the LM score
verifies which substitution best fits the context.

Two baselines for comparison:

* :func:`resolve_pronoun_recency` — return the most-recent
  compatible candidate. This is the heuristic most coreference
  systems start with; it's strong because subjects in the prior
  sentence tend to be referenced again, but it fails when the
  ground-truth referent is *not* the most recent compatible
  entity (e.g. when pragmatics or syntactic structure overrides
  recency).
* :func:`resolve_pronoun_class_only` — return the first
  compatible candidate (ignores LM and recency). Pure
  class-compatibility baseline.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .episodic import EpisodicBuffer
from .lm_synthetic import (
    PRONOUN_CLASSES,
    Tokenizer,
    is_compatible_referent,
    noun_class_of,
)


__all__ = [
    "ResolutionResult",
    "candidates_from_buffer",
    "candidates_from_token_history",
    "resolve_pronoun",
    "resolve_pronoun_recency",
    "resolve_pronoun_class_only",
]


class ResolutionResult:
    """Trace of one pronoun-resolution decision."""

    def __init__(
        self, pronoun: str, predicted_referent: str | None,
        candidates: list[str], scores: dict[str, float],
        route: str,
    ) -> None:
        self.pronoun = pronoun
        self.predicted_referent = predicted_referent
        self.candidates = candidates
        self.scores = scores
        self.route = route

    def __repr__(self) -> str:
        return (
            f"ResolutionResult(pronoun={self.pronoun!r}, "
            f"pred={self.predicted_referent!r}, "
            f"candidates={self.candidates}, route={self.route!r})"
        )


# ─────────────────────────────────────────────────────────────────
# Candidate collection
# ─────────────────────────────────────────────────────────────────


def candidates_from_buffer(
    buffer: EpisodicBuffer, pronoun: str,
    *, max_lookback: int = 10,
) -> list[tuple[str, int]]:
    """Return ``[(noun, timestamp), …]`` from the last
    ``max_lookback`` episodes, filtered by pronoun class/gender
    compatibility (using ``metadata['entity']``)."""
    candidates: list[tuple[str, int]] = []
    seen_nouns: set[str] = set()
    records = buffer.recall_most_recent(k=max_lookback)
    # Iterate newest-first so the same noun (if it appears
    # multiple times) is recorded at its most-recent timestamp.
    for r in reversed(records):
        entity = r.metadata.get("entity")
        if entity is None or entity in seen_nouns:
            continue
        if is_compatible_referent(pronoun, entity):
            candidates.append((entity, r.timestamp))
            seen_nouns.add(entity)
    return candidates


def candidates_from_token_history(
    tokens: list[str], pronoun_position: int, pronoun: str,
) -> list[tuple[str, int]]:
    """Fallback when no buffer is available: scan ``tokens[
    :pronoun_position]`` for compatible nouns. Returns
    ``[(noun, position), …]`` ordered by position (oldest first)."""
    out: list[tuple[str, int]] = []
    seen: set[str] = set()
    for i, t in enumerate(tokens[:pronoun_position]):
        if t in seen:
            continue
        if is_compatible_referent(pronoun, t):
            out.append((t, i))
            seen.add(t)
    return out


# ─────────────────────────────────────────────────────────────────
# Hypothesis-verify resolver
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def resolve_pronoun(
    tokens: list[str], pronoun_position: int,
    lm, tokenizer: Tokenizer,
    *,
    buffer: EpisodicBuffer | None = None,
    candidates: list[tuple[str, int]] | None = None,
    max_len: int = 16,
    device: str = "cpu",
    scoring: str = "target_position",
) -> ResolutionResult:
    """Hypothesis-verify: for each candidate, score the
    substitution under the LM, return the highest-scoring
    candidate.

    Two scoring modes:

    * ``"target_position"`` (default, the discriminative one):
      score ``log P(candidate | context_before_pronoun)``. This
      isolates the **prediction at the pronoun position** — the
      *only* token that genuinely differs between substitutions.
      Cleaner signal; matches the user's "假设验证" intuition
      more closely (the LM is asked "which entity comes next?",
      not "which whole sentence is more likely?").

    * ``"full_sentence"``: score by perplexity of the entire
      substituted sequence. Diluted because many tokens are
      identical between substitutions.

    Args:
        tokens: full discourse token list.
        pronoun_position: index of the pronoun in ``tokens``.
        lm: a trained LM with ``__call__(x)`` returning logits.
        tokenizer: the LM's tokenizer.
        buffer: episodic buffer for candidate generation (preferred).
        candidates: explicit candidate list — if provided,
            overrides ``buffer``-based extraction.
        max_len: padding length for tokenisation.
        device: torch device.
        scoring: ``"target_position"`` | ``"full_sentence"``.
    """
    pronoun = tokens[pronoun_position]
    if candidates is None:
        if buffer is not None:
            candidates = candidates_from_buffer(buffer, pronoun)
        else:
            candidates = candidates_from_token_history(
                tokens, pronoun_position, pronoun,
            )
    if not candidates:
        return ResolutionResult(
            pronoun=pronoun, predicted_referent=None,
            candidates=[], scores={}, route="no_candidate",
        )
    lm.eval()
    scores: dict[str, float] = {}
    if scoring == "target_position":
        # All substitutions share tokens[:pronoun_position], so we
        # encode that prefix once, take logits at position
        # pronoun_position - 1 (next-token-prediction at that
        # step), and gather each candidate's logit.
        prefix_tokens = tokens[:pronoun_position]
        if not prefix_tokens:
            # No context before pronoun — fall back to recency.
            best, _ = max(candidates, key=lambda kv: kv[1])
            return ResolutionResult(
                pronoun=pronoun, predicted_referent=best,
                candidates=[c for c, _ in candidates],
                scores={c: 0.0 for c, _ in candidates},
                route="hypothesis_verify_no_context",
            )
        prefix_ids = tokenizer.encode(
            prefix_tokens, max_len=max_len, add_eos=False,
        )
        x = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        logits = lm(x)
        # logits[0, t, v] = P(token at position t+1 = v | first t+1 tokens)
        # We want P(token at index pronoun_position = candidate),
        # which is predicted by logits[0, pronoun_position - 1].
        target_logits = logits[0, pronoun_position - 1, :]
        log_probs = F.log_softmax(target_logits, dim=-1)
        for cand, _ in candidates:
            cid = tokenizer.stoi.get(cand, tokenizer.unk_id)
            scores[cand] = float(log_probs[cid].item())
    elif scoring == "full_sentence":
        for cand, _ in candidates:
            substituted = list(tokens)
            substituted[pronoun_position] = cand
            ids = tokenizer.encode(substituted, max_len=max_len)
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
            logits = lm(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                ignore_index=tokenizer.pad_id,
                reduction="sum",
            )
            n = int((y != tokenizer.pad_id).sum().item())
            ppl = math.exp(float(loss.item()) / max(n, 1))
            scores[cand] = -ppl  # higher = better
    else:
        raise ValueError(f"unknown scoring mode {scoring!r}")
    best = max(scores, key=scores.get)
    return ResolutionResult(
        pronoun=pronoun, predicted_referent=best,
        candidates=[c for c, _ in candidates],
        scores=scores, route=f"hypothesis_verify_{scoring}",
    )


# ─────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────


def resolve_pronoun_recency(
    tokens: list[str], pronoun_position: int,
    *,
    buffer: EpisodicBuffer | None = None,
    candidates: list[tuple[str, int]] | None = None,
) -> ResolutionResult:
    """Most-recent compatible candidate wins. Standard
    coreference baseline."""
    pronoun = tokens[pronoun_position]
    if candidates is None:
        if buffer is not None:
            candidates = candidates_from_buffer(buffer, pronoun)
        else:
            candidates = candidates_from_token_history(
                tokens, pronoun_position, pronoun,
            )
    if not candidates:
        return ResolutionResult(
            pronoun=pronoun, predicted_referent=None,
            candidates=[], scores={}, route="no_candidate",
        )
    # Pick max-timestamp candidate
    best, _ = max(candidates, key=lambda kv: kv[1])
    return ResolutionResult(
        pronoun=pronoun, predicted_referent=best,
        candidates=[c for c, _ in candidates],
        scores={c: float(t) for c, t in candidates},
        route="recency",
    )


def resolve_pronoun_class_only(
    tokens: list[str], pronoun_position: int,
    *,
    buffer: EpisodicBuffer | None = None,
    candidates: list[tuple[str, int]] | None = None,
) -> ResolutionResult:
    """Return the first class-compatible candidate (deterministic
    but uninformed)."""
    pronoun = tokens[pronoun_position]
    if candidates is None:
        if buffer is not None:
            candidates = candidates_from_buffer(buffer, pronoun)
        else:
            candidates = candidates_from_token_history(
                tokens, pronoun_position, pronoun,
            )
    if not candidates:
        return ResolutionResult(
            pronoun=pronoun, predicted_referent=None,
            candidates=[], scores={}, route="no_candidate",
        )
    best, _ = candidates[0]
    return ResolutionResult(
        pronoun=pronoun, predicted_referent=best,
        candidates=[c for c, _ in candidates], scores={},
        route="class_only",
    )
