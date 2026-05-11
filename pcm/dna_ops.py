"""pcm.dna_ops — minimal primitive op registry for Tier-D cooking.

This module is the PCM port of pcm-agent's much larger
``pcm_agent.cognition.causal.dna_ops`` (1607 lines). We strip it down to
the smallest set required for the Tier-D bit-identity claim D1 (see
``docs/PCM_NODE_AS_FUNCTION_DESIGN.md`` §4.1):

- A small handful of `pure` tensor / scalar ops.
- Exactly one ``collapse`` op (``muscle.collapse_facet``) — the only PCM-
  specific primitive that touches ``ConceptGraph.bundle_pool``.
- Exactly one ``invoke_module`` op (``muscle.invoke_module``) — the
  bridge that lets cook subgraphs delegate the heavy MLP-style backbone
  to a registered ``nn.Module``.

The dispatcher in :mod:`pcm.graph_eval` injects extra context (caller,
tick, concept_graph, module_registry) only for ``collapse`` and
``invoke_module`` ops; ``pure`` ops keep the legacy ``fn(*args) -> value``
contract.

Adding a new op is two lines: define the function and add the entry to
``DNA_OPS`` (and ``OP_KIND`` if it needs context kwargs).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .concept_graph import ConceptGraph

__all__ = [
    "DNA_OPS",
    "OP_KIND",
    "register_op",
    "op_kind_of",
]


# ---------------------------------------------------------------------------
# Op-kind classification (PCM minimal subset).
# ---------------------------------------------------------------------------

OP_KIND: dict[str, str] = {}


def op_kind_of(op_name: str) -> str:
    """Return the dispatch kind of ``op_name``.

    Defaults to ``"pure"`` so newly added ops keep the simplest contract.
    """
    return OP_KIND.get(str(op_name), "pure")


# ---------------------------------------------------------------------------
# pure — stateless tensor / scalar ops.
# ---------------------------------------------------------------------------


def _add(a: Any, b: Any) -> Any:
    """Element-wise / scalar addition. Falls back to Python ``+`` so the op
    works for both ``torch.Tensor`` and ``int`` / ``float``.
    """
    return a + b


def _mul(a: Any, b: Any) -> Any:
    return a * b


def _concat(*args: Any, dim: int = -1) -> torch.Tensor:
    """Concatenate tensor arguments along ``dim`` (default -1).

    Some cook subgraphs pass ``dim`` as the last positional arg to keep the
    metadata schema flat. We accept both forms (kwarg or trailing int) to
    avoid metadata churn when subgraphs are hand-written by humans.
    """
    if len(args) >= 2 and isinstance(args[-1], int):
        *tensors, real_dim = args
        return torch.cat(tuple(tensors), dim=int(real_dim))
    return torch.cat(tuple(args), dim=int(dim))


def _l2_normalize(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(t, dim=int(dim))


def _embedding_lookup(slots: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Pure ``F.embedding`` for cases where the caller already has a long
    tensor of slot indices. The PCM hot path normally uses
    ``muscle.collapse_facet`` (which also writes attribution); this op is
    here for non-attribution lookups.
    """
    if not isinstance(slots, torch.Tensor):
        slots = torch.as_tensor(slots, dtype=torch.long, device=table.device)
    return F.embedding(slots.long(), table)


def _scalar_const(value: Any) -> Any:
    """Identity / typed pass-through. Useful for cook subgraphs that need
    to inject a literal that depends on a runtime input rather than on
    ``metadata.constants``.
    """
    return value


# ---------------------------------------------------------------------------
# collapse — the one PCM-specific primitive that touches bundle_pool.
# ---------------------------------------------------------------------------


def _muscle_collapse_facet(
    facet: str,
    concept_ids: list[str],
    *,
    caller: str,
    tick: int,
    concept_graph: "ConceptGraph",
    shape: tuple[int, ...] | None = None,
    init: str = "normal_small",
) -> torch.Tensor:
    """Cook-time wrapper around :meth:`ConceptGraph.collapse_batch`.

    The shape kwarg is normally optional: if the facet pool already
    exists in ``concept_graph.bundle_pool``, the existing shape is used.
    A fresh facet (first ever cook of this name) requires either an
    explicit ``shape`` argument from the cook node's ``args`` list or a
    pre-warmup step that has already allocated the pool. The Tier-D
    reference heads always go through the warmup path so ``shape`` stays
    inferred.
    """
    if facet in concept_graph.bundle_pool:
        existing = concept_graph.bundle_pool[facet]
        inferred_shape = tuple(int(s) for s in existing.shape[1:])
    elif shape is not None:
        inferred_shape = tuple(int(s) for s in shape)
    else:
        raise ValueError(
            f"muscle.collapse_facet on facet {facet!r}: pool not yet allocated "
            "and no shape argument provided. Either warm-up forward first or "
            "pass shape=(D,) as a third positional arg."
        )
    return concept_graph.collapse_batch(
        caller=caller,
        facet=facet,
        concept_ids=list(concept_ids),
        shape=inferred_shape,
        tick=tick,
        init=init,
    )


# ---------------------------------------------------------------------------
# invoke_module — call an external nn.Module from cook subgraphs.
# ---------------------------------------------------------------------------


def _muscle_invoke_module(
    name_or_module: Any,
    *args: Any,
    module_registry: dict[str, nn.Module],
    caller: str,
    tick: int,
) -> Any:
    """Look up an ``nn.Module`` by name (or use the passed instance) and
    forward-call it with ``*args``.

    Supports two argument forms:

    - ``("module_name", *positional_args)`` — preferred form, the name is
      a key into ``module_registry``.
    - ``(module_instance, *positional_args)`` — useful for tests that
      don't want to plumb a registry; the first arg is already an
      ``nn.Module``.
    """
    if isinstance(name_or_module, nn.Module):
        module = name_or_module
    elif isinstance(name_or_module, str):
        if name_or_module not in module_registry:
            raise KeyError(
                f"muscle.invoke_module: name {name_or_module!r} not in "
                f"module_registry (have: {sorted(module_registry)!r}); "
                f"caller={caller!r} tick={tick}"
            )
        module = module_registry[name_or_module]
    else:
        raise TypeError(
            f"muscle.invoke_module: first arg must be str or nn.Module, "
            f"got {type(name_or_module).__name__}"
        )
    return module(*args)


# ---------------------------------------------------------------------------
# Tier-G abstract-concept ops (D95 sleep abstraction pass).
# ---------------------------------------------------------------------------


def _concept_codebook_lookup(
    facet: str,
    codebook_id: str,
    *,
    caller: str,
    tick: int,
    concept_graph: "ConceptGraph",
) -> torch.Tensor:
    """Single-row collapse for a Tier-G abstract prototype slot.

    Equivalent to ``muscle.collapse_facet(facet, [codebook_id])`` but
    spelled separately so cook DAGs can document intent ("anchor"
    vs "residual") and so attribution can stamp a distinct caller
    label without aliasing with the muscle that originally trained
    the underlying member rows.

    Returns a ``(1, D)`` tensor (single row, batch axis kept so the
    relation_apply op below can elementwise-combine it with a
    ``muscle.collapse_facet`` output of the same shape).
    """
    if str(facet) not in concept_graph.bundle_pool:
        raise KeyError(
            f"concept.codebook_lookup: facet {facet!r} not yet in "
            "bundle_pool; warm up via collapse_batch / sleep pass first"
        )
    inferred_shape = tuple(
        int(s) for s in concept_graph.bundle_pool[str(facet)].shape[1:]
    )
    return concept_graph.collapse_batch(
        caller=caller,
        facet=str(facet),
        concept_ids=[str(codebook_id)],
        shape=inferred_shape,
        tick=tick,
    )


def _concept_relation_apply(
    anchor: torch.Tensor,
    residual: torch.Tensor,
    mode: str = "add",
) -> torch.Tensor:
    """Combine an abstract-prototype row with a per-member residual.

    ``mode`` accepts:
        ``"add"``      — anchor + residual (default; reconstructs original row
                         when residual was set to ``row - anchor`` at sleep time).
        ``"mul"``      — anchor * residual (gain modulation).
        ``"concat"``   — torch.cat([anchor, residual], dim=-1).

    The function is intentionally unparameterised: any learnable
    transformation should live in the residual itself (which is a
    standard ``bundle_pool`` row that AdamW can update freely).
    """
    if mode != "concat" and residual.shape != anchor.shape:
        raise ValueError(
            f"concept.relation_apply: anchor shape {tuple(anchor.shape)} "
            f"!= residual shape {tuple(residual.shape)} for mode {mode!r}; "
            "use mode='concat' for shape-mismatched composition."
        )
    if mode == "add":
        return anchor + residual
    if mode == "mul":
        return anchor * residual
    if mode == "concat":
        return torch.cat([anchor, residual], dim=-1)
    raise ValueError(
        f"concept.relation_apply: unknown mode {mode!r}; "
        "expected 'add' / 'mul' / 'concat'"
    )


# ---------------------------------------------------------------------------
# Registry assembly + helpers.
# ---------------------------------------------------------------------------


DNA_OPS: dict[str, Callable[..., Any]] = {
    # pure
    "add": _add,
    "mul": _mul,
    "concat": _concat,
    "l2_normalize": _l2_normalize,
    "embedding_lookup": _embedding_lookup,
    "scalar_const": _scalar_const,
    "concept.relation_apply": _concept_relation_apply,
    # collapse
    "muscle.collapse_facet": _muscle_collapse_facet,
    "concept.codebook_lookup": _concept_codebook_lookup,
    # invoke_module
    "muscle.invoke_module": _muscle_invoke_module,
}

OP_KIND.update({
    "muscle.collapse_facet": "collapse",
    "concept.codebook_lookup": "collapse",
    "muscle.invoke_module": "invoke_module",
})


def register_op(
    name: str,
    fn: Callable[..., Any],
    kind: str = "pure",
) -> None:
    """Register a new primitive op. ``kind`` must be one of
    ``"pure" | "collapse" | "invoke_module"``.

    Naming convention: ``namespace.verb`` (e.g. ``arith.add``,
    ``muscle.collapse_facet``).
    """
    if kind not in ("pure", "collapse", "invoke_module"):
        raise ValueError(
            f"register_op: unknown kind {kind!r}; expected pure / collapse / "
            "invoke_module"
        )
    DNA_OPS[name] = fn
    if kind != "pure":
        OP_KIND[name] = kind
