"""pcm.diagnostics — falsifiable causal-ablation protocol for PCM domains.

PAPER §6.6 / §6.7 / §6.8 / §6.9 / §7.4 / §7.5 / §7.5-color / §7.5-space
all instantiate the same five-condition ablation pattern. This
module promotes that pattern to a reusable API:

* :class:`CausalLayer` — abstract base class for B / C / D layer
  implementations (biological prior, ecological statistics,
  task-driven asymmetry).
* :class:`AblationCondition` — one of the five conditions with
  its (B, C, D) flags.
* :class:`CausalAblationProtocol` — standard 5-condition × N-seed
  driver with consistent diagnostic output.
* :func:`run_causal_ablation` — convenience entry-point that
  takes a domain runner callback and produces a summary dict
  identical in shape to the §6.8 / §7.4 / §6.9 / §7.5-space
  scripts.

The driver is **deliberately minimal**: it does not assume a
specific PCM architecture, head shape, or evaluation metric. The
caller supplies a ``run_one(seed, layers)`` callable that builds
the graph, applies any active layers, trains, and returns a
metric dict; the protocol just orchestrates the 5×N grid and
aggregates statistics.

Used together with the existing experiment scripts
(`experiments/sleep_color_primaries.py`,
`experiments/sleep_number_decimal.py`,
`experiments/sleep_phoneme_transfer.py`,
`experiments/sleep_number_extrapolate.py`,
`experiments/sleep_color_holdout.py`,
`experiments/sleep_space_extrapolate.py`) this protocol gives a
cross-domain reusable abstraction without requiring those
scripts to be rewritten.

Example::

    from pcm.diagnostics import run_causal_ablation

    def my_domain_run_one(seed, layers, **kwargs):
        '''Build a PCM graph, train with the given B/C/D layer
        flags active, return a metric dict.'''
        ...
        return {"target_acc": 0.83, "rho": +0.12, ...}

    summary = run_causal_ablation(
        my_domain_run_one,
        n_seeds=5,
        seed_base=92000,
        primary_metrics=["target_acc", "rho"],
    )
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


__all__ = [
    "CAUSAL_LAYERS",
    "DEFAULT_CONDITIONS",
    "AblationCondition",
    "AblationLayers",
    "CausalAblationProtocol",
    "run_causal_ablation",
    "summarise_per_seed",
]


# ---------------------------------------------------------------------------
# Layer + condition definitions.
# ---------------------------------------------------------------------------


CAUSAL_LAYERS: tuple[str, str, str] = ("B", "C", "D")
"""Canonical PCM causal-ablation layers, in dominant-order under the
task-symmetry × dominant-layer principle (PAPER §6.9):
* **B** — biological / hardware prior (e.g. LMS centroid, decimal
  cones, articulator cones, cardinal axes).
* **C** — ecological / environmental statistics (e.g. green-peak
  hue, round-number sampling, Zipf phonotactics, centre bias).
* **D** — task-driven asymmetry (e.g. ripe-fruit head, last-digit
  head, minimal-pair head, row-index head)."""


@dataclass(frozen=True)
class AblationLayers:
    """Per-condition B / C / D activation flags.

    Each flag is a free-form ``str | bool | None`` whose meaning is
    interpreted by the domain-specific ``run_one`` callback. The
    typical convention used by the bundled experiment scripts is:

    * ``None`` / ``False`` — layer disabled.
    * ``"random"`` / ``True`` — layer enabled in its default form.
    * Specific string — selects a named variant of the layer (e.g.
      ``"lms"`` or ``"decimal_cones"`` for B; ``"green_peak"`` or
      ``"round_number"`` for C).
    """
    B: str | bool | None = None
    C: str | bool | None = None
    D: str | bool | None = None

    def is_active(self, layer: str) -> bool:
        v = getattr(self, layer)
        return v is not None and v is not False

    def to_dict(self) -> dict[str, str | bool | None]:
        return {"B": self.B, "C": self.C, "D": self.D}


@dataclass(frozen=True)
class AblationCondition:
    """A named ablation condition (one row of the 5-condition table)."""
    name: str
    layers: AblationLayers
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layers": self.layers.to_dict(),
            "description": self.description,
        }


def _condition(name: str, B=None, C=None, D=None,
               description: str = "") -> AblationCondition:
    return AblationCondition(
        name=name,
        layers=AblationLayers(B=B, C=C, D=D),
        description=description,
    )


# Default 5-condition canonical set used in PAPER §6.6/6.7/6.8/6.9/7.4/7.5*.
DEFAULT_CONDITIONS: tuple[AblationCondition, ...] = (
    _condition(
        "A_baseline",
        description="Random centroid + uniform sampling + no auxiliary "
                    "head. Symmetric task only.",
    ),
    _condition(
        "B_prior", B=True,
        description="Biological / hardware prior layer enabled.",
    ),
    _condition(
        "C_statistics", C=True,
        description="Ecological / environmental sampling statistics "
                    "layer enabled.",
    ),
    _condition(
        "D_task", D=True,
        description="Task-driven asymmetric auxiliary head enabled.",
    ),
    _condition(
        "BCD_combined", B=True, C=True, D=True,
        description="All three causal layers stacked.",
    ),
)


# ---------------------------------------------------------------------------
# Statistics helpers.
# ---------------------------------------------------------------------------


def summarise_per_seed(values: Iterable[float]) -> dict[str, float]:
    """Mean / std / min / max / n over a numeric iterable, NaN-safe."""
    xs = [float(x) for x in values
          if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "n": 0,
                "min": float("nan"), "max": float("nan")}
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
    return {"mean": m, "std": sd, "min": min(xs), "max": max(xs),
            "n": len(xs)}


# ---------------------------------------------------------------------------
# Protocol driver.
# ---------------------------------------------------------------------------


RunOneCallback = Callable[..., dict[str, Any]]
"""Signature: ``(seed: int, layers: AblationLayers, **kwargs) -> metric_dict``.

The metric dict must have flat numeric leaves at the keys listed
in ``primary_metrics``. Additional keys (e.g. wall_s, n_train,
config) are preserved verbatim into the per-seed records."""


@dataclass
class CausalAblationProtocol:
    """5-condition × N-seed driver for PCM causal ablations."""

    run_one: RunOneCallback
    n_seeds: int = 5
    seed_base: int = 90000
    conditions: tuple[AblationCondition, ...] = DEFAULT_CONDITIONS
    primary_metrics: tuple[str, ...] = ("acc",)
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    verbose: bool = True

    def run(self) -> dict[str, Any]:
        """Run all conditions × seeds. Returns a JSON-serialisable
        summary dict in the same shape produced by the bundled
        experiment scripts (`outputs/<run>/summary.json`)."""
        out: dict[str, Any] = {
            "config": {
                "n_seeds": self.n_seeds,
                "seed_base": self.seed_base,
                "primary_metrics": list(self.primary_metrics),
                "conditions": [c.to_dict() for c in self.conditions],
                "extra_kwargs": dict(self.extra_kwargs),
            },
            "by_condition": {},
        }

        for cond in self.conditions:
            if self.verbose:
                print(f"\n── {cond.name}: {cond.layers.to_dict()} ──")
            rows: list[dict[str, Any]] = []
            for si in range(self.n_seeds):
                seed = self.seed_base + si
                metrics = self.run_one(
                    seed=seed, layers=cond.layers, **self.extra_kwargs,
                )
                row = {"seed": seed, "condition": cond.name, **metrics}
                rows.append(row)
                if self.verbose:
                    extras = " ".join(
                        f"{k}={metrics[k]:.3f}" if isinstance(metrics.get(k), (int, float))
                        and metrics.get(k) is not None
                        and not (isinstance(metrics.get(k), float) and math.isnan(metrics[k]))
                        else f"{k}={metrics.get(k)}"
                        for k in self.primary_metrics
                        if k in metrics
                    )
                    print(f"  [seed={seed}] {extras}")

            cond_summary: dict[str, Any] = {
                "config": cond.to_dict(),
                "per_seed": rows,
            }
            for metric in self.primary_metrics:
                cond_summary[metric] = summarise_per_seed(
                    r[metric] for r in rows if metric in r
                )
            out["by_condition"][cond.name] = cond_summary
            if self.verbose:
                stats_str = "  ".join(
                    f"{k}={cond_summary[k]['mean']:+.3f}"
                    f"±{cond_summary[k]['std']:.3f}"
                    for k in self.primary_metrics
                )
                print(f"  → {stats_str}")

        return out


def run_causal_ablation(
    run_one: RunOneCallback,
    *,
    n_seeds: int = 5,
    seed_base: int = 90000,
    conditions: Iterable[AblationCondition] | None = None,
    primary_metrics: Iterable[str] = ("acc",),
    verbose: bool = True,
    **extra_kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper around :class:`CausalAblationProtocol`.

    The ``run_one`` callable is invoked with positional ``seed``
    and keyword ``layers`` (an :class:`AblationLayers` instance),
    plus any extra kwargs forwarded from this call. It must
    return a metric dict containing each name listed in
    ``primary_metrics`` as a numeric scalar.

    Example::

        def color_run_one(seed, layers, **kw):
            from experiments.color_concept_study.train import train_one
            from experiments.color_concept_study.graph_builder import (
                make_lms_like_centroids, make_random_orthogonal_centroids,
            )
            centroids = (
                make_lms_like_centroids(N_COLORS, EMBED_DIM, seed)
                if layers.is_active("B")
                else make_random_orthogonal_centroids(N_COLORS, EMBED_DIM, seed)
            )
            r = train_one(
                "single", seed, centroids,
                mix_sample_weight=(green_peak_weights() if layers.is_active("C")
                                   else None),
                enable_ripe_head=layers.is_active("D"),
                **kw,
            )
            return {"mix_acc": r["mix_acc"], "rho_circ": ...}

        summary = run_causal_ablation(
            color_run_one,
            n_seeds=8,
            primary_metrics=("mix_acc", "rho_circ"),
            epochs=30, steps_per_epoch=200,  # forwarded to run_one
        )

    See ``experiments/sleep_color_primaries.py`` for a fully
    worked-through caller pattern.
    """
    proto = CausalAblationProtocol(
        run_one=run_one,
        n_seeds=n_seeds,
        seed_base=seed_base,
        conditions=tuple(conditions) if conditions is not None
        else DEFAULT_CONDITIONS,
        primary_metrics=tuple(primary_metrics),
        extra_kwargs=dict(extra_kwargs),
        verbose=verbose,
    )
    return proto.run()
