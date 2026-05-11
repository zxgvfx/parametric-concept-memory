# PCM Causal Ablation Protocol — Author's Guide

This is a 50-line reference for writing a new domain experiment
that plugs into PCM's standard `A / B / C / D / B+C+D`
five-condition causal-ablation protocol (PAPER §6.6 / §6.7 /
§6.8 / §6.9 / §7.4 / §7.5 / §7.5-color / §7.5-space).

The protocol asks, of any candidate representational primitive *Y*
in domain *X*, whether *Y* can emerge unconditionally or whether
it requires injection from one of three causal layers — biology,
ecology, or task pressure — that the cognitive-science literature
attributes to its human counterpart.

## When to use this protocol

You should run a five-condition ablation when:

1. You have a PCM-domain trainable on its own (e.g. a `train_one`
   function that takes a seed and returns metrics).
2. You can identify three plausible causal layers for the
   primitive under investigation:
   - **B** (hardware prior) — a non-uniform centroid / receptor
     structure that supplies *Y* as input geometry. Examples:
     LMS-like centroid for colour (§6.8), decimal cones for
     numbers (§7.4), articulator cones for phonemes (§6.9),
     cardinal-axis cones for space (§7.5-space).
   - **C** (ecological statistic) — non-uniform input
     distribution that over-represents *Y*-relevant regions.
     Examples: green-peak hue sampling (§6.8), round-number
     boost (§7.4), Zipf phonotactics (§6.9), centre-bias
     spatial sampling (§7.5-space).
   - **D** (task asymmetry) — a small auxiliary task whose loss
     is non-trivial only on a *Y*-aligned subset of inputs.
     Examples: ripe-fruit head (§6.8), last-digit head (§7.4),
     minimal-pair head (§6.9), row-index head (§7.5-space).
3. You want a falsifiable lower bound on which layer is
   necessary, sufficient, and which compose additively.

## Three-step recipe

Step 1 — declare conditions:

```python
from pcm.diagnostics import (
    AblationCondition, AblationLayers, run_causal_ablation,
)

CONDITIONS = (
    AblationCondition("A_baseline", AblationLayers()),
    AblationCondition("B_<prior>", AblationLayers(B="<prior_mode>")),
    AblationCondition("C_<statistic>", AblationLayers(C="<sample_mode>")),
    AblationCondition("D_<task>", AblationLayers(D=True)),
    AblationCondition("BCD_combined",
                      AblationLayers(B="<prior_mode>",
                                     C="<sample_mode>", D=True)),
)
```

Step 2 — write a domain dispatcher:

```python
def _run_one(seed: int, layers: AblationLayers, **kw) -> dict:
    centroids = (build_lms(seed) if layers.B == "lms"
                 else build_random(seed))
    weights = (zipf_weights(N) if layers.C == "zipf" else None)
    head_extra = MyHead() if layers.is_active("D") else None
    r = train_one(seed, centroids, weights=weights,
                  extra_head=head_extra, **kw)
    return {
        "primary_metric": r["accuracy"],
        "geometry_metric": r["rho"],
        "ood_metric": r["ood_acc"],
        # any additional fields are preserved verbatim per-seed
    }
```

Step 3 — drive the ablation with `run_causal_ablation`:

```python
def main():
    summary = run_causal_ablation(
        _run_one,
        n_seeds=5,
        seed_base=90000,
        conditions=CONDITIONS,
        primary_metrics=("primary_metric", "geometry_metric", "ood_metric"),
        # any extra kwargs forwarded to _run_one as **kw
        epochs=30, steps_per_epoch=200,
    )
    # summary["by_condition"][cond_name] now has:
    #   "config":        the AblationCondition.to_dict()
    #   "per_seed":      list of metric dicts (one per seed)
    #   "<metric_name>": {"mean", "std", "min", "max", "n"}
    json.dump(summary, open("summary.json", "w"))
```

## Output schema

`run_causal_ablation` produces a JSON-serialisable summary dict
identical in shape to the schema produced by all bundled
experiment scripts (`outputs/<exp_name>/summary.json`):

```json
{
  "config": {
    "n_seeds": 5,
    "seed_base": 90000,
    "primary_metrics": ["primary_metric", "geometry_metric", "ood_metric"],
    "conditions": [...condition.to_dict() for each cond...],
    "extra_kwargs": {...}
  },
  "by_condition": {
    "A_baseline": {
      "config": {"name": "A_baseline", "layers": {"B": null, "C": null, "D": null}, "description": ""},
      "per_seed": [
        {"seed": 90000, "condition": "A_baseline", "primary_metric": 0.85, ...},
        ...
      ],
      "primary_metric": {"mean": 0.85, "std": 0.02, "min": 0.83, "max": 0.87, "n": 5},
      "geometry_metric": {"mean": 0.91, "std": 0.01, ...},
      ...
    },
    "B_<prior>": {...},
    ...
  }
}
```

This schema is read directly by all bundled F11–F17 figure
renderers without modification — your new experiment can reuse
their plotting code by just pointing `--in` at your
`summary.json`.

## Optional cond-level post-aggregation

Some experiments need cond-level fields that are **not** simple
per-seed numeric metrics (e.g. `fraction_equidistant`,
`perceptual_prior_match_counts` from §6.7 / §6.8). For those,
augment `summary["by_condition"][name]` after `run_causal_ablation`
returns:

```python
summary = run_causal_ablation(...)
for cond in CONDITIONS:
    cs = summary["by_condition"][cond.name]
    rows = cs["per_seed"]
    cs["fraction_equidistant"] = sum(1 for r in rows
                                      if r.get("is_equidistant")) / len(rows)
    cs["custom_count_of_X"] = ...
```

## What you do not need to do

1. **Do not re-implement** the 5 × N nested loop, per-condition
   stat aggregation, or summary-dict assembly — that is what
   `run_causal_ablation` exists for.
2. **Do not write** a domain-specific `_stats(xs)` helper —
   `pcm.diagnostics.summarise_per_seed` is NaN-safe and does the
   `mean / std / min / max / n` computation correctly.
3. **Do not invent new condition names** — the
   `A_baseline / B_<prior> / C_<statistic> / D_<task> / BCD_combined`
   convention enables cross-domain comparison (PAPER §6.9
   dominant-layer table, §7.5 / §7.5-color / §7.5-space ceiling
   taxonomy).

## Six worked examples

The six existing experiment scripts that use this protocol
(after the F34 dogfood refactor) are the canonical reference:

| Script | Domain | Layers (B / C / D) |
|---|---|---|
| `experiments/sleep_color_primaries.py` (§6.8) | colour 12-hue | LMS centroid / green-peak / RipeFruitHead |
| `experiments/sleep_color_holdout.py` (§7.5-color) | colour hue holdout | (same as §6.8) |
| `experiments/sleep_number_decimal.py` (§7.4) | number arithmetic | decimal cones / round-number / LastDigitHead |
| `experiments/sleep_number_extrapolate.py` (§7.5) | number length-OOD | (same as §7.4) |
| `experiments/sleep_phoneme_transfer.py` (§6.9) | phoneme cross-language | articulator cones / Zipf phonotactics / MinimalPairHead |
| `experiments/sleep_space_extrapolate.py` (§7.5-space) | space 5x5 → 7x7 | cardinal cones / centre-bias / RowIndexHead |

Each is **<250 lines total** including domain-specific layer
implementations. A new domain that follows the same template
should take ~50 lines of glue code on top of a working
`train_one(seed, ...)`.

## Reading list

- **PAPER §6.6 / §6.7 / §6.8**: full colour-domain
  three-causal-layer story.
- **PAPER §7.4**: full number-domain analogue.
- **PAPER §6.9**: phoneme cross-language transfer + dominant-layer
  switch principle.
- **PAPER §7.5 / §7.5-color / §7.5-space**: four-type ceiling
  taxonomy (input-side / output-side / symmetric-OOD /
  asymmetric-OOD).
- **PAPER §7.5-space addendum (F37)**: refined ceiling taxonomy
  showing which ceilings are PCM-fundamental vs head-distribution-
  coverage.
- **`pcm/diagnostics.py`**: source of truth for the protocol API.
- **`tests/test_diagnostics.py`**: D1–D5 falsifiable contract
  tests for the protocol itself.

## When this protocol is *not* the right tool

The five-condition protocol assumes:

- A single PCM domain with one or two heads.
- A clean separation between B / C / D layers (i.e. you can
  inject one layer without leaking signal from another).
- A primary scalar metric (or a small set) that aggregates
  meaningfully across seeds.

For experiments that violate one of these assumptions:

- **Long-running continual-learning ablations** with many
  sleep passes: write a custom driver; the protocol's batch
  5 × N parallelism wastes wall-time.
- **Augmentation rate sweeps** (like F37): use a single
  `BCD_combined` condition × multiple rates instead — the
  protocol is overkill.
- **Multi-domain joint training**: the protocol's per-condition
  aggregation does not handle multi-domain joint metrics.

In those cases, write a custom driver but keep the
`AblationLayers` / `summarise_per_seed` helpers from
`pcm.diagnostics` for partial reuse.
