"""tests/test_diagnostics.py — pcm.diagnostics.causal_ablation API.

Smoke-level coverage of the falsifiable causal-ablation protocol
(PAPER §6.6 / §6.8 / §7.4 / §6.9 / §7.5-space). The protocol
itself is a thin orchestrator over a domain-specific ``run_one``
callback, so we test:

  * D1 — DEFAULT_CONDITIONS shape (5 conditions named exactly
        A_baseline / B_prior / C_statistics / D_task / BCD_combined).
  * D2 — AblationLayers.is_active behaves correctly across
        ``None / False / True / "string"`` flag values.
  * D3 — Driver invokes ``run_one`` ``5 × n_seeds`` times with the
        right (seed, layers) pairs.
  * D4 — Returned summary has the canonical
        ``config / by_condition / per_seed / mean / std`` layout
        used downstream by F11 / F12 / F15 / F16 figure renderers.
  * D5 — summarise_per_seed is NaN-safe and returns
        ``{mean, std, min, max, n}`` for normal numeric input.
"""
from __future__ import annotations

import math
import unittest

from pcm.diagnostics import (
    CAUSAL_LAYERS,
    DEFAULT_CONDITIONS,
    AblationCondition,
    AblationLayers,
    CausalAblationProtocol,
    run_causal_ablation,
    summarise_per_seed,
)


class TestD1ConditionShape(unittest.TestCase):
    def test_default_conditions_have_canonical_5_names(self) -> None:
        names = [c.name for c in DEFAULT_CONDITIONS]
        self.assertEqual(names, [
            "A_baseline", "B_prior", "C_statistics", "D_task", "BCD_combined",
        ])

    def test_canonical_layer_order_is_BCD(self) -> None:
        self.assertEqual(CAUSAL_LAYERS, ("B", "C", "D"))

    def test_each_condition_is_immutable_and_serialisable(self) -> None:
        for cond in DEFAULT_CONDITIONS:
            self.assertIsInstance(cond, AblationCondition)
            d = cond.to_dict()
            self.assertEqual(set(d.keys()), {"name", "layers", "description"})
            for layer in CAUSAL_LAYERS:
                self.assertIn(layer, d["layers"])


class TestD2LayerActive(unittest.TestCase):
    def test_none_and_false_are_inactive(self) -> None:
        layers = AblationLayers(B=None, C=False, D=False)
        for letter in CAUSAL_LAYERS:
            self.assertFalse(layers.is_active(letter))

    def test_true_and_string_are_active(self) -> None:
        layers = AblationLayers(B=True, C="green_peak", D="ripe_head")
        for letter in CAUSAL_LAYERS:
            self.assertTrue(layers.is_active(letter))

    def test_default_baseline_all_inactive(self) -> None:
        baseline = next(c for c in DEFAULT_CONDITIONS if c.name == "A_baseline")
        for letter in CAUSAL_LAYERS:
            self.assertFalse(baseline.layers.is_active(letter))

    def test_default_bcd_all_active(self) -> None:
        bcd = next(c for c in DEFAULT_CONDITIONS if c.name == "BCD_combined")
        for letter in CAUSAL_LAYERS:
            self.assertTrue(bcd.layers.is_active(letter))


class TestD3DriverInvocation(unittest.TestCase):
    def test_run_one_called_5_times_n_seeds(self) -> None:
        calls: list[tuple[int, dict]] = []

        def fake_run_one(seed: int, layers: AblationLayers, **kw):
            calls.append((seed, layers.to_dict()))
            # Return a deterministic metric driven by which layers are on.
            score = 0.1 + 0.2 * sum(1 for L in CAUSAL_LAYERS
                                    if layers.is_active(L))
            return {"target_acc": score, "rho": score - 0.05}

        proto = CausalAblationProtocol(
            run_one=fake_run_one,
            n_seeds=3,
            seed_base=12300,
            primary_metrics=("target_acc", "rho"),
            verbose=False,
        )
        summary = proto.run()

        self.assertEqual(len(calls), 5 * 3)
        seen_seeds = sorted({s for s, _ in calls})
        self.assertEqual(seen_seeds, [12300, 12301, 12302])
        # All 5 condition layers each appear exactly n_seeds times.
        layer_dicts = [tuple(sorted(d.items())) for _, d in calls]
        from collections import Counter
        cnt = Counter(layer_dicts)
        self.assertEqual(set(cnt.values()), {3})


class TestD4SummaryShape(unittest.TestCase):
    def test_summary_layout_matches_paper_outputs(self) -> None:
        def fake_run_one(seed: int, layers: AblationLayers, **kw):
            return {"acc": 0.5 + 0.1 * (seed % 5)}

        summary = run_causal_ablation(
            fake_run_one,
            n_seeds=4,
            seed_base=42000,
            primary_metrics=("acc",),
            verbose=False,
        )

        self.assertEqual(set(summary.keys()), {"config", "by_condition"})
        for cond_name in ["A_baseline", "B_prior", "C_statistics",
                          "D_task", "BCD_combined"]:
            self.assertIn(cond_name, summary["by_condition"])
            cs = summary["by_condition"][cond_name]
            self.assertIn("config", cs)
            self.assertIn("per_seed", cs)
            self.assertEqual(len(cs["per_seed"]), 4)
            self.assertIn("acc", cs)
            stats = cs["acc"]
            self.assertEqual(set(stats.keys()),
                             {"mean", "std", "min", "max", "n"})
            self.assertEqual(stats["n"], 4)


class TestD5SummariseStats(unittest.TestCase):
    def test_basic(self) -> None:
        s = summarise_per_seed([0.0, 1.0, 2.0, 3.0])
        self.assertAlmostEqual(s["mean"], 1.5)
        self.assertAlmostEqual(s["std"],
                               math.sqrt(((-1.5) ** 2 + (-0.5) ** 2 +
                                          (0.5) ** 2 + (1.5) ** 2) / 3))
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["min"], 0.0)
        self.assertEqual(s["max"], 3.0)

    def test_nan_safe(self) -> None:
        s = summarise_per_seed([0.0, float("nan"), 1.0, None])
        self.assertEqual(s["n"], 2)
        self.assertAlmostEqual(s["mean"], 0.5)

    def test_empty(self) -> None:
        s = summarise_per_seed([])
        self.assertEqual(s["n"], 0)
        for k in ("mean", "std", "min", "max"):
            self.assertTrue(math.isnan(s[k]))


if __name__ == "__main__":
    unittest.main()
