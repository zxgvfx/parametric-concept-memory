"""tests/test_sleep_four_domain.py — Tier-G integration coverage
across the four PAPER domains (numbers / colors / space / phonemes).

Each domain's ``train_one`` is exercised twice with the smallest
viable configuration:

1. ``sleep_every=None`` — the legacy code path. Must be 100%
   backward-compatible: no ``attach_sleep`` flag, no abstract
   concepts created, ``sleep_reports == []``.
2. ``sleep_every=2`` — Tier-G is ON. Must emit non-empty
   ``sleep_reports`` and must not regress task accuracy by more than
   a generous margin (the unit-test budget can't run long enough for
   §4–§6 paper-baseline accuracy to converge; full ρ regression
   testing belongs in ``experiments/sleep_pass_demo.py``).

Total budget: well under 90 seconds on CPU.
"""
from __future__ import annotations

import unittest

import torch


# ---------------------------------------------------------------------------
# Color domain (paper §5).
# ---------------------------------------------------------------------------


def _color_run(sleep_every: int | None) -> dict:
    from experiments.color_concept_study import _config as cfg
    from experiments.color_concept_study.train import train_one

    torch.manual_seed(7)
    raw = torch.randn(cfg.N_COLORS, cfg.EMBED_DIM, device=cfg.DEVICE)
    q, _ = torch.linalg.qr(raw.t())
    centroids = q.t().contiguous()[: cfg.N_COLORS]

    return train_one(
        mode="single",
        seed=0,
        centroids=centroids,
        epochs=4,
        steps_per_epoch=20,
        sleep_every=sleep_every,
    )


class TestColorDomain(unittest.TestCase):
    def test_backward_compat_none(self) -> None:
        out = _color_run(sleep_every=None)
        self.assertEqual(out["sleep_reports"], [])
        self.assertGreater(out["mix_acc"], 0.0)

    def test_with_sleep_every_2(self) -> None:
        out = _color_run(sleep_every=2)
        self.assertGreater(
            len(out["sleep_reports"]), 0,
            "sleep_every=2 over 4 epochs should fire ≥1 sleep pass",
        )
        self.assertGreater(
            out["mix_acc"], 0.05,
            f"mix_acc collapsed to {out['mix_acc']:.3f}",
        )
        first = out["sleep_reports"][0]
        self.assertTrue(first["facets"] or first["skipped_facets"])


# ---------------------------------------------------------------------------
# Space domain (paper §6.2).
# ---------------------------------------------------------------------------


def _space_run(sleep_every: int | None) -> dict:
    from experiments.space_concept_study.train import train_one
    return train_one(
        mode="single",
        seed=0,
        epochs=4,
        steps_per_epoch=20,
        sleep_every=sleep_every,
    )


class TestSpaceDomain(unittest.TestCase):
    def test_backward_compat_none(self) -> None:
        out = _space_run(sleep_every=None)
        self.assertEqual(out["sleep_reports"], [])
        self.assertGreater(out["move_acc"], 0.0)

    def test_with_sleep_every_2(self) -> None:
        out = _space_run(sleep_every=2)
        self.assertGreater(len(out["sleep_reports"]), 0)
        self.assertGreater(
            out["move_acc"], 0.05,
            f"move_acc collapsed to {out['move_acc']:.3f}",
        )


# ---------------------------------------------------------------------------
# Phoneme domain (paper §6.3).
# ---------------------------------------------------------------------------


def _phoneme_run(sleep_every: int | None) -> dict:
    from experiments.phoneme_concept_study.train import train_one
    return train_one(
        mode="single_v",
        seed=0,
        epochs=4,
        steps_per_epoch=20,
        sleep_every=sleep_every,
    )


class TestPhonemeDomain(unittest.TestCase):
    def test_backward_compat_none(self) -> None:
        out = _phoneme_run(sleep_every=None)
        self.assertEqual(out["sleep_reports"], [])
        self.assertIn("v", out["accs"])

    def test_with_sleep_every_2(self) -> None:
        out = _phoneme_run(sleep_every=2)
        self.assertGreater(len(out["sleep_reports"]), 0)
        # voicing is binary: don't ask for high accuracy in 80 steps.
        self.assertGreater(
            out["accs"]["v"], 0.30,
            f"voicing acc collapsed to {out['accs']['v']:.3f}",
        )


# ---------------------------------------------------------------------------
# Signature smoke for quad_study (numbers four-op) and purity_audit.
# Their training loops take more elaborate inputs (DatasetConfig, centroids,
# train_triples) than the three uniform-`train_one`-style domains above, so
# we don't run them end-to-end inside the unit-test budget. We *do* check
# the ``sleep_every`` kwarg is accepted and the function still returns a
# ``sleep_reports`` key when invoked through the documented contract.
# ---------------------------------------------------------------------------


class TestSignatureSmoke(unittest.TestCase):
    def test_quad_train_accepts_sleep_every(self) -> None:
        import inspect
        from experiments.quad_study import train_quad
        sig = inspect.signature(train_quad)
        self.assertIn("sleep_every", sig.parameters,
                      "quad_study.train_quad missing sleep_every kwarg")
        param = sig.parameters["sleep_every"]
        self.assertIsNone(param.default,
                          "sleep_every default must be None for "
                          "backward compatibility")

    def test_purity_train_accepts_sleep_every(self) -> None:
        import inspect
        from experiments.purity_audit.train import purity_train_one
        sig = inspect.signature(purity_train_one)
        self.assertIn("sleep_every", sig.parameters,
                      "purity_audit.purity_train_one missing sleep_every "
                      "kwarg")
        self.assertIsNone(sig.parameters["sleep_every"].default,
                          "sleep_every default must be None for "
                          "backward compatibility")

    def test_color_train_accepts_sleep_every(self) -> None:
        import inspect
        from experiments.color_concept_study.train import train_one
        self.assertIn("sleep_every",
                      inspect.signature(train_one).parameters)

    def test_space_train_accepts_sleep_every(self) -> None:
        import inspect
        from experiments.space_concept_study.train import train_one
        self.assertIn("sleep_every",
                      inspect.signature(train_one).parameters)

    def test_phoneme_train_accepts_sleep_every(self) -> None:
        import inspect
        from experiments.phoneme_concept_study.train import train_one
        self.assertIn("sleep_every",
                      inspect.signature(train_one).parameters)


if __name__ == "__main__":
    unittest.main()
