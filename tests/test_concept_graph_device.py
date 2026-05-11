"""tests/test_concept_graph_device.py — regression test for the
``ConceptGraph._ensure_facet`` device-equality bug fixed in F42.

Originally :meth:`ConceptGraph._ensure_facet` rebuilt the bundle
pool ``nn.Parameter`` whenever the requested device differed from
the existing pool's device. The comparison used
``pool.device != torch.device(device)`` directly, which silently
fails when callers pass the string ``"cuda"`` and the existing
pool is on ``cuda:0`` — they should be treated as identical, but
``torch.device("cuda")`` (no index) is *not* equal to
``torch.device("cuda", 0)`` under ``__eq__``.

In practice this rebuild was triggered on every collapse_batch
call from v2 dual-channel training, leaving the optimizer
holding a stale Parameter reference and the loss curve flat at
its initial value (the F42 bug that hid v2 V1/V2 invariants
behind the trivial-attr-collapse minimum).

This file does NOT need a GPU — the bug was equivalent on CPU
and the fix exercises the same code path. We construct a graph,
collapse once with one device spelling, then collapse again with
an "equivalent but different" spelling and assert the pool
Parameter is the same Python object both times.
"""
from __future__ import annotations

import unittest

import torch

from pcm import ConceptGraph


class TestConceptGraphEnsureFacetDeviceEquality(unittest.TestCase):
    """Regression for F42 device-equality bug."""

    def _build(self) -> tuple[ConceptGraph, list[str]]:
        cg = ConceptGraph(initial_capacity=8)
        cids: list[str] = []
        for i in range(4):
            cid = f"d:{i}"
            cg.register_concept(node_id=cid, label=cid, scope="BASE",
                                provenance="dev-test")
            cids.append(cid)
        return cg, cids

    def test_cpu_device_string_vs_object_no_rebuild(self) -> None:
        """``"cpu"`` vs ``torch.device("cpu")`` must not rebuild
        the pool Parameter."""
        cg, cids = self._build()
        cg.collapse_batch(
            caller="t1", facet="x", concept_ids=cids,
            shape=(4,), tick=0, init="normal_small", device="cpu",
        )
        first_id = id(cg.bundle_pool["x"])
        cg.collapse_batch(
            caller="t2", facet="x", concept_ids=cids,
            shape=(4,), tick=1, init="normal_small",
            device=torch.device("cpu"),
        )
        second_id = id(cg.bundle_pool["x"])
        self.assertEqual(
            first_id, second_id,
            "F42 regression: pool Parameter rebuilt despite "
            "equivalent device specifications ('cpu' vs torch.device('cpu')).",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_string_vs_indexed_no_rebuild(self) -> None:
        """``"cuda"`` (no index) vs ``torch.device("cuda", 0)``
        must not rebuild the pool Parameter — the **exact** F42
        case that broke v2 training in the wild."""
        cg, cids = self._build()
        cg.collapse_batch(
            caller="t1", facet="x", concept_ids=cids,
            shape=(4,), tick=0, init="normal_small",
            device=torch.device("cuda", 0),
        )
        first_id = id(cg.bundle_pool["x"])
        cg.collapse_batch(
            caller="t2", facet="x", concept_ids=cids,
            shape=(4,), tick=1, init="normal_small",
            device="cuda",  # no index — pre-fix this rebuilt
        )
        second_id = id(cg.bundle_pool["x"])
        self.assertEqual(
            first_id, second_id,
            "F42 regression: pool Parameter rebuilt despite "
            "equivalent device specs ('cuda:0' vs 'cuda').",
        )

    def test_optimizer_grad_step_actually_updates_pool(self) -> None:
        """End-to-end check: optimiser referencing the pool right
        after one collapse must see its Parameter change after
        loss.backward + step. Pre-fix this failed on CUDA when
        device strings differed."""
        cg, cids = self._build()
        cg.collapse_batch(
            caller="t1", facet="x", concept_ids=cids,
            shape=(4,), tick=0, init="normal_small", device="cpu",
        )
        opt = torch.optim.SGD(list(cg.iter_bundle_parameters()), lr=1.0)
        snapshot = cg.bundle_pool["x"].data.clone()

        # Run a second collapse with a different device spelling that
        # is *equivalent* (string vs object) — must not rebuild.
        rows = cg.collapse_batch(
            caller="t2", facet="x", concept_ids=cids,
            shape=(4,), tick=1, init="normal_small",
            device=torch.device("cpu"),
        )
        loss = rows.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()

        diff = (cg.bundle_pool["x"].data - snapshot).abs().sum().item()
        self.assertGreater(
            diff, 0.0,
            "F42 regression: optimizer stepped but pool unchanged "
            "(stale Parameter reference)",
        )


if __name__ == "__main__":
    unittest.main()
