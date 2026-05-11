"""scripts/bench_dense_pool.py — micro-benchmark the dense-pool fast path
vs the legacy ``for cid in concept_ids: collapse(...).as_tensor()`` loop.

Reports steps/s and wall time for both paths over identical workload, so
the speed-up claim of the upgrade plan (Tier A · 2-4×) is measurable on
any host.

Usage::

    python -m scripts.bench_dense_pool --N 30 --batch 128 --steps 500
    python -m scripts.bench_dense_pool --device cpu --N 30 --batch 128 --steps 100
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from pcm import ConceptGraph
from pcm.heads import ArithmeticHeadV2


def _legacy_loop(cg: ConceptGraph, head: ArithmeticHeadV2, ids: list[str]) -> torch.Tensor:
    """Legacy hot path: per-cid collapse + ``torch.stack``."""
    rows = []
    for cid in ids:
        cc = cg.concepts[cid].collapse(
            caller="ArithmeticHeadV2", facet="arithmetic_bias",
            shape=(head.bias_dim,), tick=0, init="normal_small",
            device=next(head.parameters()).device,
        )
        rows.append(cc.as_tensor())
    return torch.stack(rows, dim=0)


def _fast_path(cg: ConceptGraph, head: ArithmeticHeadV2, ids: list[str]) -> torch.Tensor:
    """New dense-pool path."""
    return cg.collapse_batch(
        caller="ArithmeticHeadV2", facet="arithmetic_bias",
        concept_ids=ids, shape=(head.bias_dim,), tick=0,
        init="normal_small", device=next(head.parameters()).device,
    )


def _bench(name: str, fn, steps: int, *args, sync: bool = False) -> float:
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        out = fn(*args)
        # touch output to make sure it's not optimised away
        _ = out.sum()
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"  {name:<24} {steps} steps in {dt:6.3f}s  →  {steps / dt:8.1f} steps/s")
    return dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=30, help="number of concepts")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"device={args.device}  N={args.N}  batch={args.batch}  steps={args.steps}")

    cg = ConceptGraph(initial_capacity=max(args.N, 32))
    cids = []
    for i in range(args.N):
        cid = f"concept:ans:{i}"
        cg.register_concept(node_id=cid, label=f"ANS_{i}", scope="BASE")
        cids.append(cid)

    head = ArithmeticHeadV2(embed_dim=128, bias_dim=64).to(args.device)

    # warm-up to materialise pool + RNG-init all rows
    head(
        torch.zeros(args.N, 128, device=args.device),
        torch.zeros(args.N, 128, device=args.device),
        torch.tensor([[1.0, 0.0]] * args.N, device=args.device),
        cids, cids, cg,
    )

    rng = torch.Generator().manual_seed(0)
    ids_a = [cids[int(torch.randint(args.N, (1,), generator=rng).item())] for _ in range(args.batch)]
    ids_b = [cids[int(torch.randint(args.N, (1,), generator=rng).item())] for _ in range(args.batch)]

    # warm-up cuda graph
    _ = _fast_path(cg, head, ids_a)
    _ = _legacy_loop(cg, head, ids_a)

    print()
    print("─── forward only ──────────────────────────────────────────────")
    t_legacy = _bench("legacy loop+stack", _legacy_loop, args.steps, cg, head, ids_a, sync=True)
    t_fast = _bench("dense-pool F.embedding", _fast_path, args.steps, cg, head, ids_a, sync=True)
    print(f"  speed-up: {t_legacy / max(t_fast, 1e-9):.2f}×")

    print()
    print("─── full forward+backward ─────────────────────────────────────")

    def forward_backward_legacy() -> torch.Tensor:
        ba = _legacy_loop(cg, head, ids_a)
        bb = _legacy_loop(cg, head, ids_b)
        op = torch.tensor([[1.0, 0.0]] * args.batch, device=args.device)
        x = torch.cat([ba, bb, op], -1)
        h = F.relu(head.fc1(x))
        h = F.relu(head.fc2(h))
        out = head.fc3(h)
        loss = out.pow(2).mean()
        loss.backward()
        for p in head.parameters():
            p.grad = None
        for pool in cg.bundle_pool.values():
            pool.grad = None
        return loss

    def forward_backward_fast() -> torch.Tensor:
        ba = _fast_path(cg, head, ids_a)
        bb = _fast_path(cg, head, ids_b)
        op = torch.tensor([[1.0, 0.0]] * args.batch, device=args.device)
        x = torch.cat([ba, bb, op], -1)
        h = F.relu(head.fc1(x))
        h = F.relu(head.fc2(h))
        out = head.fc3(h)
        loss = out.pow(2).mean()
        loss.backward()
        for p in head.parameters():
            p.grad = None
        for pool in cg.bundle_pool.values():
            pool.grad = None
        return loss

    t_legacy_fb = _bench("legacy fb", forward_backward_legacy, args.steps, sync=True)
    t_fast_fb = _bench("dense-pool fb", forward_backward_fast, args.steps, sync=True)
    print(f"  speed-up: {t_legacy_fb / max(t_fast_fb, 1e-9):.2f}×")

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"\nPeak VRAM allocated: {peak_mb:.1f} MB")


if __name__ == "__main__":
    main()
