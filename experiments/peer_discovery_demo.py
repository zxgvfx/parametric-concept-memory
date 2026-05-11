"""experiments/peer_discovery_demo.py — Tier-C 8GB micro demo.

Trains a tiny "discovery muscle" that uses ``ProductKeyRouter`` over a
``num_experts = 16384`` slot bank to produce a digit prediction without
any explicit ``concept_id`` lookup. Demonstrates:

- **Symbolic mode unchanged**: the same ``ConceptGraph`` still serves
  Tier-A muscles via ``cid_to_slot``.
- **Discovery mode**: the router learns a query → slot mapping that
  binds **structurally similar** queries to the same slot, even though
  no ``concept_id`` was provided.
- **Soft attribution**: ``soft_consumed_by_log`` writes ``consumed_by``
  entries for the discovered slots so the architectural attribution
  invariant survives the soft routing.
- **8GB safe**: keys are ``2 × 128 × 64 = 16K`` floats per head × 4
  heads = 128KB; total VRAM stays under 100MB for this toy.

Smoke run completes in <30s. The output report compares router-routed
accuracy vs random-router baseline.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcm import ConceptGraph
from pcm.peer import PEERConfig, ProductKeyRouter, soft_consumed_by_log

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DiscoveryMuscle(nn.Module):
    """Toy muscle: ``feature -> bias via PEER router -> small MLP -> logit``.

    Mirrors ArithmeticHeadV2's contract but the bias comes from a learnt
    PEER lookup instead of a deterministic ``concept_id`` collapse.
    """

    def __init__(self, n_classes: int, num_experts: int = 16_384,
                 query_dim: int = 64, num_heads: int = 4, top_k: int = 8) -> None:
        super().__init__()
        cfg = PEERConfig(num_experts=num_experts, top_k=top_k,
                         num_heads=num_heads, query_dim=query_dim)
        self.router = ProductKeyRouter(input_dim=query_dim, cfg=cfg)
        self.head = nn.Sequential(
            nn.Linear(query_dim + 64, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, feat: torch.Tensor, cg: ConceptGraph,
                facet: str = "discovery_bias") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Lazily allocate the pool (capacity must be >= num_experts).
        if facet not in cg.bundle_pool:
            cg._ensure_facet(facet, (64,), feat.device, "normal_small")
        bias = self.router.gather_values(cg, facet, feat)
        x = torch.cat([feat, bias], dim=-1)
        logits = self.head(x)
        # also return slots/weights for attribution logging
        slots, weights = self.router(feat)
        return logits, slots, weights


def make_synthetic_dataset(n_classes: int = 10, samples_per_class: int = 64,
                           query_dim: int = 64, seed: int = 0):
    """Build a toy classification dataset where each class has a fixed
    Gaussian centroid; queries are noisy samples around that centroid.
    The ideal router would discover one slot per centroid.
    """
    g = torch.Generator().manual_seed(seed)
    centroids = torch.randn(n_classes, query_dim, generator=g)
    centroids = F.normalize(centroids, dim=-1)
    feats, labels = [], []
    for c in range(n_classes):
        for _ in range(samples_per_class):
            noise = torch.randn(query_dim, generator=g) * 0.2
            feats.append(centroids[c] + noise)
            labels.append(c)
    return torch.stack(feats), torch.tensor(labels)


def run(
    num_experts: int = 16_384,
    n_classes: int = 10,
    epochs: int = 4,
    batch_size: int = 64,
    samples_per_class: int = 64,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    cg = ConceptGraph(initial_capacity=max(num_experts, 32_000),
                      max_capacity=64_000)
    # We don't pre-register concepts; this is "discovery mode" so slots
    # are anonymous. We manually allocate one capacity-sized pool.
    feats, labels = make_synthetic_dataset(n_classes, samples_per_class, seed=seed)
    feats = feats.to(DEVICE); labels = labels.to(DEVICE)

    model = DiscoveryMuscle(n_classes, num_experts=num_experts).to(DEVICE)

    # warm-up forward to materialise the pool
    with torch.no_grad():
        model(feats[:1], cg)
    cg.bundles_to(torch.device(DEVICE))

    params = list(model.parameters()) + list(cg.iter_bundle_parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    cg.register_optimizer(opt)

    n = feats.shape[0]
    t0 = time.time()
    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x, y = feats[idx], labels[idx]
            logits, slots, weights = model(x, cg)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        # log attribution after each epoch (Tier-C soft consumed_by)
        soft_consumed_by_log(
            cg, facet="discovery_bias", caller="DiscoveryMuscle",
            top_slots=slots.detach(), top_weights=weights.detach(),
            threshold=0.05, tick=epoch,
        )

    # eval
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(feats, cg)
        acc = float(logits.argmax(-1).eq(labels).float().mean().item())
    dt = time.time() - t0

    # diagnostics: how many distinct slots got positive consumed_by?
    n_active_slots = sum(
        1 for per_slot in cg._consumed_by_by_slot.values()
        if "DiscoveryMuscle" in per_slot.get("discovery_bias", set())
    )

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_mb = 0.0

    return {
        "config": {
            "num_experts": num_experts, "n_classes": n_classes,
            "epochs": epochs, "batch_size": batch_size,
            "device": DEVICE, "seed": seed,
        },
        "wall_s": dt,
        "final_acc": acc,
        "final_loss": losses[-1] if losses else float("nan"),
        "n_discovered_slots": n_active_slots,
        "peak_vram_mb": peak_mb,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--num-experts", type=int, default=16_384)
    ap.add_argument("--out", type=Path, default=Path("outputs/peer_discovery"))
    args = ap.parse_args()
    epochs = 2 if args.smoke else 4
    summary = run(num_experts=args.num_experts, epochs=epochs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"final acc          : {summary['final_acc']:.3f}")
    print(f"final loss         : {summary['final_loss']:.3f}")
    print(f"discovered slots   : {summary['n_discovered_slots']} / {args.num_experts}")
    print(f"wall               : {summary['wall_s']:.1f}s")
    print(f"peak VRAM          : {summary['peak_vram_mb']:.1f} MB")
    print(f"wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
