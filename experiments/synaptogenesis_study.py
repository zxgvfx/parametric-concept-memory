"""experiments/synaptogenesis_study.py — Tier-B developmental pruning curve.

Trains a Tier-B-enabled PCM on the §4 number arithmetic task with an
**over-allocated** concept pool: N_used = 7 task-relevant numerals
(``concept:ans:{1..7}``) plus K_extra "phantom" concepts that never
appear in any training batch. Records the per-slot gate trajectory
across training and writes:

- ``outputs/synaptogenesis/summary.json`` — final gate stats per facet,
  used vs phantom population means, fraction-pruned timeline.
- ``outputs/synaptogenesis/gate_trajectory.png`` — bionic developmental
  curve: phantom-slot mean gate vs used-slot mean gate over training
  steps. Compare to Frontiers Cell Dev Bio 2025 / bioRxiv 2025/01 for
  qualitative match (over-production peak then pruning decay).

Runs in <60s on a single 8GB GPU. ``--smoke`` reduces step count to ~10s
for CI.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from pcm import ConceptGraph, gate
from pcm.heads import ArithmeticHeadV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run(
    n_used: int = 7,
    n_phantom: int = 24,
    epochs: int = 12,
    steps_per_epoch: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    l0_lambda: float = 5e-2,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    cg = ConceptGraph(initial_capacity=max(64, n_used + n_phantom),
                      growth_factor=2.0, max_capacity=512)
    cids_used: list[str] = []
    for n in range(1, n_used + 1):
        cid = f"concept:ans:{n}"
        cg.register_concept(node_id=cid, label=f"ANS_{n}", scope="BASE",
                            provenance=f"synaptogenesis_used:n={n}")
        cids_used.append(cid)
    cids_phantom: list[str] = []
    for j in range(n_phantom):
        cid = f"concept:phantom:{j}"
        cg.register_concept(node_id=cid, label=f"PHANTOM_{j}", scope="BASE",
                            provenance="synaptogenesis_phantom")
        cids_phantom.append(cid)

    head = ArithmeticHeadV2(embed_dim=128, bias_dim=64).to(DEVICE)

    # warm-up touches **only used cids**. Phantom slots stay
    # "undifferentiated": their gate is never opened from the init
    # logit (sigmoid ≈ 0.1), and their pool row stays at 0. This is the
    # neonatal cortex state in the bionic story (see docs/PCM_BIO_PREALLOC_UPGRADE.md).
    with torch.no_grad():
        zeros = torch.zeros(n_used, 128, device=DEVICE)
        op = torch.tensor([[1.0, 0.0]] * n_used, device=DEVICE)
        head(zeros, zeros, op, cids_used, cids_used, cg)
    cg.bundles_to(torch.device(DEVICE))

    gate.attach_gates(cg)

    # random-orthogonal centroids (paper §4 setup)
    g_seed = torch.Generator().manual_seed(seed + 1)
    A = torch.randn(128, n_used, generator=g_seed)
    Q, _ = torch.linalg.qr(A)
    centroids = F.normalize(Q.t(), dim=-1).to(DEVICE)

    params = (
        list(head.parameters())
        + list(cg.iter_bundle_parameters())
        + list(cg.iter_gate_parameters())
    )
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    cg.register_optimizer(opt)

    used_slots = [cg.cid_to_slot[c] for c in cids_used]
    phantom_slots = [cg.cid_to_slot[c] for c in cids_phantom]

    trajectory: list[dict] = []
    t0 = time.time()

    def _snapshot(epoch: int, step: int) -> None:
        g = cg.bundle_gates["arithmetic_bias"]
        probs = torch.sigmoid(g.detach())
        trajectory.append({
            "epoch": epoch,
            "step": step,
            "used_mean": float(probs[used_slots].mean().item()),
            "used_min": float(probs[used_slots].min().item()),
            "phantom_mean": float(probs[phantom_slots].mean().item()),
            "phantom_max": float(probs[phantom_slots].max().item()),
            "frac_pruned": float((probs < 0.05).float().mean().item()),
        })

    # capture the initial peak (post warm-up, pre-training)
    _snapshot(epoch=0, step=0)

    for epoch in range(1, epochs + 1):
        head.train()
        for step in range(steps_per_epoch):
            # sample only used-cid pairs (phantom slots receive zero gradient
            # for the data loss → only L0 acts on them, classic pruning)
            a_l = [int(torch.randint(1, n_used + 1, (1,), generator=rng).item())
                   for _ in range(batch_size)]
            b_l = []
            c_l = []
            op_l = []
            for a in a_l:
                if a + 1 <= n_used:
                    b = int(torch.randint(0, n_used + 1 - a, (1,), generator=rng).item())
                    if b == 0:
                        b = 1
                    op_l.append("add"); c = a + b
                else:
                    b = int(torch.randint(0, a, (1,), generator=rng).item())
                    if b == 0:
                        b = 1
                    op_l.append("sub"); c = a - b
                b_l.append(b); c_l.append(c)
            ids_a = [f"concept:ans:{a}" for a in a_l]
            ids_b = [f"concept:ans:{b}" for b in b_l]
            op_oh = torch.tensor(
                [[1.0, 0.0] if o == "add" else [0.0, 1.0] for o in op_l],
                device=DEVICE,
            )
            tgt = torch.tensor([c - 1 for c in c_l], device=DEVICE)
            dummy = torch.zeros(batch_size, 128, device=DEVICE)
            pred = head(dummy, dummy, op_oh, ids_a, ids_b, cg,
                        tick=epoch * 10000 + step)
            data_loss = F.cross_entropy(pred @ centroids.t(), tgt)
            l0 = gate.gate_l0_loss(cg)
            loss = data_loss + l0_lambda * l0
            opt.zero_grad(); loss.backward(); opt.step()

        _snapshot(epoch=epoch, step=epochs * steps_per_epoch)

    head.eval()
    # final eval accuracy (paper §4 metric)
    with torch.no_grad():
        hits = total = 0
        for _ in range(40):
            a_l = [int(torch.randint(1, n_used + 1, (1,), generator=rng).item())
                   for _ in range(20)]
            b_l = []
            c_l = []
            op_l = []
            for a in a_l:
                if a + 1 <= n_used:
                    b = int(torch.randint(0, n_used + 1 - a, (1,), generator=rng).item())
                    if b == 0:
                        b = 1
                    op_l.append("add"); c = a + b
                else:
                    b = int(torch.randint(0, a, (1,), generator=rng).item())
                    if b == 0:
                        b = 1
                    op_l.append("sub"); c = a - b
                b_l.append(b); c_l.append(c)
            ids_a = [f"concept:ans:{a}" for a in a_l]
            ids_b = [f"concept:ans:{b}" for b in b_l]
            op_oh = torch.tensor(
                [[1.0, 0.0] if o == "add" else [0.0, 1.0] for o in op_l],
                device=DEVICE,
            )
            tgt = torch.tensor([c - 1 for c in c_l], device=DEVICE)
            dummy = torch.zeros(20, 128, device=DEVICE)
            pred = head(dummy, dummy, op_oh, ids_a, ids_b, cg)
            hits += (pred @ centroids.t()).argmax(-1).eq(tgt).sum().item()
            total += 20

    dt = time.time() - t0
    final_status = gate.gate_status(cg)
    return {
        "config": {
            "n_used": n_used, "n_phantom": n_phantom,
            "epochs": epochs, "steps_per_epoch": steps_per_epoch,
            "batch_size": batch_size, "lr": lr, "l0_lambda": l0_lambda,
            "seed": seed, "device": DEVICE,
        },
        "wall_s": dt,
        "final_acc": hits / max(total, 1),
        "final_status": final_status,
        "used_slots": used_slots,
        "phantom_slots": phantom_slots,
        "trajectory": trajectory,
    }


def write_plot(summary: dict, out_path: Path) -> None:
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    traj = summary["trajectory"]
    xs = [t["epoch"] for t in traj]
    used = [t["used_mean"] for t in traj]
    phantom = [t["phantom_mean"] for t in traj]
    pruned = [t["frac_pruned"] for t in traj]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(xs, used, "-o", label="used slots (gate mean)", color="tab:blue")
    ax1.plot(xs, phantom, "-s", label="phantom slots (gate mean)", color="tab:orange")
    ax1.set_xlabel("training epoch")
    ax1.set_ylabel("sigmoid(gate)")
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(loc="center right")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, pruned, "--", label="fraction pruned", color="tab:green")
    ax2.set_ylabel("fraction with gate < 0.05")
    ax2.set_ylim(-0.02, 1.05)

    ax1.set_title("PCM Tier-B: synaptogenesis ↔ pruning trajectory")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="short run (~10s)")
    ap.add_argument("--out", type=Path, default=Path("outputs/synaptogenesis"))
    ap.add_argument("--n-used", type=int, default=7)
    ap.add_argument("--n-phantom", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.smoke:
        epochs, steps = 4, 60
    else:
        epochs, steps = 12, 100

    summary = run(
        n_used=args.n_used, n_phantom=args.n_phantom,
        epochs=epochs, steps_per_epoch=steps, seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_plot(summary, args.out / "gate_trajectory.png")
    print(f"final acc        : {summary['final_acc']:.3f}")
    print(f"used   gate mean : {summary['final_status']['arithmetic_bias']['mean']:.3f}")
    print(f"used   p_open    : {summary['final_status']['arithmetic_bias']['p_open']:.3f}")
    print(f"phantom p_pruned : {summary['final_status']['arithmetic_bias']['p_pruned']:.3f}")
    print(f"wall: {summary['wall_s']:.1f}s")
    print(f"wrote {args.out / 'summary.json'}")
    print(f"wrote {args.out / 'gate_trajectory.png'}")


if __name__ == "__main__":
    main()
