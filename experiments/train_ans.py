"""train_ans.py — D88 Stage-0: 训练 NumerosityEncoder (ANS).

产出: outputs/ans_encoder/{final.pt, best.pt}

  - encoder_state   : state_dict of NumerosityEncoder
  - ds_cfg          : DatasetConfig dict
  - epoch           : int
  - spatial_invariance : float (hold-out invariance score)
  - weber           : {sim_by_distance_k: dict, monotonic_decreasing: bool}

Usage:
    python -m experiments.train_ans --epochs 30 --out outputs/ans_encoder
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import torch

from pcm.heads.numerosity_encoder import (
    DatasetConfig,
    NumerosityEncoder,
    contrastive_ordinal_loss,
    sample_batch,
)


def _eval_spatial_invariance(enc: NumerosityEncoder, cfg: DatasetConfig, n_pairs: int = 200) -> float:
    """For each count n, sample 2 different layouts → cos(emb1, emb2) should be high."""
    enc.eval()
    sims = []
    rng = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for _ in range(n_pairs):
            n = int(torch.randint(cfg.n_min, cfg.n_max + 1, (1,), generator=rng).item())
            x1, _ = sample_batch(1, cfg, rng=rng)
            x2, _ = sample_batch(1, cfg, rng=rng)
            # force same count
            from pcm.heads.numerosity_encoder import generate_dot_canvas
            x1 = generate_dot_canvas(n, cfg, rng).unsqueeze(0)
            x2 = generate_dot_canvas(n, cfg, rng).unsqueeze(0)
            e1 = enc(x1)
            e2 = enc(x2)
            sims.append((e1 * e2).sum(-1).item())
    return sum(sims) / max(len(sims), 1)


def _eval_weber(enc: NumerosityEncoder, cfg: DatasetConfig) -> dict:
    """For each dk = |n_a - n_b| in 1..6, average cos(e_a, e_b).

    Expect cos to decrease monotonically with dk (Weber's fraction / ordinal coding).
    """
    from pcm.heads.numerosity_encoder import generate_dot_canvas
    enc.eval()
    rng = torch.Generator().manual_seed(11)
    sim_by_k: dict[int, float] = {}
    with torch.no_grad():
        for dk in range(1, cfg.n_max - cfg.n_min + 1):
            sims = []
            for _ in range(80):
                a = int(torch.randint(cfg.n_min, cfg.n_max + 1 - dk, (1,), generator=rng).item())
                b = a + dk
                x1 = generate_dot_canvas(a, cfg, rng).unsqueeze(0)
                x2 = generate_dot_canvas(b, cfg, rng).unsqueeze(0)
                sims.append(((enc(x1) * enc(x2)).sum(-1)).item())
            sim_by_k[dk] = sum(sims) / len(sims)
    monotonic = all(
        sim_by_k[k] >= sim_by_k[k + 1] - 1e-4
        for k in range(1, cfg.n_max - cfg.n_min)
    )
    return {"sim_by_distance_k": sim_by_k, "monotonic_decreasing": monotonic}


def train(
    cfg: DatasetConfig,
    out_dir: Path,
    epochs: int = 30,
    steps_per_epoch: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    temperature: float = 0.1,
    ordinal_weight: float = 0.3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    enc = NumerosityEncoder().to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-4)

    best_inv = -math.inf
    for epoch in range(1, epochs + 1):
        enc.train()
        ep_loss = 0.0
        for _ in range(steps_per_epoch):
            imgs, counts = sample_batch(batch_size, cfg, rng=rng)
            imgs, counts = imgs.to(device), counts.to(device)
            emb = enc(imgs)
            loss, _ = contrastive_ordinal_loss(
                emb, counts, temperature=temperature, ordinal_weight=ordinal_weight
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        inv = _eval_spatial_invariance(enc.to("cpu"), cfg, n_pairs=100)
        enc.to(device)
        print(f"[ans][ep {epoch:02d}] loss={ep_loss/steps_per_epoch:.4f}  spatial_inv={inv:.4f}")

        ckpt = {
            "encoder_state": enc.state_dict(),
            "ds_cfg": dataclasses.asdict(cfg),
            "epoch": epoch,
            "spatial_invariance": inv,
            "weber": {},
        }
        torch.save(ckpt, out_dir / "final.pt")
        if inv > best_inv:
            best_inv = inv
            torch.save(ckpt, out_dir / "best.pt")

    weber = _eval_weber(enc.to("cpu"), cfg)
    ckpt["weber"] = weber
    torch.save(ckpt, out_dir / "final.pt")
    (out_dir / "history.json").write_text(
        json.dumps({"spatial_invariance": best_inv, "weber": weber}, indent=2)
    )
    print(f"[ans] done. best spatial_inv={best_inv:.4f}  weber monotonic={weber['monotonic_decreasing']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=Path, default=Path("outputs/ans_encoder"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    cfg = DatasetConfig()
    train(
        cfg,
        out_dir=a.out,
        epochs=a.epochs,
        steps_per_epoch=a.steps_per_epoch,
        batch_size=a.batch_size,
        lr=a.lr,
        device=a.device,
        seed=a.seed,
    )


if __name__ == "__main__":
    main()
