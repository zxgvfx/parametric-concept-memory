"""compositional_number_study.py — D93a 原型: 用 10 个 digit atom + 少量
position atom 组合任意数字的 bundle, 测试真·外推到训练中从未塌缩过的
数字上.

与 D91 flat ConceptGraph 的核心区别:

  D91 (baseline):
      concept:ans:20 是独立 ConceptNode, 独立 64 维 bundle
      → N=100 需要 100 个 ConceptNode.
      → 未注册数字 (e.g. 150) 根本没 bundle, **无法外推**.

  D93a (本脚本):
      concept:digit:0..9      → 10 个 ConceptNode
      concept:pos:0..max_pos-1 → 3 个 ConceptNode (units, tens, hundreds)
      bundle(N) = Σ_i digit_bundle[d_i] ⊙ pos_bundle[i]   (Hadamard)
      → 13 个 atom, 表达 0..999 所有数字
      → 训练时只见过 N∈[1,99] 的 a+b 运算, 测试时仍能"合成" bundle(150),
        bundle(237) 等 — 架构首次支持**真·数字外推**.

Hadamard 必须, 因为加性 `digit_b+pos_b` permutation-invariant:
  bundle(50) = (digit_5+pos_0) + (digit_0+pos_1) + (digit_0+pos_2)
  bundle(5)  = (digit_5+pos_0) + (digit_0+pos_1) + (digit_0+pos_2)
  两者相等 (因为 pad 了 digit_0). 必须 position-dependent scaling 才能
  打破对称.

Target loss: self-referential cross-entropy
  pred = head(compose(a), compose(b), op_onehot)
  candidates = [compose(c) for c in 0..C_max]
  loss = CE(cosine_similarity(pred, candidates) * temp, true_c)

好处: candidate 集合可扩展到任意 C_eval_max, 不需要训练时见过 target c
就能评估 — 这正是外推测试需要的.

运行:

  python -m experiments.compositional_number_study --smoke
  python -m experiments.compositional_number_study --n-seeds 3
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BIAS_DIM = 64


# ────────────────────────────────────────────────────────────────────────
# D93a composer
# ────────────────────────────────────────────────────────────────────────


class DigitPosComposer(nn.Module):
    """Position-by-slot concatenation: bundle(n) 把每个 digit position 的
    digit_bundle 填到 bias_dim 的一个固定 block.

        bundle_dim = max_pos × per_digit_dim     (per_digit_dim = bias_dim // max_pos)
        bundle(n)[i*block : (i+1)*block] = digit_bundles[d_i(n)]

    Structural properties:
      - Position 由 slot 索引强制区分 (permutation symmetry 天然打破).
      - digit_bundles 共享 across positions → 强迫 position-invariant 的
        digit 语义 (5 at units vs 5 at tens 表示同样的"五").
      - 外推时只要训练覆盖过所有 (d, p) 组合, 就能直接把 digit_bundles
        插到新 position 上组合出未见过的 number.
    """

    def __init__(self, n_digits: int = 10, max_pos: int = 3,
                 per_digit_dim: int = 24, seed: int = 0) -> None:
        super().__init__()
        self.n_digits = n_digits
        self.max_pos = max_pos
        self.per_digit_dim = per_digit_dim
        self.bias_dim = max_pos * per_digit_dim
        g = torch.Generator().manual_seed(seed)
        self.digit_bundles = nn.Parameter(
            torch.randn(n_digits, per_digit_dim, generator=g) * 0.1
        )

    def compose(self, ns) -> torch.Tensor:
        """ns: list[int] 或 1D long Tensor. Returns (B, bias_dim)."""
        if isinstance(ns, list):
            ns = torch.tensor(ns, device=self.digit_bundles.device, dtype=torch.long)
        ns = ns.long()
        B = ns.shape[0]
        blocks = []
        base = 1
        for i in range(self.max_pos):
            digits_i = (ns // base) % 10
            blocks.append(self.digit_bundles[digits_i])   # (B, per_digit_dim)
            base *= 10
        return torch.cat(blocks, dim=-1)                   # (B, bias_dim)


class CompHead(nn.Module):
    """回归 head: 输出 bias_dim 维向量, 与 candidate bundle 做 cosine 匹配.

    Flat 版本: 全连接 MLP, 对 slot-wise pattern 没有 equivariance.
    """

    def __init__(self, bias_dim: int, hidden: int = 128, n_ops: int = 2) -> None:
        super().__init__()
        in_dim = 2 * bias_dim + n_ops
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, bias_dim)

    def forward(self, bias_a: torch.Tensor, bias_b: torch.Tensor,
                op: torch.Tensor) -> torch.Tensor:
        x = torch.cat([bias_a, bias_b, op], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class SlotEquivariantHead(nn.Module):
    """Slot-共享 MLP + 双向 carry pass.

    对每个 slot i 独立运行相同的 MLP, 输入: (digit_a_i, digit_b_i, op, carry_in).
    输出: (digit_c_i, carry_out). Carry 从低位向高位 ripple, 类似硬件加法器.

    这种架构对 digit-pos 有完整 equivariance: slot i 学到的运算规则对所有
    slot 都适用 → digit_bundle[d] 在任意新 slot 下行为一致.
    """

    def __init__(self, per_digit_dim: int, max_pos: int,
                 hidden: int = 64, n_ops: int = 2, carry_dim: int = 8) -> None:
        super().__init__()
        self.per_digit_dim = per_digit_dim
        self.max_pos = max_pos
        self.carry_dim = carry_dim
        in_dim = 2 * per_digit_dim + n_ops + carry_dim
        self.slot_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, per_digit_dim + carry_dim),
        )

    def forward(self, bias_a: torch.Tensor, bias_b: torch.Tensor,
                op: torch.Tensor) -> torch.Tensor:
        B = bias_a.shape[0]
        device = bias_a.device
        carry = torch.zeros(B, self.carry_dim, device=device)
        outs = []
        for i in range(self.max_pos):
            a_i = bias_a[:, i * self.per_digit_dim:(i + 1) * self.per_digit_dim]
            b_i = bias_b[:, i * self.per_digit_dim:(i + 1) * self.per_digit_dim]
            x = torch.cat([a_i, b_i, op, carry], dim=-1)
            y = self.slot_mlp(x)
            digit_out = y[:, :self.per_digit_dim]
            carry = y[:, self.per_digit_dim:]
            outs.append(digit_out)
        return torch.cat(outs, dim=-1)


# ────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────


OPS = ["add", "sub"]
OP_IDX = {op: i for i, op in enumerate(OPS)}


def _valid_triples(a_range: tuple[int, int], b_range: tuple[int, int],
                   ops: list[str], c_max: int) -> list[tuple[int, int, str, int]]:
    triples = []
    for a in range(*a_range):
        for b in range(*b_range):
            for op in ops:
                if op == "add":
                    c = a + b
                elif op == "sub":
                    if a < b: continue
                    c = a - b
                else:
                    raise ValueError(op)
                if 0 <= c <= c_max:
                    triples.append((a, b, op, c))
    return triples


def op_onehot_tensor(ops: list[str], device: str) -> torch.Tensor:
    t = torch.zeros(len(ops), len(OPS), device=device)
    for i, op in enumerate(ops):
        t[i, OP_IDX[op]] = 1.0
    return t


# ────────────────────────────────────────────────────────────────────────
# Training & eval
# ────────────────────────────────────────────────────────────────────────


def train_d93a(
    seed: int,
    train_triples: list,
    c_max_train: int,
    *,
    epochs: int = 40,
    steps_per_epoch: int = 200,
    batch_size: int = 64,
    lr: float = 3e-3,
    temp: float = 10.0,
    max_pos: int = 3,
    head_type: str = "flat",
) -> tuple[DigitPosComposer, nn.Module, list[float]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    composer = DigitPosComposer(max_pos=max_pos, seed=seed).to(DEVICE)
    if head_type == "flat":
        head = CompHead(bias_dim=composer.bias_dim).to(DEVICE)
    elif head_type == "slot":
        head = SlotEquivariantHead(
            per_digit_dim=composer.per_digit_dim, max_pos=max_pos
        ).to(DEVICE)
    else:
        raise ValueError(head_type)
    params = list(composer.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    all_cs_train = torch.arange(c_max_train + 1, device=DEVICE)  # 0..c_max_train
    loss_history = []

    for epoch in range(epochs):
        composer.train(); head.train()
        epoch_losses = []
        for step in range(steps_per_epoch):
            batch = [train_triples[rng.randrange(len(train_triples))]
                     for _ in range(batch_size)]
            a_t = torch.tensor([t[0] for t in batch], device=DEVICE, dtype=torch.long)
            b_t = torch.tensor([t[1] for t in batch], device=DEVICE, dtype=torch.long)
            op_l = [t[2] for t in batch]
            c_t = torch.tensor([t[3] for t in batch], device=DEVICE, dtype=torch.long)

            bias_a = composer.compose(a_t)
            bias_b = composer.compose(b_t)
            op_oh = op_onehot_tensor(op_l, DEVICE)
            pred = head(bias_a, bias_b, op_oh)                  # (B, bias_dim)

            candidates = composer.compose(all_cs_train)          # (C+1, bias_dim)
            pred_n = F.normalize(pred, dim=-1)
            cand_n = F.normalize(candidates, dim=-1)
            logits = pred_n @ cand_n.t() * temp                  # (B, C+1)
            loss = F.cross_entropy(logits, c_t)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_losses.append(loss.item())
        loss_history.append(sum(epoch_losses) / len(epoch_losses))
    return composer, head, loss_history


@torch.no_grad()
def eval_triples(
    composer: DigitPosComposer,
    head: CompHead,
    triples: list,
    c_max_eval: int,
    batch_size: int = 128,
) -> dict:
    composer.eval(); head.eval()
    all_cs_eval = torch.arange(c_max_eval + 1, device=DEVICE)
    candidates = composer.compose(all_cs_eval)
    cand_n = F.normalize(candidates, dim=-1)

    by_op: dict[str, list[int]] = {op: [] for op in OPS}
    err_dists: list[int] = []

    for i in range(0, len(triples), batch_size):
        batch = triples[i: i + batch_size]
        a_t = torch.tensor([t[0] for t in batch], device=DEVICE, dtype=torch.long)
        b_t = torch.tensor([t[1] for t in batch], device=DEVICE, dtype=torch.long)
        op_l = [t[2] for t in batch]
        c_t = torch.tensor([t[3] for t in batch], device=DEVICE, dtype=torch.long)
        bias_a = composer.compose(a_t); bias_b = composer.compose(b_t)
        op_oh = op_onehot_tensor(op_l, DEVICE)
        pred = head(bias_a, bias_b, op_oh)
        pred_n = F.normalize(pred, dim=-1)
        sim = pred_n @ cand_n.t()
        pred_c = sim.argmax(-1)
        hits = pred_c.eq(c_t)
        err = (pred_c - c_t).abs()
        for h, er, op in zip(hits.cpu().tolist(), err.cpu().tolist(), op_l):
            by_op[op].append(int(h))
            if not h:
                err_dists.append(er)

    per_op = {op: (sum(hs) / max(len(hs), 1)) if hs else float("nan")
              for op, hs in by_op.items()}
    overall = sum(sum(hs) for hs in by_op.values()) / max(
        sum(len(hs) for hs in by_op.values()), 1)
    return {
        "per_op": per_op,
        "overall": overall,
        "mean_err_dist": sum(err_dists) / len(err_dists) if err_dists else 0.0,
        "n": sum(len(hs) for hs in by_op.values()),
    }


# ────────────────────────────────────────────────────────────────────────
# Main experiment runner
# ────────────────────────────────────────────────────────────────────────


def run_seed(seed: int, *, epochs: int, steps_per_epoch: int,
             ood_ratio: float = 0.15, extended: bool = False,
             head_type: str = "flat") -> dict:
    """两种 regime:

      default (extended=False): train a,b ∈ [1,99]. 暴露 "digit-pos coverage"
        缺失问题: pos=2 只见 d=0..1, L1/L2 崩溃.
      extended (extended=True): train a,b ∈ [1,999] (所有 d×p 都见过),
        test a,b ∈ [1000, 4999]. 这是 D93a 架构真正承诺的测试.
    """
    rng = random.Random(seed)

    if not extended:
        in_range_triples = _valid_triples((1, 100), (1, 100), OPS, c_max=198)
        rng.shuffle(in_range_triples)
        n_ood = int(len(in_range_triples) * ood_ratio)
        l0_triples = in_range_triples[:n_ood]
        train_triples = in_range_triples[n_ood:]

        c_max_train, c_max_eval = 198, 800
        max_pos = 3

        l1_a_big = _valid_triples((100, 300), (1, 100), OPS, c_max=c_max_eval)
        l1_b_big = _valid_triples((1, 100), (100, 300), OPS, c_max=c_max_eval)
        l1_triples = l1_a_big + l1_b_big
        rng.shuffle(l1_triples); l1_triples = l1_triples[:1000]
        l2_triples = _valid_triples((100, 300), (100, 300), OPS, c_max=c_max_eval)
        rng.shuffle(l2_triples); l2_triples = l2_triples[:1000]
    else:
        max_pos = 4
        c_max_train, c_max_eval = 1998, 9998
        train_a_max, test_a_max = 999, 4999

        sampled = set()
        train_triples = []
        while len(train_triples) < 40000:
            a = rng.randint(1, train_a_max)
            b = rng.randint(1, train_a_max)
            key = (a, b)
            if key in sampled:
                continue
            sampled.add(key)
            for op in OPS:
                if op == "add":
                    c = a + b
                else:
                    if a < b: continue
                    c = a - b
                if c <= c_max_train:
                    train_triples.append((a, b, op, c))

        rng.shuffle(train_triples)
        n_ood = int(len(train_triples) * ood_ratio)
        l0_triples = train_triples[:n_ood]
        train_triples = train_triples[n_ood:]

        def _gen_range(a_range, b_range, target_n=1000):
            out = []
            tries = 0
            while len(out) < target_n and tries < target_n * 40:
                tries += 1
                a = rng.randint(*a_range)
                b = rng.randint(*b_range)
                op = rng.choice(OPS)
                if op == "add":
                    c = a + b
                    if c > c_max_eval: continue
                else:
                    if a < b: continue
                    c = a - b
                out.append((a, b, op, c))
            return out

        l1_triples = _gen_range((1000, test_a_max + 1), (1, train_a_max + 1), 500)
        l1_triples += _gen_range((1, train_a_max + 1), (1000, test_a_max + 1), 500)
        l2_triples = _gen_range((1000, test_a_max + 1), (1000, test_a_max + 1), 1000)

    t0 = time.time()
    composer, head, loss_hist = train_d93a(
        seed, train_triples, c_max_train,
        epochs=epochs, steps_per_epoch=steps_per_epoch, max_pos=max_pos,
        head_type=head_type,
    )
    dt = time.time() - t0

    # Training acc (sanity) — subsample train
    rng.shuffle(train_triples)
    train_eval = eval_triples(composer, head, train_triples[:1000], c_max_train)
    l0_eval = eval_triples(composer, head, l0_triples, c_max_train)
    l1_eval = eval_triples(composer, head, l1_triples, c_max_eval)
    l2_eval = eval_triples(composer, head, l2_triples, c_max_eval)

    return {
        "seed": seed,
        "n_train": len(train_triples), "n_ood_l0": len(l0_triples),
        "n_ood_l1": len(l1_triples),   "n_ood_l2": len(l2_triples),
        "train": train_eval,
        "L0_in_range_pair_OOD": l0_eval,
        "L1_one_out_of_range": l1_eval,
        "L2_both_out_of_range": l2_eval,
        "final_loss": loss_hist[-1],
        "wall_s": dt,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps-per-epoch", type=int, default=250)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--extended", action="store_true",
                    help="Train on [1,999] with max_pos=4, test on [1000,4999]. "
                         "Ensures all digit×position combos are seen.")
    ap.add_argument("--head", choices=["flat", "slot"], default="flat",
                    help="flat = standard MLP; slot = slot-equivariant MLP + carry.")
    ap.add_argument("--out", type=Path, default=Path("outputs/compositional_number"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.n_seeds = 1
        args.epochs = 20
        args.steps_per_epoch = 150

    rows = []
    for si in range(args.n_seeds):
        seed = 90000 + si
        print(f"─── seed {seed} ───")
        row = run_seed(seed, epochs=args.epochs,
                       steps_per_epoch=args.steps_per_epoch,
                       extended=args.extended,
                       head_type=args.head)
        rows.append(row)
        print(f"  final_loss={row['final_loss']:.4f}  wall={row['wall_s']:.1f}s")
        print(f"  train:         {row['train']['overall']:.3f}  "
              f"per-op {row['train']['per_op']}")
        print(f"  L0 in-range:   {row['L0_in_range_pair_OOD']['overall']:.3f}  "
              f"per-op {row['L0_in_range_pair_OOD']['per_op']}")
        print(f"  L1 one-out:    {row['L1_one_out_of_range']['overall']:.3f}  "
              f"per-op {row['L1_one_out_of_range']['per_op']}  "
              f"|err|~{row['L1_one_out_of_range']['mean_err_dist']:.1f}")
        print(f"  L2 both-out:   {row['L2_both_out_of_range']['overall']:.3f}  "
              f"per-op {row['L2_both_out_of_range']['per_op']}  "
              f"|err|~{row['L2_both_out_of_range']['mean_err_dist']:.1f}")

    def _mean(key_path: list[str]) -> float:
        vals = []
        for r in rows:
            x = r
            for k in key_path:
                x = x[k]
            if isinstance(x, float) and not (x != x):
                vals.append(x)
        return sum(vals) / max(len(vals), 1) if vals else float("nan")

    summary = {
        "n_seeds": args.n_seeds,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "extended": args.extended,
        "head_type": args.head,
        "per_seed": rows,
        "avg": {
            "train":   _mean(["train", "overall"]),
            "L0":      _mean(["L0_in_range_pair_OOD", "overall"]),
            "L1":      _mean(["L1_one_out_of_range", "overall"]),
            "L2":      _mean(["L2_both_out_of_range", "overall"]),
            "L1_err":  _mean(["L1_one_out_of_range", "mean_err_dist"]),
            "L2_err":  _mean(["L2_both_out_of_range", "mean_err_dist"]),
        },
    }

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.out / 'summary.json'}")
    print(f"\n=== Average over {args.n_seeds} seeds ===")
    for k, v in summary["avg"].items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
