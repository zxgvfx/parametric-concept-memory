"""space_concept_study.py — A2: PCM 在 2D 格点 (product-order) domain 的验证.

核心问题: 数字 (linear) 和颜色 (circular) 都是 **1D 拓扑**. PCM 能否
emerge 高维结构化几何? 我们用 5×5 整数格点做最小 2D 测试:
- 拓扑是**半序 / product order** — (r1,c1) ≤ (r2,c2) iff r1≤r2 且 c1≤c2.
- 两个正交 metric 轴 (row / column), 几何 anisotropy 直接可测.

实验:
- **25 cells** 在 5×5 网格上, `concept:space:r_c`.
- **MoveHead** (消费 `motion_bias`, 64d): (a, b) → {up, down, left, right, same}
  5-class (只用相邻 pair + 自身).
- **DistanceHead** (消费 `distance_offset`, 8d): (a, b) → L1 距离 ∈ {0..8}
  9-class (全 625 pair).

关键指标:
- **ρ_L1**: Spearman(off-diag cos, −L1(a,b)). 整体 2D metric 贴合度.
- **ρ_row_within**: 在同行 pair 内测 ρ (cos vs −|c_a − c_b|).
- **ρ_col_within**: 同列.
- **ρ_linear**: Spearman(cos, −|i−j|) 按 flattened index. 1D 扁平化 control,
  若 ρ_linear > ρ_L1 则 PCM 被 1D bias 住了 (falsifier).
- **Procrustes grid fit**: MDS 投射到 2D 后, 与 ground-truth 5×5 坐标
  做 Procrustes align, 给出 disparity (0=完美, 1=正交).
- **cross-facet alignment**: Spearman(vec(cos_motion), vec(cos_dist))
  + permutation test.

运行:
    python -m experiments.space_concept_study --n-seeds 5

Smoke:
    python -m experiments.space_concept_study --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from pcm.concept_graph import ConceptGraph

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Experiment config ────────────────────────────────────────────────
N_ROWS = 5
N_COLS = 5
N_CELLS = N_ROWS * N_COLS          # 25
EMBED_DIM = 128
MOTION_DIM = 64
DIST_DIM = 8
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 30
STEPS_PER_EPOCH = 200

# Move classes (5-class)
MOVE_CLASSES = ["up", "down", "left", "right", "same"]
CLS_UP, CLS_DOWN, CLS_LEFT, CLS_RIGHT, CLS_SAME = 0, 1, 2, 3, 4

# Muscle registry
CALLER_MOVE = "MoveHead"
FACET_MOVE = "motion_bias"
CALLER_DIST = "DistanceHead"
FACET_DIST = "distance_offset"


# ─── Topology helpers ────────────────────────────────────────────────


def cid_of(r: int, c: int) -> str:
    return f"concept:space:{r}_{c}"


def rc_of_idx(idx: int) -> tuple[int, int]:
    return divmod(idx, N_COLS)


def idx_of_rc(r: int, c: int) -> int:
    return r * N_COLS + c


def l1_dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def move_class(a: tuple[int, int], b: tuple[int, int]) -> Optional[int]:
    """a → b 的方向. 只对 L1=1 相邻 + same-cell 返回 class, 否则 None."""
    dr, dc = b[0] - a[0], b[1] - a[1]
    if (dr, dc) == (0, 0):
        return CLS_SAME
    if (dr, dc) == (-1, 0):
        return CLS_UP
    if (dr, dc) == (1, 0):
        return CLS_DOWN
    if (dr, dc) == (0, -1):
        return CLS_LEFT
    if (dr, dc) == (0, 1):
        return CLS_RIGHT
    return None


def enumerate_move_triples() -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    out = []
    for r1 in range(N_ROWS):
        for c1 in range(N_COLS):
            for r2 in range(N_ROWS):
                for c2 in range(N_COLS):
                    cls = move_class((r1, c1), (r2, c2))
                    if cls is not None:
                        out.append(((r1, c1), (r2, c2), cls))
    return out


def enumerate_distance_triples() -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    out = []
    for r1 in range(N_ROWS):
        for c1 in range(N_COLS):
            for r2 in range(N_ROWS):
                for c2 in range(N_COLS):
                    out.append(((r1, c1), (r2, c2), l1_dist((r1, c1), (r2, c2))))
    return out


# ─── Muscles ──────────────────────────────────────────────────────────


class MoveHead(nn.Module):
    """(cell_a, cell_b) → 5-class direction logits, 消费 motion_bias."""

    def __init__(self, facet_dim: int = MOTION_DIM, hidden: int = 128,
                 n_classes: int = 5) -> None:
        super().__init__()
        self.facet_dim = facet_dim
        self.fc1 = nn.Linear(2 * facet_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, ids_a, ids_b, cg, tick=0):
        ba = self._collapse(ids_a, cg, tick)
        bb = self._collapse(ids_b, cg, tick)
        x = torch.cat([ba, bb], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def _collapse(self, ids, cg, tick):
        device = next(self.parameters()).device
        rows = []
        for cid in ids:
            cc = cg.concepts[cid].collapse(
                caller=CALLER_MOVE, facet=FACET_MOVE, shape=(self.facet_dim,),
                tick=tick, init="normal_small", device=device,
            )
            rows.append(cc.as_tensor())
        return torch.stack(rows, dim=0)


class DistanceHead(nn.Module):
    """(cell_a, cell_b) → 9-class L1 distance logits, 消费 distance_offset."""

    def __init__(self, facet_dim: int = DIST_DIM, hidden: int = 64,
                 n_classes: int = 9) -> None:
        super().__init__()
        self.facet_dim = facet_dim
        self.fc1 = nn.Linear(2 * facet_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, ids_a, ids_b, cg, tick=0):
        oa = self._collapse(ids_a, cg, tick)
        ob = self._collapse(ids_b, cg, tick)
        x = torch.cat([oa, ob], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def _collapse(self, ids, cg, tick):
        device = next(self.parameters()).device
        rows = []
        for cid in ids:
            cc = cg.concepts[cid].collapse(
                caller=CALLER_DIST, facet=FACET_DIST, shape=(self.facet_dim,),
                tick=tick, init="normal_small", device=device,
            )
            rows.append(cc.as_tensor())
        return torch.stack(rows, dim=0)


# ─── Graph builder ────────────────────────────────────────────────────


def build_space_graph() -> ConceptGraph:
    cg = ConceptGraph(feat_dim=EMBED_DIM)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            cg.register_concept(
                node_id=cid_of(r, c),
                label=f"SPACE_{r}_{c}",
                scope="BASE",
                provenance=f"space_study:r={r},c={c}",
            )
    return cg


def _apply_shuffle(ids: list[str], sm: dict[int, int] | None) -> list[str]:
    """shuffle map: flattened_index (0..24) → shuffled_flattened_index."""
    if sm is None:
        return ids
    out = []
    for cid in ids:
        _, rc = cid.rsplit(":", 1)
        r_s, c_s = rc.split("_")
        idx = idx_of_rc(int(r_s), int(c_s))
        new_idx = sm[idx]
        nr, nc = rc_of_idx(new_idx)
        out.append(cid_of(nr, nc))
    return out


# ─── Training ─────────────────────────────────────────────────────────


def train_one(
    mode: str,                        # "single" (move only) or "dual"
    seed: int,
    shuffle_map: dict[int, int] | None = None,
    epochs: int = EPOCHS,
    steps_per_epoch: int = STEPS_PER_EPOCH,
) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    move_triples = enumerate_move_triples()
    dist_triples = enumerate_distance_triples()

    cg = build_space_graph()
    head_move = MoveHead().to(DEVICE)
    head_dist: DistanceHead | None = None
    if mode == "dual":
        head_dist = DistanceHead().to(DEVICE)

    with torch.no_grad():
        for r in range(N_ROWS):
            for c in range(N_COLS):
                cn = cg.concepts[cid_of(r, c)]
                cn.collapse(CALLER_MOVE, FACET_MOVE, (MOTION_DIM,),
                            tick=0, device=DEVICE, init="normal_small")
                if mode == "dual":
                    cn.collapse(CALLER_DIST, FACET_DIST, (DIST_DIM,),
                                tick=0, device=DEVICE, init="normal_small")
    cg.bundles_to(torch.device(DEVICE))

    params = list(head_move.parameters()) + list(cg.iter_bundle_parameters())
    if head_dist is not None:
        params = (list(head_move.parameters())
                  + list(head_dist.parameters())
                  + list(cg.iter_bundle_parameters()))
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        head_move.train()
        if head_dist is not None:
            head_dist.train()
        for step_i in range(steps_per_epoch):
            batch = [move_triples[rng.randrange(len(move_triples))]
                     for _ in range(BATCH_SIZE)]
            ids_a = _apply_shuffle(
                [cid_of(*t[0]) for t in batch], shuffle_map)
            ids_b = _apply_shuffle(
                [cid_of(*t[1]) for t in batch], shuffle_map)
            tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
            pred = head_move(ids_a, ids_b, cg, tick=epoch * 10000 + step_i)
            loss_m = F.cross_entropy(pred, tgt)
            total = loss_m

            if head_dist is not None:
                batch2 = [dist_triples[rng.randrange(len(dist_triples))]
                          for _ in range(BATCH_SIZE)]
                ids_a2 = _apply_shuffle(
                    [cid_of(*t[0]) for t in batch2], shuffle_map)
                ids_b2 = _apply_shuffle(
                    [cid_of(*t[1]) for t in batch2], shuffle_map)
                tgt2 = torch.tensor([t[2] for t in batch2], device=DEVICE)
                logits = head_dist(ids_a2, ids_b2, cg,
                                   tick=epoch * 10000 + step_i)
                total = total + F.cross_entropy(logits, tgt2)

            opt.zero_grad(); total.backward(); opt.step()

    head_move.eval()
    if head_dist is not None:
        head_dist.eval()

    hits_m = 0
    with torch.no_grad():
        for i in range(0, len(move_triples), 64):
            batch = move_triples[i:i + 64]
            ids_a = _apply_shuffle(
                [cid_of(*t[0]) for t in batch], shuffle_map)
            ids_b = _apply_shuffle(
                [cid_of(*t[1]) for t in batch], shuffle_map)
            tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
            pred = head_move(ids_a, ids_b, cg)
            hits_m += pred.argmax(-1).eq(tgt).sum().item()
    move_acc = hits_m / len(move_triples)

    dist_acc: Optional[float] = None
    if head_dist is not None:
        hits_d = 0
        with torch.no_grad():
            for i in range(0, len(dist_triples), 64):
                batch = dist_triples[i:i + 64]
                ids_a = _apply_shuffle(
                    [cid_of(*t[0]) for t in batch], shuffle_map)
                ids_b = _apply_shuffle(
                    [cid_of(*t[1]) for t in batch], shuffle_map)
                tgt = torch.tensor([t[2] for t in batch], device=DEVICE)
                logits = head_dist(ids_a, ids_b, cg)
                hits_d += logits.argmax(-1).eq(tgt).sum().item()
        dist_acc = hits_d / len(dist_triples)

    bundle_state = {
        cid: {k: v.detach().cpu() for k, v in c.bundle.state_dict().items()}
        for cid, c in cg.concepts.items()
    }
    return {
        "mode": mode, "seed": seed,
        "shuffled": shuffle_map is not None,
        "move_acc": move_acc, "dist_acc": dist_acc,
        "bundle_state": bundle_state,
    }


# ─── Metrics ──────────────────────────────────────────────────────────


def _cos_matrix(bs: dict, facet: str) -> torch.Tensor:
    rows = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            rows.append(bs[cid_of(r, c)][f"params.{facet}"])
    M = F.normalize(torch.stack(rows), dim=-1)
    return M @ M.t()


def _rho_L1(cos: torch.Tensor) -> float:
    mask = ~torch.eye(N_CELLS, dtype=torch.bool)
    off = cos[mask].numpy()
    d = torch.zeros(N_CELLS, N_CELLS)
    for i in range(N_CELLS):
        for j in range(N_CELLS):
            d[i, j] = -l1_dist(rc_of_idx(i), rc_of_idx(j))
    return float(spearmanr(off, d[mask].numpy())[0])


def _rho_linear_flat(cos: torch.Tensor) -> float:
    """1D-flattened control: ρ(cos, −|i−j|). If > ρ_L1, PCM 退化到了 1D bias."""
    mask = ~torch.eye(N_CELLS, dtype=torch.bool)
    off = cos[mask].numpy()
    d = torch.tensor(
        [[-abs(i - j) for j in range(N_CELLS)] for i in range(N_CELLS)],
        dtype=torch.float,
    )
    return float(spearmanr(off, d[mask].numpy())[0])


def _rho_row_within(cos: torch.Tensor) -> float:
    """同行 pair 内: ρ(cos_ab, −|c_a − c_b|)."""
    xs, ys = [], []
    for r in range(N_ROWS):
        for c1 in range(N_COLS):
            for c2 in range(N_COLS):
                if c1 == c2:
                    continue
                i = idx_of_rc(r, c1); j = idx_of_rc(r, c2)
                xs.append(cos[i, j].item())
                ys.append(-abs(c1 - c2))
    if len(xs) < 3:
        return float("nan")
    return float(spearmanr(xs, ys)[0])


def _rho_col_within(cos: torch.Tensor) -> float:
    xs, ys = [], []
    for c in range(N_COLS):
        for r1 in range(N_ROWS):
            for r2 in range(N_ROWS):
                if r1 == r2:
                    continue
                i = idx_of_rc(r1, c); j = idx_of_rc(r2, c)
                xs.append(cos[i, j].item())
                ys.append(-abs(r1 - r2))
    return float(spearmanr(xs, ys)[0])


def _cross_facet_align(bs: dict, f1: str, f2: str) -> float:
    ca = _cos_matrix(bs, f1)
    cb = _cos_matrix(bs, f2)
    mask = ~torch.eye(N_CELLS, dtype=torch.bool)
    return float(spearmanr(ca[mask].numpy(), cb[mask].numpy())[0])


def _mds_grid_fit(bs: dict, facet: str, seed: int = 0) -> dict:
    """把 bundle cos 矩阵用 MDS 投到 2D, 和 GT grid 坐标做 Procrustes align.

    返回 disparity (∈ [0, 1]; 0=完美 grid, 1=最差) + 每点残差.
    """
    from sklearn.manifold import MDS
    from scipy.spatial import procrustes
    import numpy as np

    cos = _cos_matrix(bs, facet)
    cos_np = cos.numpy()
    # 用 (1 − cos) 作为 dissimilarity, clip ≥ 0
    diss = np.clip(1.0 - cos_np, 0.0, 2.0)
    np.fill_diagonal(diss, 0.0)
    mds = MDS(
        n_components=2, dissimilarity="precomputed",
        random_state=seed, normalized_stress="auto",
        n_init=4, max_iter=500,
    )
    coords = mds.fit_transform(diss)  # (N_CELLS, 2)

    # GT 5×5 grid coords
    gt = np.array(
        [[r, c] for r in range(N_ROWS) for c in range(N_COLS)],
        dtype=float,
    )

    gt_n, coords_n, disparity = procrustes(gt, coords)
    residuals = np.linalg.norm(gt_n - coords_n, axis=1)
    return {
        "disparity": float(disparity),
        "mean_residual": float(residuals.mean()),
        "max_residual": float(residuals.max()),
        "mds_stress": float(mds.stress_),
    }


# ─── Experiments ──────────────────────────────────────────────────────


def run_e1_multi_seed(
    n_seeds: int,
    seed_base: int = 1000,
    epochs: int = EPOCHS,
    steps_per_epoch: int = STEPS_PER_EPOCH,
) -> dict:
    rows = []
    for i in range(n_seeds):
        seed = seed_base + i
        t0 = time.time()
        s = train_one("single", seed, epochs=epochs, steps_per_epoch=steps_per_epoch)
        d = train_one("dual",   seed, epochs=epochs, steps_per_epoch=steps_per_epoch)
        rho_L1_s = _rho_L1(_cos_matrix(s["bundle_state"], FACET_MOVE))
        rho_lin_s = _rho_linear_flat(_cos_matrix(s["bundle_state"], FACET_MOVE))
        rho_row_s = _rho_row_within(_cos_matrix(s["bundle_state"], FACET_MOVE))
        rho_col_s = _rho_col_within(_cos_matrix(s["bundle_state"], FACET_MOVE))

        rho_L1_d = _rho_L1(_cos_matrix(d["bundle_state"], FACET_MOVE))
        rho_L1_d_dist = _rho_L1(_cos_matrix(d["bundle_state"], FACET_DIST))
        align = _cross_facet_align(d["bundle_state"], FACET_MOVE, FACET_DIST)

        mds_fit_single = _mds_grid_fit(s["bundle_state"], FACET_MOVE, seed=seed)
        mds_fit_dual_move = _mds_grid_fit(d["bundle_state"], FACET_MOVE, seed=seed)

        dt = time.time() - t0
        print(f"[E1 seed={seed}] single move_acc={s['move_acc']:.3f}  "
              f"ρ_L1={rho_L1_s:+.3f}  ρ_lin={rho_lin_s:+.3f}  "
              f"ρ_row={rho_row_s:+.3f}  ρ_col={rho_col_s:+.3f}  "
              f"MDS_disp={mds_fit_single['disparity']:.3f}  | "
              f"dual move_acc={d['move_acc']:.3f} dist_acc={d['dist_acc']:.3f}  "
              f"ρ_mov={rho_L1_d:+.3f}  ρ_dst={rho_L1_d_dist:+.3f}  "
              f"align={align:+.3f}  ({dt:.1f}s)")
        rows.append({
            "seed": seed,
            "single_move_acc": s["move_acc"],
            "single_rho_L1": rho_L1_s,
            "single_rho_linear_flat": rho_lin_s,
            "single_rho_row_within": rho_row_s,
            "single_rho_col_within": rho_col_s,
            "single_mds_fit": mds_fit_single,
            "dual_move_acc": d["move_acc"],
            "dual_dist_acc": d["dist_acc"],
            "dual_rho_motion_L1": rho_L1_d,
            "dual_rho_distance_L1": rho_L1_d_dist,
            "dual_cross_facet_align": align,
            "dual_mds_fit_motion": mds_fit_dual_move,
            "wall_s": dt,
        })

    def _stats(xs):
        xs = list(xs)
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
        return {"mean": m, "std": sd, "min": min(xs), "max": max(xs)}

    return {
        "n_seeds": n_seeds,
        "per_seed": rows,
        "single_rho_L1": _stats([r["single_rho_L1"] for r in rows]),
        "single_rho_linear_flat": _stats([r["single_rho_linear_flat"] for r in rows]),
        "single_rho_row_within": _stats([r["single_rho_row_within"] for r in rows]),
        "single_rho_col_within": _stats([r["single_rho_col_within"] for r in rows]),
        "single_mds_disparity": _stats([r["single_mds_fit"]["disparity"] for r in rows]),
        "dual_rho_motion_L1": _stats([r["dual_rho_motion_L1"] for r in rows]),
        "dual_rho_distance_L1": _stats([r["dual_rho_distance_L1"] for r in rows]),
        "dual_cross_facet_align": _stats([r["dual_cross_facet_align"] for r in rows]),
        "dual_mds_disparity_motion": _stats(
            [r["dual_mds_fit_motion"]["disparity"] for r in rows]),
    }


def run_e2_shuffled(
    n_seeds: int,
    seed_base: int = 2000,
    epochs: int = EPOCHS,
    steps_per_epoch: int = STEPS_PER_EPOCH,
) -> dict:
    """Shuffle concept_id → bundle 映射, ρ 应大幅塌缩."""
    rows = []
    for i in range(n_seeds):
        seed = seed_base + i
        perm = list(range(N_CELLS))
        random.Random(seed).shuffle(perm)
        sm = {k: perm[k] for k in range(N_CELLS)}
        t0 = time.time()
        s = train_one("single", seed, shuffle_map=sm,
                      epochs=epochs, steps_per_epoch=steps_per_epoch)
        rho_L1 = _rho_L1(_cos_matrix(s["bundle_state"], FACET_MOVE))
        rho_row = _rho_row_within(_cos_matrix(s["bundle_state"], FACET_MOVE))
        rho_col = _rho_col_within(_cos_matrix(s["bundle_state"], FACET_MOVE))
        mds_fit = _mds_grid_fit(s["bundle_state"], FACET_MOVE, seed=seed)
        dt = time.time() - t0
        print(f"[E2 seed={seed} shuffled] move_acc={s['move_acc']:.3f}  "
              f"ρ_L1={rho_L1:+.3f} ρ_row={rho_row:+.3f} ρ_col={rho_col:+.3f}  "
              f"MDS_disp={mds_fit['disparity']:.3f}  ({dt:.1f}s)")
        rows.append({
            "seed": seed, "shuffle_map": sm,
            "move_acc": s["move_acc"],
            "rho_L1_raw_order": rho_L1,
            "rho_row_raw_order": rho_row,
            "rho_col_raw_order": rho_col,
            "mds_fit": mds_fit,
            "wall_s": dt,
        })

    def _stats(xs):
        xs = [abs(x) for x in xs]
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
        return {"mean": m, "std": sd, "min": min(xs), "max": max(xs)}

    return {
        "n_seeds": n_seeds,
        "per_seed": rows,
        "abs_rho_L1_stats": _stats([r["rho_L1_raw_order"] for r in rows]),
        "notes": "shuffle_map 破坏 concept_id→bundle 身份映射, |ρ_L1| 应 ≈ 0.",
    }


def run_e4_permutation(dual_bundle: dict, n_perm: int = 1000) -> dict:
    cos_m = _cos_matrix(dual_bundle, FACET_MOVE)
    cos_d = _cos_matrix(dual_bundle, FACET_DIST)
    mask = ~torch.eye(N_CELLS, dtype=torch.bool)
    om = cos_m[mask].numpy()
    od = cos_d[mask].numpy()
    observed = float(spearmanr(om, od)[0])

    rng = random.Random(0)
    ge = 0
    null = []
    for _ in range(n_perm):
        perm = list(range(N_CELLS))
        rng.shuffle(perm)
        cos_d_perm = cos_d[perm][:, perm]
        odp = cos_d_perm[mask].numpy()
        r = float(spearmanr(om, odp)[0])
        null.append(r)
        if abs(r) >= abs(observed):
            ge += 1
    p = (ge + 1) / (n_perm + 1)
    return {
        "observed_cross_facet_rho": observed,
        "n_permutations": n_perm,
        "p_value": p,
        "null_mean": sum(null) / len(null),
        "null_std": math.sqrt(
            sum((r - sum(null) / len(null)) ** 2 for r in null)
            / max(len(null) - 1, 1)
        ),
        "conclusion": (
            "significant (p<0.01)" if p < 0.01
            else "significant (p<0.05)" if p < 0.05
            else "not significant"
        ),
    }


# ─── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    ap.add_argument("--out", type=Path, default=Path("outputs/space_concept"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], help="e1 / e2 / e4")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.n_seeds = 1
        args.epochs = 10
        args.steps_per_epoch = 80

    summary: dict = {
        "n_seeds": args.n_seeds,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "device": DEVICE,
        "n_cells": N_CELLS,
        "grid": [N_ROWS, N_COLS],
    }

    if "e1" not in args.skip:
        print("=" * 60); print("E1: Multi-seed (single vs dual)"); print("=" * 60)
        summary["E1_multi_seed"] = run_e1_multi_seed(
            args.n_seeds, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch
        )

    if "e2" not in args.skip:
        print("=" * 60); print("E2: Shuffled concept counterfactual"); print("=" * 60)
        summary["E2_shuffled"] = run_e2_shuffled(
            args.n_seeds, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch
        )

    if "e4" not in args.skip:
        print("=" * 60); print("E4: Cross-facet permutation test"); print("=" * 60)
        d = train_one("dual", 1000,
                      epochs=args.epochs, steps_per_epoch=args.steps_per_epoch)
        summary["E4_permutation"] = run_e4_permutation(d["bundle_state"])

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60); print("SUMMARY"); print("=" * 60)
    if "E1_multi_seed" in summary:
        e1 = summary["E1_multi_seed"]
        sL = e1["single_rho_L1"]; sl = e1["single_rho_linear_flat"]
        sr = e1["single_rho_row_within"]; sc = e1["single_rho_col_within"]
        md = e1["single_mds_disparity"]
        al = e1["dual_cross_facet_align"]
        print(f"E1  single ρ_L1         = {sL['mean']:+.3f} ± {sL['std']:.3f}")
        print(f"E1  single ρ_linear_flat= {sl['mean']:+.3f} ± {sl['std']:.3f}"
              f"  (should be << ρ_L1)")
        print(f"E1  single ρ_row_within = {sr['mean']:+.3f} ± {sr['std']:.3f}")
        print(f"E1  single ρ_col_within = {sc['mean']:+.3f} ± {sc['std']:.3f}")
        print(f"E1  single MDS disp     = {md['mean']:.3f} ± {md['std']:.3f}"
              f"  (0=perfect grid)")
        print(f"E1  dual cross-facet align = {al['mean']:+.3f} ± {al['std']:.3f}")
    if "E2_shuffled" in summary:
        e2 = summary["E2_shuffled"]
        print(f"E2  |ρ_L1| shuffled = {e2['abs_rho_L1_stats']['mean']:.3f} "
              f"± {e2['abs_rho_L1_stats']['std']:.3f}")
    if "E4_permutation" in summary:
        e4 = summary["E4_permutation"]
        print(f"E4  observed = {e4['observed_cross_facet_rho']:+.3f}  "
              f"null_mean = {e4['null_mean']:+.3f}  p = {e4['p_value']:.3f}  "
              f"→ {e4['conclusion']}")


if __name__ == "__main__":
    main()
