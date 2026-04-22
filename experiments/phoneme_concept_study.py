"""phoneme_concept_study.py — A3: PCM on a discrete (non-metric) domain.

核心问题: PCM 的 geometry emergence 在 metric-like topology (linear /
circular / lattice) 上都成立; 它能否外推到 **非连续 / 无自然度量** 的
离散 domain?

更关键的命题 (H5'' 的充要性方向):
    同一份 phonemes, 同时训练 3 个**正交** muscle (voicing / manner /
    place), 三个 facet bundle 的 cos geometry 应该**互不相关** (cross-
    facet align ≈ 0). 这会补齐 paper §3.4 里 H5'' 的 "not-if" 方向:
    任务 algebra 正交 ⇒ facet 几何不对齐.

实验:
- **20 phonemes** (SPE 简化) × 3 正交属性: ±voice / 4-way manner /
  4-way place.
- 3 个单输入 binary / multi-class muscle:
  - VoicingHead (facet `voice_bias`) — binary (±voice)
  - MannerHead  (facet `manner_bias`) — 4-class (stop/fric/nas/apr)
  - PlaceHead   (facet `place_bias`) — 4-class (lab/cor/dor/glt)

关键指标:
- **ρ_same_voicing** etc: ρ(cos, −I[same_class]) per facet on its axis.
- **intra vs inter class cos gap** per facet.
- **cross-facet align** (3 pairs):
  - voice ↔ manner
  - voice ↔ place
  - manner ↔ place
  - 预测全部 ≈ 0 (p > 0.05) — 与 color/number 对照
- **shuffle counterfactual**: |ρ| 全塌缩.
- **ρ_hamming (总体)**: Spearman(off-diag cos, −hamming). 单 facet
  应只对**自己的轴**强, 跨轴弱, 组合多 facet 后对全 hamming 强 →
  这是"PCM 在 categorical domain 也 emerge geometry"的正面 claim.

运行:
    python -m experiments.phoneme_concept_study --n-seeds 3
    python -m experiments.phoneme_concept_study --smoke
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from pcm.concept_graph import ConceptGraph

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Phoneme feature matrix ──────────────────────────────────────────
# 20 phonemes, 3 attributes: (voice, manner, place)
# voice: 0 = voiceless, 1 = voiced
# manner: 0=STOP, 1=FRIC, 2=NAS, 3=APR
# place: 0=LAB, 1=COR, 2=DOR, 3=GLT

PHONEMES: list[tuple[str, int, int, int]] = [
    # label, voice, manner, place
    ("p",  0, 0, 0),
    ("b",  1, 0, 0),
    ("t",  0, 0, 1),
    ("d",  1, 0, 1),
    ("k",  0, 0, 2),
    ("g",  1, 0, 2),
    ("q",  0, 0, 3),    # /ʔ/ glottal stop
    ("f",  0, 1, 0),
    ("v",  1, 1, 0),
    ("s",  0, 1, 1),
    ("z",  1, 1, 1),
    ("x",  0, 1, 2),    # velar fricative
    ("h",  0, 1, 3),
    ("m",  1, 2, 0),
    ("n",  1, 2, 1),
    ("N",  1, 2, 2),    # /ŋ/
    ("w",  1, 3, 0),
    ("l",  1, 3, 1),
    ("r",  1, 3, 1),
    ("j",  1, 3, 2),    # palatal approximant
]
N_PH = len(PHONEMES)  # 20

N_VOICE, N_MANNER, N_PLACE = 2, 4, 4

# ─── Experiment hyperparams ──────────────────────────────────────────
EMBED_DIM = 128
VOICE_DIM = 16
MANNER_DIM = 16
PLACE_DIM = 16
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 60
STEPS_PER_EPOCH = 120

CALLER_V, FACET_V = "VoicingHead", "voice_bias"
CALLER_M, FACET_M = "MannerHead", "manner_bias"
CALLER_P, FACET_P = "PlaceHead", "place_bias"


# ─── Helpers ──────────────────────────────────────────────────────────


def cid_of(idx: int) -> str:
    lbl = PHONEMES[idx][0]
    return f"concept:phoneme:{lbl}"


def feat_of(idx: int) -> tuple[int, int, int]:
    """→ (voice, manner, place)."""
    return PHONEMES[idx][1], PHONEMES[idx][2], PHONEMES[idx][3]


def hamming(i: int, j: int) -> int:
    """Count of differing attributes across (voice, manner, place), 0..3."""
    a = feat_of(i); b = feat_of(j)
    return sum(1 for x, y in zip(a, b) if x != y)


def build_phoneme_graph() -> ConceptGraph:
    cg = ConceptGraph(feat_dim=EMBED_DIM)
    for i, (lbl, v, m, p) in enumerate(PHONEMES):
        cg.register_concept(
            node_id=cid_of(i),
            label=f"PHONEME_{lbl}",
            scope="BASE",
            provenance=f"phoneme_study:voice={v},manner={m},place={p}",
        )
    return cg


def _apply_shuffle(ids: list[str], sm: dict[int, int] | None) -> list[str]:
    if sm is None:
        return ids
    out = []
    for cid in ids:
        # lookup idx by label
        lbl = cid.rsplit(":", 1)[-1]
        for i, row in enumerate(PHONEMES):
            if row[0] == lbl:
                out.append(cid_of(sm[i]))
                break
    return out


# ─── Muscles (single-input classifiers, one per attribute axis) ──────


class _SingleInputHead(nn.Module):
    """(phoneme) → n_classes logits, consuming a single facet."""

    def __init__(self, caller: str, facet: str, facet_dim: int,
                 n_classes: int, hidden: int = 64) -> None:
        super().__init__()
        self.caller = caller
        self.facet = facet
        self.facet_dim = facet_dim
        self.fc1 = nn.Linear(facet_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, ids: list[str], cg: ConceptGraph, tick: int = 0) -> torch.Tensor:
        x = self._collapse(ids, cg, tick)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def _collapse(self, ids: list[str], cg: ConceptGraph, tick: int) -> torch.Tensor:
        device = next(self.parameters()).device
        rows = []
        for cid in ids:
            cc = cg.concepts[cid].collapse(
                caller=self.caller, facet=self.facet, shape=(self.facet_dim,),
                tick=tick, init="normal_small", device=device,
            )
            rows.append(cc.as_tensor())
        return torch.stack(rows, dim=0)


def build_voicing_head() -> _SingleInputHead:
    return _SingleInputHead(CALLER_V, FACET_V, VOICE_DIM,  N_VOICE)


def build_manner_head() -> _SingleInputHead:
    return _SingleInputHead(CALLER_M, FACET_M, MANNER_DIM, N_MANNER)


def build_place_head() -> _SingleInputHead:
    return _SingleInputHead(CALLER_P, FACET_P, PLACE_DIM,  N_PLACE)


# ─── Training ─────────────────────────────────────────────────────────


def train_one(
    mode: str,                          # "single_v" / "single_m" / "single_p" / "triple"
    seed: int,
    shuffle_map: dict[int, int] | None = None,
    epochs: int = EPOCHS,
    steps_per_epoch: int = STEPS_PER_EPOCH,
) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    cg = build_phoneme_graph()
    heads: dict[str, _SingleInputHead] = {}
    if mode in ("single_v", "triple"):
        heads["v"] = build_voicing_head().to(DEVICE)
    if mode in ("single_m", "triple"):
        heads["m"] = build_manner_head().to(DEVICE)
    if mode in ("single_p", "triple"):
        heads["p"] = build_place_head().to(DEVICE)

    # lazy-init facets by calling collapse once per phoneme per active head
    with torch.no_grad():
        for i in range(N_PH):
            cn = cg.concepts[cid_of(i)]
            if "v" in heads:
                cn.collapse(CALLER_V, FACET_V, (VOICE_DIM,),
                            tick=0, device=DEVICE, init="normal_small")
            if "m" in heads:
                cn.collapse(CALLER_M, FACET_M, (MANNER_DIM,),
                            tick=0, device=DEVICE, init="normal_small")
            if "p" in heads:
                cn.collapse(CALLER_P, FACET_P, (PLACE_DIM,),
                            tick=0, device=DEVICE, init="normal_small")
    cg.bundles_to(torch.device(DEVICE))

    params: list = []
    for h in heads.values():
        params += list(h.parameters())
    params += list(cg.iter_bundle_parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        for h in heads.values():
            h.train()
        for step_i in range(steps_per_epoch):
            # sample a batch of phoneme indices (uniform)
            idx_batch = [rng.randrange(N_PH) for _ in range(BATCH_SIZE)]
            ids = _apply_shuffle([cid_of(i) for i in idx_batch], shuffle_map)

            total = 0.0
            count = 0
            if "v" in heads:
                tgt = torch.tensor(
                    [feat_of(i)[0] for i in idx_batch], device=DEVICE
                )
                logits = heads["v"](ids, cg, tick=epoch * 10000 + step_i)
                total = total + F.cross_entropy(logits, tgt); count += 1
            if "m" in heads:
                tgt = torch.tensor(
                    [feat_of(i)[1] for i in idx_batch], device=DEVICE
                )
                logits = heads["m"](ids, cg, tick=epoch * 10000 + step_i)
                total = total + F.cross_entropy(logits, tgt); count += 1
            if "p" in heads:
                tgt = torch.tensor(
                    [feat_of(i)[2] for i in idx_batch], device=DEVICE
                )
                logits = heads["p"](ids, cg, tick=epoch * 10000 + step_i)
                total = total + F.cross_entropy(logits, tgt); count += 1

            opt.zero_grad(); total.backward(); opt.step()

    for h in heads.values():
        h.eval()

    accs: dict[str, float] = {}
    with torch.no_grad():
        all_ids = _apply_shuffle([cid_of(i) for i in range(N_PH)], shuffle_map)
        for key, h in heads.items():
            logits = h(all_ids, cg)
            axis = {"v": 0, "m": 1, "p": 2}[key]
            tgt = torch.tensor([feat_of(i)[axis] for i in range(N_PH)], device=DEVICE)
            accs[key] = float(logits.argmax(-1).eq(tgt).sum().item()) / N_PH

    bundle_state = {
        cid: {k: v.detach().cpu() for k, v in c.bundle.state_dict().items()}
        for cid, c in cg.concepts.items()
    }
    return {
        "mode": mode, "seed": seed,
        "shuffled": shuffle_map is not None,
        "accs": accs,
        "bundle_state": bundle_state,
    }


# ─── Metrics ──────────────────────────────────────────────────────────


def _cos_matrix(bs: dict, facet: str) -> torch.Tensor:
    rows = []
    for i in range(N_PH):
        rows.append(bs[cid_of(i)][f"params.{facet}"])
    M = F.normalize(torch.stack(rows), dim=-1)
    return M @ M.t()


def _rho_hamming_total(cos: torch.Tensor) -> float:
    """Spearman(off-diag cos, −hamming): 总体 categorical metric 贴合度."""
    mask = ~torch.eye(N_PH, dtype=torch.bool)
    off = cos[mask].numpy()
    d = torch.tensor(
        [[-hamming(i, j) for j in range(N_PH)] for i in range(N_PH)],
        dtype=torch.float,
    )
    return float(spearmanr(off, d[mask].numpy())[0])


def _rho_same_axis(cos: torch.Tensor, axis: int) -> float:
    """Spearman(off-diag cos, indicator[same axis value]).

    axis ∈ {0, 1, 2} for (voice, manner, place).
    高 ρ ⇒ facet 几何把同 class 集合塞得更近.
    """
    mask = ~torch.eye(N_PH, dtype=torch.bool)
    off = cos[mask].numpy()
    ind = torch.tensor(
        [[1.0 if feat_of(i)[axis] == feat_of(j)[axis] else 0.0
          for j in range(N_PH)] for i in range(N_PH)],
        dtype=torch.float,
    )
    return float(spearmanr(off, ind[mask].numpy())[0])


def _intra_vs_inter_gap(cos: torch.Tensor, axis: int) -> dict:
    intra_vals, inter_vals = [], []
    for i in range(N_PH):
        for j in range(N_PH):
            if i == j:
                continue
            c = cos[i, j].item()
            if feat_of(i)[axis] == feat_of(j)[axis]:
                intra_vals.append(c)
            else:
                inter_vals.append(c)

    def _m(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    return {
        "intra_mean": _m(intra_vals),
        "inter_mean": _m(inter_vals),
        "gap": _m(intra_vals) - _m(inter_vals),
        "n_intra": len(intra_vals),
        "n_inter": len(inter_vals),
    }


def _cross_facet_align(bs: dict, f1: str, f2: str) -> float:
    ca = _cos_matrix(bs, f1)
    cb = _cos_matrix(bs, f2)
    mask = ~torch.eye(N_PH, dtype=torch.bool)
    return float(spearmanr(ca[mask].numpy(), cb[mask].numpy())[0])


def _perm_test_align(bs: dict, f1: str, f2: str, n_perm: int = 1000) -> dict:
    ca = _cos_matrix(bs, f1)
    cb = _cos_matrix(bs, f2)
    mask = ~torch.eye(N_PH, dtype=torch.bool)
    oa = ca[mask].numpy()
    ob = cb[mask].numpy()
    observed = float(spearmanr(oa, ob)[0])

    rng = random.Random(0)
    ge = 0
    null = []
    for _ in range(n_perm):
        perm = list(range(N_PH))
        rng.shuffle(perm)
        cb_perm = cb[perm][:, perm]
        obp = cb_perm[mask].numpy()
        r = float(spearmanr(oa, obp)[0])
        null.append(r)
        if abs(r) >= abs(observed):
            ge += 1
    p = (ge + 1) / (n_perm + 1)
    return {
        "observed": observed,
        "p_value": p,
        "null_mean": sum(null) / len(null),
        "conclusion": (
            "significant (p<0.01)" if p < 0.01
            else "significant (p<0.05)" if p < 0.05
            else "not significant"
        ),
    }


# ─── Experiments ──────────────────────────────────────────────────────


def run_e1_multi_seed(n_seeds: int, seed_base: int = 1000,
                      epochs: int = EPOCHS,
                      steps_per_epoch: int = STEPS_PER_EPOCH) -> dict:
    rows = []
    for i in range(n_seeds):
        seed = seed_base + i
        t0 = time.time()
        t = train_one("triple", seed, epochs=epochs,
                      steps_per_epoch=steps_per_epoch)
        bs = t["bundle_state"]

        # per-facet geometry: ρ on its own axis vs other axes
        rho_same_v = _rho_same_axis(_cos_matrix(bs, FACET_V), axis=0)
        rho_same_m = _rho_same_axis(_cos_matrix(bs, FACET_M), axis=1)
        rho_same_p = _rho_same_axis(_cos_matrix(bs, FACET_P), axis=2)
        # leakage: does voice facet carry manner info? etc.
        rho_v_on_m = _rho_same_axis(_cos_matrix(bs, FACET_V), axis=1)
        rho_m_on_v = _rho_same_axis(_cos_matrix(bs, FACET_M), axis=0)
        rho_v_on_p = _rho_same_axis(_cos_matrix(bs, FACET_V), axis=2)
        rho_p_on_v = _rho_same_axis(_cos_matrix(bs, FACET_P), axis=0)
        rho_m_on_p = _rho_same_axis(_cos_matrix(bs, FACET_M), axis=2)
        rho_p_on_m = _rho_same_axis(_cos_matrix(bs, FACET_P), axis=1)

        # intra/inter gap
        gap_v = _intra_vs_inter_gap(_cos_matrix(bs, FACET_V), axis=0)
        gap_m = _intra_vs_inter_gap(_cos_matrix(bs, FACET_M), axis=1)
        gap_p = _intra_vs_inter_gap(_cos_matrix(bs, FACET_P), axis=2)

        # cross-facet alignment (3 pairs)
        align_vm = _cross_facet_align(bs, FACET_V, FACET_M)
        align_vp = _cross_facet_align(bs, FACET_V, FACET_P)
        align_mp = _cross_facet_align(bs, FACET_M, FACET_P)

        # total hamming correlation per facet (single-axis facet should be weak here)
        rho_hm_v = _rho_hamming_total(_cos_matrix(bs, FACET_V))
        rho_hm_m = _rho_hamming_total(_cos_matrix(bs, FACET_M))
        rho_hm_p = _rho_hamming_total(_cos_matrix(bs, FACET_P))

        dt = time.time() - t0
        print(f"[E1 seed={seed}] accs(v/m/p)={t['accs']['v']:.2f}/"
              f"{t['accs']['m']:.2f}/{t['accs']['p']:.2f}  "
              f"ρ_same(v/m/p)={rho_same_v:+.3f}/{rho_same_m:+.3f}/{rho_same_p:+.3f}  "
              f"align(v-m/v-p/m-p)={align_vm:+.3f}/{align_vp:+.3f}/{align_mp:+.3f}  "
              f"({dt:.1f}s)")

        rows.append({
            "seed": seed,
            "accs": t["accs"],
            "rho_same_axis": {"v": rho_same_v, "m": rho_same_m, "p": rho_same_p},
            "rho_leakage": {
                "v_on_m": rho_v_on_m, "m_on_v": rho_m_on_v,
                "v_on_p": rho_v_on_p, "p_on_v": rho_p_on_v,
                "m_on_p": rho_m_on_p, "p_on_m": rho_p_on_m,
            },
            "intra_inter_gap": {"v": gap_v, "m": gap_m, "p": gap_p},
            "cross_facet_align": {
                "v_m": align_vm, "v_p": align_vp, "m_p": align_mp,
            },
            "rho_hamming_total": {"v": rho_hm_v, "m": rho_hm_m, "p": rho_hm_p},
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
        "rho_same_axis_v": _stats([r["rho_same_axis"]["v"] for r in rows]),
        "rho_same_axis_m": _stats([r["rho_same_axis"]["m"] for r in rows]),
        "rho_same_axis_p": _stats([r["rho_same_axis"]["p"] for r in rows]),
        "cross_facet_align_vm": _stats([r["cross_facet_align"]["v_m"] for r in rows]),
        "cross_facet_align_vp": _stats([r["cross_facet_align"]["v_p"] for r in rows]),
        "cross_facet_align_mp": _stats([r["cross_facet_align"]["m_p"] for r in rows]),
        "gap_v": _stats([r["intra_inter_gap"]["v"]["gap"] for r in rows]),
        "gap_m": _stats([r["intra_inter_gap"]["m"]["gap"] for r in rows]),
        "gap_p": _stats([r["intra_inter_gap"]["p"]["gap"] for r in rows]),
    }


def run_e2_shuffled(n_seeds: int, seed_base: int = 2000,
                    epochs: int = EPOCHS,
                    steps_per_epoch: int = STEPS_PER_EPOCH) -> dict:
    rows = []
    for i in range(n_seeds):
        seed = seed_base + i
        perm = list(range(N_PH))
        random.Random(seed).shuffle(perm)
        sm = {k: perm[k] for k in range(N_PH)}
        t0 = time.time()
        t = train_one("triple", seed, shuffle_map=sm,
                      epochs=epochs, steps_per_epoch=steps_per_epoch)
        bs = t["bundle_state"]
        rho_v = _rho_same_axis(_cos_matrix(bs, FACET_V), axis=0)
        rho_m = _rho_same_axis(_cos_matrix(bs, FACET_M), axis=1)
        rho_p = _rho_same_axis(_cos_matrix(bs, FACET_P), axis=2)
        dt = time.time() - t0
        print(f"[E2 seed={seed} shuffled] accs(v/m/p)="
              f"{t['accs']['v']:.2f}/{t['accs']['m']:.2f}/{t['accs']['p']:.2f}  "
              f"ρ_same(v/m/p)={rho_v:+.3f}/{rho_m:+.3f}/{rho_p:+.3f}  "
              f"({dt:.1f}s)")
        rows.append({
            "seed": seed, "shuffle_map": sm,
            "accs": t["accs"],
            "rho_same_raw_order_v": rho_v,
            "rho_same_raw_order_m": rho_m,
            "rho_same_raw_order_p": rho_p,
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
        "abs_rho_same_v_stats": _stats([r["rho_same_raw_order_v"] for r in rows]),
        "abs_rho_same_m_stats": _stats([r["rho_same_raw_order_m"] for r in rows]),
        "abs_rho_same_p_stats": _stats([r["rho_same_raw_order_p"] for r in rows]),
        "notes": "shuffle concept_id→bundle 映射后, 单 facet 在 raw 轴上的 ρ 应塌缩",
    }


# ─── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    ap.add_argument("--out", type=Path, default=Path("outputs/phoneme_concept"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], help="e1 / e2 / e4")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.n_seeds = 1
        args.epochs = 20
        args.steps_per_epoch = 60

    summary: dict = {
        "n_seeds": args.n_seeds,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "device": DEVICE,
        "n_phonemes": N_PH,
        "phoneme_set": [
            {"label": p[0], "voice": p[1], "manner": p[2], "place": p[3]}
            for p in PHONEMES
        ],
    }

    if "e1" not in args.skip:
        print("=" * 60); print("E1: Multi-seed (triple orthogonal muscles)"); print("=" * 60)
        summary["E1_multi_seed"] = run_e1_multi_seed(
            args.n_seeds, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch
        )

    if "e2" not in args.skip:
        print("=" * 60); print("E2: Shuffled concept counterfactual"); print("=" * 60)
        summary["E2_shuffled"] = run_e2_shuffled(
            args.n_seeds, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch
        )

    if "e4" not in args.skip:
        print("=" * 60); print("E4: Cross-facet permutation tests"); print("=" * 60)
        t = train_one("triple", 1000, epochs=args.epochs,
                      steps_per_epoch=args.steps_per_epoch)
        bs = t["bundle_state"]
        summary["E4_permutation"] = {
            "v_m": _perm_test_align(bs, FACET_V, FACET_M),
            "v_p": _perm_test_align(bs, FACET_V, FACET_P),
            "m_p": _perm_test_align(bs, FACET_M, FACET_P),
        }

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60); print("SUMMARY"); print("=" * 60)
    if "E1_multi_seed" in summary:
        e1 = summary["E1_multi_seed"]
        print("E1 — per-facet geometry on own axis (ρ_same):")
        for ax in ("v", "m", "p"):
            s = e1[f"rho_same_axis_{ax}"]
            print(f"    {ax}: {s['mean']:+.3f} ± {s['std']:.3f}")
        print("E1 — cross-facet alignment (expect ≈ 0):")
        for pair in ("vm", "vp", "mp"):
            s = e1[f"cross_facet_align_{pair}"]
            print(f"    {pair}: {s['mean']:+.3f} ± {s['std']:.3f}")
        print("E1 — intra/inter class cos gap (expect positive):")
        for ax in ("v", "m", "p"):
            s = e1[f"gap_{ax}"]
            print(f"    {ax}: {s['mean']:+.3f} ± {s['std']:.3f}")
    if "E2_shuffled" in summary:
        e2 = summary["E2_shuffled"]
        print("E2 — |ρ_same| shuffled (expect ≈ 0):")
        for ax in ("v", "m", "p"):
            s = e2[f"abs_rho_same_{ax}_stats"]
            print(f"    {ax}: {s['mean']:.3f} ± {s['std']:.3f}")
    if "E4_permutation" in summary:
        e4 = summary["E4_permutation"]
        print("E4 — cross-facet permutation tests:")
        for pair, v in e4.items():
            print(f"    {pair}: ρ = {v['observed']:+.3f}  "
                  f"null_mean = {v['null_mean']:+.3f}  "
                  f"p = {v['p_value']:.3f}  → {v['conclusion']}")


if __name__ == "__main__":
    main()
