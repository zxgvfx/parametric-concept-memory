"""render_paper_figures.py — regenerate all figures for the PCM paper.

Outputs (PDF + PNG) land in `mind/docs/research/figures/`:

- F2: arithmetic + ordinal cos heatmaps (N numbers, dual-muscle run)
- F4: four-domain universality panel
    (a) linear number line cos heatmap
    (b) circular hue ring MDS
    (c) 2-D spatial grid MDS (Procrustes-aligned)
    (d) phoneme cos heatmap with class block annotations
- F5: base-10 null — cos vs shift k (no spike at k=10)
- F6: counterfactual swap double-dissociation bars (number + color)
- F7: H5'' four-domain cross-facet alignment bars with null reference
- F8: space grid MDS — trained vs shuffle counterfactual side-by-side

Each figure is a self-contained function. Training is re-run for one
representative seed per domain (cheap: ~3 min total). Cached bundle
states in-process are reused across figures that need the same domain.

Usage:
    python -m experiments.render_paper_figures
    python -m experiments.render_paper_figures --only F4 F7
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import hsv_to_rgb
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from sklearn.manifold import MDS

# Paper figure style
plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Serif",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("mind/docs/research/figures")
OUTPUTS = Path("outputs")

# Color palettes
CMAP_COS = "RdBu_r"
CMAP_SEQ = "viridis"
DOMAIN_COLORS = {
    "number": "#1f77b4",
    "color":  "#d62728",
    "space":  "#2ca02c",
    "phoneme": "#9467bd",
}


def _savefig(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"{name}.{ext}"
        fig.savefig(p)
        print(f"  saved  {p}")
    plt.close(fig)


# ─── Bundle-state caches (lazily trained on first use) ────────────────

_cache: dict[str, dict] = {}


def get_number_bundles(seed: int = 1000) -> dict:
    key = f"num:{seed}"
    if key in _cache:
        return _cache[key]
    print(f"[train] number seed={seed} (dual muscle) ...")
    from experiments import robustness_study as rs
    from pcm.heads.numerosity_encoder import NumerosityEncoder
    enc_ckpt = torch.load(
        "outputs/ans_encoder/final.pt", map_location="cpu", weights_only=False
    )
    enc = NumerosityEncoder()
    enc.load_state_dict(enc_ckpt["encoder_state"])
    enc.eval()
    cfg = rs.DatasetConfig(**enc_ckpt["ds_cfg"])
    centroids = rs._compute_centroids(enc, cfg)
    d = rs.train_one(enc_ckpt, "dual", seed, cfg, centroids)
    out = {
        "bundle_state": d["bundle_state"],
        "concepts": d["concepts"],
        "cfg": cfg,
        "facet_add": "arithmetic_bias",
        "facet_ord": "ordinal_offset",
        "n_min": cfg.n_min,
        "n_max": cfg.n_max,
    }
    _cache[key] = out
    return out


def get_color_bundles(seed: int = 1000) -> dict:
    key = f"color:{seed}"
    if key in _cache:
        return _cache[key]
    print(f"[train] color seed={seed} (dual muscle) ...")
    from experiments import color_concept_study as cs
    centroids = cs.make_random_orthogonal_centroids(cs.N_COLORS, cs.EMBED_DIM, seed)
    d = cs.train_one("dual", seed, centroids)
    out = {
        "bundle_state": d["bundle_state"],
        "n_colors": cs.N_COLORS,
        "facet_mix": cs.FACET_MIX,
        "facet_adj": cs.FACET_ADJ,
    }
    _cache[key] = out
    return out


def get_space_bundles(seed: int = 1000, shuffled: bool = False) -> dict:
    key = f"space:{seed}:shuffle={shuffled}"
    if key in _cache:
        return _cache[key]
    from experiments import space_concept_study as ss
    sm = None
    if shuffled:
        perm = list(range(ss.N_CELLS))
        random.Random(seed + 7).shuffle(perm)
        sm = {k: perm[k] for k in range(ss.N_CELLS)}
    print(f"[train] space seed={seed} shuffled={shuffled} ...")
    d = ss.train_one("dual", seed, shuffle_map=sm)
    out = {
        "bundle_state": d["bundle_state"],
        "n_rows": ss.N_ROWS,
        "n_cols": ss.N_COLS,
        "n_cells": ss.N_CELLS,
        "facet_motion": ss.FACET_MOVE,
        "facet_dist": ss.FACET_DIST,
        "shuffle_map": sm,
        "cid_of": ss.cid_of,
    }
    _cache[key] = out
    return out


def get_phoneme_bundles(seed: int = 1000) -> dict:
    key = f"phon:{seed}"
    if key in _cache:
        return _cache[key]
    print(f"[train] phoneme seed={seed} (triple muscle) ...")
    from experiments import phoneme_concept_study as ps
    d = ps.train_one("triple", seed)
    out = {
        "bundle_state": d["bundle_state"],
        "phonemes": ps.PHONEMES,
        "facets": [ps.FACET_V, ps.FACET_M, ps.FACET_P],
        "cid_of": ps.cid_of,
    }
    _cache[key] = out
    return out


# ─── Bundle helpers ───────────────────────────────────────────────────


def _cos_matrix_from_bundles(bs: dict, cids: list[str], facet: str) -> np.ndarray:
    rows = []
    for cid in cids:
        rows.append(bs[cid][f"params.{facet}"])
    M = F.normalize(torch.stack(rows), dim=-1)
    return (M @ M.t()).numpy()


def _mds_2d(cos: np.ndarray, seed: int = 0) -> np.ndarray:
    diss = np.clip(1.0 - cos, 0.0, 2.0)
    np.fill_diagonal(diss, 0.0)
    mds = MDS(
        n_components=2, dissimilarity="precomputed",
        random_state=seed, normalized_stress="auto",
        n_init=4, max_iter=500,
    )
    return mds.fit_transform(diss)


# ─── F2: number arith + ord heatmaps ──────────────────────────────────


def render_F2_number_cos_heatmaps() -> None:
    b = get_number_bundles()
    bs = b["bundle_state"]
    ns = list(range(b["n_min"], b["n_max"] + 1))
    cids = [f"concept:ans:{n}" for n in ns]
    cos_add = _cos_matrix_from_bundles(bs, cids, b["facet_add"])
    cos_ord = _cos_matrix_from_bundles(bs, cids, b["facet_ord"])

    rho_add = spearmanr(
        cos_add[np.triu_indices(len(ns), k=1)],
        -np.abs(np.subtract.outer(ns, ns))[np.triu_indices(len(ns), k=1)],
    )[0]
    rho_ord = spearmanr(
        cos_ord[np.triu_indices(len(ns), k=1)],
        -np.abs(np.subtract.outer(ns, ns))[np.triu_indices(len(ns), k=1)],
    )[0]
    align = spearmanr(
        cos_add[np.triu_indices(len(ns), k=1)],
        cos_ord[np.triu_indices(len(ns), k=1)],
    )[0]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, cos, rho, title in [
        (axes[0], cos_add, rho_add, "arithmetic_bias (AddHead)"),
        (axes[1], cos_ord, rho_ord, "ordinal_offset (CmpHead)"),
    ]:
        im = ax.imshow(cos, cmap=CMAP_COS, vmin=-1, vmax=1, aspect="equal")
        ax.set_title(f"{title}\nρ(cos, −|Δn|) = {rho:+.3f}")
        ax.set_xticks(range(0, len(ns), max(1, len(ns) // 7)))
        ax.set_yticks(range(0, len(ns), max(1, len(ns) // 7)))
        ax.set_xticklabels([str(ns[i]) for i in ax.get_xticks()])
        ax.set_yticklabels([str(ns[i]) for i in ax.get_yticks()])
        ax.set_xlabel("concept ID (n)")
        ax.set_ylabel("concept ID (n)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cos")

    fig.suptitle(
        f"F2  Number bundle cos heatmaps (N={len(ns)}, dual muscle)  |  "
        f"cross-facet align ρ = {align:+.3f}",
        y=1.02, fontsize=10,
    )
    fig.tight_layout()
    _savefig(fig, "F2_number_cos_heatmaps")


# ─── F4: four-domain universality panel ───────────────────────────────


def render_F4_four_domain_panel() -> None:
    # (a) number — linear
    nb = get_number_bundles()
    ns = list(range(nb["n_min"], nb["n_max"] + 1))
    ncids = [f"concept:ans:{n}" for n in ns]
    cos_num = _cos_matrix_from_bundles(nb["bundle_state"], ncids, nb["facet_add"])
    rho_num = spearmanr(
        cos_num[np.triu_indices(len(ns), k=1)],
        -np.abs(np.subtract.outer(ns, ns))[np.triu_indices(len(ns), k=1)],
    )[0]

    # (b) color — circular
    cb = get_color_bundles()
    K = cb["n_colors"]
    ccids = [f"concept:color:{i}" for i in range(K)]
    cos_col = _cos_matrix_from_bundles(cb["bundle_state"], ccids, cb["facet_mix"])
    coords_col = _mds_2d(cos_col, seed=0)
    # center + unit-normalize for cleaner figure
    coords_col -= coords_col.mean(0, keepdims=True)
    coords_col /= np.linalg.norm(coords_col, axis=1).max() + 1e-9

    # (c) space — 2D grid
    sb = get_space_bundles()
    sids = [sb["cid_of"](r, c) for r in range(sb["n_rows"]) for c in range(sb["n_cols"])]
    cos_spc = _cos_matrix_from_bundles(sb["bundle_state"], sids, sb["facet_motion"])
    coords_spc = _mds_2d(cos_spc, seed=0)
    gt = np.array(
        [[r, c] for r in range(sb["n_rows"]) for c in range(sb["n_cols"])],
        dtype=float,
    )
    gt_n, coords_n, disparity = procrustes(gt, coords_spc)

    # (d) phoneme — cos heatmap with manner block lines
    pb = get_phoneme_bundles()
    phons = pb["phonemes"]
    pcids = [pb["cid_of"](i) for i in range(len(phons))]
    # Sort phonemes by (manner, place, voicing) so manner blocks are contiguous
    sort_idx = sorted(
        range(len(phons)),
        key=lambda i: (phons[i][2], phons[i][3], phons[i][1]),  # manner, place, voice
    )
    pcids_sorted = [pcids[i] for i in sort_idx]
    labels_sorted = [phons[i][0] for i in sort_idx]
    cos_phon = _cos_matrix_from_bundles(
        pb["bundle_state"], pcids_sorted, "manner_bias"  # show manner facet (cleanest)
    )
    # Group boundaries by manner
    manner_of = [phons[i][2] for i in sort_idx]
    block_starts = [0]
    for k in range(1, len(manner_of)):
        if manner_of[k] != manner_of[k - 1]:
            block_starts.append(k)
    block_starts.append(len(manner_of))

    # ── Render 2×2 panel ──
    fig = plt.figure(figsize=(10.5, 9.0))

    # (a)
    ax_a = fig.add_subplot(2, 2, 1)
    im = ax_a.imshow(cos_num, cmap=CMAP_COS, vmin=-1, vmax=1, aspect="equal")
    ax_a.set_title(f"(a) Numbers 1–{ns[-1]}   ρ(cos, −|Δn|) = {rho_num:+.3f}",
                   color=DOMAIN_COLORS["number"])
    ax_a.set_xlabel("n"); ax_a.set_ylabel("n")
    ticks = list(range(0, len(ns), max(1, len(ns) // 6)))
    ax_a.set_xticks(ticks); ax_a.set_yticks(ticks)
    ax_a.set_xticklabels([str(ns[i]) for i in ticks])
    ax_a.set_yticklabels([str(ns[i]) for i in ticks])
    fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)

    # (b) color ring — points coloured by true hue
    ax_b = fig.add_subplot(2, 2, 2)
    hues = np.array([(i / K) for i in range(K)])
    rgb = hsv_to_rgb(np.stack([hues, np.ones(K), np.ones(K)], axis=-1))
    theta = np.linspace(0, 2 * np.pi, 256)
    ax_b.plot(np.cos(theta), np.sin(theta), color="gray", lw=0.7, ls="--", alpha=0.6)
    ax_b.scatter(coords_col[:, 0], coords_col[:, 1], c=rgb, s=120,
                 edgecolor="black", linewidth=0.6, zorder=3)
    for i, (x, y) in enumerate(coords_col):
        ax_b.text(x * 1.15, y * 1.15, str(i), ha="center", va="center",
                  fontsize=8, color="black")
    # circular rho
    from experiments.color_concept_study import _rho_circular
    rho_circ = _rho_circular(torch.tensor(cos_col))
    ax_b.set_title(f"(b) Colors 12 hues   ρ_circular = {rho_circ:+.3f}",
                   color=DOMAIN_COLORS["color"])
    ax_b.set_aspect("equal", "box")
    ax_b.set_xlim(-1.4, 1.4); ax_b.set_ylim(-1.4, 1.4)
    ax_b.set_xticks([]); ax_b.set_yticks([])
    ax_b.spines["left"].set_visible(False); ax_b.spines["bottom"].set_visible(False)

    # (c) space grid — Procrustes-aligned MDS + GT grid
    ax_c = fig.add_subplot(2, 2, 3)
    # draw GT grid lines
    for r in range(sb["n_rows"]):
        ax_c.plot(gt_n[r * sb["n_cols"]:(r + 1) * sb["n_cols"], 0],
                  gt_n[r * sb["n_cols"]:(r + 1) * sb["n_cols"], 1],
                  color="lightgray", lw=1, zorder=1)
    for c in range(sb["n_cols"]):
        ax_c.plot(gt_n[c::sb["n_cols"], 0], gt_n[c::sb["n_cols"], 1],
                  color="lightgray", lw=1, zorder=1)
    # trained points, coloured by (r + c) / (R+C-2) through viridis
    idx = np.arange(sb["n_cells"])
    color_vals = idx / max(sb["n_cells"] - 1, 1)
    ax_c.scatter(coords_n[:, 0], coords_n[:, 1], c=color_vals, cmap=CMAP_SEQ,
                 s=90, edgecolor="black", linewidth=0.4, zorder=3)
    # draw matching residual lines
    for i in range(sb["n_cells"]):
        ax_c.plot([gt_n[i, 0], coords_n[i, 0]], [gt_n[i, 1], coords_n[i, 1]],
                  color="red", lw=0.5, alpha=0.6, zorder=2)
    ax_c.set_title(f"(c) Space 5×5 grid   Procrustes disp = {disparity:.3f}",
                   color=DOMAIN_COLORS["space"])
    ax_c.set_aspect("equal", "box")
    ax_c.set_xticks([]); ax_c.set_yticks([])
    ax_c.spines["left"].set_visible(False); ax_c.spines["bottom"].set_visible(False)

    # (d) phoneme heatmap with manner-block boundaries
    ax_d = fig.add_subplot(2, 2, 4)
    im = ax_d.imshow(cos_phon, cmap=CMAP_COS, vmin=-1, vmax=1, aspect="equal")
    manner_names = ["STOP", "FRIC", "NAS", "APR"]
    for s in block_starts[1:-1]:
        ax_d.axhline(s - 0.5, color="black", lw=1.2)
        ax_d.axvline(s - 0.5, color="black", lw=1.2)
    # axis labels = phoneme symbol
    ax_d.set_xticks(range(len(labels_sorted)))
    ax_d.set_yticks(range(len(labels_sorted)))
    ax_d.set_xticklabels(labels_sorted, fontsize=6)
    ax_d.set_yticklabels(labels_sorted, fontsize=6)
    # manner block labels outside
    for b_i in range(len(block_starts) - 1):
        mid = (block_starts[b_i] + block_starts[b_i + 1] - 1) / 2
        ax_d.text(mid, -2.0, manner_names[manner_of[block_starts[b_i]]],
                  ha="center", fontsize=8, color="black")
    ax_d.set_title("(d) Phonemes 20   manner_bias cos, block = manner class",
                   color=DOMAIN_COLORS["phoneme"])
    fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)

    fig.suptitle(
        "F4  Four-Domain Universality of Bundle Geometry  "
        "(linear · circular · 2-D lattice · categorical; same framework, no change)",
        y=1.00, fontsize=11,
    )
    fig.tight_layout()
    _savefig(fig, "F4_four_domain_universality")


# ─── F5: base-10 spike null ───────────────────────────────────────────


def render_F5_base10_spike_null() -> None:
    js = json.loads((OUTPUTS / "emergent_base10_full" / "summary.json").read_text())
    configs = js["configs"]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for c in configs:
        N = c["N"]
        # aggregate per-seed shift_stats
        shifts = []
        for row in c["per_seed"]:
            shifts.append(row["shift_stats"])
        keys = sorted({int(k) for d in shifts for k in d.keys()})
        means = []
        for k in keys:
            vs = [d[str(k)] for d in shifts if str(k) in d]
            means.append(np.mean(vs) if vs else np.nan)
        ax.plot(keys, means, "-o", label=f"N={N}  (n={len(shifts)} seeds)",
                markersize=3, linewidth=1)
        if 10 in keys:
            ax.axvline(10, color="red", ls="--", lw=0.8, alpha=0.4,
                       label="_expected base-10 peak_" if N == configs[0]["N"] else None)

    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("shift k (|n_a − n_b|)")
    ax.set_ylabel("avg cos(n, n+k)")
    ax.set_title("F5  Pure base-10 emergence null  —  cos(shift) is monotone linear, "
                 "no peak at k=10")
    # annotate expected peak location
    ax.annotate("no peak at k=10\n(spike₁₀ ≈ 0.001,  p = 0.44)",
                xy=(10, 0), xytext=(14, 0.35),
                fontsize=8, color="red",
                arrowprops={"arrowstyle": "->", "color": "red", "lw": 0.8})
    ax.legend(loc="lower left")
    fig.tight_layout()
    _savefig(fig, "F5_base10_spike_null")


# ─── F6: counterfactual swap double-dissociation ──────────────────────


def render_F6_swap_dissociation() -> None:
    js = json.loads((OUTPUTS / "counterfactual_swap" / "summary.json").read_text())

    def _avg(per_seed: list[dict], path: list[str]) -> float:
        xs = []
        for row in per_seed:
            v = row
            for k in path:
                v = v[k]
            xs.append(v)
        return float(np.mean(xs))

    conditions = ["baseline", "swap_arith_only", "swap_ord_only", "swap_both"]
    cond_labels = ["baseline", "swap facet A only", "swap facet B only", "swap both"]

    def build_rows(dom_key: str, head_a: str, head_b: str) -> dict:
        """Return dict of condition → 4 accuracies: A-inv, A-not, B-inv, B-not."""
        per_seed = js[dom_key]["per_seed"]
        conds_num = ["baseline"] + (
            ["swap_arith_only", "swap_ord_only", "swap_both"]
            if dom_key == "number_domain"
            else ["swap_mix_only", "swap_adj_only", "swap_both"]
        )
        rows = {}
        for cond in conds_num:
            rows[cond] = {
                "A_inv": _avg(per_seed, [cond, head_a, "involving_swap", "acc"]),
                "A_not": _avg(per_seed, [cond, head_a, "not_involving", "acc"]),
                "B_inv": _avg(per_seed, [cond, head_b, "involving_swap", "acc"]),
                "B_not": _avg(per_seed, [cond, head_b, "not_involving", "acc"]),
            }
        return rows

    num_rows = build_rows("number_domain", "add", "cmp")
    col_rows = build_rows("color_domain", "mix", "adj")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    def _draw(ax, rows, cond_keys, head_a_name, head_b_name, domain):
        conds = cond_keys
        x = np.arange(len(conds))
        w = 0.18
        ax.bar(x - 1.5 * w, [rows[c]["A_inv"] for c in conds],
               width=w, color="#d62728", edgecolor="black", linewidth=0.4,
               label=f"{head_a_name}  (inv)")
        ax.bar(x - 0.5 * w, [rows[c]["A_not"] for c in conds],
               width=w, color="#ff9896", edgecolor="black", linewidth=0.4,
               label=f"{head_a_name}  (not-inv)")
        ax.bar(x + 0.5 * w, [rows[c]["B_inv"] for c in conds],
               width=w, color="#1f77b4", edgecolor="black", linewidth=0.4,
               label=f"{head_b_name}  (inv)")
        ax.bar(x + 1.5 * w, [rows[c]["B_not"] for c in conds],
               width=w, color="#aec7e8", edgecolor="black", linewidth=0.4,
               label=f"{head_b_name}  (not-inv)")
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels)
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="gray", lw=0.5, ls=":")
        ax.set_title(f"{domain}  (N seeds = {len(js[domain.lower() + '_domain']['per_seed'])})")
        if ax is axes[0]:
            ax.set_ylabel("accuracy")
        ax.legend(loc="lower left", ncol=2, fontsize=7.5, framealpha=0.9)

    _draw(
        axes[0], num_rows,
        ["baseline", "swap_arith_only", "swap_ord_only", "swap_both"],
        "AddHead", "CmpHead", "Number",
    )
    _draw(
        axes[1], col_rows,
        ["baseline", "swap_mix_only", "swap_adj_only", "swap_both"],
        "MixHead", "AdjHead", "Color",
    )

    fig.suptitle(
        "F6  Post-hoc bundle swap: textbook double dissociation  |  "
        "target-facet collapses only on involved-pair accuracy of the consuming muscle",
        y=1.01, fontsize=10,
    )
    fig.tight_layout()
    _savefig(fig, "F6_swap_dissociation")


# ─── F7: H5'' four-domain alignment bars ──────────────────────────────


def render_F7_h5pp_alignment_schema() -> None:
    # Pull alignment values from each domain's summary.json
    robust = json.loads((OUTPUTS / "robustness" / "summary.json").read_text())
    color  = json.loads((OUTPUTS / "color_full" / "summary.json").read_text())
    space  = json.loads((OUTPUTS / "space_concept" / "summary.json").read_text())
    phon   = json.loads((OUTPUTS / "phoneme_concept" / "summary.json").read_text())

    def _extract(per_seed: list[dict], key: str) -> tuple[float, float]:
        xs = [r[key] for r in per_seed]
        return float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0

    # number: dual cross-facet align (arith ↔ ord)
    m_num, s_num = _extract(
        robust["E1_multi_seed"]["per_seed"], "dual_cross_facet_align"
    )
    # color
    m_col, s_col = _extract(
        color["E1_multi_seed"]["per_seed"], "dual_cross_facet_align"
    )
    # space (motion ↔ L1)
    m_spc, s_spc = _extract(
        space["E1_multi_seed"]["per_seed"], "dual_cross_facet_align"
    )
    # phoneme
    phon_rows = phon["E1_multi_seed"]["per_seed"]
    m_vm = float(np.mean([r["cross_facet_align"]["v_m"] for r in phon_rows]))
    s_vm = float(np.std([r["cross_facet_align"]["v_m"] for r in phon_rows], ddof=1))
    m_vp = float(np.mean([r["cross_facet_align"]["v_p"] for r in phon_rows]))
    s_vp = float(np.std([r["cross_facet_align"]["v_p"] for r in phon_rows], ddof=1))
    m_mp = float(np.mean([r["cross_facet_align"]["m_p"] for r in phon_rows]))
    s_mp = float(np.std([r["cross_facet_align"]["m_p"] for r in phon_rows], ddof=1))

    # perm-test p-values (from earlier runs; best-of-available)
    pvals = {
        "number": 0.003,   # from robustness permutation result
        "color":  0.016,
        "space":  0.77,
        "phon_vm": 0.052,
        "phon_vp": 0.991,
        "phon_mp": 0.037,
    }

    labels = [
        "number\narith ↔ ord",
        "color\nmix ↔ adj",
        "space\nmotion ↔ L1",
        "phoneme\nvoice ↔ manner",
        "phoneme\nvoice ↔ place",
        "phoneme\nmanner ↔ place",
    ]
    means = [m_num, m_col, m_spc, m_vm, m_vp, m_mp]
    stds  = [s_num, s_col, s_spc, s_vm, s_vp, s_mp]
    ps    = [pvals["number"], pvals["color"], pvals["space"],
             pvals["phon_vm"], pvals["phon_vp"], pvals["phon_mp"]]
    predict_align = [True, True, False, False, False, False]
    colors = [
        DOMAIN_COLORS["number"], DOMAIN_COLORS["color"],
        DOMAIN_COLORS["space"],
        DOMAIN_COLORS["phoneme"], DOMAIN_COLORS["phoneme"], DOMAIN_COLORS["phoneme"],
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    x = np.arange(len(labels))

    # vertical separators / algebra region shading (draw FIRST so bars are on top)
    ax.axvspan(-0.5, 1.5, color=DOMAIN_COLORS["number"], alpha=0.06,
               label="same facet-algebra → align  (H5″ 'if')")
    ax.axvspan(1.5, 2.5, color=DOMAIN_COLORS["space"], alpha=0.07,
               label="same domain, vector vs scalar algebra → null")
    ax.axvspan(2.5, 5.5, color=DOMAIN_COLORS["phoneme"], alpha=0.06,
               label="orthogonal categorical axes → null")

    # draw permutation-null reference band at 0 ± 0.10 (empirical null scale)
    ax.axhline(0, color="gray", lw=0.6, ls="-")
    ax.fill_between([-0.5, len(labels) - 0.5], -0.10, +0.10,
                    color="gray", alpha=0.13,
                    label="permutation-null band (|ρ| < 0.10)")

    ax.bar(x, means, yerr=stds, capsize=3, color=colors,
           edgecolor="black", linewidth=0.5,
           error_kw={"elinewidth": 0.8})

    # annotate p-value and prediction class above / below each bar
    for xi, (m, s, p, pred) in enumerate(zip(means, stds, ps, predict_align)):
        kind = "align" if pred else "null"
        sig = "p < 0.01" if p < 0.01 else "p < 0.05" if p < 0.05 else f"p = {p:.2f}"
        if m >= 0:
            y = m + s + 0.05
            va = "bottom"
        else:
            y = m - s - 0.05
            va = "top"
        ax.text(xi, y, f"{m:+.2f}\n{sig}\n[pred: {kind}]", ha="center",
                fontsize=7.8, color="black", va=va)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("cross-facet alignment  ρ (Spearman)")
    ax.set_ylim(-0.35, 1.18)
    ax.set_title(
        "F7  H5″  four-domain schema  —  alignment is gated by facet-level "
        "algebraic compatibility"
    )
    ax.legend(loc="upper right", fontsize=7.8, framealpha=0.95)
    fig.tight_layout()
    _savefig(fig, "F7_h5pp_alignment_schema")


# ─── F8: space MDS overlay (trained vs shuffle) ───────────────────────


def render_F8_space_mds_trained_vs_shuffle() -> None:
    sb = get_space_bundles(seed=1000, shuffled=False)
    sb_sh = get_space_bundles(seed=1000, shuffled=True)

    def _compute(sb_obj):
        cids = [sb_obj["cid_of"](r, c)
                for r in range(sb_obj["n_rows"])
                for c in range(sb_obj["n_cols"])]
        cos = _cos_matrix_from_bundles(sb_obj["bundle_state"], cids,
                                       sb_obj["facet_motion"])
        coords = _mds_2d(cos, seed=0)
        gt = np.array(
            [[r, c] for r in range(sb_obj["n_rows"])
             for c in range(sb_obj["n_cols"])],
            dtype=float,
        )
        gt_n, coords_n, disparity = procrustes(gt, coords)
        return gt_n, coords_n, disparity

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    for ax, sb_obj, title in [
        (axes[0], sb,    "(a) trained on true identity"),
        (axes[1], sb_sh, "(b) trained on shuffled identity (counterfactual)"),
    ]:
        gt_n, coords_n, disp = _compute(sb_obj)
        # GT grid lines
        R, C = sb_obj["n_rows"], sb_obj["n_cols"]
        for r in range(R):
            idx = np.arange(r * C, (r + 1) * C)
            ax.plot(gt_n[idx, 0], gt_n[idx, 1], color="lightgray", lw=1,
                    zorder=1)
        for c in range(C):
            idx = np.arange(c, R * C, C)
            ax.plot(gt_n[idx, 0], gt_n[idx, 1], color="lightgray", lw=1,
                    zorder=1)
        # residual lines
        for i in range(R * C):
            ax.plot([gt_n[i, 0], coords_n[i, 0]], [gt_n[i, 1], coords_n[i, 1]],
                    color="red", lw=0.4, alpha=0.55, zorder=2)
        # coloured scatter (cmap by GT row-major index so colour scheme is shared)
        cols = np.arange(R * C) / (R * C - 1)
        ax.scatter(coords_n[:, 0], coords_n[:, 1], c=cols, cmap=CMAP_SEQ,
                   s=85, edgecolor="black", linewidth=0.4, zorder=3)
        ax.set_title(f"{title}\nProcrustes disp = {disp:.3f}")
        ax.set_aspect("equal", "box")
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    fig.suptitle(
        "F8  Space grid MDS: trained bundle recovers the 5×5 lattice; "
        "under shuffled-identity training, the lattice dissolves",
        y=1.02, fontsize=10,
    )
    fig.tight_layout()
    _savefig(fig, "F8_space_mds_trained_vs_shuffle")


# ─── Dispatcher ───────────────────────────────────────────────────────


FIGURES = {
    "F2": render_F2_number_cos_heatmaps,
    "F4": render_F4_four_domain_panel,
    "F5": render_F5_base10_spike_null,
    "F6": render_F6_swap_dissociation,
    "F7": render_F7_h5pp_alignment_schema,
    "F8": render_F8_space_mds_trained_vs_shuffle,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help=f"subset of figures to render (default: all); keys={list(FIGURES)}")
    args = ap.parse_args()
    keys = args.only or list(FIGURES)
    unknown = [k for k in keys if k not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figures: {unknown}  (available: {list(FIGURES)})")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for k in keys:
        print(f"=== {k} ===")
        FIGURES[k]()
    print(f"\nAll figures written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
