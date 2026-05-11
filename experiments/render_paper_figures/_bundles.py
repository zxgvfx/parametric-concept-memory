"""Lazy bundle-state caches used by the F2..F8 figure renderers.

Each ``get_*_bundles`` runs the corresponding training pipeline once
(seed-locked, ~30s on CPU), caches the result in-process, and returns
a dict with ``bundle_state`` plus per-domain metadata. Used by every
figure renderer that needs a per-concept cosine matrix.

Also hosts the small numerical helpers ``_cos_matrix_from_bundles``
and ``_mds_2d`` so each figure module can stay narrow.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import MDS


__all__ = [
    "get_number_bundles",
    "get_color_bundles",
    "get_space_bundles",
    "get_phoneme_bundles",
    "cos_matrix_from_bundles",
    "mds_2d",
]


# In-process LRU is overkill; a plain dict survives the lifetime of the
# CLI run and that's all we need.
_cache: dict[str, dict] = {}


def get_number_bundles(seed: int = 1000) -> dict:
    """Return a dual-muscle (Add + Cmp) bundle snapshot for paper §4 N=7."""
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


# ─── Numerical helpers ───────────────────────────────────────────────


def cos_matrix_from_bundles(bs: dict, cids: list[str], facet: str) -> np.ndarray:
    rows = []
    for cid in cids:
        rows.append(bs[cid][f"params.{facet}"])
    M = F.normalize(torch.stack(rows), dim=-1)
    return (M @ M.t()).numpy()


def mds_2d(cos: np.ndarray, seed: int = 0) -> np.ndarray:
    diss = np.clip(1.0 - cos, 0.0, 2.0)
    np.fill_diagonal(diss, 0.0)
    mds = MDS(
        n_components=2, dissimilarity="precomputed",
        random_state=seed, normalized_stress="auto",
        n_init=4, max_iter=500,
    )
    return mds.fit_transform(diss)
