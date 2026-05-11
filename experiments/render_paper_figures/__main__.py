"""CLI entry: ``python -m experiments.render_paper_figures [--only F4 F7 ...]``."""
from __future__ import annotations

import argparse

from . import FIGURES
from ._style import OUT_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help=f"subset of figures to render (default: all); "
                         f"keys={list(FIGURES)}")
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
