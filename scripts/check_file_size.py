"""scripts/check_file_size.py — enforce the 500-line cap on source files.

Walks the repository (defaults to ``pcm/`` + ``experiments/`` +
``scripts/`` + ``tests/``) and exits non-zero with a tabulated report
when any ``.py`` file exceeds the configured line cap.

Usage::

    python -m scripts.check_file_size
    python -m scripts.check_file_size --cap 500
    python -m scripts.check_file_size --paths pcm experiments
    python -m scripts.check_file_size --report-all   # show top-10 even if PASS

PHILOSOPHY red line: "单文件 ≤ 500 行" — this script is the cheap CI
guard for that. Run it locally before pushing or wire it into a
pre-commit hook / GitHub Actions step.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


__all__ = ["count_lines", "scan", "main"]


DEFAULT_PATHS = ["pcm", "experiments", "scripts", "tests"]
DEFAULT_CAP = 500
EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".git", "outputs"}


def count_lines(path: Path) -> int:
    """Return the line count of a text file, defensively decoding."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    except UnicodeDecodeError:
        # Fall back to latin-1 so we never crash on a stray binary blob.
        with path.open("r", encoding="latin-1") as fh:
            return sum(1 for _ in fh)


def scan(roots: list[Path]) -> list[tuple[int, Path]]:
    """Return ``[(line_count, path), ...]`` sorted desc, excluding cache dirs."""
    out: list[tuple[int, Path]] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if any(part in EXCLUDED_DIRS for part in p.parts):
                continue
            out.append((count_lines(p), p))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP,
                    help=f"max allowed lines per .py file (default: {DEFAULT_CAP})")
    ap.add_argument("--paths", nargs="*", default=DEFAULT_PATHS,
                    help="root directories to scan (relative to repo root)")
    ap.add_argument("--report-all", action="store_true",
                    help="always print the top-10 largest files")
    args = ap.parse_args()

    roots = [Path(p) for p in args.paths]
    rows = scan(roots)
    over = [(n, p) for n, p in rows if n > args.cap]

    if over:
        print(f"FAIL: {len(over)} file(s) exceed the {args.cap}-line cap "
              f"(scanned: {[str(r) for r in roots]}):")
        print()
        print(f"{'lines':>6}  path")
        print(f"{'-' * 6}  {'-' * 60}")
        for n, p in over:
            print(f"{n:>6}  {p.as_posix()}")
        print()
        print("Refactor each over-cap file by splitting it along clear "
              "responsibility lines. See ``pcm/concept_graph/`` and the "
              "per-study packages under ``experiments/`` for the standard "
              "split pattern (one mixin / one figure / one runner per file).")
        return 1

    print(f"PASS: all {len(rows)} .py files in "
          f"{[str(r) for r in roots]} are ≤ {args.cap} lines.")
    if args.report_all and rows:
        print()
        print(f"  top-10 largest:")
        print(f"  {'lines':>6}  path")
        print(f"  {'-' * 6}  {'-' * 60}")
        for n, p in rows[:10]:
            print(f"  {n:>6}  {p.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
