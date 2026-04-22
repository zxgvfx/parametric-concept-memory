# Submission packages

Two parallel LaTeX builds of the paper for different submission
channels. Both produce a 22–23 page PDF of the full four-domain
Parametric Concept Memory paper.

```
submission/
├── arxiv/        non-blind version with author info + GitHub link
│                 → upload here: https://arxiv.org/submit
├── neurips/      double-blind version (Anonymous authors, no repo URL)
│                 → upload to OpenReview when NeurIPS 2026 opens
└── _shared/      canonical .tex source; both builds copy from here
```

Both versions share the same body (`paper_body.tex`), abstract
(`abstract.tex`), teaser figure block (`teaser.tex`), and preamble
(`preamble_common.tex`). They differ only in `main.tex`:

- `arxiv/main.tex` — uses `\documentclass{article}`, puts
  **Xugang Zhang** + GitHub URL in the author block.
- `neurips/main.tex` — identical body, but
  `\author{Anonymous authors \\ Paper under double-blind review}`
  and a commented-out anonymous-code-URL block. Switch to the
  official `neurips_2026.sty` class when NeurIPS publishes it
  (see the top-of-file comment in `main.tex`).

## Prerequisites

- **XeLaTeX** (handles Unicode natively — needed for ρ, ≤, ≥, ±, ≈,
  subscripts, arrows, Greek, etc.)
- `latexmk` to drive the build
- `texlive-fonts-recommended` + `texlive-latex-extra`
- DejaVu Serif / DejaVu Sans / DejaVu Sans Mono + Latin Modern Math
  (installed by `texlive-fonts-*` and `fonts-dejavu`)

On Debian/Ubuntu:

```bash
apt-get install --no-install-recommends \
    texlive-xetex texlive-latex-recommended texlive-latex-extra \
    texlive-fonts-recommended fonts-dejavu latexmk lmodern
```

Or use **Overleaf** (free, browser-based): drag the `arxiv/` or
`neurips/` folder into a new Overleaf project, set compiler to
XeLaTeX in Menu → Compiler → *XeLaTeX*, hit Recompile. No local
install needed.

## Build locally

```bash
cd submission/arxiv         # or submission/neurips
latexmk -xelatex main.tex   # produces main.pdf
latexmk -C                  # clean all build artefacts
```

## Uploading to arXiv

1. Create an arXiv account at <https://arxiv.org/user/register> if
   you don't have one. New accounts on `cs.LG` need an
   **endorsement** — ask someone in your institution who has already
   posted to `cs.LG` or `cs.AI` for one; once endorsed you can post
   freely.
2. Go to <https://arxiv.org/submit>.
3. Choose **Submit a new article**.
4. Fields:
   - **Title**: *Concepts Collapse into Muscles: Domain-Topology-
     Adaptive Parametric Concept Memory*
   - **Authors**: Xugang Zhang
   - **Abstract**: paste `_shared/abstract.tex` (plain-text version
     — arXiv strips most LaTeX; use the raw text of the abstract
     with Unicode characters intact)
   - **Primary category**: `cs.LG`
   - **Cross-list**: `cs.AI`, `q-bio.NC` (cognitive-science audience)
   - **MSC / ACM**: leave blank
   - **Journal ref**: leave blank (fill after acceptance)
   - **DOI**: leave blank
   - **License**: CC BY 4.0 (recommended for open science)
5. Upload: **zip the whole `arxiv/` directory** (excluding `main.pdf`
   and build artefacts, so the contents are just the `.tex` files
   and `figures/`), plus `main.pdf` separately as the compiled PDF.
   arXiv will re-compile on its side with `pdflatex`. If your local
   build uses XeLaTeX (which is the default here for Unicode
   support), also include a file named `00README.XXX` with:

   ```
   fileformat pdflatex
   ```

   → arXiv will then trust your shipped PDF and skip re-compilation.
6. Preview, confirm author info, then **Submit**. Moderation takes
   1–3 business days. You'll get an arXiv ID like `2604.XXXXX`.

### What to include in the zip

```
arxiv_submission.zip
├── main.tex
├── preamble_common.tex
├── paper_body.tex
├── abstract.tex
├── teaser.tex
├── main.pdf              ← your compiled PDF (optional if arXiv re-builds)
├── 00README.XXX          ← "fileformat pdflatex" if shipping PDF as final
└── figures/*.png         ← all F2–F8 figures
```

You do **not** need to include `main.log`, `main.aux`, `main.out`,
`.fls`, `.fdb_latexmk`, `.xdv` — these are build artefacts.

## Uploading to NeurIPS (when the call opens)

NeurIPS 2026 is expected to open submissions around
**May 15, 2026** (abstract deadline typically ~1 week before full
paper deadline). Consult
<https://neurips.cc/Conferences/2026> for exact dates.

1. Download the official **`neurips_2026.sty`** style file from
   NeurIPS once published; swap it in as explained at the top of
   `neurips/main.tex`.
2. Create an **anonymous mirror** of the code repo (while your
   `github.com/zxgvfx/parametric-concept-memory` is private during
   review). Easiest: <https://anonymous.4open.science/> lets you
   upload a ZIP of the repo and get an anonymous URL.
3. Uncomment the `\section*{Code availability}` block at the bottom
   of `neurips/main.tex` and substitute the anonymous URL.
4. Re-compile NeurIPS PDF.
5. Create an OpenReview account if needed, then submit the PDF via
   the NeurIPS 2026 submission portal.
6. Keep your GitHub repo **private** during the review period (May
   → Aug/Sep 2026). Once notification arrives, flip public and
   update `PAPER.md` to use your name.

## Double-checking blind-ness

Before submitting the NeurIPS copy, `grep -iE 'xugang|zhang|zxgvfx|
github' neurips/*.tex figures/` should return **no matches**
(except possibly from figure filenames that mention e.g.
"F8_space_mds" — those are fine, they don't identify the author).
Body text and captions are already clean.

## arXiv + NeurIPS parallel timeline (recommended)

```
2026-04    · PAPER.md consistent; v0.1.0 tagged; repo private
2026-04-29 · this build → arXiv submission (takes 1–3 days to appear)
2026-04-30 · arXiv v1 live — shareable citation handle
2026-05-01 · start NeurIPS format polish (style file, page limits)
2026-05-15 · NeurIPS 2026 submission (double-blind)
2026-05-15 → 08-XX  · keep GitHub repo private
2026-08-XX · NeurIPS decisions
2026-09-XX · flip repo public, update PAPER.md author line,
             push an arXiv v2 with any post-review improvements
```
