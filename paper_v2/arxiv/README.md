# arXiv source: "Write Where Short Memory Cannot Reach"

Build (TeX Live 2021 or later; needs natbib, hyperref, booktabs, cleveref, listings, microtype, lmodern):

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Files:
- `main.tex`: the paper.
- `repro_appendix.tex`: the reproducibility appendix (`\input` by `main.tex`).
- `references.bib`: the bibliography. Every entry was checked against its primary source; the URL checked is in each entry's `url` field.
- `fig_*.png`: the figures, made by `../plot_kh1.py` and the plot scripts next to it.
- `main.bbl`: included so the arXiv build does not need to run BibTeX.

`make_arxiv_tarball.sh` packs exactly these files for upload.

Every number in the paper was audited against the raw files under `results/`; see the research log `MAMBA3_HAIS_POC_PLAN.md` §8.52.
