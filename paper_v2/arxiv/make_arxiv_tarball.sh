#!/bin/bash
# Pack the arXiv source (run after a full build so main.bbl is current).
cd "$(dirname "$0")"
tar czf kh1_paper_arxiv_source.tar.gz main.tex repro_appendix.tex references.bib main.bbl fig_recall_vs_length.png fig_sweep5_far_drills.png fig_kh1_recall.png
tar tzf kh1_paper_arxiv_source.tar.gz
