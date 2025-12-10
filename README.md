# Deal-Origination-Analytics

**Deal Origination Blind Spots in Private Equity: Alternative Data Reveals Hidden PE Targets**

Author: Ayankhan Pathan (ESSEC Strategic Business Analytics)

This repository accompanies the ESSEC project investigating whether multi-signal alternative data can surface mid-market targets earlier than traditional origination channels. The materials are academic in style but designed with industry applicability: code, synthetic datasets, visualizations, and a LaTeX appendix suitable for inclusion in a project report.

## Contents
- `code/` : scripts to generate synthetic alt-data signals, build indices, and train a simple early-warning model.
- `data/` : synthetic datasets (signals + a small simulated deal pipeline).
- `images/` : pre-generated charts for inclusion in the report.
- `report/` : LaTeX appendix with figures (compile with pdflatex).
- `appendix/` : code snippets and tables used as appendices in the report.

## Quick start
1. Install Python dependencies:
```bash
pip install -r requirements.txt
python code/generate_visuals.py
pdflatex report/main.tex
