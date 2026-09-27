# MRBI-stabilized implicit equilibrium features for simulated QNN readouts

Minimal reproducibility package for the manuscript:

**MRBI-Stabilized Implicit Equilibrium Features for Simulated Quantum Neural Network Readouts**

This package checks manuscript-level numerical summaries with a lightweight MRBI implementation. Recovered working-source candidates and supporting raw results are archived separately. Exact provenance of the nine-task main benchmark is not yet established.

## Contents

- `src/mrbi.py`: compact MRBI/hybrid implementation for smoke testing, not a copy of the historical experiment module.
- `experiments/source_snapshot/`: recovered historical code and a documented scale-loop issue.
- `results/supporting_raw/`: supplied Spambase and multistart raw and summary CSV files.
- `scripts/check_supporting_raw.py`: recomputes the supporting numerical checks from these raw files.
- `results/summary_tables/`: archived CSV summaries used for the manuscript tables and lightweight figures.
- `scripts/check_consistency.py`: verifies that archived summary values match the manuscript-level reported numbers.
- `scripts/reproduce_tables.py`: regenerates LaTeX table snippets from the archived CSV summaries.
- `scripts/make_rho_sensitivity_figure.py`: regenerates the spectral-radius sensitivity figure from CSV data.
- `scripts/make_qubit_width_heatmap.py`: regenerates the qubit-width sensitivity heatmap from CSV data.
- `scripts/run_mrbi_smoke_test.py`: runs a fast sanity check of the MRBI/hybrid solver code.
- `docs/related_numerical_algorithms_article.md`: reference note for the related Numerical Algorithms article.
- `paper/references_to_add.bib`: BibTeX entry for the related article.

This repository is intentionally minimal. It is not a full experiment dump. The QNN experiments in the manuscript were simulated classically; no quantum speedup, quantum advantage, or hardware-level result is claimed.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The core checks do not require PennyLane or PyTorch. For optional QNN experimentation, install:

```bash
pip install -r requirements-qnn.txt
```

## Quick checks

Run from the repository root:

```bash
python scripts/run_mrbi_smoke_test.py
python scripts/check_consistency.py
python scripts/check_supporting_raw.py
python scripts/reproduce_tables.py
python scripts/make_rho_sensitivity_figure.py
python scripts/make_qubit_width_heatmap.py
```

Generated outputs are written to `outputs/`.

## What is reproduced here

The scripts reproduce manuscript table snippets and lightweight diagnostic figures from archived summary CSV files. This is the intended minimal reproducibility layer for checking the numerical values used in the manuscript.

The full simulated-QNN sweeps are computationally slower and depend on the exact QNN software stack. They are therefore not the default quick path. The archived summaries preserve the values used in the manuscript, while `src/mrbi.py` exposes the solver-aware initialization method itself.

## Experimental provenance and limitations

The recovered `mrbi.py` code is preserved without changing its numerical behavior. Its scale loop passes the final sigma on every iteration, so it does not implement nominal descending-scale continuation. The separate compact `src/mrbi.py` uses a different Newton correction and hybrid policy. Neither version has been independently certified as the source of the nine-task manuscript results. See `experiments/source_snapshot/README.md`.

The supplied raw Spambase CSV confirms the current mean balanced accuracies 0.8667 (PCA-QNN), 0.8511 (Zero-QNN), and 0.8617 (Best MRBI-QNN). Best MRBI-QNN is the best of two profiles selected **per seed**, not the average of a single fixed profile. The archived multistart raw file also matches the supplementary table. The original nine-task per-seed `article_qnn_final_raw1.csv` and `article_qnn_final_raw2.csv` files have not been supplied.

The main-table mean `delta_pca` is -0.0196 when computed from the nine displayed task rows. The earlier -0.0198 was inconsistent with those rows; only the summary value and its check were corrected.

## Main manuscript summaries represented by the CSV files

- `main_qnn_results.csv`: main nine-task QNN comparison.
- `pca4_control.csv`: fair input-dimensionality control.
- `main_statistical_summary.csv`: dataset-level Wilcoxon summary.
- `classical_readout_check.csv`: logistic regression, SVM-RBF, MLP, RF, and GBM readout check.
- `spambase_external.csv`: external numeric Spambase check for the current NPL submission version.
- `rho_sensitivity_delta_summary_for_plot.csv`: spectral-radius sensitivity figure data.
- `qubit_width_sensitivity_summary_for_plot.csv`: qubit-width sensitivity heatmap data.

## Related numerical-method background

The multiscale residual-detector idea is related to:

> Vlahek, D. *A hybrid gaussian–wavelet multiscale algorithm for zero localization in oscillatory functions*. Numerical Algorithms (2026). https://doi.org/10.1007/s11075-026-02484-8

That article studies Gaussian-wavelet multiscale zero localization for oscillatory scalar functions and its use as preprocessing for basin identification and initialization of classical root-finding methods. The present repository uses the same broad numerical motivation in a different application: initialization of a classical implicit equilibrium feature layer before a simulated QNN readout.

## Interpretation note

MRBI changes the initialization of the implicit equilibrium solve. It does not change the downstream QNN architecture. The best-profile MRBI columns in the manuscript are upper-envelope diagnostics over predefined MRBI profiles, not separately validated deployment models. A deployment-oriented version should use a fixed standard profile or select a profile on a validation split.
