# MRBI-stabilized implicit equilibrium features for simulated QNN readouts

Minimal reproducibility package for the manuscript:

**MRBI-Stabilized Implicit Equilibrium Features for Simulated Quantum Neural Network Readouts**

The package contains a compact, review-friendly code path for checking the numerical values and core implementation used in the manuscript.

## Contents

- `src/mrbi.py`: reusable implementation of Multiscale Residual-Based Initialization (MRBI), root solving, and the hybrid trigger-and-accept wrapper.
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
python scripts/reproduce_tables.py
python scripts/make_rho_sensitivity_figure.py
python scripts/make_qubit_width_heatmap.py
```

Generated outputs are written to `outputs/`.

## What is reproduced here

The scripts reproduce manuscript table snippets and lightweight diagnostic figures from archived summary CSV files. This is the intended minimal reproducibility layer for checking the numerical values used in the manuscript.

The full simulated-QNN sweeps are computationally slower and depend on the exact QNN software stack. They are therefore not the default quick path. The archived summaries preserve the values used in the manuscript, while `src/mrbi.py` exposes the solver-aware initialization method itself.

## Main manuscript summaries represented by the CSV files

- `main_qnn_results.csv`: main nine-task QNN comparison.
- `pca4_control.csv`: fair input-dimensionality control.
- `main_statistical_summary.csv`: dataset-level Wilcoxon summary.
- `classical_readout_check.csv`: logistic regression, SVM-RBF, MLP, RF, and GBM readout check.
- `spambase_external.csv`: external numeric Spambase check.
- `rho_sensitivity_delta_summary_for_plot.csv`: spectral-radius sensitivity figure data.
- `qubit_width_sensitivity_summary_for_plot.csv`: qubit-width sensitivity heatmap data.

## Related numerical-method background

The multiscale residual-detector idea is related to:

> Vlahek, D. *A hybrid gaussian–wavelet multiscale algorithm for zero localization in oscillatory functions*. Numerical Algorithms (2026). https://doi.org/10.1007/s11075-026-02484-8

That article studies Gaussian-wavelet multiscale zero localization for oscillatory scalar functions and its use as preprocessing for basin identification and initialization of classical root-finding methods. The present repository uses the same broad numerical motivation in a different application: initialization of a classical implicit equilibrium feature layer before a simulated QNN readout.

## Interpretation note

MRBI changes the initialization of the implicit equilibrium solve. It does not change the downstream QNN architecture. The best-profile MRBI columns in the manuscript are upper-envelope diagnostics over predefined MRBI profiles, not separately validated deployment models.
