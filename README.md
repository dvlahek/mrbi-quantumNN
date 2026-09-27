# MRBI implicit features for simulated QNN readouts

Code and result checks for *MRBI-Stabilized Implicit Equilibrium Features for Simulated Quantum Neural Network Readouts*, prepared for Neural Processing Letters.

The study compares feature representations under the same compact, classically simulated QNN readout. The implicit operator is fixed. The experiments do not establish quantum advantage or hardware performance.

## What is in this repository

- `experiments/main_qnn_benchmark.py` — the recovered nine-task benchmark with its original numerical routines and a cleaned command-line interface. The default arguments match the recorded main experiment.
- `experiments/mrbi.py` — the recovered MRBI solver module used by the benchmark candidate. Only an inaccurate comment was corrected; its numerical operations were left unchanged.
- `experiments/spambase_external.py` and `experiments/spambase_external_config.json` — the Spambase external-check runner and its supplied configuration.
- `experiments/run_multistart_sanity_check.py` — the multistart diagnostic used in the supplementary analysis.
- `results/raw/` — the two supplied main-benchmark CSVs with the original result columns; line endings have been normalized to LF.
- `results/supporting_raw/` — the Spambase and multistart raw and summary CSVs.
- `results/summary_tables/` — archived manuscript tables and data for the spectral-radius and qubit-width plots.
- `scripts/` — checks and regeneration of table snippets and diagnostic figures.

Earlier prototypes and files outside the submitted paper are kept in the [`archive/pre-npl-cleanup-20260927`](https://github.com/dvlahek/mrbi-quantumNN/tree/archive/pre-npl-cleanup-20260927) branch. The working files were not deleted from Git history.

## Install and verify the reported summaries

Use Python 3.11 or a compatible environment:

```bash
python -m pip install -r requirements.txt
python scripts/run_mrbi_smoke_test.py
python scripts/check_main_raw.py
python scripts/check_consistency.py
python scripts/check_supporting_raw.py
python scripts/reproduce_tables.py
python scripts/check_generated_tables.py
python scripts/make_rho_sensitivity_figure.py
python scripts/make_qubit_width_heatmap.py
```

Regenerated tables and plots are written to `outputs/`. GitHub Actions runs the same checks. They do not retrain the QNN.

### Coverage of the supplied main-benchmark raw files

`article_qnn_final_raw.csv` and `article_qnn_final_raw2.csv` contain all five seeds for six of the nine tasks: `breast_cancer`, `wine_binary`, `digits_2_vs_7`, `digits_3_vs_8`, `digits_4_vs_9`, and `digits_5_vs_6`. For each of these, the reported PCA, zero-initialized and best-MRBI QNN means can be recalculated from the raw data.

`wine_0_vs_2` currently has only seed 0. The supplied files have no rows for `wine_1_vs_2` or `digits_1_vs_7`. The nine-task mean, Wilcoxon statistic and those three dataset rows therefore remain checks against archived summary tables, not full raw-data reproductions. `check_main_raw.py` prints this coverage explicitly.

For the main table, the selected MRBI profile is the method with the highest **mean over five seeds within a dataset**. The Spambase external-check summary instead uses the best of two profiles **within each seed**. The latter is a descriptive upper envelope, not a validated single deployment configuration.

### Source behavior and full reruns

The recovered `experiments/mrbi.py` passes the final scale to each optimization pass, even though its configuration lists several scales. It must not be described as implementing descending-scale continuation. The unedited working-file version is preserved on the archive branch. The previous, smaller `src/mrbi.py` used a different Newton proxy and has also been moved out of `main`.

To run the simulated-QNN benchmark, install the optional requirements and start the benchmark script:

```bash
python -m pip install -r requirements-qnn.txt
python experiments/main_qnn_benchmark.py --help
python experiments/main_qnn_benchmark.py --datasets breast_cancer --seeds 0
```

The last command is a single-dataset, single-seed example, not the full experiment. The full default run uses nine tasks, five seeds, four qubits, 16 implicit dimensions, spectral radius 2.0, at most 80 samples per class and 60 QNN epochs. It is substantially slower than the summary checks. Exact software versions and the missing per-seed CSVs for three tasks have not been recovered, so full end-to-end reproduction of all manuscript figures is not yet certified.

## Data availability and interpretation

The repository contains the available per-seed results, archived manuscript-level summaries, the relevant numerical code, and scripts for checking tables and regenerating diagnostic figures. Only the checks identified above have been verified against per-seed raw output. The simulated QNN and the fixed implicit operator define the scope of the study.

The scalar Gaussian–wavelet zero-localization principle is described in Vlahek, D., *A hybrid gaussian–wavelet multiscale algorithm for zero localization in oscillatory functions*, Numerical Algorithms (2026), https://doi.org/10.1007/s11075-026-02484-8. The present study adapts its numerical motivation to vector-valued implicit features before a simulated QNN readout.
