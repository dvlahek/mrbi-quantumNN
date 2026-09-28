# MRBI implicit features for simulated QNN readouts

Code and numerical results for *MRBI-Stabilized Implicit Equilibrium Features for Simulated Quantum Neural Network Readouts*, prepared for Neural Processing Letters.

The study compares representations under a fixed implicit operator and the same compact, classically simulated QNN readout. It does not claim quantum advantage or hardware performance.

## Files

- `experiments/mrbi.py`: solver-aware MRBI implementation with warm-started optimization at each decreasing Gaussian probe scale and an optional final refinement.
- `experiments/main_qnn_benchmark.py`: nine-task benchmark, including the fixed QNN architecture, classifier baselines, MRBI profiles and hybrid settings.
- `experiments/spambase_external.py` and `experiments/run_multistart_sanity_check.py`: supporting experiment entry points.
- `results/raw/`: the supplied historical per-seed main-benchmark CSVs.
- `results/summary_tables/article_qnn_final_summary1.csv`: recovered five-dataset summary, including the three tasks not fully represented in the raw CSVs.
- `results/summary_tables/`: manuscript tables and sensitivity-plot inputs.
- `results/supporting_raw/`: supplied Spambase and multistart data.
- `scripts/`: automated verification, figure/table regeneration and the resumable corrected-continuation campaign.

The complete pre-cleanup repository is preserved in [the archive branch](https://github.com/dvlahek/mrbi-quantumNN/tree/archive/pre-npl-cleanup-20260927).

## Install

Use Python 3.11 in WSL or another supported Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-qnn.txt
```

The numerical-summary checks only require `requirements.txt`. The full simulated-QNN experiment additionally requires PyTorch and PennyLane. The campaign records the Git commit, Python version, package versions and `pip freeze` in its output directory.

## Verify the available results

From the repository root:

```bash
python scripts/run_mrbi_smoke_test.py
python scripts/check_continuation.py
python scripts/check_main_raw.py
python scripts/check_recovered_summary.py
python scripts/check_consistency.py
python scripts/check_supporting_raw.py
python scripts/reproduce_tables.py
python scripts/check_generated_tables.py
python scripts/make_rho_sensitivity_figure.py
python scripts/make_qubit_width_heatmap.py
```

These are also run in GitHub Actions. The checks do not retrain the QNN.

The two historical main raw files have all five seeds for six tasks and only seed 0 for `wine_0_vs_2`. The recovered `article_qnn_final_summary1.csv` contains five-seed means for `breast_cancer`, `wine_binary`, `wine_0_vs_2`, `wine_1_vs_2` and `digits_1_vs_7`. It agrees exactly with 194 fully observed dataset-method means in the raw files. Together, these data verify all nine reported task-level QNN means. Full historical per-seed CSVs for three tasks are still unavailable.

The historical main table selects the best MRBI method by its five-seed **dataset-level mean**. The separate Spambase external check selects the best of two MRBI profiles within each seed, so its best-profile column is a descriptive upper envelope.

## Lower-cost, fixed-method experiment

A separate [core design](docs/core_design.md) evaluates eight fixed-after-pilot QNN representations (25 total readout-method rows) for every dataset and seed. It uses the same corrected continuation solver and QNN training settings as the full campaign. The experiment is isolated on branch `experiment/continuation-core-20260927`, so an already-running full campaign remains untouched.

From a separate checkout of that branch, run `python scripts/run_core_campaign.py --max-new-jobs 1` and then resume with `python scripts/run_core_campaign.py`. Once every job has completed, run `python scripts/summarize_core_campaign.py --require-complete`. The new result package has its own output directory and explicit `campaign_design` tag.

This core design was fixed after reviewing the first corrected-continuation pilot and historical results, so it is not a prospective preregistration. The campaign gives up the 97-method search and cannot support the old nine-task upper-envelope claim without updating the paper's result tables and interpretation.

## Overnight core run (approximately eight hours)

Use a separate checkout of the core branch; do not modify the Git commit or packages in an active campaign. From the core checkout, activate your existing Python environment and start:

```bash
mkdir -p outputs/core_continuation_v1
nohup python -u scripts/run_core_campaign.py --max-wall-hours 8 --summarize \
  > outputs/core_continuation_v1/night.log 2>&1 < /dev/null &
echo $! > outputs/core_continuation_v1/night.pid
disown
```

The runner checks its budget between jobs, allowing the current job to finish before stopping. It keeps each completed dataset/seed CSV and rebuilds `main_raw.csv`. With `--summarize`, it also writes method means and tables for any dataset with five completed seeds. This is an approximate duration, not a hard cutoff. Windows must remain awake, and the computer should be plugged in.

The next morning, see `outputs/core_continuation_v1/night.log`. Resume with `python scripts/run_core_campaign.py --summarize`, or recover aggregated data without launching a job using `python scripts/run_core_campaign.py --collect-only --summarize`. The runner checks the Git commit and package versions before it resumes.

## Fixed-profile scale ablation

The [paired final-sigma ablation](docs/final_sigma_ablation.md), on a separate
`experiment/continuation-final-sigma-ablation-20260928` branch, compares the
existing `forced_full_balanced_qnn` continuation results with one new QNN
control that optimizes at the final scale throughout the same number of
L-BFGS-B stages. It reads the completed core `main_raw.csv` as a reference;
it does **not** retrain the existing continuation arm. The design is
pilot-informed and its test is exploratory. The control records actual
objective calls and feature computation time, since matching stagewise
iteration ceilings does not guarantee identical work.

## Reproduce the corrected multiscale method

The current `experiments/mrbi.py` now optimizes `L_sigma` consecutively at the configured decreasing scales. Each stage starts at the preceding stage's candidate and uses `maxiter_per_scale`. An optional final pass at the smallest scale uses `refinement_iters`. The actual stage order and iteration budgets are tested by `scripts/check_continuation.py`.

This numerical change requires a new set of QNN results. The historical results above remain available for comparison, but should not be presented as results obtained with the corrected-continuation code.

Start with a single resumable job:

```bash
python scripts/run_continuation_campaign.py --max-new-jobs 1
```

Then resume the full predefined nine-task, five-seed campaign:

```bash
python scripts/run_continuation_campaign.py
python scripts/summarize_continuation.py --require-complete
```

Each of the 45 jobs has separate raw, summary, configuration and log files under `outputs/continuation_v1/jobs/`. The driver verifies a completed file before skipping it and refuses to mix results from a different implementation, Git commit or package environment. The full run is computationally expensive and is not part of CI.

After completion, `outputs/continuation_v1/main_raw.csv`, `main_qnn_results.csv`, `selected_profiles.csv`, `main_statistics.json`, `environment.json` and `environment.freeze.txt` provide the new result package. Review these results before updating the manuscript's historical tables. The nominal Wilcoxon result for a best-profile upper envelope is not corrected for profile selection.

## Method background

The Gaussian–wavelet zero-localization principle is developed in Vlahek, D., *A hybrid gaussian–wavelet multiscale algorithm for zero localization in oscillatory functions*, Numerical Algorithms (2026), https://doi.org/10.1007/s11075-026-02484-8. This study adapts its numerical motivation to vector-valued implicit equilibrium features before a simulated QNN readout.
