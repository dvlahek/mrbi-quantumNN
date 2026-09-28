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

## Current full corrected-continuation campaign

The current evidence is the completed **full** benchmark: nine tasks, five seeds,
97 readout-method rows per task and seed (32 QNN representations), totaling
4,365 raw rows. The complete run used the corrected descending-scale MRBI
implementation and the fixed experiment settings in
`scripts/run_continuation_campaign.py`.

The best-per-task QNN upper envelope improves mean balanced accuracy over
zero-initialized QNN by +0.0211 on eight of nine tasks. This is **not** the
performance of one fixed method; the nominal Wilcoxon test is not adjusted
for choosing a profile separately on each task. PCA-QNN still has a higher
mean balanced accuracy on every task.

The full raw CSV and its three source-shard environments were generated on
the Ryzen workstation. The `results/raw/` and `results/summary_tables/`
files checked in before this campaign are historical audit material, **not**
the corrected full campaign's current results. Do not reproduce the paper's
current claims from those archived files.

## Fixed-profile final-sigma ablation

The [paired full-campaign ablation](docs/final_sigma_ablation.md) uses the
full benchmark's `forced_full_balanced_qnn` rows as its fixed continuation
reference and computes only the repeated-final-sigma control. Across nine
tasks, the mean balanced-accuracy difference (continuation minus control)
is +0.0023, with an exploratory two-sided Wilcoxon p=0.5703. The continuation
arm makes more objective calls. This does not establish an independent
advantage of descending-scale optimization for this fixed profile.

## Stage-level multiscale diagnosis

A [label-free, stage-level audit](docs/multiscale_stage_audit.md) on branch
`experiment/multiscale-stage-audit-20260928` compares matched continuation
and repeated-final-sigma trajectories on a fixed subset of the *full*
benchmark's training inputs. It does not retrain the QNN, alter the
completed full campaign or treat diagnostic samples as confirmatory data.

## Exploratory smoothed-residual numerical pilot

A separate [opt-in homotopy pilot](docs/smoothed_homotopy_pilot.md) compares
smoothed and plain residual objectives with descending and repeated
final-sigma schedules. It reuses the full 97-method campaign only for
provenance, performs **no QNN training**, and never changes the
completed benchmark or its reported results. The four arms have the
same objective/residual evaluation ceilings but can consume different
realized budgets; those counts are measured before drawing conclusions.

## Stage-one checkpoint pilot

A separate [solver-aware checkpoint development pilot](docs/checkpoint_homotopy_pilot.md)
retains an early successful root solve instead of discarding it during
subsequent continuation stages. It applies the same checkpoint and root
check to four matched arms, counts all F/J and root evaluations, and
skips MRBI altogether when zero initialization already converges.
The old stage diagnostic exposed this possibility; the checkpoint
pilot is exploratory and does not retrain QNN or change full results.

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
