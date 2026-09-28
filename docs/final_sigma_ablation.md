# Paired final-scale ablation for the corrected MRBI method

This branch adds one fixed, pilot-informed comparison: the `forced_full_balanced_qnn`
continuation result from the **completed** eight-QNN core campaign versus a new
`final_sigma_repeated_full_balanced_qnn` control. No observations are used to tune
the implicit operator, QNN, or MRBI profile. The existing core campaign and
historical results are never overwritten.

## What is held fixed?

Both arms use the same breast-cancer, wine and digit tasks, seeds 0–4, dataset
splits, fixed implicit operator, root solver and forced acceptance policy. The
QNN uses four qubits, two layers, 60 epochs and seed `dataset_seed + 777`.
`full_balanced` uses `sigmas=(0.70, 0.25, 0.08, 0.02)`, 10 antithetic
Gaussian detector probes, four L-BFGS-B stages of at most 60 iterations and a
fifth refinement stage of at most 60 iterations.

The reference continuation optimizes at `0.70 → 0.25 → 0.08 → 0.02 → 0.02`.
The final-sigma control optimizes at `0.02 → 0.02 → 0.02 → 0.02 → 0.02`.
The control pre-generates the probe sets for all four nominal scales, preserving
the same detector probe draws and subsequent sample RNG state as the continuation
arm. The baseline's final-scale probes are the same ones as in the reference.
The code change is opt-in; ordinary corrected continuation retains its original
stage order and behavior.

**Computational matching is by five L-BFGS-B stages and equal per-stage iteration
ceilings.** Actual objective evaluations and wall times can differ because the
optimizer may converge early. We measure and report both; do not claim exact
evaluation-budget equality. A strict equal-evaluation-budget experiment would
require additional controls.

Only one forced profile is tested to isolate the scale schedule. The hybrid
trigger policies, alternative profiles and detector-off conditions are not part
of this comparison. The design was fixed after inspecting the earlier campaign
and should be reported as an exploratory ablation.

## Local WSL commands

Use a **new** checkout, keeping the successful core run untouched. Activate the
**same venv** used for the core experiment:

```bash
cd ~
git clone --branch experiment/continuation-final-sigma-ablation-20260928 \
  --single-branch https://github.com/dvlahek/mrbi-quantumNN.git mrbi-qnn-final-ablation
cd ~/mrbi-qnn-final-ablation
source ~/mrbi-qnn-continuation-run/.venv/bin/activate
python scripts/test_final_sigma_ablation.py

REF="$HOME/mrbi-qnn-core-night-8h/outputs/core_continuation_v1/main_raw.csv"
python scripts/run_final_sigma_ablation.py --reference-raw "$REF" \
  --max-new-jobs 1 --summarize
```

The reference environment is read from `environment.json` next to the
reference raw CSV. The driver checks that Python package versions match. It
also records the reference CSV SHA-256, source Git commit, control Git commit
and package versions. It refuses to resume with a different reference or
environment. Every completed control gets a one-row raw CSV, config and log,
and is skipped on resumption.

After inspecting the pilot, continue all pending tasks:

```bash
python scripts/run_final_sigma_ablation.py --reference-raw "$REF" --summarize
```

To run for approximately eight hours, add `--max-wall-hours 8`. This limit
is checked between jobs; the current job may finish after the deadline. An
interrupted run resumes by issuing the same command, without rerunning valid
jobs. `--collect-only --summarize` rebuilds the aggregate from saved files.

After all 45 controls complete, the driver generates
`outputs/final_sigma_ablation_v1/main_raw.csv`,
`paired_seed_results.csv`, `dataset_comparison.csv` and
`ablation_statistics.json`. A two-sided Wilcoxon value is descriptive and
exploratory; the dataset, not the individual seed, is the independent
comparison unit. Positive QNN differences mean continuation outperformed the
final-scale control.

## What to send for review

After the first job, send
`outputs/final_sigma_ablation_v1/jobs/breast_cancer_seed0_raw.csv` and its
`.log`. After completion, send the aggregated `main_raw.csv`,
`dataset_comparison.csv` and `ablation_statistics.json`.
