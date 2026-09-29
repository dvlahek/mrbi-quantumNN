# Frozen one-stage MRBI–QNN comparison: new seeds 10–14

**Status:** prespecified experiment; no results from seeds 10–14 had
been inspected when the code and endpoints were frozen. This is a
separate exploratory-to-validation QNN campaign. The previously
completed 97-method campaign, final-sigma ablation, smoothed-residual
stage audit, numerical checkpoint study and single-stage numerical
holdout all remain untouched.

## Research question and limits

Does the **one-stage smoothed coarse initialization** improve the
downstream, classically simulated QNN readout over a matched one-stage
smoothed *fine* initialization? Compare also with **Zero-QNN** and
**PCA-QNN** under the same seeded data splits and training recipe.

The numerical experiment with seeds 5–9 found 26 successful coarse
rescues versus 11 fine rescues among 175 zero-root failures. That
result selected the fixed method before QNN seed 10–14 execution. It
does not itself establish a balanced-accuracy improvement. All nine
datasets are the same benchmark datasets used during development;
the new seeds are *new splits/implicit randomizations*, not new
independent datasets. Simulated QNN comparisons are not evidence of
quantum advantage.

## Four methods (fixed, no best-per-dataset selection)

1. \`pca_qnn\`: original four-dimensional PCA input to the original
   \`TorchQNN\` with four qubits and two circuit layers.
2. \`zero_qnn\`: same original latent-16 implicit representation
   solved from zero for every train/test input.
3. \`smoothed_coarse_qnn\`: solve from zero; when it fails, perform
   **one** 480-objective-call MRBI stage using sigma **0.70** and
   smoothed-residual weight **0.75**, then solve the original equation
   from the stage candidate. Keep the result under the unchanged
   forced-acceptance rule.
4. \`smoothed_fine_qnn\`: identical to the coarse arm except
   sigma **0.02**.

Both MRBI arms use the original fixed \`full_balanced\` numerical
weights, search radius, 10 shared antithetic Gaussian probes and the
same 5,760 primitive-F ceiling per optimization stage. Their
per-input probe seeds follow
\`seed + 2000 + 1000003 * index\`, with train indices
\`0..N_train-1\` and test indices
\`N_train..N_train+N_test-1\`. The helper providing stage
initialization and all F/J accounting is the same one used in the
successful numerical seed-holdout study, with an opt-in
\`return_solution\` output; its existing two-value interface remains
unchanged.

Use the original nine tasks and new seeds **10–14**, with 80
examples per class, stratified 70/30 split, training-only standardization
and PCA, latent dimension 16, spectral radius 2, input scale 1.1,
the hard implicit layer and the original QNN readout. Train **60
epochs** with the original Adam/minibatch schedule, four qubits, two
circuit layers, and an identical per-job readout seed
\`seed + 777\` independently reset for each arm. The classical
input-projection parameter shape necessarily differs for PCA (4
inputs) and the implicit representations (16 inputs); it is not
identical parameter initialization across different input widths.
The three implicit arms use exactly the same input width and
readout seeding. No external test-set evaluation is used for selecting
or modifying the numerical method.

## Frozen endpoints

**Primary:** paired balanced-accuracy difference
\`smoothed_coarse_qnn - smoothed_fine_qnn\` across 45
dataset–seed jobs.

**Secondary:** paired balanced-accuracy differences coarse minus
zero and coarse minus PCA, strict root success among zero-start
failures on the *test* representation, training/test root success,
measured test-time F/J and root calls, optimizer objective calls,
QNN training time and feature-construction time. Report all nine
per-dataset five-seed means and the overall paired mean, including
negative and tied outcomes. Do not select a favorable task or
method after seeing test accuracy. Analyze at the dataset and
dataset–seed level; do not treat test inputs from one dataset
as independent datasets.

PCA input preprocessing is shared by all arms and not charged
as an implicit root-solver cost. Solver F/J counts include the
zero-root attempt once per method, and the MRBI arms' recorded
\`algorithm_F_calls\`/\`algorithm_J_calls\` exclude the
stage logger's 11-F/1-J diagnostic evaluations. The
\`total_F_calls\`/\`total_J_calls\` fields retain those actual
instrumentation calls. Feature-construction wall times are
instrumented and must not be interpreted as strict equal-time
comparisons; QNN training times are recorded separately.

## Reproducibility and safe first run on Ryzen WSL

Create a **new checkout**. The first experiment does one full job
including **four 60-epoch QNN trainings** on
\`breast_cancer, seed=10\`; do not add \`--seeds 5 6 7 8 9\`,
retune the model or use prior test-accuracy results to choose
a new subset.

\`\`\`bash
cd ~
git clone --branch experiment/frozen-single-stage-qnn-20260929 \\
  --single-branch https://github.com/dvlahek/mrbi-quantumNN.git \\
  mrbi-qnn-frozen-single-stage
cd ~/mrbi-qnn-frozen-single-stage
source ~/mrbi-qnn-full-ryzen/.venv/bin/activate

# Synthetic checks only: do not open results for new seeds.
python scripts/test_frozen_single_stage_qnn.py || exit 1

# 45-job frozen coverage and provenance, without running the QNN.
python scripts/run_frozen_single_stage_qnn.py --dry-run || exit 1

# One dataset–seed job with its 4 fixed readouts.
python -u scripts/run_frozen_single_stage_qnn.py --max-new-jobs 1
\`\`\`

The experiment never asks for the old 97-method raw CSV because its
seed coverage does not include 10–14. The per-run
\`environment.json\` records Git SHA, Python/platform/package
versions, parameters, tasks and seeds. Each complete job has
\`_metrics.csv\`, \`_features.csv\`, and a final
\`_manifest.json\` with hashes of both CSVs, train/test PCA inputs
and labels (provenance only) and the implicit layer. Resume rejects
a changed Git SHA, plan, environment or modified/incomplete job.
Job manifests are written *last*.

Outputs:
\`outputs/frozen_one_stage_qnn_v1/qnn_raw.csv\`,
\`feature_raw.csv\`, \`paired_job_results.csv\`,
\`dataset_summary.csv\` and \`qnn_summary.json\`.

After first-job inspection, rerun
\`python -u scripts/run_frozen_single_stage_qnn.py\` from the
**same checkout and virtual environment** to finish remaining jobs.
Only one writer may use an output directory at a time.
To rebuild summaries from completed jobs only, use
\`--collect-only\`. The first QNN job may take substantially
longer than the label-free numerical pilot, because each method
trains a classically simulated circuit for 60 epochs. No runtime
prediction is made without actual timing.
