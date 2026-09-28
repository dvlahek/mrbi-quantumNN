# Held-out single-stage MRBI: frozen numerical gate

This is a new, prespecified **numerical** comparison using seeds **5–9**
on the same nine datasets. It tests the development finding that a
successful **first coarse-scale candidate** can be lost during later
continuation. The previous full campaign and stage/checkpoint pilots used
seeds 0–4 and informed the method choice. The fresh seeds are a new
randomization of the same underlying datasets, not independent external
datasets, and may reuse individual observations in different splits.

The branch \`experiment/heldout-single-stage-20260928\` does not change
the production MRBI optimizer, full raw results, old pilots, QNN
architecture or manuscript. It runs **no QNN training**.

## Fixed design, before inspecting new-seed outcomes

Primary contrast: \`smoothed_coarse\` (sigma 0.70) against
\`smoothed_fine\` (sigma 0.02). Each uses one L-BFGS-B stage from zero,
exactly the same fixed \`full_balanced\` objective and the same
smoothed-residual weight **0.75**. Only sigma differs.

Secondary contrasts: \`plain_coarse\` and \`plain_fine\` with weight
0, otherwise using the same profile, first-stage procedure and shared
antithetic probes. These controls separate the contribution of a
coarse detector scale from the additional smoothed leading residual.
All four arms share the same Gaussian \`U\` **within each input**;
\`seed + 2000 + 1000003 * train_index\` determines U independently
for each new dataset–seed–input. No result-based profile selection.

For each input, attempt the original root solver from zero once. If it
succeeds, all four policies keep that result without MRBI work. If
zero-root fails, optimize **one stage** under the frozen per-stage
ceiling of 480 objective calls and 5,760 primitive F calls and attempt
the original strict root solver from that candidate. Use the
unchanged forced-acceptance rule to choose the returned root solution.
There is no continuation and no refinement stage in this new design.
The root solver always solves the *original*, unsmoothed equation.

Dataset plan: nine original benchmark datasets, new seeds 5–9, the
first eight **training** inputs after the original seeded stratified
split and fitted PCA transform. The split is label-stratified, but the
numerical optimizer, selection and tests never use held-out labels.
No test accuracy is calculated.

## Endpoints, cost and safeguards

Primary endpoint is the paired difference in strict accepted root
success among cases where zero-root failed. Because success from zero
is shared, also report total accepted successes over all inputs.
Secondary endpoints are measured primitive F and J calls and root
calls **including the shared zero solve, allocated once to each arm**,
plus optimizer objective calls, budget hits, residual and wall time.
Summarize per dataset and dataset–seed pair. Do not treat training
inputs from one dataset as independent datasets.

The inherited optimization stage logger evaluates **11 additional F**
and **one additional J** per attempted arm for diagnostic fields.
\`total_F_calls\`/\`total_J_calls\` include these real measured costs.
\`algorithm_F_calls\`/\`algorithm_J_calls\` subtract just those
instrumentation-only evaluations. The optimizer and root calls remain
counted. Per-stage *ceilings* are equal, not the realized number of
calls; comparisons must report both success and observed work.
Do not call the result an equal-realized-budget advantage.

The manifest records the Git commit, Python/platform and pinned
installed package versions, config and root tolerances. Each job
records hashes of its PCA-transformed training inputs and generated
implicit layer, its config and hashes of its CSV outputs. Resume
rejects changed Git SHA, parameters or partial/altered job files.
The old 97-method reference raw CSV covers seeds 0–4 only; the new
runner **does not require or claim source coverage for seeds 5–9**.
The data and layers for new seeds are generated reproducibly from
the frozen benchmark functions.

## Ryzen WSL

Use a fresh checkout so the completed \`~/mrbi-qnn-checkpoint\`
results and their provenance are left untouched:

\`\`\`bash
cd ~
git clone --branch experiment/heldout-single-stage-20260928 \\
  --single-branch https://github.com/dvlahek/mrbi-quantumNN.git \\
  mrbi-qnn-single-stage
cd ~/mrbi-qnn-single-stage
source ~/mrbi-qnn-full-ryzen/.venv/bin/activate

# Only synthetic tests: the CI does not evaluate held-out seeds.
python scripts/test_heldout_single_stage.py || exit 1

# Starts breast_cancer seed 5, eight paired training inputs only.
python -u scripts/run_heldout_single_stage.py --max-new-jobs 1
\`\`\`

For the first job, send
\`outputs/heldout_single_stage_v1/validation_statistics.json\`,
\`dataset_summary.csv\` and \`paired_sample_results.csv\`.
After checking output coverage/accounting, resume all remaining
jobs by running \`python -u scripts/run_heldout_single_stage.py\`
without \`--max-new-jobs 1\` in the same checkout. A completed job
is skipped, not rerun. Use \`--collect-only\` to regenerate aggregate
files without executing pending jobs. Output also includes
\`stage_raw.csv\`, \`environment.json\`, and per-job SHA manifests.

\`\`\`bash
cd ~/mrbi-qnn-single-stage
pgrep -af '[r]un_heldout_single_stage.py' || echo "Process inactive"
find outputs/heldout_single_stage_v1/jobs \\
  -name '*_manifest.json' 2>/dev/null | wc -l
\`\`\`

Expected total: **45 jobs**. Run just one process writing to the
same output directory at a time. On an interrupted *partial* job,
inspect its files before removing only that job's incomplete
\`_stages.csv\` / \`_pairs.csv\` / \`_manifest.json\` and retrying.

## Interpretation gate

Compare the prespecified smoothed coarse–fine paired success
difference and F/J cost on **all** fresh seeds; do not reselect
scales, weights, seeds, input indices or individual datasets after
seeing outcomes. A result on fresh seeds can support robustness
to new random initialization and splits on these nine tasks. An
external independent benchmark and, if needed, a further strict
equal-realized-work study would be required for broader
generalization or a budget-matched claim. Any QNN training is a
separately planned later study, not included here.
