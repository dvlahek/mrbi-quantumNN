# Opt-in smoothed-residual homotopy: numerical pilot

This is **development code**, not the method used in the completed full
97-method corrected-continuation campaign. No existing raw results,
manuscript metrics, implicit layer, train/test split or QNN implementation
is modified. Work in the separate branch
\`experiment/smoothed-residual-homotopy-20260928\`.

## Hypothesis

The completed label-free stage audit found that continuation and repeated
final-sigma optimization converge to nearly the same final candidates.
Changing the Gaussian scale only in the weak detector term is usually
insufficient to alter the optimization basin.

For the new opt-in arm, at stage \`j\`, replace the **leading** residual
term with

\`\`\`text
alpha * [(1-w_j) ||F(z)||_2
         + w_j ||(1/K) sum_{u in U} F(z + sigma_j u)||_2].
\`\`\`

Use the fixed, predeclared weights \`w=(0.75,0.50,0.25,0,0)\` and the
original \`full_balanced\` scales \`(0.70,0.25,0.08,0.02,0.02)\`,
including final refinement. At both final-scale stages \`w=0\`, so the
original unsmoothed residual, Newton proxy and regularizers define the
objective. The implicit equation \`F(z;x)=0\` and root solver never change.

Use one **common antithetic Gaussian probe set U** per input in every
stage and arm. Both smoothed and plain arms use the same detector
estimator, so the new experiment does not conflate changing the
residual term with resampling Gaussian probes.

## Four paired arms, not a profile search

- \`smoothed_continuation\`: smooth weights and descending sigma.
- \`smoothed_final_sigma\`: identical weights, sigma=0.02 throughout.
- \`plain_continuation\`: weights all zero, descending sigma.
- \`plain_final_sigma\`: weights all zero, sigma=0.02 throughout.

The first paired contrast isolates the role of scale in the smoothed
objective. The second contrast tests smoothing against a plain objective
at the *same* scale schedule and shared-probe estimator. The plain
arms are **new common-probe controls**, not a replacement for historical
full-campaign rows, which used independent probes by sigma.

Each arm has five L-BFGS-B stages and identical objective and primitive
residual evaluation **ceilings** per stage: by default 480 objective
calls and 5760 residual evaluations (12 residual calls per objective
for full_balanced). If L-BFGS-B converges early, the realized counts
may differ. The runner records realized objective, residual and Jacobian
calls, plus budget-hit frequency. It does **not** claim exact
equal-realized-budget matching. One final objective scoring evaluation
after the five stages is included in the reported total but excluded
from stage caps. Added diagnostic residuals and stage-root solves are
not counted as optimizer work.

## Scope and safeguards

Only the first eight **training** inputs from the original seeded
split/PCA transform are used for each dataset and seed. This reproduces
the full stage audit's fixed sampling rule, not a new random search.
No QNN is trained and labels are not used by any numerical algorithm.
Inspect candidate-root and *forced accepted* success separately:
an MRBI candidate is not an improvement if zero initialization already
finds an accepted solution. The 45 jobs are repeated development
measurements across 9 tasks and 5 seeds. The dataset, not individual
training inputs, is the unit for later inference.

The runner validates the original full 97-method reference CSV and all
three shard environment JSON files. It hashes the source CSV and records
the source and pilot Git SHA, package versions and cap parameters.
Changing the branch, files or parameters requires a new output
directory. Outputs never overwrite the full campaign or final-sigma
ablation.

## Ryzen WSL — first numerical job

\`\`\`bash
cd ~
git clone --branch experiment/smoothed-residual-homotopy-20260928 \\
  --single-branch https://github.com/dvlahek/mrbi-quantumNN.git \\
  mrbi-qnn-smoothed-homotopy
cd ~/mrbi-qnn-smoothed-homotopy
source ~/mrbi-qnn-full-ryzen/.venv/bin/activate
python scripts/test_smoothed_homotopy.py

FULL="$HOME/mrbi-qnn-full-ryzen/outputs/full_ryzen"
REF="$HOME/mrbi-qnn-full-ryzen/outputs/continuation_v1/main_raw.csv"
python -u scripts/run_smoothed_homotopy.py \\
  --reference-raw "$REF" \\
  --reference-env "$FULL/shard1/environment.json" \\
                  "$FULL/shard2/environment.json" \\
                  "$FULL/shard3/environment.json" \\
  --samples-per-job 8 --max-new-jobs 1
\`\`\`

The first job produces one pair row per input (with four arms),
\`stage_raw.csv\`, \`stage_summary.csv\`, \`dataset_summary.csv\` and
\`paired_sample_results.csv\` in
\`outputs/smoothed_residual_pilot_v1/\`. Inspect these before any
full development sweep. Then run the same command without
\`--max-new-jobs 1\`, or use \`--collect-only\` to regenerate summaries
from completed job files.

## Interpretation and next gate

Primary **exploratory** numerical endpoint: paired candidate-root
success of smoothed continuation minus smoothed final sigma, together
with the full distribution of realized residual evaluations.
Secondary: forced accepted success, residual, candidate distance, stage
root outcomes and the plain-objective controls. No QNN result is
available from this pilot. Do not optimize the method by selecting a
winner from individual datasets' test-label accuracies.

If there is a coherent numerical signal, run a separately versioned
strict equal-*realized*-budget test (or report success vs measured
evaluation budget) and hold out fresh seeds/tasks for validation before
claiming a multiscale improvement. Retain negative results.
