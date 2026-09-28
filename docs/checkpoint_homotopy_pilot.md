# Solver-aware stage-one checkpoint: matched numerical pilot

**Exploratory development branch:** \`experiment/checkpoint-homotopy-20260928\`.
This does not replace the completed full corrected-continuation benchmark,
the 45-pair QNN final-sigma ablation, or the finished 360-input diagnostic
pilot. No QNN training or test-label tuning occurs here.

## Motivation: a successful early root must not be thrown away

The complete 360-input diagnostic pilot found 239 successful root solves
from the smoothed-continuation first-stage candidate and 188 from its
last-stage candidate. Among 117 zero-start failures, the coarse checkpoint
rescued 24 cases; the smoothed repeated-final-sigma checkpoint rescued
3. The original final-only acceptance retained only 4 successes in
that difficult subset. First-stage inspection and the resulting policy
are **post hoc on the development sample**, so the same sample cannot
confirm generalization or establish a new statistical result.

## Predeclared checkpoint policy for all four arms

Each arm uses the original fixed full_balanced profile, the existing
solver tolerance, one fixed common antithetic probe set per input, and
the same per-stage objective/residual evaluation ceilings.

1. Run the original zero-root solver. If it succeeds, return its
   solution immediately; do not spend MRBI work on an already solved
   input.
2. Only if zero-root fails, optimize the first stage and perform **one**
   root solve from its candidate. If it meets the original strict
   root-success criterion, accept it and skip the remaining stages.
3. If the checkpoint solve fails, complete the four remaining stages
   (including refinement) and run the final root solve. Apply the
   existing forced-acceptance rule; never use a class label or an
   outcome from another arm to make a selection.

Exactly the same first-stage root check and early-exit rule is used
for \`smoothed_continuation\`, \`smoothed_final_sigma\`,
\`plain_continuation\`, and \`plain_final_sigma\`. The two-by-two
design preserves the original pilot's stage weights and scales.
It does not increase the number of starts or search profiles.

## Cost measurement and interpretation

Record optimization objective calls, primitive F and J calls
(including stage diagnostics), zero/checkpoint/final root calls,
and wall time. The zero-root result is shared within each paired
input and its cost is included once **per arm** in the comparison.
An arm that exits after stage one avoids all later optimizer work.
Both arms have equal allowed per-stage ceilings and identical root
check rules; **realized** cost may differ. Do not call this a
strict equal-realized-budget experiment.

Each input has a deterministic independent probe seed
\`seed + 2000 + 1000003 * train_index\`, reused unchanged across
the four arms. This avoids RNG stream changes when one method
exits early. It differs from the previous continuous per-seed
probe stream, so the new paired experiment must be interpreted
as a new version, not merged into older raw tables.

Use the same first eight **training** samples per dataset and seed.
This is a feasibility and instrumentation check on already
explored data. A new prespecified hold-out experiment will be
needed before performance claims, followed by a QNN experiment
only if numerical and cost endpoints justify it.

## Ryzen WSL — run one job first

\`\`\`bash
cd ~
git clone --branch experiment/checkpoint-homotopy-20260928 \\
  --single-branch https://github.com/dvlahek/mrbi-quantumNN.git \\
  mrbi-qnn-checkpoint
cd ~/mrbi-qnn-checkpoint
source ~/mrbi-qnn-full-ryzen/.venv/bin/activate
python scripts/test_checkpoint_homotopy.py

FULL="$HOME/mrbi-qnn-full-ryzen/outputs/full_ryzen"
REF="$HOME/mrbi-qnn-full-ryzen/outputs/continuation_v1/main_raw.csv"
python -u scripts/run_checkpoint_homotopy.py \\
  --reference-raw "$REF" \\
  --reference-env "$FULL/shard1/environment.json" \\
                  "$FULL/shard2/environment.json" \\
                  "$FULL/shard3/environment.json" \\
  --samples-per-job 8 --max-new-jobs 1
\`\`\`

The runner verifies 97 methods and 32 QNN rows in each requested full
reference job; the three shard manifests and package versions; the
reference SHA-256; and the current code commit. The manifest and
completed per-job data must remain unchanged when resuming.

The output directory is
\`outputs/checkpoint_homotopy_v1/\`, with
\`paired_sample_results.csv\`, \`stage_raw.csv\`,
\`dataset_summary.csv\` and \`checkpoint_statistics.json\`.
Review those files before running the 44 remaining jobs.

To continue the same full 45-job numerical pilot, rerun the command
without \`--max-new-jobs 1\`. To rebuild outputs without executing
new samples, add \`--collect-only\`. The completed full QNN benchmark
and the older pilot remain untouched.

## Gate

Compare successes specifically among zero-start failures **and**
the complete 360-input set, alongside total measured F/J cost and
root calls. Only promote this checkpoint method if a separate,
predeclared held-out experiment supports improved successful
solves at acceptable cost. The old 267/360 count from diagnostic
stage roots is an oracle bound, not a deployment measurement.
