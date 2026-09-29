# Certified one-stage MRBI–QNN: locked follow-up after seeds 10–14

## Why this is the next experiment

The completed frozen QNN campaign \`frozen_one_stage_qnn_seeds10to14_v1\`
has 45 jobs (nine datasets, five seeds) and 180 QNN readouts.
Across jobs, mean balanced accuracy is PCA-QNN **0.9749074**,
Zero-QNN **0.9033025**, ungated smoothed-coarse QNN
**0.8877028**, ungated smoothed-fine QNN **0.8956526**.
The primary coarse-minus-fine BA contrast is **−0.0079497**.
Those data do not support claiming a downstream QNN gain from
coarse MRBI. The earlier label-free numerical holdout on seeds
5–9 did support more strict root rescues from a coarse stage.

The \`feature_raw.csv\` from seeds 10–14 reveals a concrete
algorithmic flaw in the previous representation policy:
the coarse method chose an MRBI solution on **638** test inputs,
but **498** of these replacements still had
\`accepted_success=0\`. On training inputs, **1,103** of
**1,402** selected coarse replacements were also unsuccessful.
The old forced-accept rule may replace a failed zero solve by a
failed MRBI solve if its (still nonzero) residual is lower.
The hypothesis is that some of these uncertified feature changes
harm stability and classification. This is a proposed mechanism,
not an established cause. The complete seed-10–14 campaign is
now development data for this new rule and must **not** be reused
as its validation set.

## Frozen rule (no tuning after the new run)

Run the original shared zero-root attempt and the original
one-stage smoothed-coarse / smoothed-fine checkpoints with
identical antithetic probes, sigma **0.70** or **0.02**,
smoothed weight **0.75**, 480 objective/5,760 optimizer F-call
ceilings, original profile and original strict root solver.

The **certified-gate** arm retains the MRBI root **only when
the original solver declares strict root success** (solver flag,
finite root, finite residual and the frozen residual threshold
\`1e-8\`). Otherwise, return the already computed zero-root
feature, even if the unsuccessful MRBI attempt has a lower
nonzero residual. The gate uses **no labels**, confidence
threshold chosen on evaluation data, fitted predictor, extra
root attempts or additional optimization. Zero-root successes
still skip MRBI entirely. All solver work spent on unsuccessful
attempts remains fully charged to the gated arm.

For transparent attribution, **six** fixed QNN readouts are
compared using the same data splits, implicit layer and original
QNN training code:

* PCA-QNN
* Zero-QNN
* ungated smoothed-coarse MRBI-QNN
* ungated smoothed-fine MRBI-QNN
* certified smoothed-coarse MRBI-QNN
* certified smoothed-fine MRBI-QNN

The ungated and gated features share *exactly* the same computed
zero and checkpoint roots. Only the feature-selection policy
changes. Each arm separately fits its input \`StandardScaler\`
on the training split, and the original QNN uses
\`seed+777\` independently reset for each readout.
No test labels are used in preprocessing or solver selection.

## Evaluation: seeds 15–19, same nine tasks

The validation uses **seeds 15–19**, five per dataset, the
original cap of 80 per class and 70/30 stratified split, four PCA
input features, latent dimension 16, hard implicit spectral
radius 2.0, four qubits, two circuit layers and 60 QNN training
epochs. This is a seed holdout on the *same underlying datasets*,
not validation on unseen dataset families.

**Primary endpoint:** paired
\`certified_coarse_qnn − zero_qnn\` balanced accuracy over all
45 dataset–seed jobs. The primary question is if strict
solver-certified rescue improves the implicit QNN over its
zero-init counterpart.

**Secondary endpoints:** certified coarse versus its ungated
counterpart, certified coarse versus certified fine, descriptive
comparison to PCA-QNN, test root-success rescue counts, actual
optimizer F/J/root calls and full feature and readout time.
Report all positive, negative and tied dataset-level effects.
Do not cherry-pick successful datasets or adjust training
parameters after observing any seed-15–19 outcomes.

The gate **does not guarantee a BA improvement**. A root solution
can still be non-informative for the class label, and the zero
fallback can also be nonconverged. PCA-QNN is already near the
ceiling on these nine datasets and may remain stronger. If the
primary effect is absent, keep the result and stop searching
these dataset/seed splits for a favorable story; the completed
root-convergence improvement may motivate a solver-focused
article with an explicitly reported negative QNN result.

## Reproduction on Ryzen WSL

The numerical helper still has its previous two-return-value
API; the new QNN gate is derived from its existing strict
success fields and feature arrays. CI runs **synthetic-only**
tests and a \`--dry-run\` that does not train or load seed-15–19
tasks. All new scientific outcomes must run on Ryzen, with
the old full-campaign virtual environment including PyTorch
and PennyLane.

\`\`\`bash
cd ~
git clone --branch experiment/certified-gate-qnn-20260929 \\
  --single-branch https://github.com/dvlahek/mrbi-quantumNN.git \\
  mrbi-qnn-certified-gate
cd ~/mrbi-qnn-certified-gate
source ~/mrbi-qnn-full-ryzen/.venv/bin/activate
python scripts/test_certified_gate_qnn.py || exit 1
python scripts/run_certified_gate_qnn.py --dry-run || exit 1

OUT="outputs/certified_gate_qnn_v1"
mkdir -p "$OUT"
nohup python -u scripts/run_certified_gate_qnn.py \\
  > "$OUT/certified_gate.log" 2>&1 < /dev/null &
echo $! > "$OUT/certified_gate.pid"
disown
\`\`\`

This runs all 45 prespecified jobs *without examining pilot
performance midway*. Only one process may write to the output
directory. Resume uses the same command after a clean failure,
skipping completed jobs and rejecting mismatched/partial job
files. A completed job is committed with a manifest containing
the Git SHA and hashes of metrics/features outputs. The run
manifest records exact Python and installed package versions,
arms, seeds and numerical parameters. Do not edit this checkout
or update the branch in the middle of the experiment.

Monitor the log for exceptions and *job counts*, not for
per-dataset BA rankings:

\`\`\`bash
cd ~/mrbi-qnn-certified-gate
pgrep -af '[r]un_certified_gate_qnn.py' || echo 'Nema aktivnog procesa'
find outputs/certified_gate_qnn_v1/jobs \\
  -name '*_manifest.json' 2>/dev/null | wc -l
tail -n 8 outputs/certified_gate_qnn_v1/certified_gate.log
\`\`\`

After **45/45**, inspect aggregate
\`qnn_summary.json\`, \`paired_job_results.csv\`,
\`dataset_summary.csv\`, \`qnn_raw.csv\`,
\`feature_raw.csv\` and \`environment.json\`. No part
of this QNN study claims quantum advantage.
