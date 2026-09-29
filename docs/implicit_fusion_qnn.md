# PCA–implicit MRBI fusion for a classically simulated QNN

## Scientific question

Previous fixed-seed numerical studies show that the coarse
smoothed-residual MRBI stage increases the fraction of strict
roots recovered from a hard implicit layer. In the completed
implicit-*replacement* QNN campaigns (seeds 10–19), more strict
roots did **not** improve average balanced accuracy against
Zero-QNN. The certified replacement study on seeds 15–19
obtained 0.90017 versus 0.90055 BA for certified MRBI-QNN
and Zero-QNN, respectively. The original PCA-QNN readout
was near 0.976 BA. Replacing PCA with an often-nonconverged
implicit representation may discard discriminative information.
That is a mechanistic *hypothesis*, not an established cause.

The new test asks a different question: **can an MRBI-selected
implicit state supply discriminative information to a quantum
readout when the original PCA features remain available?**

For the fixed implicit map \`F(z; x_PCA)=0\`, the fusion
representation is

\`h(x) = [x_PCA, StandardScaler_train(z(x))] ∈ R^20\`.

Both branches are provided simultaneously to the **same original**
four-qubit, two-layer, 60-epoch classically simulated QNN.
The original first four PCA coordinates are copied exactly,
without fitting a second PCA/scale transform on test data.
Only the 16 latent dimensions receive an independent
training-fitted \`StandardScaler\`. There is no quantum
advantage claim.

## Four frozen comparisons (all nine tasks, new seeds 20–24)

1. \`pca_qnn\`: the original four-dimensional PCA input.
   This is a descriptive performance reference. Its input
   projection is 4→4 rather than 20→4, so parameter
   counts and initialization distribution differ from fusion.
2. \`fusion_zero_qnn\`: PCA(4) concatenated with the original
   zero-initialized implicit latent state(16).
3. \`fusion_mrbi_qnn\`: PCA(4) concatenated with the state
   produced by the **one-stage smoothed-coarse** MRBI
   procedure and the original forced-accept rule. This
   *ungated* arm may substitute failed root attempts.
4. \`fusion_certified_mrbi_qnn\`: same coarse MRBI
   attempt, probes and root solve, but the MRBI latent
   state is used **only if the original root solver declares
   strict success**. If the checkpoint fails, the
   corresponding zero-root latent state is reused, without
   any label-dependent decision or extra root solve.

The **three fusion QNNs** have exactly 20 input features and
the same fixed four-qubit/two-layer QNN architecture,
60 epochs, minibatch schedule, Adam parameters and
training seed \`dataset_seed+777\`, reset independently
for every method. The three methods share exactly the
same sampled train/test split, PCA transform, implicit
layer and per-input zero-root results. Coarse MRBI uses
the previously frozen \`full_balanced\` profile,
\`sigma=0.70\`, smooth-residual weight \`0.75\`,
10 shared antithetic probes, maximum 480 optimizer
objective calls and maximum 5760 optimizer F calls
per attempted input. All root-solves target the original
unsmoothed \`F=0\`. No training/validation/test label
is used in MRBI feature selection.

\`fusion_mrbi_qnn\` and \`fusion_certified_mrbi_qnn\`
use identical computed candidate roots. The gating
difference, not a change in numerical search, defines
their controlled comparison.

## Prespecified endpoints and interpretation

The **primary endpoint** is the paired mean difference in
test balanced accuracy \`fusion_certified_mrbi_qnn -
fusion_zero_qnn\` over 45 dataset–seed jobs
(nine tasks × five fresh seeds 20–24). This directly
tests the value of strictly certified MRBI states in
a QNN that retains the original PCA information.

Secondary endpoints are the corresponding ungated
fusion-MRBI-minus-zero difference, certified-minus-
ungated fusion difference, descriptive comparison to
PCA-QNN, and strict root-success rates. Actual F/J,
feature preparation and QNN training times are
reported for reproducibility **but are not the
claimed scientific contribution**.

Report all tasks and seeds, positive/negative/tied
paired differences and per-task five-seed means.
No best-per-dataset profile selection, test-set
feature selection, post hoc scale/weight tuning
or subgroup-only headline. Seeds 0–19 were used
for method development or earlier evaluation;
seeds **20–24 are frozen before inspection**.
Since the nine datasets are the same previously
examined dataset families, this is a new-seed
validation of a new fusion mechanism, **not**
independent-domain generalization. A stronger
claim would require genuinely new dataset
families. PCA-QNN's four-feature input size
is not a perfectly architecture-matched
20-feature fusion control.

A positive root-rescue result alone does **not**
establish a discriminative or quantum benefit.
If the primary QNN balanced-accuracy effect is
absent, report the negative result and do not
repeat seed searches on these same datasets
to manufacture a positive outcome.

## Ryzen WSL execution

Create a fresh checkout, isolated from all
previous numerical and QNN runs.

\`\`\`bash
cd ~
git clone --branch experiment/implicit-fusion-qnn-20260929 \\
  --single-branch https://github.com/dvlahek/mrbi-quantumNN.git \\
  mrbi-qnn-fusion-s20-s24

cd ~/mrbi-qnn-fusion-s20-s24
source ~/mrbi-qnn-full-ryzen/.venv/bin/activate
python scripts/test_implicit_fusion_qnn.py
python scripts/run_implicit_fusion_qnn.py --dry-run
\`\`\`

CI runs the synthetic tests and dry-run *without*
opening seed-20–24 results. When both commands
succeed, run all 45 jobs and examine the
aggregate endpoints only after completion:

\`\`\`bash
cd ~/mrbi-qnn-fusion-s20-s24
source ~/mrbi-qnn-full-ryzen/.venv/bin/activate
OUT="outputs/implicit_fusion_qnn_v1"
mkdir -p "$OUT"
if pgrep -f '[r]un_implicit_fusion_qnn.py' >/dev/null; then
    echo 'Pokus već radi; ne pokrećem drugi proces.'
else
    nohup python -u scripts/run_implicit_fusion_qnn.py \\
      > "$OUT/fusion.log" 2>&1 < /dev/null &
    echo $! > "$OUT/fusion.pid"
    disown
fi
\`\`\`

Check *only* process/job coverage while it runs:

\`\`\`bash
cd ~/mrbi-qnn-fusion-s20-s24
pgrep -af '[r]un_implicit_fusion_qnn.py' || echo "Nema aktivnog procesa"
find outputs/implicit_fusion_qnn_v1/jobs \\
  -name '*_manifest.json' 2>/dev/null | wc -l
tail -n 8 outputs/implicit_fusion_qnn_v1/fusion.log
\`\`\`

Completion is \`45/45\` jobs and
\`180\` QNN readout results. The run creates
\`environment.json\`, \`qnn_raw.csv\`,
\`paired_job_results.csv\`, \`dataset_summary.csv\`,
\`feature_raw.csv\`, \`qnn_summary.json\` and
per-job manifest/CSV triplets. Completed jobs
are checksum-validated on resume. A partial
job stops the runner for inspection instead
of silently overwriting files. Run one
writer per output directory.
