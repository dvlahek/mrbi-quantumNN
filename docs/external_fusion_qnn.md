# Final independent-source MRBI–implicit-layer QNN validation

## Motivation and locked design

The first nine tasks are derived from only breast-cancer, handwritten
digits and wine source datasets. On fresh seeds 20–24 of those tasks,
the fixed PCA+certified-MRBI implicit fusion QNN had mean paired
balanced accuracy **+0.012328** over the equal-width PCA+zero-implicit
fusion QNN. This positive controlled *new-seed* result is not
independent-source confirmation, and the original 4D PCA-QNN still had
a higher mean BA. The prior negative implicit-replacement QNN
experiments and the positive fusion experiment remain unmodified.

This final experiment tests the **same frozen fusion mechanism**
on *three additional source datasets*, neither derived from
breast cancer, digits nor wine. Data selection, six QNN inputs,
numeric profile, acceptance rule, seed list and primary endpoint
are fixed in this document before inspecting any result.

## Three UCI source datasets and acquisition

Download once using the provided data-acquisition script,
then run only on locally validated source bytes. Files are
**not** bundled into this repository or fetched during training.

| Dataset | Official raw file | Full source shape | UCI DOI |
|---|---|---:|---|
| Banknote Authentication | \`data_banknote_authentication.txt\` | 1,372 × 4 + binary label | [10.24432/C55P57](https://doi.org/10.24432/C55P57) |
| Ionosphere | \`ionosphere.data\` | 351 × 34 + good/bad label | [10.24432/C5W01B](https://doi.org/10.24432/C5W01B) |
| Connectionist Bench (Sonar, Mines vs. Rocks) | \`sonar.all-data\` | 208 × 60 + mine/rock label | [10.24432/C5T01Q](https://doi.org/10.24432/C5T01Q) |

Official UCI pages: [Banknote](https://archive.ics.uci.edu/dataset/267/banknote+authentication),
[Ionosphere](https://archive.ics.uci.edu/dataset/52/ionosphere)
and [Sonar](https://archive.ics.uci.edu/dataset/151/connectionist).
All three UCI pages state a CC BY 4.0 dataset license.
The script uses the repository's official UCI raw-file
URLs, checks exact shapes and label dictionaries, rejects
nonfinite or invalid CSV, and records SHA-256 of the actual
downloaded bytes in \`data/external_fusion/data_manifest.json\`.
Every run/resume verifies the same bytes and includes them
in \`environment.json\` and the per-job manifest. If UCI
serves a changed file, the existing source is **not**
silently replaced.

All datasets have at least 80 examples in both classes.
For each dataset and seed, the unchanged baseline
\`balanced_subsample\` draws **80 per class**, followed
by the unchanged stratified 70/30 split. All four PCA
coordinates are fitted using training data only; the 16D
hard implicit layer and QNN training settings are exactly
as in the completed fusion campaign. The new frozen seeds
are **25, 26, 27, 28, 29**, five per dataset, yielding
**15 jobs and 90 QNN trainings**.

## Six fixed arms and two information controls

The same four-qubit, two-layer classically simulated
QNN trains for 60 epochs on each arm with identically
reset \`seed+777\`. Only the input representation
differs.

1. \`pca_qnn\`: original PCA(4), a descriptive 4D
   performance comparator with different input
   projection parameterization.
2. \`pca_padded_qnn\`: PCA(4) plus **16 literal zeros**;
   its QNN input width is exactly 20D, matched
   to every following fusion arm. This tests the
   impact of extra input coordinates/parameterization
   without adding implicit information.
3. \`fusion_zero_qnn\`: PCA(4) concatenated with the
   train-standardized zero-root latent(16).
4. \`fusion_mrbi_qnn\`: PCA(4) plus train-standardized
   smoothed-coarse MRBI latent(16), using the
   original ungated candidate-acceptance rule.
5. \`fusion_certified_mrbi_qnn\`: exactly the same
   MRBI candidate calculation, accepted as a
   substitute only when the original root solver
   declares strict success. Otherwise retain
   the original zero-root latent(16).
6. \`fusion_permuted_certified_qnn\`: PCA(4) plus
   certified MRBI latent rows **deranged without
   replacement** within each train and test split,
   separately. Sattolo permutations depend only
   on split size and fixed seed offsets; never
   on labels or QNN scores. The marginal latent
   distribution is exactly unchanged but its
   sample-level pairing with PCA is broken.
   This is a *batch-level, label-blind diagnostic
   negative control*, **not** a deployable
   per-sample inference algorithm. In particular,
   the permuted latent need not satisfy
   \`F(z; x_PCA)=0\` for the receiving input;
   do not attribute source-root success to the
   receiving input.

The three implicit representations from zero, ungated
MRBI and certified MRBI are calculated only **once**
per input and then reused by all readout controls.
The original \`full_balanced\` profile, one-stage
sigma **0.70**, smoothed weight **0.75**,
10 antithetic probes and 480 objective /
5,760 optimizer F-call ceilings remain fixed.
Neither data labels nor readout predictions
affect numerical root acceptance. The original
PCA coordinates are bitwise unchanged for
every 20D arm, and each latent scaler is
fitted on the corresponding training states
only.

## Prespecified endpoints and claim boundaries

**Primary:** paired test balanced accuracy
\`fusion_certified_mrbi_qnn -
fusion_zero_qnn\`, reported for **each of
the three source datasets** as the five-seed
mean and for all 15 paired jobs. The
three-source average is the descriptive
external-source effect; seeds within each
source are not 15 independent datasets.

**Secondary (fixed):** certified minus
zero-padded PCA, certified minus label-blind
deranged-latent fusion, certified minus
ungated fusion, ungated minus zero
fusion, and the descriptive gap to
four-dimensional PCA-QNN. Include
actual source-specific scores, all
negative/tied outcomes, strict root-success
rates, and runtime/F/J diagnostics for
reproducibility. Compute no subgroup-only
headline or after-the-fact best arm.

Do not interpret a positive paired mean
as proof of causal information gain from
individual root rescues. The padded
control isolates input-width changes;
the deranged control tests loss of
sample-level pairing conditional on a
matched latent marginal distribution,
not a stronger general statistical
independence assertion. Three source
datasets are more informative than
three binary tasks on the same source,
but still a small external validation
set. No hardware execution or quantum
advantage is claimed.

If the primary effect is absent on
these three new sources, report
the negative result. Do not add
source datasets or seeds until a
favorable contrast appears.

## Run in Ryzen WSL

Create a fresh checkout separate from the
old 45-job fusion run. The data acquisition
step requires network access **once**
to the UCI raw endpoints. CI has no
network dependency; its tests use
synthetic fixtures.

\`\`\`bash
cd ~
git clone --branch experiment/external-fusion-controls-qnn-20260929 \\
  --single-branch https://github.com/dvlahek/mrbi-quantumNN.git \\
  mrbi-qnn-external-fusion
cd ~/mrbi-qnn-external-fusion
source ~/mrbi-qnn-full-ryzen/.venv/bin/activate

python -u scripts/test_external_fusion_qnn.py
python -u scripts/run_external_fusion_qnn.py --dry-run

python -u scripts/download_external_fusion_data.py
python -u scripts/download_external_fusion_data.py --offline
\`\`\`

If the downloader fails, do **not**
start the QNN campaign. Inspect the
error and the UCI URL; if necessary,
manually download the three exact
official raw files into the named
\`data/external_fusion\` folder and
re-run the downloader to validate
them and create the manifest.
Never bypass the schema and SHA
checks.

After all checks succeed, start a
**single** asynchronous worker:

\`\`\`bash
cd ~/mrbi-qnn-external-fusion
source ~/mrbi-qnn-full-ryzen/.venv/bin/activate
OUT="outputs/external_fusion_qnn_v1"
mkdir -p "$OUT"

if pgrep -f '[r]un_external_fusion_qnn.py' >/dev/null; then
    echo "Vanjski QNN pokus već radi."
else
    nohup python -u scripts/run_external_fusion_qnn.py \\
      > "$OUT/external.log" 2>&1 < /dev/null &
    echo $! > "$OUT/external.pid"
    disown
fi
\`\`\`

Track only job coverage and errors during
the run (no trial-based retuning):

\`\`\`bash
cd ~/mrbi-qnn-external-fusion
pgrep -af '[r]un_external_fusion_qnn.py' ||
  echo "Proces nije aktivan."
find outputs/external_fusion_qnn_v1/jobs \\
  -name '*_manifest.json' 2>/dev/null | wc -l
tail -n 12 outputs/external_fusion_qnn_v1/external.log
\`\`\`

Completion is **15/15 job manifests**,
**90 QNN metric records** and **7,200
per-input numerical diagnostic rows**
(three methods × 160 samples × 15
jobs). The same output directory
is resumable after a clean process
stop; unexpected partial jobs,
changed code, changed package
versions or changed source bytes
are rejected rather than overwritten.

After completion inspect
\`qnn_summary.json\`,
\`paired_job_results.csv\`,
\`dataset_summary.csv\`,
\`qnn_raw.csv\`,
\`feature_raw.csv\`,
\`environment.json\`
and the source
\`data/external_fusion/data_manifest.json\`.
