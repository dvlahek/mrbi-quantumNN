# Frozen PCA–implicit MRBI fusion QNN: completed seeds-20–24 results

**Date:** 2026-09-29. **Plan:** `implicit_fusion_qnn_seeds20to24_v1`.
**Status:** full, prespecified new-seed campaign; all 45 dataset–seed
jobs complete. This document records measured outcomes, including the
remaining PCA-QNN gap and the limits of the source datasets. It does not
modify the method, code, earlier campaigns, or manuscript claims.

## Problem and fixed comparison

The preceding implicit-*replacement* QNN campaigns found that
smoothed-coarse MRBI rescued additional strict implicit roots but did
not increase mean QNN balanced accuracy. The new mechanism supplies
both the original PCA information and the implicit state to the
classically simulated QNN, testing the added value of a solver-selected
implicit feature **conditional on retaining the original features**:

`h(x)=[x_PCA (4), StandardScaler_train(z(x)) (16)] ∈ R^20`.

The four prespecified readouts are `pca_qnn` (4D input),
`fusion_zero_qnn` (20D), `fusion_mrbi_qnn` (20D,
original forced acceptance), and
`fusion_certified_mrbi_qnn` (20D, retaining the MRBI
candidate only after strict root success; otherwise reusing the
identical zero-start feature). All fusion arms share the same
PCA preprocessing, implicit layer, four-qubit/two-layer QNN,
60 training epochs and per-job training seed. The original
`full_balanced` coarse MRBI scale and smooth weighting were
frozen at sigma 0.70 and weight 0.75 before this run.
No best-per-task selection, no tuning on test labels, no
quantum-advantage claim.

The **primary prespecified endpoint** was paired test
balanced accuracy of certified-MRBI-fusion minus Zero-fusion
over nine tasks × five seeds **20–24**.

## Full numerical outcomes

Uploaded `qnn_raw.csv`, `paired_job_results.csv`,
`dataset_summary.csv`, `feature_raw.csv` and
`qnn_summary.json` contain **180 QNN rows**, 45 complete
dataset–seed pairs and **19,740 feature diagnostic rows**
(6,580 input records per fusion method across train/test).
Recomputed aggregate means match the supplied JSON and
dataset summaries.

| Method | Mean test balanced accuracy, 45 jobs |
|---|---:|
| PCA-QNN, 4D input | 0.974083 |
| Zero-fusion QNN, 20D | 0.948801 |
| Ungated-MRBI-fusion QNN, 20D | 0.954665 |
| Certified-MRBI-fusion QNN, 20D | **0.961129** |

The **primary paired mean difference** is **+0.012328**
(+1.2328 percentage points). Of 45 paired jobs, **14**
have positive differences, **26** are tied and **5**
negative. Across nine five-seed task averages, **seven**
are positive, **one** tied and **one** negative.
The descriptive secondary differences are
certified-minus-ungated **+0.006464**,
ungated-minus-zero **+0.005864**, and
certified-minus-PCA **−0.012954**.

| Dataset | Five-seed certified-minus-zero BA |
|---|---:|
| breast_cancer | +0.029167 |
| digits_1_vs_7 | −0.008333 |
| digits_2_vs_7 | +0.016667 |
| digits_3_vs_8 | +0.004167 |
| digits_4_vs_9 | 0.000000 |
| digits_5_vs_6 | +0.025000 |
| wine_0_vs_2 | +0.006667 |
| wine_1_vs_2 | +0.016190 |
| wine_binary | +0.021429 |

The positive mean persists in the three **source families**
breast cancer (+0.029167), digits (+0.007500 across five
binary digit tasks) and wine (+0.014762 across three binary
wine tasks). Nevertheless, **nine tasks are derived from
only three source datasets**, and the within-source binary
tasks have overlapping observations. The nominal two-sided
Wilcoxon signed-rank test over nine task-mean differences
returns **p=0.0390625**; do not present this as evidence
from nine independent external datasets or as a corrected,
multi-study confirmatory significance result. The
two-sided sign test over eight non-tied task means gives
p=0.0703125. A descriptive dataset-resampling 95% interval
for the mean of the nine task averages is about
[+0.0045,+0.0198], but also assumes exchangeability of
task units that is imperfect for shared-source datasets.
The primary outcome is an encouraging new-*seed* result
for a different, fixed **fusion** mechanism, not
independent-domain confirmation.

Seed-specific mean certified-minus-zero differences
over nine tasks are +0.037081 (seed 20),
+0.005401 (21), +0.005732 (22),
+0.013426 (23), and 0.000000 (24).
Removing seed 20 reduces the descriptive primary
mean to +0.006140, still positive. For seed 24,
all three fusion arms have identical BA on each
of the nine tasks, despite one new strict-root
rescue in that seed's test inputs. Do not claim
a uniform benefit on every seed.

## Numerically verified solver behavior

The feature table has no duplicated
(dataset,seed,split,input,method) key. All fusion
arms agree on the original zero-success flags.
The certified and ungated MRBI variants have
**identical per-input F/J, root and optimizer
work**; the certification rule only determines
which previously computed latent state becomes
the QNN input. Certified MRBI substitutions occur
**exactly on strict checkpoint root success**,
and MRBI is skipped after successful zero solves.
The attempted MRBI probe hashes match
between the two fusion-MRBI readouts.
The QNN raw table verifies all fusion input
dimensions are 20 (PCA comparator 4), the same
four-qubit/two-layer/60-epoch architecture and
`qnn_seed=dataset_seed+777`.

On **1,980 test inputs**, zero-root strict success
is **1,767** and certified MRBI rescues an
additional **48**, reaching **1,815** strict
roots. The corresponding training totals are
**4,148/4,600** zero successes and **4,254/4,600**
certified successes (+106 rescues).
The ungated MRBI variant selects failed
MRBI root features on **123** test and
**282** training inputs. The certified gate
reuses the zero feature on its **165**
test/**346** training failed checkpoints;
a reused zero feature can itself be
nonconverged. Thus the gate certifies only
the *MRBI substitutions*, not every feature.

On test inputs, average deployable F calls
are approximately 30.3 for zero-fusion
and 654.3 for certified/ungated MRBI-fusion.
These costs are reported transparently but
are not the scientific hypothesis being tested.
The key controlled comparison is
**representational value conditional on
retaining the original four PCA coordinates**,
with equal QNN input dimensions and training
settings across fusion methods.

## Claims, limitations, publication path

**Supported:** on new seeds 20–24 of the same
nine benchmark tasks, a QNN receiving both PCA
and strictly certified MRBI implicit features
achieves a +1.23-point paired mean BA relative
to the architecture-matched PCA+zero-implicit
fusion control. The strict certification rule
has an additional +0.65-point descriptive
mean over otherwise identical ungated fusion.
The measured improvement occurs despite a
modest additional 48 strict test roots, so
the data establish association between the
fusion representation policy and readout
performance, not a mechanism connecting
individual root rescues to specific corrected
classification errors.

**Not supported:** superiority to PCA-QNN,
a universal improvement on all tasks/seeds,
quantum advantage, state-of-the-art accuracy,
independent-dataset generalization, or
conclusions about individual prediction flips.
The PCA comparator uses 4D vs 20D input
projection and thus has a different
classical projection parameter count; the
three 20D fusion methods are architecture
matched. The supplied outputs contain
aggregate QNN metrics and per-input solver
diagnostics but not per-input QNN logits,
class probabilities or latent feature
arrays, so the hypothesized precise
information-retention mechanism is not
directly established.

For a journal submission emphasizing
MRBI + implicit-layer + QNN integration,
retain the original negative
implicit-replacement and solver-certificate
experiments as methodological motivation and
report them without hiding unsuccessful
outcomes. The new fusion result provides
a defensible *controlled* positive
representation-level contribution. A
stronger generalization argument would
use previously unused dataset families
and a prespecified test of how much
QNN performance depends on MRBI rescue
and PCA retention, without choosing datasets
or hyperparameters based on held-out test
results.
