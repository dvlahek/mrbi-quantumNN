# Independent-source PCA–MRBI fusion QNN: frozen 25–29 results

**Study:** `external_fusion_qnn_three_sources_seeds25to29_v1`.
**Date:** 2026-09-29. **Scope:** three new UCI source families,
five seeds each, 15 completed jobs and 90 QNN results.
**Status:** all uploaded paired/metric/feature CSVs and
summary JSON have been inspected. This records a negative
prespecified primary result; do not retune the same test sets.

## Problem and motivation

The earlier 45-job fusion campaign on seeds 20–24 of
nine tasks from three pre-explored source datasets found
a **+0.012328** paired mean balanced-accuracy advantage
for PCA+certified-MRBI over PCA+zero-latent fusion.
That new-seed result motivated an independent-source
replication with the **unchanged** coarse-smoothing,
strict solver certificate, four-qubit/two-layer QNN
and 60 training epochs.

The external experiment freezes Banknote Authentication,
Ionosphere and Sonar, 80 examples per class, seeds 25–29,
and six QNN methods. Five 20D input methods are
architecture-matched; the original PCA-only QNN
has a 4D input projection and is descriptive.
The two new 20D controls are a PCA+16-literal-zero
input and a label-blind train/test within-split
derangement of certified latent rows. All QNNs
are run by classical simulation, not quantum hardware.

## Primary and secondary outcomes

The **prespecified primary endpoint** was the paired
test balanced-accuracy difference
`fusion_certified_mrbi_qnn − fusion_zero_qnn`
over the 15 jobs. The observed mean is **−0.00555556**
(−0.56 percentage point). It is positive in 5 jobs,
negative in 7 and tied in 3. All **three**
source-specific five-seed means are negative:
Banknote −0.004167, Ionosphere −0.004167
and Sonar −0.008333. The fixed certification-fusion
hypothesis from the earlier study therefore **did not
replicate on these independent source families**.

| Mean BA, 15 jobs | Value |
|---|---:|
| PCA-QNN, 4D | 0.836111 |
| PCA+16 zero padding, 20D | 0.818056 |
| PCA+zero-latent fusion, 20D | 0.779167 |
| PCA+ungated-MRBI fusion, 20D | 0.791667 |
| PCA+certified-MRBI fusion, 20D | 0.773611 |
| PCA+deranged certified latent, 20D | 0.680556 |

| Source | PCA 4D | Padded PCA 20D | Zero-fusion 20D | Ungated MRBI-fusion 20D | Certified MRBI-fusion 20D | Deranged certified latent 20D |
|---|---:|---:|---:|---:|---:|---:|
| Banknote | 0.950000 | 0.954167 | 0.904167 | 0.904167 | 0.900000 | 0.812500 |
| Ionosphere | 0.812500 | 0.787500 | 0.754167 | 0.745833 | 0.750000 | 0.687500 |
| Sonar | 0.745833 | 0.712500 | 0.679167 | 0.725000 | 0.670833 | 0.541667 |

Prespecified secondary mean differences for
certified-MRBI fusion are **−0.044444** versus
20D zero-padded PCA, **+0.093056** versus the
20D deranged-latent diagnostic and **−0.018056**
versus otherwise numerically identical ungated
MRBI-fusion. The *ungated* MRBI-fusion has a
positive **+0.012500** descriptive difference
versus Zero-fusion (mostly Sonar), but that is
a secondary contrast, not the frozen primary
hypothesis and not a reason to choose a
different 'winning' method post hoc. It does
not exceed PCA-QNN or the 20D zero-padded
PCA baseline on the full external suite.

The derangement control deliberately breaks
the relationship between each test input's
PCA and certified latent state while retaining
the within-split latent marginal distribution.
Its lower BA indicates that sample-level
pairing matters for that representation.
It does **not** show that the certified
MRBI state contributes useful *incremental*
class information over PCA, since both
zero-fusion and certified fusion underperform
the equal-width padded-PCA control.

## Numerical and provenance checks

The uploaded files have **15/15 jobs**,
**90 QNN rows**, **7,200 numerical-feature rows**
(three methods × 160 inputs × 15 jobs).
No duplicate dataset–seed–method or
dataset–seed–split–input–method keys appear.
Recomputed mean balanced accuracies, paired
deltas and per-source summaries match the
submitted summary JSON. All 20D methods
use four qubits, two circuit layers and
60 training epochs with the original
`qnn_seed=seed+777`.

The three numerical arms have identical
per-input zero-success flags; certified and
ungated MRBI arms have identical per-input
checkpoint flags, probe hashes and measured
F/J/root work. The certified arm selects
MRBI exactly upon strict checkpoint success.
Thus no bug in the *recorded certificate
and pairing policy* explains the primary
negative QNN outcome. The underlying
latent arrays, class predictions, full
environment manifest, official source
byte files and source SHA manifest were
**not uploaded**; their exact contents
and raw-source provenance cannot be
independently reverified from these
aggregate files alone.

Strict test roots: zero **407/720**,
certified MRBI **450/720** (+43
additional strict roots). Strict training
roots: zero **983/1680**,
certified MRBI **1102/1680** (+119).
The 20D certified QNN consequently
receives additional valid implicit
states, but those rescues did not produce
an external-source balanced-accuracy
improvement. Computing the certified
candidate carries extra numerical work;
cost is a reported limitation, **not
the primary science objective**.

## Research and manuscript implications

**Supported:** a reproducible MRBI-root
rescue effect on distinct source families,
controlled fusion/QNN evaluation on
fresh seeds and a diagnostic indication
that sample-level latent pairing matters.
The prior +1.23-point fusion gain is
a valid outcome **for the previously
examined nine tasks and their seed holdout**.

**Not supported:** a general mean
balanced-accuracy gain from certified
MRBI fusion across new datasets,
superiority over the original PCA-QNN
or equal-width padded PCA, an
universal benefit of the certificate,
or quantum advantage. We must not
report only the secondary ungated
Sonar gain or the derangement contrast
as confirmation of the frozen
certified-MRBI primary question.

For an NPL manuscript centered on
MRBI + implicit layer + QNN, frame
the contribution as a controlled
fusion/solver integration, an
experimentally documented
representation-dependent result
and its genuine generalization
limitation, not as consistently
improved classification accuracy.
Any fundamentally new scientific
mechanism should be specified using
training-only/development datasets
and subsequently tested on genuinely
unused source families. Do not
retroactively optimize toward
these three external held-out
source results.
