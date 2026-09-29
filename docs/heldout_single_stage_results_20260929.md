# Frozen one-stage coarse/fine numerical check — seeds 5–9

## Scope

The prespecified, label-free experiment
`experiment/heldout-single-stage-20260928` used the original nine
benchmark datasets, the *new* seeds 5–9, and eight training inputs
per dataset–seed after the original split and PCA transform. The
uploaded `paired_sample_results.csv`, `stage_raw.csv`, and
`dataset_summary.csv` give 45 complete jobs, **360 paired inputs**
and **700 one-stage records**, with no duplicate keys. On **185/360**
inputs, zero initialization succeeds and all arms skip MRBI. Each of
the remaining **175** inputs has one stage for all four arms.

These are new seeded splits and randomly initialized implicit layers
of the *same underlying datasets*, not independent new datasets.
Prior development on seeds 0–4 chose the frozen stage-one policy,
weight and coarse/fine sigma comparison. The full 97-method campaign
raw file is **not** a reference for the new seeds.

Only the three uploaded CSVs were available for this analysis. Their
internal coverage, arithmetic and method/probe consistency were
verified. The separately produced run-environment and per-job
SHA manifests were not uploaded, so the CSVs alone do not
independently verify the code revision or environment hashes.

## Frozen four-arm results

| Arm | Successful checkpoint roots among 175 zero failures | Accepted successful inputs / 360 | Mean deployable primitive F calls / input |
|---|---:|---:|---:|
| Smoothed coarse, sigma=0.70 | **26** | **211** | **2870.90** |
| Smoothed fine, sigma=0.02 | 11 | 196 | 2867.82 |
| Plain coarse, sigma=0.70 | 14 | 199 | 2867.37 |
| Plain fine, sigma=0.02 | 12 | 197 | 2868.11 |

The **primary prespecified contrast** smoothed coarse minus smoothed
fine gives 15 more accepted successes among 360 inputs, equivalently
26 vs 11 successful checkpoint rescues among 175 zero failures.
The paired discordance is **20 coarse-only versus 5 fine-only**
(6 both and 144 neither). This is not an across-the-board
per-input dominance. For the secondary smoothed coarse versus
plain coarse comparison, 18 cases are smoothed-only and 6
plain-only, net +12 successes.

At the dataset level, primary rescue counts are higher on six
tasks, lower on one (breast_cancer: 1 coarse versus 2 fine), and
tied on two. At the 45 dataset–seed job level, 11 differences
are positive, 1 is negative and 33 are zero. This is an
exploratory measurement of robustness across fresh seeds on
previously explored datasets; it should not be reported as 360
statistically independent tasks. No QNN was trained, and there
are no downstream classification-accuracy results.

## Actual work and integrity checks

All 175 attempted samples in **each** arm hit the identical
predeclared **480 objective calls / 5,760 optimizer F calls**
stage ceiling. Therefore the realized *optimizer* objective
and primitive F budgets are exactly equal in this comparison.
The root solver still uses a data- and arm-dependent number of
F/J evaluations. Mean deployable total F calls are 2870.90
for smoothed coarse and 2867.82 for smoothed fine, a difference
of **3.08 calls/input (+0.107%)**. These totals include the
shared zero-root attempt, allocated once per arm, and exclude
the inherited stage logger's 11 F and 1 J purely diagnostic
evaluations. The raw `total_F_calls`/`total_J_calls` fields
include them. Real wall times may reflect instrumentation and
machine variation and are not an equal-time guarantee.

The 700 stage records all have stage=1, arm-consistent
sigma/weight, 480 objective and 5,760 optimizer F calls,
no exceeded ceiling, and common antithetic probe hashes within
each attempted paired sample. Pair-table counts and dataset
summaries agree; missing candidate residual/hash fields are
restricted to the 185 inputs that correctly skipped MRBI.

## Interpretation and next gate

The coarse smoothed residual retains a positive numerical
**seed-holdout** signal under a genuinely matched optimizer
evaluation ceiling. This supports the **single coarse-stage
solver-aware initialization** mechanism, not a benefit from
five-stage continuation, and not a QNN classification gain.
The outcome is heterogeneous across the nine datasets, and
the within-dataset samples/seeds are correlated. Avoid
naive 360-independent-input p-values or generalization to
new data domains.

Freeze the method. Do not retune sigma, smoothed weight,
selection rule, caps or dataset exclusions using seeds 5–9.
Before extending the paper's claim, use genuinely new
independent datasets or a further prespecified validation
on fresh seeds **10–14** with the same frozen code and
evaluate aggregate paired success *and* measured F/J/root cost.
A subsequent QNN experiment would separately test if
the root-rescue improvement changes balanced accuracy,
against zero, fixed MRBI and PCA-QNN baselines. Keep the
existing full QNN results and manuscript claims unchanged
until such a test has been completed.
