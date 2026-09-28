# Complete checkpoint-homotopy numerical pilot — 28 September 2026

## Scope and reproducibility

Exploratory *development* pilot of the opt-in checkpoint policy from
`experiment/checkpoint-homotopy-20260928`. The uploaded
`paired_sample_results.csv`, `stage_raw.csv`, and
`dataset_summary.csv` contain 9 tasks × 5 seeds × 8 training
inputs = **360 paired inputs** in 45 complete jobs. All four arms use
identical independent per-input antithetic probe seeds and the same
first-stage root-check/early-exit policy. No QNN was trained.

There are 2,212 stage records, not 7,200: 243 inputs succeed from
zero and skip MRBI in all arms, and some of the remaining 117 exit after
stage one. The only missing entries in the paired table correspond to
candidate-root and probe fields for those 243 zero-success inputs.
There are no duplicate sample/stage keys, the stage-row and candidate
evaluation counts agree with the paired table, the identical-probe
check passes, and no stage exceeds 480 objective or 5,760 primitive
residual evaluations. The source reference manifests and
`checkpoint_statistics.json` were not included in the uploaded files;
the uploaded CSVs alone do not independently re-verify the source
reference Git SHA or environment.

## Observed outcomes

The zero-initialized root solver succeeds on **243/360** inputs and
fails on 117.

| Arm | Successful stage-one checkpoints among 117 zero failures | Accepted successes / 360 | Mean all-in primitive F calls / input | Mean optimizer objective calls / input |
|---|---:|---:|---:|---:|
| Smoothed continuation | 26 | 269 | 6,552.57 | 540.08 |
| Smoothed repeated-final sigma | 2 | 245 | 6,530.68 | 538.09 |
| Plain continuation | 2 | 245 | 6,481.01 | 533.99 |
| Plain repeated-final sigma | 2 | 245 | 5,018.42 | 412.07 |

The three controls succeed on the **same two** first-stage inputs.
Against the smoothed repeated-final control, smoothed continuation
rescues 24 additional cases and loses none among 117 zero failures.
The extra rescues span all nine datasets and 17/45 dataset–seed
jobs (28 ties, no negative jobs). These are *development-set*
observations, not independent statistical confirmation. The prior
stage-audit and pilot on these same training inputs motivated the
checkpoint rule. Do not report an inferential p-value based on 360
independent observations.

Among cases failing the checkpoint, the fifth-stage root succeeds
on **zero** cases in every arm (91 full fallbacks for smoothed
continuation; 115 for each control). Consequently, all accepted
success gains in this pilot come from stage-one checkpoints, not
from subsequent fine-scale continuation. This does **not** establish
that later stages are never useful on fresh seeds or different
equilibrium problems.

Mean primitive F work for smoothed continuation is 0.34% above its
smoothed repeated-final control, reflecting early exit in 26 cases;
the former makes 0.067 fewer root attempts per input on average.
Conditional on the 91 inputs where smoothed continuation's checkpoint
fails, it consumes roughly 24,178 F calls/input versus 20,178 in
the smoothed repeated-final control. Thus the nearly identical
overall mean hides substantial wasted late-stage computation on
the unsuccessful subset. Relative to plain repeated-final sigma,
smoothed continuation uses ~30.6% more primitive F calls on average.
Equal stage **ceilings** do not equal realized computational cost.

The new independent per-input probe seed rule differs from the
previous pilot's sequential per-seed probe stream. The old stage
audit's 24 early rescues and this run's 26 are separate exploratory
runs and should not be merged as replicates or an enlarged sample.

## Predeclared next gate

The completed pilot supports a concrete numerical follow-up:
**coarse-scale one-stage checkpoint**, not an unqualified claim of
multiscale continuation benefit. Freeze the existing smoothed
weight=0.75, sigma=0.70, 480-call stage cap, probes, strict
root-success threshold and zero-root gate. Use an identical
one-stage checkpoint policy at sigma=0.02 (same weight and common
probes) and a plain-objective control. Measure actual F/J/root
work, wall time and success after the zero-root failure.

Run on fresh, prespecified seeds, e.g. 5–9, before looking at any
outcomes, and preferably additional independent tasks, because
seeds 0–4 / the first eight training inputs already informed
method development. A runner for new seeds must record its *own*
provenance; do not mislabel the old seeds-0–4 full-campaign reference
CSV as covering new seeds. All candidate comparisons must be
paired; summarize at the dataset and dataset–seed levels, not by
treating 360 correlated training inputs as independent tasks.
A strict equal-realized-work experiment may require explicit
budget-controlled controls; otherwise report paired success **and**
measured work without claiming budget equality.

Do not launch a new QNN campaign or revise the manuscript's
published numerical claims from these development-only results.
