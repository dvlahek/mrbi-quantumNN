# Full smoothed-residual homotopy pilot — 28 September 2026

## Provenance and scope

Exploratory development results from branch
`experiment/smoothed-residual-homotopy-20260928`. The paired pilot
reuses the completed full 97-method corrected-continuation campaign
only as its reference and does not train a QNN. It uses the first eight
training samples after the benchmark split/PCA transform for each of
nine datasets and five seeds: 45 jobs, 360 paired samples, and 7,200
stage records (four arms times five stages). The uploaded CSVs have
no missing values and exactly 20 stage rows for each paired sample.

The opt-in smoothed objective has fixed weights
`(0.75, 0.50, 0.25, 0, 0)` and scales
`(0.70, 0.25, 0.08, 0.02, 0.02)`. Its paired repeated-final
control uses sigma 0.02 at every stage with the same smoothing
weights. Two plain-objective controls test descending versus repeated
final sigma, also with the same antithetic probe sample shared across
all stages and arms. The results below apply to this development
sample, not to held-out accuracy or the original full campaign.

## Final-stage outcomes

| Arm | Candidate root successes / 360 | Accepted successes / 360 | Mean optimizer objective calls per sample |
|---|---:|---:|---:|
| Smoothed continuation | 188 | 247 | 2209.93 |
| Smoothed final sigma | 187 | 245 | 1996.40 |
| Plain continuation | 188 | 245 | 1992.23 |
| Plain final sigma | 186 | 245 | 1790.82 |

Smoothed continuation versus smoothed final sigma changes candidate
success in 11 positive and 10 negative pairs (339 ties). Accepted
success changes in three positive and one negative pair (356 ties).
The final accepted difference is two of 360 cases. Smoothed
continuation requires 10.70% more objective calls on average than its
smoothed final-scale control, and hits the stage budget in 1,532 of
1,800 stages, compared with 1,214 of 1,800 for the control. The pilot
matches *ceilings*, not realized counts; an equal-realized-work
advantage is **not** established. Both arms' final-stage candidate
success is approximately 52%, and no QNN accuracy is available.

The smoothed continuation candidate residual before the root solver
has mean 0.11716, compared with 0.14675 for smoothed final sigma,
but the paired median difference (smoothed continuation minus
control) is positive `3.89e-7`. This mean advantage is driven by
a minority of large differences, not consistent paired improvement.
The root-success comparison above is the more relevant endpoint.

## The first-stage checkpoint result

The **first** coarse stage of smoothed continuation yields a root
success on 239 of 360 samples, compared with 188 after the fifth
stage. Exactly 52 samples have first-stage success and fifth-stage
failure, and one sample changes from first-stage failure to
fifth-stage success. In contrast, the smoothed final-sigma control
has 190 first-stage and 187 fifth-stage root successes.

The zero-start solver fails on 117 of 360 samples. Of those,
smoothed continuation's first coarse-stage candidate yields a
successful root in **24** cases, compared with **3** for the
smoothed final-sigma checkpoint, **3** for plain continuation and
**2** for plain final sigma. The full five-stage smoothed
continuation retains only **4** successes in the zero-failure
subset. The first-stage rescue count exceeds the smoothed-final
control on each of the nine development datasets, but this
observation emerged from post hoc stage inspection; it is not
an independent confirmatory test.

An oracle that adds the already observed successful first-stage
roots to the zero-start result would cover 267/360 inputs,
compared with 247/360 for the current final accepted method.
**267/360 is not an observed deployment result.** Implementing
this requires actually executing/checking a first-stage root solve
and counting its F/J evaluations, or using a separately validated
predictor; the existing audit's stage-root solves were diagnostic
and not part of the original optimizer's computation budget.

## Next version: predeclared, matched checkpoint policy

Do **not** tune the homotopy weights to these 360 samples and do not
retrain QNN from this pilot. Implement an opt-in, solver-aware
checkpoint experiment:

1. Execute the same zero-root solve first. If it succeeds, return
   the zero solution without paying for MRBI. If it fails, optimize
   through the **first** stage and run one root solve from the
   checkpoint. If the checkpoint solve meets the existing strict
   root-success criterion, retain that solved state immediately.
   Otherwise finish the remaining stages and run the final root
   solve. Never pick a candidate by the test label.
2. Apply exactly the same checkpoint and early-exit policy to the
   smoothed-final and both plain-objective arms. Keep each arm's
   original, fixed stage weights, Gaussian probes, root tolerance
   and cap parameters. Ensure each sample's paired probe stream
   is unchanged by another arm's early exit.
3. Count actual optimization objective evaluations, primitive
   F/J calls during optimization, zero and checkpoint/final root
   solves, number of root calls, and total wall time. Identical
   per-stage ceilings do not guarantee matched realized budget.
   Compare root success **at measured cost** and later run a
   strict equal-work validation if a positive result remains.
4. Use these same nine-task data only to test feasibility and
   correctness. Predefine primary root-success and cost endpoints
   and reserve fresh seeds or independent tasks for validation.
   Only after that consider QNN training using fixed features and
   a prespecified inference procedure.

The completed full corrected-continuation and final-sigma QNN
campaigns remain the canonical paper results until a separately
validated new numerical method justifies a new benchmark.
