# MRBI-QNN: diagnostic path for a genuine multiscale contribution

Status: exploratory follow-up after the **full**, corrected-continuation
97-method campaign and the 45-pair, fixed-profile final-sigma ablation.
The previous reduced method sweep is not a source of current claims.

## What the completed ablation establishes

For the fixed `forced_full_balanced_qnn` profile, four tasks favor genuine
continuation, four favor repeated final-sigma optimization, and one ties.
The mean paired balanced-accuracy difference is +0.0023 with exploratory
two-sided Wilcoxon p=0.5703. Mean root success differs by +0.00083.
Continuation uses around 2729 objective calls versus 2175 for the
final-scale control. Both arms had equal stage count and iteration ceilings,
**not** equal realized evaluation budgets. Neither a benefit of the scale
schedule nor the independent contribution of the detector is established.

## Next experiment: diagnose the mechanism before optimizing it

1. **Stage-level audit, no QNN retraining.** On a reproducibly chosen,
   outcome-independent sample of training inputs from every task and seed,
   log each stage's `sigma`, candidate `z`, residual, Newton proxy norm,
   detector term, full/final-scale objective, Jacobian conditioning, actual
   objective calls, stage convergence and final root-solve outcome. Retain
   intermediate candidates. Check whether the early scales materially
   change the candidate or merely spend extra evaluations before the last
   stage reconstructs the same point. Compare outcomes with repeated
   final-sigma control on the identical inputs.

2. **Remove avoidable Monte Carlo jumps.** Current fixed probes are
   generated independently for each sigma. Test an ablation that uses one
   antithetic Gaussian probe set `U` at all scales and changes only
   `sigma*U`. Keep the probe sample count and seeds fixed and compare
   the induced detector gradient directions, variance and candidate path.
   This is a change to the method and must have a new experiment version.

3. **Measure term strength and basin transitions.** Evaluate the relative
   magnitudes and finite-difference directional gradients of the residual,
   Newton, detector and regularization terms. Report how often each
   continuation stage changes the solver's basin or converts a failed zero
   solve into a converged solve. If the detector term is negligible or
   almost sigma-invariant, adding more scales cannot plausibly help.

4. **Test one motivated change at a time.** After the audit, compare either
   a probe-coupled adaptive scale schedule with a fixed schedule, or
   retaining a small set of stage candidates chosen by **label-free**
   final-scale residual and conditioning. Match actual objective evaluation
   and root-solve budgets against final-sigma controls. Do not change the
   implicit layer, label splits or QNN architecture to favor the method.

5. **Validate on held-out experiments.** Predefine the primary endpoint
   (root success at a fixed budget), secondary QNN balanced accuracy, the
   unit of inference (dataset), uncertainty intervals and a fresh
   validation set before examining its results. Use the existing nine
   tasks for diagnostics and development, not as fresh confirmatory
   evidence for a tuned method. Retain negative task-level results.

## Decision rule for the paper

If a budget-matched, independently validated scale schedule fails to
improve root success or downstream QNN accuracy, present solver-aware
initialization as the contribution and the multiscale schedule as an
evaluated but unconfirmed design choice. Do not turn the exploratory
best-profile upper envelope into a fixed-method claim.
