# Fixed-profile robustness protocol

Status: fixed before inspecting any new robustness outcome.

## Common rules

- The dataset-specific MRBI method is the map already selected on development seeds 0-4 and stored in `experiments/selected_profile_confirmation_lock_v1.json`.
- No MRBI profile, hybrid threshold, dataset, seed, readout, or spectral-radius value is selected from the new robustness results.
- Datasets: all nine benchmark tasks in the fixed profile map.
- Seeds: confirmation seeds 40-49.
- Any implementation bug found after the first result is produced must be documented, fixed, and followed by a complete rerun of the affected analysis. No partial result-dependent patching is allowed.
- Both analyses below are secondary descriptive checks. They do not replace the primary confirmation comparison.

## A. Spectral-radius sensitivity

### Question
Does the MRBI-minus-Zero classification difference change as the same implicit operator is made harder by increasing its spectral radius?

### Fixed design

- Spectral radii: 1.20, 1.60, 2.00, 2.25.
- Datasets: all nine tasks.
- Seeds: 40-49.
- Readout: the same 4-qubit, 2-layer simulated QNN and 60-epoch optimization used in confirmation.
- MRBI method: the fixed dataset-specific method from the development selection map.
- Primary descriptive quantity at each radius: balanced-accuracy difference `selected MRBI-QNN - Zero-QNN`.
- Additional quantities: Zero and MRBI solver success rates, balanced accuracies, and residual summaries.
- No formal hypothesis test is run separately at each radius. Report dataset-level mean, median, 95% dataset bootstrap interval, and positive/neutral/negative dataset counts.

### Operator construction

The base layer uses the exact layer seed used by the existing confirmation at rho=2.0:

`base_seed = seed + 1000 + latent_dim + int(100 * 2.0)`.

The target rho is never part of this seed. This preserves the existing confirmation operator at rho=2.0 and decouples operator generation from the sensitivity variable.

Let `W_base, U, b` denote the generated base layer. Define the spectral radius by

`rho(W) = max(abs(eigvals(W)))`.

For rho=2.0, use `W_base` directly, with no additional multiplication, to preserve the existing operator exactly. For every other target radius,

`W_rho = (rho_target / rho(W_base)) * W_base`.

`U` and `b` are copied unchanged. Solver RNG offsets and QNN seeds are also independent of the target rho. Thus only the magnitude of `W` changes across the four conditions.

### Regression anchor

The rho=2.0 run must reproduce the existing confirmation result for every dataset and seed. The runner compares QNN metrics and strict solver-success values with `outputs/locked_selected_profile_confirmation_v1/seed*/locked_result.json` and fails if the difference exceeds 1e-12. The generated base `W`, `U`, and `b` are also checked against the original confirmation layer construction.

The anchor cache stores PCA, Zero, and selected-MRBI train/test features at rho=2.0 for the classical-readout analysis below.

### Reporting

Report all four radii, all nine datasets, and all ten seeds. The main descriptive plot should show mean dataset-level MRBI-minus-Zero balanced-accuracy difference versus rho with a 95% dataset bootstrap interval. Individual dataset curves may be shown in the supplementary plot or table. Do not report per-rho p-values.

## B. Classical-readout check

### Question
Is the selected-MRBI versus Zero feature difference visible under standard classical readouts, or is it restricted to the simulated QNN readout?

### Fixed design

- Use only the rho=2.0 feature cache from the sensitivity anchor.
- Datasets: all nine tasks.
- Seeds: 40-49.
- Representations: PCA reference, Zero implicit features, and the fixed selected-MRBI implicit features.
- Classical readouts: logistic regression, RBF SVM, MLP, random forest, and gradient boosting.
- Hyperparameters and random-state conventions are the existing definitions in `experiments/main_qnn_benchmark.py`; no tuning is performed for this check.
- The selected MRBI method is not changed by readout.

### Reporting

For each readout, average each dataset over the ten confirmation seeds first. Then report across the nine dataset means:

- mean and median `selected MRBI - Zero` balanced-accuracy difference;
- 95% dataset bootstrap interval;
- positive/neutral/negative dataset counts;
- full per-dataset values in a supplementary table.

No formal p-value is used for this secondary analysis. PCA is reported only as a context reference; the central comparison remains selected MRBI versus Zero under the same readout.

## Analyses not repeated

The PCA4 dimension control, qubit-width sensitivity, and Spambase benchmark are not repeated. They do not address the current primary comparison as directly as the two checks above, and the old versions used result-dependent profile selection that is not part of the current fixed-profile protocol.


## Implementation amendment 2026-10-01

The first rho=1.20 run exposed a serialization-only edge case. For a hybrid profile that never invoked the detector, the diagnostic `hybrid_mean_sigma_min_det` was undefined and represented internally as `NaN`. Strict JSON output (`allow_nan=False`) therefore stopped the runner after computation. The numerical result itself was not invalid.

The runner now converts non-finite auxiliary diagnostics to JSON `null` before persistence. No solver rule, MRBI objective, trigger, profile, random seed, operator, QNN setting, dataset, spectral-radius value, or statistical analysis was changed. The incomplete rho=1.20 / seed 40 output must be discarded and rerun from the start under the patched runner. This amendment was made before inspecting any completed rho=1.20 sensitivity job.
