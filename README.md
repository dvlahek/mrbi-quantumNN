# MRBI-QNN reproducibility

Code and aggregate results for MRBI-stabilized implicit equilibrium features with simulated QNN readouts.

All QNN experiments use classical simulation. No quantum-hardware or quantum-advantage claim is made.

## Final protocol

The implicit layer is

```text
z = tanh(W z + U x + b)
```

MRBI supplies solver-aware initializations before the nonlinear root solve.

Dataset-specific MRBI profiles were selected on development seeds 0–4 and then fixed in `experiments/selected_profile_confirmation_lock_v1.json`. The reported confirmation uses seeds 40–49 with no profile reselection.

The main comparison is selected MRBI-QNN versus Zero-QNN. PCA-QNN and random-5 multistart QNN are reference methods.

Two fixed secondary checks are included:

- spectral-radius sensitivity at rho(W) = 1.20, 1.60, 2.00, and 2.25, with the same base operator for a given seed and only W rescaled;
- classical readouts at rho(W) = 2.00 using logistic regression, RBF SVM, MLP, random forest, and gradient boosting.

The rho(W)=2.00 sensitivity anchor reproduces the confirmation run.

## Aggregate results

Confirmation over nine dataset-level means:

- PCA-QNN balanced accuracy: **0.9722**
- Zero-QNN: **0.9215**
- selected MRBI-QNN: **0.9299**
- selected MRBI - Zero: **+0.00835**
- positive / neutral / negative datasets: **6 / 1 / 2**
- exact one-sided signed-rank p-value: **0.1016**
- selected MRBI - random-5 multistart: **+0.01611**
- exact one-sided signed-rank p-value versus multistart: **0.0234**
- solver success: Zero **0.7161**, selected MRBI **0.7364**

In the spectral-radius check, the MRBI-minus-Zero solver-success gain increases from approximately **0.05 pp** at rho(W)=1.20 to **3.51 pp** at rho(W)=2.25. The balanced-accuracy difference is largest at the confirmation setting rho(W)=2.00.

Tracked aggregate results are under `results/final/`.

## Install

Lightweight checks and summaries:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Simulated QNN runs additionally require:

```bash
python -m pip install -r requirements-qnn.txt
```

## Quick checks

These commands do not retrain the QNN:

```bash
python scripts/run_mrbi_smoke_test.py
python scripts/check_continuation.py
python scripts/check_final_results.py
python scripts/test_fixed_profile_robustness_protocol.py
python scripts/run_locked_selected_profile_confirmation.py --dry-run --seed 40
python scripts/run_fixed_profile_rho_sensitivity.py --dry-run --seed 40 --rho 2.0
```

## Reproduce confirmation

```bash
for seed in {40..49}; do
    python -u scripts/run_locked_selected_profile_confirmation.py --run --seed "$seed"
done

python scripts/summarize_locked_selected_profile_confirmation.py
```

Outputs are written under `outputs/locked_selected_profile_confirmation_v1/`.

## Reproduce spectral-radius sensitivity

Run the four fixed radii for seeds 40–49:

```bash
for rho in 1.2 1.6 2.0 2.25; do
    for seed in {40..49}; do
        python -u scripts/run_fixed_profile_rho_sensitivity.py --run --seed "$seed" --rho "$rho"
    done
done

python scripts/summarize_fixed_profile_rho_sensitivity.py
```

The rho(W)=2.00 run also writes the feature cache used by the classical-readout check.

## Reproduce classical readouts

```bash
for seed in {40..49}; do
    python -u scripts/run_fixed_profile_classical_readouts.py --run --seed "$seed"
done

python scripts/summarize_fixed_profile_classical_readouts.py
```

## Recreate robustness figures

```bash
python scripts/figures/make_fixed_profile_robustness_figures.py
```

The figure script reads the tracked aggregate summaries in `results/final/` and writes PNG/PDF files under `outputs/fixed_profile_robustness_figures/`.

## Development archive

The main branch contains only the final reproducibility path and aggregate results. Development campaigns, exploratory variants, and intermediate diagnostics are preserved separately:

- `archive/research-history-20261001`
- `archive/fixed-profile-robustness-20261002`
