# Robustness figure code

This directory contains plotting code only. It reads code-generated robustness
summaries and does not run MRBI, retrain QNNs, or change the fixed analysis
protocol.

## Script

`make_fixed_profile_robustness_figures.py`

Default inputs:

- `outputs/fixed_profile_rho_sensitivity_v1/rho_dataset_summary.csv`
- `outputs/fixed_profile_rho_sensitivity_v1/rho_overall_summary.csv`
- `outputs/fixed_profile_classical_readouts_v1/classical_dataset_summary.csv`
- `outputs/fixed_profile_classical_readouts_v1/classical_overall_summary.csv`
- `results/final/confirmation_summary.json`

The script checks that the rho=2.0 classification mean matches the canonical
confirmation summary before it writes any figure.

Default output directory:

`outputs/fixed_profile_robustness_figures/`

Generated figures, each as PNG and PDF:

- `rho_sensitivity_fixed_profiles`: two-panel classification and solver-success sensitivity.
- `rho_sensitivity_by_dataset`: per-dataset classification differences across rho.
- `classical_readout_fixed_profiles`: forest plot for QNN plus five fixed classical readouts.

Run from the repository root:

```bash
python scripts/figures/make_fixed_profile_robustness_figures.py
```

All figure outputs are under `outputs/`, which is ignored by Git.
