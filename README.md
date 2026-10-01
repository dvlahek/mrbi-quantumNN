# MRBI-stabilized implicit equilibrium features for simulated QNN readouts

Minimal reproducibility package for the Neural Processing Letters manuscript.

## Scope

This repository contains only the code, frozen configuration, and aggregate results needed to reproduce the manuscript's final MRBI-QNN development and confirmation protocol. Earlier diagnostic work (caps, gates, rescue-only variants, trainable-angle prototypes, homotopy experiments, external-fusion studies, and other follow-up branches) is preserved separately on the archive branch:

`archive/pre-npl-final-cleanup-20261001`

The study uses classically simulated quantum neural-network readouts. It does not claim quantum advantage or hardware performance.

## Final experimental design

The implicit layer is

```text
z = tanh(W z + U x + b)
```

and MRBI constructs solver-aware initializations through corrected descending-scale continuation.

The final protocol has two stages.

1. **Development selection** — corrected-continuation campaign on seeds 0–4 across nine binary classification tasks. For each dataset, the MRBI-QNN method with the highest five-seed mean balanced accuracy is selected.
2. **Frozen confirmation** — the selected dataset-specific methods are frozen in `experiments/selected_profile_confirmation_lock_v1.json` and evaluated without reselection on new seeds 40–49.

The primary confirmatory comparison is selected MRBI-QNN versus Zero-QNN. PCA-QNN and random-5 multistart QNN are secondary references.

## Confirmation result

Across the nine dataset-level means:

- PCA-QNN mean balanced accuracy: **0.9722**
- Zero-QNN: **0.9215**
- selected MRBI-QNN: **0.9299**
- random-5 multistart QNN: **0.9137**
- selected MRBI − Zero: **+0.00835**
- positive / neutral / negative datasets versus Zero: **6 / 1 / 2**
- one-sided Wilcoxon selected MRBI > Zero: **p = 0.1016**
- selected MRBI − random-5 multistart: **+0.01611**
- one-sided Wilcoxon selected MRBI > random-5 multistart: **p = 0.0195**
- solver success: Zero **0.7161**, selected MRBI **0.7364**

The primary MRBI-versus-Zero result is therefore a positive prospective effect, but not a statistically significant superiority claim at the nine-dataset level. Runtime is recorded in the result files as a secondary implementation characteristic and is not the main contribution of the study.

Canonical aggregate results are under `results/final/`.

## Install

For checks and summaries:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the simulated QNN runs:

```bash
python -m pip install -r requirements-qnn.txt
```

## Quick reproducibility checks

```bash
python scripts/run_mrbi_smoke_test.py
python scripts/check_continuation.py
python scripts/check_final_results.py
python scripts/run_locked_selected_profile_confirmation.py --dry-run --seed 40
```

These checks do not retrain the QNN.

## Reproduce the development selection

This is computationally expensive.

```bash
python scripts/run_continuation_campaign.py
python scripts/summarize_continuation.py --require-complete
```

The campaign writes outputs under `outputs/continuation_v1/`. The frozen lock records the SHA-256 hashes of the development raw table and selected-profile table used for the manuscript confirmation.

## Reproduce the frozen confirmation

```bash
for seed in {40..49}; do
    python -u scripts/run_locked_selected_profile_confirmation.py --run --seed "$seed"
done

python scripts/summarize_locked_selected_profile_confirmation.py
```

Generated outputs are written under `outputs/locked_selected_profile_confirmation_v1/` and are ignored by Git.

## Provenance

The frozen selected-profile map was verified against the complete corrected-continuation development campaign before confirmation was run. The committed final aggregate result records:

- development raw SHA-256: `e1a3917be493485156e87e79c921d2a91a54cf715aa55443516059fc3d370cd6`
- development selected-profile SHA-256: `4cb9ad804076dea25820b73c1640a81e85c7d3b4ad77d16341555269da707908`
- development seeds: 0–4
- confirmation seeds: 40–49

The full research path before final cleanup remains available on the archive branch above.
