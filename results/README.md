# Final manuscript results

This directory contains only the aggregate evidence used by the final manuscript protocol.

- `final/confirmation_summary.json` — dataset-level and overall confirmation statistics.
- `final/confirmation_dataset_summary.csv` — one row per dataset, averaged over frozen confirmation seeds 40–49.

The dataset-specific MRBI methods are frozen in:

`experiments/selected_profile_confirmation_lock_v1.json`

The full per-seed generated outputs are reproducible with `scripts/run_locked_selected_profile_confirmation.py` and are intentionally not tracked on `main`. Earlier historical and diagnostic result files are preserved on `archive/pre-npl-final-cleanup-20261001`.
