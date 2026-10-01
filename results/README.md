# Final manuscript results

This directory contains the aggregate results used in the final manuscript.

- `final/confirmation_summary.json` — dataset-level and overall confirmation statistics.
- `final/confirmation_dataset_summary.csv` — one row per dataset, averaged over confirmation seeds 40–49.

The dataset-specific MRBI methods are fixed in:

`experiments/selected_profile_confirmation_lock_v1.json`

Per-seed outputs can be regenerated with `scripts/run_locked_selected_profile_confirmation.py` and are not tracked on `main`. Earlier historical and diagnostic result files are kept on `archive/pre-npl-final-cleanup-20261001`.
