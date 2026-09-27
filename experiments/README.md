# Experiment entry points

`main_qnn_benchmark.py` and `mrbi.py` are the recovered working-source candidate for the nine-task experiment. The benchmark's former unconditional debugger overrides were replaced with ordinary command-line defaults; the numerical routines were not modified. The original source is retained on the archive branch.

The six complete datasets in `results/raw/` reproduce their displayed means from that benchmark's reported method names and selection rule. That agreement does not prove the exact original software environment or provide the absent seeds for the remaining three datasets.

The recovered MRBI source has a known scale-loop discrepancy: it iterates over the configured scales but passes `final_sigma` to every optimization call. It also uses a direct Newton solve with fallback, not the damped normal-equation proxy used by the former compact smoke-test module. This behavior is preserved. Any corrected-scale implementation would be a different experiment and must be evaluated separately.

`spambase_external.py` is the supplied external-check candidate with its entry point and default arguments cleaned for the archived Spambase configuration. The source data for the current published Spambase summary can be checked independently in `results/supporting_raw/`. `run_multistart_sanity_check.py` is the supplementary diagnostic runner. Neither of these scripts runs in CI; CI checks their archived numeric output.
