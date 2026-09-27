# MRBI-QNN experiment entry points

`main_qnn_benchmark.py` runs the nine datasets with five seeds, four qubits, latent dimension 16, spectral radius 2.0, maximum 80 samples per class and 60 QNN epochs by default. The implicit operator is fixed. The code uses a classically simulated QNN readout.

`mrbi.py` performs warm-started optimization at the configured decreasing Gaussian probe scales. Each stage uses the current sigma and its own iteration budget, followed by an optional final-scale refinement. The schedule is checked in CI. The historical source before this correction remains in the archive branch; the new campaign writes version-tagged output files separate from the historical CSVs.

`spambase_external.py` and `run_multistart_sanity_check.py` contain supporting experiments. Their archived numerical results are checked by `scripts/check_supporting_raw.py` without re-running the QNN.

Use `python scripts/run_continuation_campaign.py --max-new-jobs 1` from the repository root for a first complete dataset/seed job. Run the same script without the limit to resume all 45 jobs.
