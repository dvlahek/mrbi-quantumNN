# MRBI-QNN recovered experiment source

These source files were supplied during the NPL manuscript audit. They are archived as working-source candidates, not as an independently verified reproduction of the nine-task headline results.

- `mrbi.py` is the historical uploaded MRBI module (source SHA-256: `608617aba7d964fe9642d9315930de6d6e48100a61018d5efc8188de5f3ae374`).
- `qnn_mrbi_hard_hybrid_benchmark_repaired.py` was recovered from earlier working files (source SHA-256: `4092d1e288f8e80bc79c560b7d7c1434d223db4103de50eaf1eaa97ca5153993`). It supports user-specified datasets and numerical settings, but its defaults do not define the nine-task NPL benchmark. The exact source revision for the published main table is unverified.
- `qnn_mrbi_article_benchmark.py` is an earlier article benchmark with a different default task set. The standalone Spambase runner, Spambase configuration and multistart diagnostic script are also preserved.

**Known numerical discrepancy:** the recovered `mrbi.py` loops over `sigmas`, but its `MRBIOptimizer.optimize()` passes `sigma=final_sigma` on each iteration. The Newton proxy tries a direct solve of `Js=F`, followed by shifted-J fallbacks. The compact public `src/mrbi.py` instead passes the current sigma and uses damped normal equations. Do not silently repair the historical source or claim that the compact version produced the archived QNN results.

The supplied Spambase and multistart raw data are archived under `results/supporting_raw/`; their aggregates are checked in CI. CSV line endings may be normalized by repository import, but numerical rows are unchanged. The raw nine-task files `article_qnn_final_raw1.csv` and `article_qnn_final_raw2.csv` were not supplied. No full simulated-QNN rerun was performed during this source audit.
