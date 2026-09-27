# Results used in the manuscript

`raw/` contains the two supplied main-experiment CSV files. They cover six full five-seed datasets and seed 0 for `wine_0_vs_2`. `scripts/check_main_raw.py` checks each complete dataset against the published table and prints the remaining coverage gaps.

`supporting_raw/` contains the current three-seed Spambase result and the three-dataset multistart diagnostic. `scripts/check_supporting_raw.py` recomputes their archived summaries.

`summary_tables/` contains the reported manuscript summaries and input tables for the sensitivity plots. A successful check of these summaries is not a substitute for missing main-benchmark per-seed output.
