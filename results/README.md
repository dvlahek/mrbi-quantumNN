# Numerical results and provenance

**Current manuscript results** are the completed full corrected-continuation
benchmark (9 tasks × 5 seeds × 97 methods) and its paired full-reference
final-sigma ablation (45 control QNN rows). They were generated on the Ryzen
workstation in `outputs/continuation_v1/`,
`outputs/full_ryzen/shard1|shard2|shard3/` and
`outputs/final_sigma_ablation_full_ryzen_v1/` in their respective checkouts.

The existing CSVs under this GitHub directory are **historical audit data**.
They are not the corrected full campaign and are not current evidence for
the manuscript. In particular, old best-profile tables and sensitivity
plots should not be reproduced as the corrected full method's output.

`raw/article_qnn_final_raw.csv` and `raw/article_qnn_final_raw2.csv` are the available historical per-seed main-benchmark results. They provide all five seeds for six tasks and one seed for `wine_0_vs_2`.

`summary_tables/article_qnn_final_summary1.csv` is the recovered five-dataset summary. It contains 97 methods per dataset, each with five runs, including `wine_0_vs_2`, `wine_1_vs_2` and `digits_1_vs_7`. Together with the six complete raw datasets, it supports a numerical check of all nine historical task-level QNN means. The missing per-seed CSVs remain unavailable for three tasks.

`supporting_raw/` contains Spambase and multistart diagnostic raw files. Other `summary_tables/` files contain the manuscript summaries and inputs for the diagnostic figures.

The corrected-continuation campaign writes its new, version-tagged outputs to `outputs/continuation_v1/`. The corrected full campaign has now completed and been checked. Its complete raw CSV and paired ablation provenance still need to be deposited into this GitHub repository; do not overwrite legacy audit files with non-matching data.

