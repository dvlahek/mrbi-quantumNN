# Numerical results

`raw/article_qnn_final_raw.csv` and `raw/article_qnn_final_raw2.csv` are the available historical per-seed main-benchmark results. They provide all five seeds for six tasks and one seed for `wine_0_vs_2`.

`summary_tables/article_qnn_final_summary1.csv` is the recovered five-dataset summary. It contains 97 methods per dataset, each with five runs, including `wine_0_vs_2`, `wine_1_vs_2` and `digits_1_vs_7`. Together with the six complete raw datasets, it supports a numerical check of all nine historical task-level QNN means. The missing per-seed CSVs remain unavailable for three tasks.

`supporting_raw/` contains Spambase and multistart diagnostic raw files. Other `summary_tables/` files contain the manuscript summaries and inputs for the diagnostic figures.

The corrected-continuation campaign writes its new, version-tagged outputs to `outputs/continuation_v1/`. Historical CSVs are not replaced with corrected-method results until the new campaign has completed and has been checked.
