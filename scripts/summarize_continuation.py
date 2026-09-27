"""Summarize a completed or partial corrected-continuation MRBI-QNN campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "continuation_v1"
DATASETS = (
    "breast_cancer", "wine_binary", "wine_0_vs_2", "wine_1_vs_2",
    "digits_1_vs_7", "digits_2_vs_7", "digits_3_vs_8",
    "digits_4_vs_9", "digits_5_vs_6",
)
SEEDS = {0, 1, 2, 3, 4}
VERSION = "mrbi_continuation_v1"
EXPECTED_METHODS = 97


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_OUT / "main_raw.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--require-complete", action="store_true",
                        help="Fail unless all nine datasets have all five seeds.")
    opts = parser.parse_args()

    data = pd.read_csv(opts.raw)
    required = {"dataset", "seed", "method", "readout",
                "implementation_version", "balanced_accuracy"}
    if not required.issubset(data.columns):
        raise SystemExit(f"Missing raw columns: {sorted(required-set(data.columns))}")
    if data.duplicated(["dataset", "seed", "method"]).any():
        raise SystemExit("Duplicate dataset, seed and method in corrected raw file.")
    if not data["implementation_version"].eq(VERSION).all():
        raise SystemExit("Refusing to combine historical and corrected MRBI data.")
    if not set(data["dataset"]).issubset(DATASETS):
        raise SystemExit("Raw file contains an unexpected dataset.")
    complete = []
    missing = {}
    for dataset in DATASETS:
        group = data[data["dataset"] == dataset]
        counts = group.groupby("seed")["method"].nunique().to_dict()
        if set(counts) == SEEDS and all(value == EXPECTED_METHODS for value in counts.values()):
            complete.append(dataset)
        else:
            missing[dataset] = {
                "missing_seeds": sorted(SEEDS - set(counts)),
                "method_counts_by_available_seed": counts,
            }
    if opts.require_complete and missing:
        raise SystemExit(f"Campaign incomplete: {json.dumps(missing, indent=2)}")
    if not complete:
        raise SystemExit("No complete five-seed dataset to summarize.")

    qnn = data[(data["dataset"].isin(complete)) & (data["readout"] == "qnn")]
    output = []
    chosen = []
    deltas = []
    for dataset in complete:
        group = qnn[qnn["dataset"] == dataset]
        scores = group.groupby("method")["balanced_accuracy"].mean()
        if not {"pca_qnn", "implicit_zero_qnn"}.issubset(scores.index):
            raise SystemExit(f"QNN baselines absent for {dataset}")
        candidates = scores.drop(index=["pca_qnn", "implicit_zero_qnn"])
        if candidates.empty:
            raise SystemExit(f"No MRBI candidates for {dataset}")
        best_method = candidates.idxmax()
        pca, zero, best = (float(scores["pca_qnn"]),
                           float(scores["implicit_zero_qnn"]),
                           float(candidates.loc[best_method]))
        output.append({"dataset": dataset, "pca_qnn": pca, "zero_qnn": zero,
                       "best_mrbi_qnn": best, "delta_zero": best-zero,
                       "delta_pca": best-pca})
        chosen.append({"dataset": dataset, "best_method": best_method,
                       "best_mrbi_qnn": best, "n_seeds": 5})
        deltas.append(best-zero)
    result = pd.DataFrame(output)
    if len(complete) == len(DATASETS):
        means = result.drop(columns="dataset").mean().to_dict()
        result = pd.concat([result, pd.DataFrame([{"dataset": "Mean", **means}])],
                           ignore_index=True)
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(opts.out_dir / "main_qnn_results.csv", index=False, float_format="%.6f")
    pd.DataFrame(chosen).to_csv(opts.out_dir / "selected_profiles.csv", index=False)
    stat = {
        "implementation_version": VERSION,
        "n_complete_datasets": len(complete),
        "complete_datasets": complete,
        "incomplete_datasets": missing,
        "selection_rule": "Select the highest five-seed mean per dataset and QNN method.",
        "mean_delta_zero": float(np.mean(deltas)),
        "median_delta_zero": float(np.median(deltas)),
        "positive": int(sum(d > 1e-12 for d in deltas)),
        "zero": int(sum(abs(d) <= 1e-12 for d in deltas)),
        "negative": int(sum(d < -1e-12 for d in deltas)),
        "wilcoxon_p_nominal_one_sided": None,
        "note": "Best-profile upper envelope; nominal test is not adjusted for selection.",
    }
    if len(complete) == len(DATASETS) and any(abs(d) > 1e-12 for d in deltas):
        stat["wilcoxon_p_nominal_one_sided"] = float(wilcoxon(
            deltas, alternative="greater", zero_method="wilcox").pvalue)
    (opts.out_dir / "main_statistics.json").write_text(
        json.dumps(stat, indent=2) + "\n", encoding="utf-8")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Complete datasets: {len(complete)}/9")
    print("Saved corrected-continuation table, selected profiles and statistics.")
    if missing:
        print("Incomplete tasks:", ", ".join(missing))


if __name__ == "__main__":
    main()
