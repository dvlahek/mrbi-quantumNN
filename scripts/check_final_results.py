"""Verify the frozen profile map and committed final aggregate results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "experiments" / "selected_profile_confirmation_lock_v1.json"
SUMMARY = ROOT / "results" / "final" / "confirmation_summary.json"
DATASET = ROOT / "results" / "final" / "confirmation_dataset_summary.csv"

EXPECTED_RAW_SHA = "e1a3917be493485156e87e79c921d2a91a54cf715aa55443516059fc3d370cd6"
EXPECTED_SELECTED_SHA = "4cb9ad804076dea25820b73c1640a81e85c7d3b4ad77d16341555269da707908"
EXPECTED_DATASETS = (
    "breast_cancer", "wine_binary", "wine_0_vs_2", "wine_1_vs_2",
    "digits_1_vs_7", "digits_2_vs_7", "digits_3_vs_8",
    "digits_4_vs_9", "digits_5_vs_6",
)


def close(label: str, actual: float, expected: float, tol: float = 1e-12):
    if not np.isclose(float(actual), float(expected), atol=tol, rtol=0.0):
        raise AssertionError(f"{label}: {actual} != {expected}")


def main():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    data = pd.read_csv(DATASET)

    if lock["lock_schema"] != "mrbi_selected_profile_confirmation_lock_v1":
        raise AssertionError("Unexpected lock schema")
    evidence = lock["development_evidence"]
    if evidence["main_raw_sha256"] != EXPECTED_RAW_SHA:
        raise AssertionError("Development raw SHA drift")
    if evidence["selected_profiles_sha256"] != EXPECTED_SELECTED_SHA:
        raise AssertionError("Selected-profile SHA drift")
    if tuple(evidence["datasets"]) != EXPECTED_DATASETS:
        raise AssertionError("Locked dataset set/order drift")
    if tuple(lock["confirmation_plan"]["seeds"]) != tuple(range(40, 50)):
        raise AssertionError("Confirmation seed drift")

    if tuple(data["dataset"]) != EXPECTED_DATASETS or len(data) != 9:
        raise AssertionError("Final dataset summary must contain the nine locked tasks")

    locked_methods = {
        row["dataset"]: row["method"] for row in lock["selected_profiles"]
    }
    got_methods = dict(zip(data["dataset"], data["selected_method"]))
    if got_methods != locked_methods:
        raise AssertionError("Final dataset methods differ from the frozen lock")

    dz = data["delta_selected_zero"].to_numpy(float)
    dm = data["delta_selected_multistart5"].to_numpy(float)

    close("mean selected MRBI BA", data["selected_mrbi"].mean(),
          summary["overall"]["selected_mrbi_ba_mean"])
    close("mean Zero BA", data["zero"].mean(),
          summary["overall"]["zero_ba_mean"])
    close("mean PCA BA", data["pca"].mean(),
          summary["overall"]["pca_ba_mean"])
    close("mean multistart BA", data["multistart5"].mean(),
          summary["overall"]["multistart5_ba_mean"])
    close("mean MRBI-Zero delta", dz.mean(),
          summary["overall"]["delta_selected_zero_mean"])
    close("mean MRBI-multistart delta", dm.mean(),
          summary["overall"]["delta_selected_multistart5_mean"])
    close("mean solver gain",
          (data["selected_success"] - data["zero_success"]).mean(),
          summary["overall"]["selected_solver_success_gain_vs_zero_mean"])

    positive = int(np.sum(dz > 1e-12))
    neutral = int(np.sum(np.abs(dz) <= 1e-12))
    negative = int(np.sum(dz < -1e-12))
    if (positive, neutral, negative) != (6, 1, 2):
        raise AssertionError(
            f"Unexpected MRBI-Zero sign count: {(positive, neutral, negative)}"
        )

    wz = wilcoxon(dz, alternative="greater", zero_method="wilcox").pvalue
    wm = wilcoxon(dm, alternative="greater", zero_method="wilcox").pvalue
    close("Wilcoxon selected > Zero", wz,
          summary["overall"]["wilcoxon_selected_gt_zero"]["pvalue"])
    close("Wilcoxon selected > multistart", wm,
          summary["overall"]["wilcoxon_selected_gt_multistart5"]["pvalue"])

    if summary["primary_comparison"] != "selected MRBI-QNN minus Zero-QNN":
        raise AssertionError("Primary comparison drift")
    if summary["n_dataset_seed_pairs"] != 90:
        raise AssertionError("Expected 90 frozen confirmation dataset-seed pairs")

    print(
        "FINAL_RESULTS_CHECK_OK "
        "datasets=9 confirmation_seeds=40-49 "
        f"delta_zero={dz.mean():+.6f} "
        f"wilcoxon_zero={wz:.7f} "
        f"delta_multistart={dm.mean():+.6f} "
        f"wilcoxon_multistart={wm:.7f}"
    )


if __name__ == "__main__":
    main()
