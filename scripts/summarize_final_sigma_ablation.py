"""Paired continuation vs repeated-final-sigma analysis on completed five-seed tasks.

The primary comparison is a fixed forced full_balanced QNN profile. Positive
paired differences mean continuation exceeds the final-sigma control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DATASETS = (
    "breast_cancer", "wine_binary", "wine_0_vs_2", "wine_1_vs_2",
    "digits_1_vs_7", "digits_2_vs_7", "digits_3_vs_8",
    "digits_4_vs_9", "digits_5_vs_6",
)
SEEDS = {0, 1, 2, 3, 4}
PLAN = "final_sigma_repeated_vs_continuation_v1"
SOURCE_PLAN = "core_fixed_after_pilot_v1"
FULL_SOURCE_PLAN = "full_corrected_continuation_v1"
VERSION = "mrbi_continuation_v1"
SOURCE_METHOD = "forced_full_balanced_qnn"
CONTROL_METHOD = "final_sigma_repeated_full_balanced_qnn"
METRICS = {
    "balanced_accuracy": "qnn_ba",
    "stat_test_forced_success_rate": "root_success",
    "stat_test_forced_final_mean_residual": "final_residual",
    "stat_test_forced_mean_objective_calls": "objective_calls",
    "stat_feature_total_time_sec": "feature_time_sec",
}


def analyse(control_raw: Path, reference_raw: Path, out_dir: Path, require_complete=False):
    control = pd.read_csv(control_raw)
    source = pd.read_csv(reference_raw)
    required = {"dataset", "seed", "method", "implementation_version", "balanced_accuracy"}
    if not required.issubset(control.columns) or not required.issubset(source.columns):
        raise SystemExit("Both raw CSVs must contain dataset, seed, method, implementation_version and balanced_accuracy")
    if not set(METRICS).issubset(control.columns) or not set(METRICS).issubset(source.columns):
        raise SystemExit("Missing solver or computation-budget metrics in one of the raw CSVs")
    if not {"ablation_design", "ablation_arm"}.issubset(control.columns):
        raise SystemExit("The final-sigma raw CSV lacks the ablation labels")
    if "campaign_design" in source.columns:
        if not source["campaign_design"].eq(SOURCE_PLAN).all():
            raise SystemExit("Unexpected core reference campaign design")
        source_plan = SOURCE_PLAN
    else:
        source_plan = FULL_SOURCE_PLAN
    if (not control["implementation_version"].eq(VERSION).all()
            or not control["ablation_design"].eq(PLAN).all()
            or not control["ablation_arm"].eq("final_sigma_repeated").all()
            or not control["method"].eq(CONTROL_METHOD).all()):
        raise SystemExit("Control CSV is not the fixed final-sigma ablation")
    if not source["implementation_version"].eq(VERSION).all():
        raise SystemExit("Reference is not the corrected continuation implementation")
    if "source_campaign_design" in control.columns and not control["source_campaign_design"].eq(source_plan).all():
        raise SystemExit("Control reference label does not match supplied reference")
    if control.duplicated(["dataset", "seed"]).any():
        raise SystemExit("Duplicate dataset/seed in final-sigma controls")
    reference = source[source["method"].eq(SOURCE_METHOD)].copy()
    if reference.duplicated(["dataset", "seed"]).any():
        raise SystemExit("Duplicate dataset/seed in continuation reference")
    if (not set(control["dataset"]).issubset(DATASETS)
            or not set(control["seed"]).issubset(SEEDS)):
        raise SystemExit("Unexpected control dataset or seed")
    for field in METRICS:
        if not np.isfinite(pd.to_numeric(control[field], errors="coerce")).all():
            raise SystemExit(f"Non-finite final-sigma metric: {field}")
        if not np.isfinite(pd.to_numeric(reference[field], errors="coerce")).all():
            raise SystemExit(f"Non-finite continuation metric: {field}")

    missing = {}
    complete = []
    for ds in DATASETS:
        seeds = set(control.loc[control["dataset"].eq(ds), "seed"])
        if seeds == SEEDS:
            complete.append(ds)
        else:
            missing[ds] = sorted(SEEDS - seeds)
    if require_complete and missing:
        raise SystemExit(f"Incomplete paired campaign: {json.dumps(missing)}")
    if not complete:
        raise SystemExit("No dataset has all five final-sigma seed controls yet")
    fields = ["dataset", "seed"] + list(METRICS)
    for extra in ("n_qubits", "latent_dim", "spectral_radius",
                  "input_scale", "max_samples_per_class"):
        if extra not in control.columns or extra not in reference.columns:
            raise SystemExit(f"Missing configuration column: {extra}")
        fields.append(extra)
    control = control.loc[control["dataset"].isin(complete), fields]
    reference = reference.loc[reference["dataset"].isin(complete), fields]
    merged = reference.merge(control, on=["dataset", "seed"],
                             suffixes=("_cont", "_final"), how="inner",
                             validate="one_to_one")
    if len(merged) != len(complete) * len(SEEDS):
        raise SystemExit("Missing continuation/control seed pairs")
    for extra in ("n_qubits", "latent_dim", "spectral_radius",
                  "input_scale", "max_samples_per_class"):
        left = pd.to_numeric(merged[f"{extra}_cont"], errors="coerce")
        right = pd.to_numeric(merged[f"{extra}_final"], errors="coerce")
        if not np.allclose(left, right, atol=1e-12, rtol=0):
            raise SystemExit(f"Paired configuration mismatch: {extra}")
    paired = merged[["dataset", "seed"]].copy()
    for raw_field, short in METRICS.items():
        paired[f"{short}_cont"] = pd.to_numeric(merged[f"{raw_field}_cont"], errors="raise")
        paired[f"{short}_final"] = pd.to_numeric(merged[f"{raw_field}_final"], errors="raise")
        paired[f"delta_{short}_cont_minus_final"] = (
            paired[f"{short}_cont"] - paired[f"{short}_final"]
        )
    paired = paired.sort_values(["dataset", "seed"]).reset_index(drop=True)
    summary_cols = [column for column in paired if column not in ("dataset", "seed")]
    ds_means = paired.groupby("dataset", sort=False)[summary_cols].mean().reset_index()
    ds_means["n_seeds"] = len(SEEDS)
    ds_means = ds_means.set_index("dataset").reindex(complete).reset_index()
    ds_std = paired.groupby("dataset")["delta_qnn_ba_cont_minus_final"].std(ddof=1)
    ds_means["delta_qnn_ba_sd_across_seeds"] = ds_means["dataset"].map(ds_std)
    differences = ds_means["delta_qnn_ba_cont_minus_final"].to_numpy()
    p = None
    if len(complete) == len(DATASETS) and np.any(np.abs(differences) > 1e-12):
        p = float(wilcoxon(differences, alternative="two-sided", zero_method="wilcox").pvalue)
    stat = {
        "ablation_design": PLAN,
        "source_campaign_design": source_plan,
        "profile": "forced_full_balanced_qnn",
        "n_complete_datasets": len(complete),
        "complete_datasets": complete,
        "missing_seeds_by_dataset": missing,
        "n_paired_seeds": int(len(paired)),
        "primary_endpoint": "balanced_accuracy, paired continuation minus final-sigma, averaged per dataset over five seeds",
        "mean_dataset_delta_qnn_ba": float(np.mean(differences)),
        "median_dataset_delta_qnn_ba": float(np.median(differences)),
        "positive_datasets": int((differences > 1e-12).sum()),
        "tied_datasets": int((np.abs(differences) <= 1e-12).sum()),
        "negative_datasets": int((differences < -1e-12).sum()),
        "wilcoxon_two_sided_p_exploratory": p,
        "mean_dataset_delta_root_success": float(
            ds_means["delta_root_success_cont_minus_final"].mean()
        ),
        "mean_dataset_objective_calls_cont": float(ds_means["objective_calls_cont"].mean()),
        "mean_dataset_objective_calls_final": float(ds_means["objective_calls_final"].mean()),
        "mean_dataset_feature_time_sec_cont": float(ds_means["feature_time_sec_cont"].mean()),
        "mean_dataset_feature_time_sec_final": float(ds_means["feature_time_sec_final"].mean()),
        "limitations": [
            "The fixed profile and comparison were chosen after inspecting earlier results.",
            "Identical number of L-BFGS-B stages and iteration ceilings; realized objective calls can differ.",
            "Only forced full_balanced QNN is tested; no broad profile or deployment-policy inference.",
            "Source continuation and new final-sigma controls are from distinct Git commits; verify recorded source and control environments.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out_dir / "paired_seed_results.csv", index=False, float_format="%.9g")
    ds_means.to_csv(out_dir / "dataset_comparison.csv", index=False, float_format="%.9g")
    (out_dir / "ablation_statistics.json").write_text(
        json.dumps(stat, indent=2) + "\n", encoding="utf-8"
    )
    print(ds_means[["dataset", "qnn_ba_cont", "qnn_ba_final",
                    "delta_qnn_ba_cont_minus_final", "root_success_cont",
                    "root_success_final", "objective_calls_cont",
                    "objective_calls_final"]].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"Paired datasets: {len(complete)}/9, paired seed jobs: {len(paired)}; "
          f"mean BA difference continuation minus final sigma: {np.mean(differences):+.5f}")
    print("Saved paired_seed_results.csv, dataset_comparison.csv, ablation_statistics.json.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-raw", type=Path, required=True)
    parser.add_argument("--reference-raw", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    analyse(args.control_raw, args.reference_raw, args.out_dir, args.require_complete)


if __name__ == "__main__":
    main()
