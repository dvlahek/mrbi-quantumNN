"""Summarize the fully completed locked fixed-policy MRBI benchmark.

Requires all frozen seeds 20-29. The statistical unit for the primary
confirmatory comparison is the dataset: each dataset contributes its mean
paired BA difference across the ten frozen split seeds.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parents[1]
PLAN="locked_fixed_mrbi_full_balanced_aggressive_rho225_v1"
BASE=ROOT/"outputs"/PLAN
SEEDS=tuple(range(20,30))
DATASETS=(
    "breast_cancer",
    "wine_binary",
    "wine_0_vs_2",
    "wine_1_vs_2",
    "digits_1_vs_7",
    "digits_2_vs_7",
    "digits_3_vs_8",
    "digits_4_vs_9",
    "digits_5_vs_6",
)


def bootstrap_ci(values,seed=20260930,n_boot=20000):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    draws=rng.choice(values,size=(n_boot,len(values)),replace=True).mean(axis=1)
    return [float(np.quantile(draws,0.025)),float(np.quantile(draws,0.975))]


def safe_wilcoxon(values):
    values=np.asarray(values,dtype=float)
    if np.all(np.abs(values)<=1e-15):
        return {"statistic":0.0,"pvalue":1.0,"alternative":"greater"}
    test=wilcoxon(
        values,
        alternative="greater",
        zero_method="wilcox",
        method="auto",
    )
    return {
        "statistic":float(test.statistic),
        "pvalue":float(test.pvalue),
        "alternative":"greater",
    }


def main():
    records=[]
    commits=set()
    for seed in SEEDS:
        path=BASE/f"seed{seed}"/"locked_result.json"
        if not path.exists():
            raise SystemExit(f"Missing locked result: {path}")
        payload=json.loads(path.read_text(encoding="utf-8"))
        if payload.get("plan")!=PLAN or payload.get("seed")!=seed:
            raise SystemExit(f"Provenance mismatch: {path}")
        if tuple(payload.get("datasets",()))!=DATASETS:
            raise SystemExit(f"Dataset set drift: {path}")
        if payload["primary_method"]!={
            "profile":"full_balanced",
            "hybrid":"aggressive",
            "spectral_radius":2.25,
        }:
            raise SystemExit(f"Policy drift: {path}")
        commits.add(payload["git_commit"])
        by_name={d["dataset"]:d for d in payload["dataset_results"]}
        if tuple(by_name)!=DATASETS:
            raise SystemExit(f"Incomplete/order-drifted results: {path}")
        for dataset in DATASETS:
            d=by_name[dataset]
            records.append({
                "dataset":dataset,
                "seed":seed,
                "pca":float(d["results"]["pca"]["balanced_accuracy"]),
                "zero":float(d["results"]["zero"]["balanced_accuracy"]),
                "mrbi":float(d["results"]["mrbi"]["balanced_accuracy"]),
                "multistart":float(d["results"]["multistart"]["balanced_accuracy"]),
                "delta_mrbi_zero":float(d["paired"]["delta_mrbi_zero"]),
                "delta_mrbi_multistart":float(d["paired"]["delta_mrbi_multistart"]),
                "delta_mrbi_pca":float(d["paired"]["delta_mrbi_pca"]),
                "zero_success":float(d["test_solver_stats"]["zero_success_rate"]),
                "mrbi_success":float(d["test_solver_stats"]["mrbi_success_rate"]),
                "multistart_success":float(d["test_solver_stats"]["multistart_success_rate"]),
                "mrbi_rescues":int(d["test_solver_stats"]["mrbi_rescues"]),
                "multistart_rescues":int(d["test_solver_stats"]["multistart_rescues"]),
                "mrbi_runtime":float(d["test_solver_stats"]["mrbi_mean_runtime_sec"]),
                "multistart_runtime":float(d["test_solver_stats"]["multistart_mean_runtime_sec"]),
            })

    dataset_rows=[]
    for dataset in DATASETS:
        subset=[r for r in records if r["dataset"]==dataset]
        row={"dataset":dataset,"n_seeds":len(subset)}
        for key in (
            "pca","zero","mrbi","multistart",
            "delta_mrbi_zero","delta_mrbi_multistart","delta_mrbi_pca",
            "zero_success","mrbi_success","multistart_success",
            "mrbi_runtime","multistart_runtime",
        ):
            row[key]=float(np.mean([r[key] for r in subset]))
        row["mrbi_rescues"]=int(sum(r["mrbi_rescues"] for r in subset))
        row["multistart_rescues"]=int(sum(r["multistart_rescues"] for r in subset))
        row["positive_seed_deltas_mrbi_zero"]=int(sum(
            r["delta_mrbi_zero"]>1e-12 for r in subset
        ))
        row["negative_seed_deltas_mrbi_zero"]=int(sum(
            r["delta_mrbi_zero"]< -1e-12 for r in subset
        ))
        dataset_rows.append(row)

    dz=np.asarray([r["delta_mrbi_zero"] for r in dataset_rows],dtype=float)
    dm=np.asarray([r["delta_mrbi_multistart"] for r in dataset_rows],dtype=float)
    dp=np.asarray([r["delta_mrbi_pca"] for r in dataset_rows],dtype=float)
    sz=np.asarray([
        r["mrbi_success"]-r["zero_success"] for r in dataset_rows
    ],dtype=float)
    sm=np.asarray([
        r["mrbi_success"]-r["multistart_success"] for r in dataset_rows
    ],dtype=float)

    summary={
        "plan":PLAN,
        "frozen_seeds":list(SEEDS),
        "n_datasets":len(DATASETS),
        "n_dataset_seed_pairs":len(records),
        "commits_present":sorted(commits),
        "primary_statistical_unit":"dataset mean across ten frozen split seeds",
        "dataset_level":dataset_rows,
        "overall":{
            "pca_ba_mean":float(np.mean([r["pca"] for r in dataset_rows])),
            "zero_ba_mean":float(np.mean([r["zero"] for r in dataset_rows])),
            "mrbi_ba_mean":float(np.mean([r["mrbi"] for r in dataset_rows])),
            "multistart_ba_mean":float(np.mean([r["multistart"] for r in dataset_rows])),
            "delta_mrbi_zero_mean":float(dz.mean()),
            "delta_mrbi_zero_median":float(np.median(dz)),
            "delta_mrbi_zero_bootstrap95_dataset_ci":bootstrap_ci(dz),
            "delta_mrbi_zero_positive_datasets":int(np.sum(dz>1e-12)),
            "delta_mrbi_zero_neutral_datasets":int(np.sum(np.abs(dz)<=1e-12)),
            "delta_mrbi_zero_negative_datasets":int(np.sum(dz< -1e-12)),
            "wilcoxon_mrbi_gt_zero":safe_wilcoxon(dz),
            "delta_mrbi_multistart_mean":float(dm.mean()),
            "delta_mrbi_multistart_median":float(np.median(dm)),
            "delta_mrbi_multistart_bootstrap95_dataset_ci":bootstrap_ci(dm,seed=20260931),
            "wilcoxon_mrbi_gt_multistart":safe_wilcoxon(dm),
            "delta_mrbi_pca_mean":float(dp.mean()),
            "solver_success_gain_mrbi_minus_zero_mean":float(sz.mean()),
            "solver_success_gain_mrbi_minus_multistart_mean":float(sm.mean()),
            "mrbi_rescues_total":int(sum(r["mrbi_rescues"] for r in dataset_rows)),
            "multistart_rescues_total":int(sum(r["multistart_rescues"] for r in dataset_rows)),
        },
        "interpretation_guardrails":[
            "No profile, dataset, seed, or cap was selected using seeds 20-29.",
            "The primary MRBI policy is full_balanced plus aggressive hybrid at rho=2.25.",
            "The five-start multistart control includes the zero initialization and is not claimed to be exactly compute-matched; measured runtime is reported.",
            "PCA-QNN remains a direct compact-encoding reference and is not an implicit-solver baseline.",
        ],
    }

    out_json=BASE/"locked_summary.json"
    out_json.write_text(
        json.dumps(summary,indent=2,allow_nan=False)+"\n",
        encoding="utf-8",
    )
    out_csv=BASE/"locked_dataset_summary.csv"
    with out_csv.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(dataset_rows[0]))
        writer.writeheader()
        writer.writerows(dataset_rows)

    print("LOCKED_SUMMARY_OK",out_json,flush=True)
    print(json.dumps(summary["overall"],indent=2),flush=True)


if __name__=="__main__":
    main()
