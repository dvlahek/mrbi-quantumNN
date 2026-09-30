"""Summarize the completed locked rescue-only MRBI benchmark.

Requires all frozen seeds 30-39. The primary statistical unit is one dataset:
each of the nine manuscript datasets contributes its mean paired BA difference
across the ten frozen split seeds.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parents[1]
PLAN="locked_rescue_only_mrbi_full_balanced_rho225_v1"
BASE=ROOT/"outputs"/PLAN
SEEDS=tuple(range(30,40))
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


def bootstrap_ci(values,seed,n_boot=20000):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    draws=rng.choice(
        values,size=(n_boot,len(values)),replace=True
    ).mean(axis=1)
    return [
        float(np.quantile(draws,0.025)),
        float(np.quantile(draws,0.975)),
    ]


def safe_wilcoxon(values,alternative="greater"):
    values=np.asarray(values,dtype=float)
    if np.all(np.abs(values)<=1e-15):
        return {
            "statistic":0.0,
            "pvalue":1.0,
            "alternative":alternative,
        }
    test=wilcoxon(
        values,
        alternative=alternative,
        zero_method="wilcox",
        method="auto",
    )
    return {
        "statistic":float(test.statistic),
        "pvalue":float(test.pvalue),
        "alternative":alternative,
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
            raise SystemExit(f"Dataset drift: {path}")
        primary=payload.get("primary_method",{})
        expected={
            "profile":"full_balanced",
            "trigger":"zero strict solve failure only",
            "acceptance":"MRBI strict solve success only",
            "fallback":"original zero terminal state",
            "spectral_radius":2.25,
        }
        if primary!=expected:
            raise SystemExit(f"Primary-rule drift: {path}")
        commits.add(payload["git_commit"])
        by_name={d["dataset"]:d for d in payload["dataset_results"]}
        if tuple(by_name)!=DATASETS:
            raise SystemExit(f"Incomplete/order-drifted results: {path}")

        for dataset in DATASETS:
            d=by_name[dataset]
            ts=d["test_solver_stats"]
            records.append({
                "dataset":dataset,
                "seed":seed,
                "pca":float(d["results"]["pca"]["balanced_accuracy"]),
                "zero":float(d["results"]["zero"]["balanced_accuracy"]),
                "mrbi_rescue":float(
                    d["results"]["mrbi_rescue"]["balanced_accuracy"]
                ),
                "multistart_rescue":float(
                    d["results"]["multistart_rescue"]["balanced_accuracy"]
                ),
                "delta_mrbi_zero":float(
                    d["paired"]["delta_mrbi_rescue_zero"]
                ),
                "delta_ms_zero":float(
                    d["paired"]["delta_multistart_rescue_zero"]
                ),
                "delta_mrbi_ms":float(
                    d["paired"]["delta_mrbi_rescue_multistart_rescue"]
                ),
                "delta_mrbi_pca":float(
                    d["paired"]["delta_mrbi_rescue_pca"]
                ),
                "zero_success":float(ts["zero_success_rate"]),
                "mrbi_success":float(ts["mrbi_rescue_success_rate"]),
                "ms_success":float(ts["multistart_rescue_success_rate"]),
                "zero_failure_rate":float(ts["zero_failure_rate"]),
                "mrbi_rescues":int(ts["mrbi_rescues"]),
                "ms_rescues":int(ts["multistart_rescues"]),
                "mrbi_rescue_fraction":float(
                    ts["mrbi_rescue_fraction_of_zero_failures"]
                ),
                "ms_rescue_fraction":float(
                    ts["multistart_rescue_fraction_of_zero_failures"]
                ),
                "zero_runtime":float(ts["zero_mean_runtime_sec"]),
                "mrbi_runtime":float(ts["mrbi_mean_total_runtime_sec"]),
                "ms_runtime":float(
                    ts["multistart_mean_total_runtime_sec"]
                ),
            })

    dataset_rows=[]
    for dataset in DATASETS:
        subset=[r for r in records if r["dataset"]==dataset]
        row={"dataset":dataset,"n_seeds":len(subset)}
        mean_keys=(
            "pca","zero","mrbi_rescue","multistart_rescue",
            "delta_mrbi_zero","delta_ms_zero","delta_mrbi_ms",
            "delta_mrbi_pca","zero_success","mrbi_success","ms_success",
            "zero_failure_rate","mrbi_rescue_fraction","ms_rescue_fraction",
            "zero_runtime","mrbi_runtime","ms_runtime",
        )
        for key in mean_keys:
            row[key]=float(np.mean([r[key] for r in subset]))
        row["mrbi_rescues"]=int(sum(r["mrbi_rescues"] for r in subset))
        row["ms_rescues"]=int(sum(r["ms_rescues"] for r in subset))
        row["positive_seed_deltas_mrbi_zero"]=int(sum(
            r["delta_mrbi_zero"]>1e-12 for r in subset
        ))
        row["negative_seed_deltas_mrbi_zero"]=int(sum(
            r["delta_mrbi_zero"]< -1e-12 for r in subset
        ))
        dataset_rows.append(row)

    dz=np.asarray([r["delta_mrbi_zero"] for r in dataset_rows],dtype=float)
    dm=np.asarray([r["delta_mrbi_ms"] for r in dataset_rows],dtype=float)
    dp=np.asarray([r["delta_mrbi_pca"] for r in dataset_rows],dtype=float)
    sg=np.asarray([
        r["mrbi_success"]-r["zero_success"] for r in dataset_rows
    ],dtype=float)
    sm=np.asarray([
        r["mrbi_success"]-r["ms_success"] for r in dataset_rows
    ],dtype=float)

    overall={
        "pca_ba_mean":float(np.mean([r["pca"] for r in dataset_rows])),
        "zero_ba_mean":float(np.mean([r["zero"] for r in dataset_rows])),
        "mrbi_rescue_ba_mean":float(np.mean([
            r["mrbi_rescue"] for r in dataset_rows
        ])),
        "multistart_rescue_ba_mean":float(np.mean([
            r["multistart_rescue"] for r in dataset_rows
        ])),
        "delta_mrbi_zero_mean":float(dz.mean()),
        "delta_mrbi_zero_median":float(np.median(dz)),
        "delta_mrbi_zero_bootstrap95_dataset_ci":bootstrap_ci(
            dz,seed=20261001,
        ),
        "delta_mrbi_zero_positive_datasets":int(np.sum(dz>1e-12)),
        "delta_mrbi_zero_neutral_datasets":int(
            np.sum(np.abs(dz)<=1e-12)
        ),
        "delta_mrbi_zero_negative_datasets":int(np.sum(dz< -1e-12)),
        "wilcoxon_mrbi_gt_zero":safe_wilcoxon(dz),
        "delta_mrbi_multistart_rescue_mean":float(dm.mean()),
        "delta_mrbi_multistart_rescue_median":float(np.median(dm)),
        "delta_mrbi_multistart_rescue_bootstrap95_dataset_ci":bootstrap_ci(
            dm,seed=20261002,
        ),
        "wilcoxon_mrbi_gt_multistart_rescue":safe_wilcoxon(dm),
        "delta_mrbi_pca_mean":float(dp.mean()),
        "solver_success_gain_mrbi_minus_zero_mean":float(sg.mean()),
        "solver_success_gain_mrbi_minus_multistart_rescue_mean":float(
            sm.mean()
        ),
        "mrbi_rescues_total":int(sum(r["mrbi_rescues"] for r in dataset_rows)),
        "multistart_rescues_total":int(
            sum(r["ms_rescues"] for r in dataset_rows)
        ),
        "mrbi_rescue_fraction_of_zero_failures_mean":float(np.mean([
            r["mrbi_rescue_fraction"] for r in dataset_rows
        ])),
        "multistart_rescue_fraction_of_zero_failures_mean":float(np.mean([
            r["ms_rescue_fraction"] for r in dataset_rows
        ])),
        "zero_runtime_sec_per_sample_mean":float(np.mean([
            r["zero_runtime"] for r in dataset_rows
        ])),
        "mrbi_rescue_runtime_sec_per_sample_mean":float(np.mean([
            r["mrbi_runtime"] for r in dataset_rows
        ])),
        "multistart_rescue_runtime_sec_per_sample_mean":float(np.mean([
            r["ms_runtime"] for r in dataset_rows
        ])),
    }

    summary={
        "plan":PLAN,
        "frozen_seeds":list(SEEDS),
        "n_datasets":len(DATASETS),
        "n_dataset_seed_pairs":len(records),
        "commits_present":sorted(commits),
        "primary_statistical_unit":"dataset mean across ten frozen split seeds",
        "dataset_level":dataset_rows,
        "overall":overall,
        "interpretation_guardrails":[
            "The rescue-only rule was defined after inspecting only partial results from seeds 20-24 of the prior aggressive-policy benchmark.",
            "Seeds 30-39 are the prospective frozen evaluation for the rescue-only rule.",
            "MRBI and random-5 rescue are triggered only after strict zero-solve failure.",
            "When Zero succeeds, all implicit methods use exactly the same raw equilibrium feature.",
            "No profile, dataset, seed, threshold, or rescue acceptance rule is selected using seeds 30-39.",
            "PCA-QNN is a compact direct-encoding reference, not an implicit-solver baseline.",
        ],
    }

    BASE.mkdir(parents=True,exist_ok=True)
    (BASE/"locked_summary.json").write_text(
        json.dumps(summary,indent=2,allow_nan=False)+"\n",
        encoding="utf-8",
    )
    with (BASE/"locked_dataset_summary.csv").open(
        "w",newline="",encoding="utf-8"
    ) as handle:
        writer=csv.DictWriter(
            handle,fieldnames=list(dataset_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(dataset_rows)

    print(
        "LOCKED_RESCUE_ONLY_SUMMARY_OK",
        BASE/"locked_summary.json",
        flush=True,
    )
    print(json.dumps(overall,indent=2),flush=True)


if __name__=="__main__":
    main()
