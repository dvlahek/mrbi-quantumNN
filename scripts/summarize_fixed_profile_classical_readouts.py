"""Summarize the fixed-profile classical readout check descriptively."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs"/"fixed_profile_classical_readouts_v1"
SEEDS=tuple(range(40,50))
READOUTS=("logreg","svm_rbf","mlp","rf","gb")
ZERO_TOL=1e-12


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


def main():
    records=[]
    datasets=None
    for seed in SEEDS:
        path=BASE/f"seed{seed}"/"result.json"
        if not path.exists():
            raise SystemExit(f"Missing classical readout result: {path}")
        payload=json.loads(path.read_text(encoding="utf-8"))
        if int(payload["seed"])!=seed:
            raise SystemExit(f"Seed mismatch in {path}")
        if tuple(payload["readouts"])!=READOUTS:
            raise SystemExit(f"Readout set changed in {path}")
        if datasets is None:
            datasets=tuple(payload["datasets"])
        if tuple(payload["datasets"])!=datasets:
            raise SystemExit("Dataset order changed across readout jobs")

        for row in payload["dataset_results"]:
            for readout in READOUTS:
                r=row["readouts"][readout]
                records.append({
                    "seed":seed,
                    "dataset":row["dataset"],
                    "selected_method":row["selected_method"],
                    "readout":readout,
                    "pca_ba":float(r["pca"]["balanced_accuracy"]),
                    "zero_ba":float(r["zero"]["balanced_accuracy"]),
                    "selected_mrbi_ba":float(
                        r["selected_mrbi"]["balanced_accuracy"]
                    ),
                    "delta_selected_zero":float(
                        r["delta_selected_zero"]
                    ),
                })

    dataset_rows=[]
    for readout in READOUTS:
        for dataset in datasets:
            sub=[
                r for r in records
                if r["readout"]==readout and r["dataset"]==dataset
            ]
            if len(sub)!=10:
                raise SystemExit(
                    f"Expected 10 seeds for {readout} / {dataset}"
                )
            methods={r["selected_method"] for r in sub}
            if len(methods)!=1:
                raise SystemExit("Selected MRBI method changed across seeds")
            dataset_rows.append({
                "readout":readout,
                "dataset":dataset,
                "selected_method":next(iter(methods)),
                "n_seeds":10,
                "pca_ba":float(np.mean([r["pca_ba"] for r in sub])),
                "zero_ba":float(np.mean([r["zero_ba"] for r in sub])),
                "selected_mrbi_ba":float(np.mean([
                    r["selected_mrbi_ba"] for r in sub
                ])),
                "delta_selected_zero":float(np.mean([
                    r["delta_selected_zero"] for r in sub
                ])),
            })

    overall=[]
    for idx,readout in enumerate(READOUTS):
        sub=[r for r in dataset_rows if r["readout"]==readout]
        delta=np.asarray([
            r["delta_selected_zero"] for r in sub
        ],dtype=float)
        ci=bootstrap_ci(delta,seed=20261200+idx)
        overall.append({
            "readout":readout,
            "n_datasets":len(sub),
            "pca_ba_mean":float(np.mean([r["pca_ba"] for r in sub])),
            "zero_ba_mean":float(np.mean([r["zero_ba"] for r in sub])),
            "selected_mrbi_ba_mean":float(np.mean([
                r["selected_mrbi_ba"] for r in sub
            ])),
            "delta_mean":float(delta.mean()),
            "delta_median":float(np.median(delta)),
            "delta_bootstrap95_low":ci[0],
            "delta_bootstrap95_high":ci[1],
            "positive_datasets":int(np.sum(delta>ZERO_TOL)),
            "neutral_datasets":int(np.sum(np.abs(delta)<=ZERO_TOL)),
            "negative_datasets":int(np.sum(delta< -ZERO_TOL)),
        })

    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/"classical_dataset_summary.csv").open(
        "w",newline="",encoding="utf-8"
    ) as f:
        writer=csv.DictWriter(f,fieldnames=list(dataset_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dataset_rows)

    with (BASE/"classical_overall_summary.csv").open(
        "w",newline="",encoding="utf-8"
    ) as f:
        writer=csv.DictWriter(f,fieldnames=list(overall[0].keys()))
        writer.writeheader()
        writer.writerows(overall)

    payload={
        "plan":"fixed_profile_classical_readouts_v1",
        "analysis_role":"secondary descriptive readout check",
        "rho":2.0,
        "seeds":list(SEEDS),
        "datasets":list(datasets),
        "readouts":list(READOUTS),
        "no_profile_reselection":True,
        "no_readout_hyperparameter_tuning":True,
        "no_formal_hypothesis_tests":True,
        "dataset_level":dataset_rows,
        "overall":overall,
    }
    (BASE/"classical_summary.json").write_text(
        json.dumps(payload,indent=2,allow_nan=False)+"\n",
        encoding="utf-8",
    )

    print(
        "FIXED_PROFILE_CLASSICAL_SUMMARY_OK",
        BASE/"classical_summary.json",
    )
    print(json.dumps(overall,indent=2))


if __name__=="__main__":
    main()
