"""Summarize fixed-profile spectral-radius sensitivity descriptively."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs"/"fixed_profile_rho_sensitivity_v1"
RHO_VALUES=(1.20,1.60,2.00,2.25)
SEEDS=tuple(range(40,50))
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


def rho_tag(rho):
    return f"{rho:.2f}".replace(".","p")


def load_all():
    records=[]
    datasets=None
    for rho in RHO_VALUES:
        for seed in SEEDS:
            path=BASE/f"rho_{rho_tag(rho)}"/f"seed{seed}"/"result.json"
            if not path.exists():
                raise SystemExit(f"Missing sensitivity result: {path}")
            payload=json.loads(path.read_text(encoding="utf-8"))
            if float(payload["rho"])!=rho or int(payload["seed"])!=seed:
                raise SystemExit(f"Result provenance mismatch: {path}")
            if datasets is None:
                datasets=tuple(payload["datasets"])
            if tuple(payload["datasets"])!=datasets:
                raise SystemExit("Dataset order changed across sensitivity jobs")
            rows=payload["dataset_results"]
            if len(rows)!=len(datasets):
                raise SystemExit(f"Incomplete sensitivity result: {path}")
            for r in rows:
                if rho==2.0 and not r.get("anchor_regression",{}).get("passed"):
                    raise SystemExit(
                        f"rho=2.0 anchor did not pass for {r['dataset']} seed={seed}"
                    )
                records.append({
                    "rho":rho,
                    "seed":seed,
                    "dataset":r["dataset"],
                    "selected_method":r["selected_method"],
                    "zero_ba":float(r["results"]["zero"]["balanced_accuracy"]),
                    "selected_mrbi_ba":float(
                        r["results"]["selected_mrbi"]["balanced_accuracy"]
                    ),
                    "delta_selected_zero":float(
                        r["paired"]["delta_selected_zero"]
                    ),
                    "zero_success":float(
                        r["solver"]["zero_test"]["zero_success_rate"]
                    ),
                    "selected_success":float(
                        r["solver"]["selected_test"][
                            "forced_success_rate"
                            if r["selected_kind"]=="forced"
                            else "hybrid_success_rate"
                        ]
                    ),
                })
    return records,datasets


def main():
    records,datasets=load_all()

    dataset_rows=[]
    for rho in RHO_VALUES:
        for dataset in datasets:
            sub=[
                r for r in records
                if r["rho"]==rho and r["dataset"]==dataset
            ]
            if len(sub)!=10:
                raise SystemExit(
                    f"Expected 10 seeds for rho={rho} dataset={dataset}"
                )
            methods={r["selected_method"] for r in sub}
            if len(methods)!=1:
                raise SystemExit("Selected method changed across sensitivity seeds")
            dataset_rows.append({
                "rho":rho,
                "dataset":dataset,
                "selected_method":next(iter(methods)),
                "n_seeds":10,
                "zero_ba":float(np.mean([r["zero_ba"] for r in sub])),
                "selected_mrbi_ba":float(np.mean([
                    r["selected_mrbi_ba"] for r in sub
                ])),
                "delta_selected_zero":float(np.mean([
                    r["delta_selected_zero"] for r in sub
                ])),
                "zero_success":float(np.mean([
                    r["zero_success"] for r in sub
                ])),
                "selected_success":float(np.mean([
                    r["selected_success"] for r in sub
                ])),
            })

    overall=[]
    for idx,rho in enumerate(RHO_VALUES):
        sub=[r for r in dataset_rows if r["rho"]==rho]
        delta=np.asarray([
            r["delta_selected_zero"] for r in sub
        ],dtype=float)
        ci=bootstrap_ci(delta,seed=20261100+idx)
        overall.append({
            "rho":rho,
            "n_datasets":len(sub),
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
            "zero_success_mean":float(np.mean([
                r["zero_success"] for r in sub
            ])),
            "selected_success_mean":float(np.mean([
                r["selected_success"] for r in sub
            ])),
        })

    final_summary_path=ROOT/"results"/"final"/"confirmation_summary.json"
    final=json.loads(final_summary_path.read_text(encoding="utf-8"))
    anchor=next(r for r in overall if r["rho"]==2.0)
    expected=float(final["overall"]["delta_selected_zero_mean"])
    if not np.isclose(anchor["delta_mean"],expected,atol=1e-12,rtol=0.0):
        raise SystemExit(
            f"rho=2.0 aggregate anchor mismatch: "
            f"{anchor['delta_mean']} != {expected}"
        )

    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/"rho_dataset_summary.csv").open(
        "w",newline="",encoding="utf-8"
    ) as f:
        writer=csv.DictWriter(f,fieldnames=list(dataset_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dataset_rows)

    with (BASE/"rho_overall_summary.csv").open(
        "w",newline="",encoding="utf-8"
    ) as f:
        writer=csv.DictWriter(f,fieldnames=list(overall[0].keys()))
        writer.writeheader()
        writer.writerows(overall)

    payload={
        "plan":"fixed_profile_rho_sensitivity_v1",
        "analysis_role":"secondary descriptive sensitivity",
        "rhos":list(RHO_VALUES),
        "seeds":list(SEEDS),
        "datasets":list(datasets),
        "no_profile_reselection":True,
        "no_per_rho_hypothesis_tests":True,
        "anchor_rho":2.0,
        "anchor_matches_confirmation":True,
        "dataset_level":dataset_rows,
        "overall":overall,
    }
    (BASE/"rho_summary.json").write_text(
        json.dumps(payload,indent=2,allow_nan=False)+"\n",
        encoding="utf-8",
    )

    x=np.asarray([r["rho"] for r in overall],dtype=float)
    y=np.asarray([r["delta_mean"] for r in overall],dtype=float)
    lo=np.asarray([r["delta_bootstrap95_low"] for r in overall],dtype=float)
    hi=np.asarray([r["delta_bootstrap95_high"] for r in overall],dtype=float)
    yerr=np.vstack([y-lo,hi-y])

    fig,ax=plt.subplots(figsize=(6.2,4.0))
    ax.errorbar(x,y,yerr=yerr,marker="o",capsize=4)
    ax.axhline(0.0,linewidth=1)
    ax.set_xlabel("Spectral radius rho(W)")
    ax.set_ylabel("Selected MRBI - Zero balanced accuracy")
    ax.set_xticks(x)
    fig.tight_layout()
    fig.savefig(BASE/"rho_sensitivity_fixed_profiles.png",dpi=220)
    plt.close(fig)

    print("FIXED_PROFILE_RHO_SUMMARY_OK",BASE/"rho_summary.json")
    print(json.dumps(overall,indent=2))


if __name__=="__main__":
    main()
