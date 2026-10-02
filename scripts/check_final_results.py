"""Verify the fixed profile map and tracked aggregate results."""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/"experiments"/"selected_profile_confirmation_lock_v1.json"
CONF_JSON=ROOT/"results"/"final"/"confirmation_summary.json"
CONF_CSV=ROOT/"results"/"final"/"confirmation_dataset_summary.csv"
RHO_OVERALL=ROOT/"results"/"final"/"rho_overall_summary.csv"
RHO_DATASET=ROOT/"results"/"final"/"rho_dataset_summary.csv"
CLASS_OVERALL=ROOT/"results"/"final"/"classical_overall_summary.csv"
CLASS_DATASET=ROOT/"results"/"final"/"classical_dataset_summary.csv"

EXPECTED_RAW_SHA="e1a3917be493485156e87e79c921d2a91a54cf715aa55443516059fc3d370cd6"
EXPECTED_SELECTED_SHA="4cb9ad804076dea25820b73c1640a81e85c7d3b4ad77d16341555269da707908"
DATASETS=(
    "breast_cancer","wine_binary","wine_0_vs_2","wine_1_vs_2",
    "digits_1_vs_7","digits_2_vs_7","digits_3_vs_8",
    "digits_4_vs_9","digits_5_vs_6",
)
RHOS=(1.20,1.60,2.00,2.25)
READOUTS=("logreg","svm_rbf","mlp","rf","gb")
TOL=1e-12


def close(label,actual,expected,tol=TOL):
    if not np.isclose(float(actual),float(expected),atol=tol,rtol=0.0):
        raise AssertionError(f"{label}: {actual} != {expected}")


def exact_signed_rank_greater(values,tol=TOL):
    values=np.asarray(values,dtype=float)
    values=values[np.abs(values)>tol]
    if len(values)==0:
        return 0.0,1.0
    ranks=rankdata(np.abs(values),method="average")
    statistic=float(np.sum(ranks[values>0]))
    null=np.fromiter(
        (
            sum(rank for rank,include in zip(ranks,bits) if include)
            for bits in product((0,1),repeat=len(ranks))
        ),
        dtype=float,
    )
    return statistic,float(np.mean(null>=statistic-1e-15))


def main():
    lock=json.loads(LOCK.read_text(encoding="utf-8"))
    summary=json.loads(CONF_JSON.read_text(encoding="utf-8"))
    conf=pd.read_csv(CONF_CSV)
    rho_overall=pd.read_csv(RHO_OVERALL)
    rho_dataset=pd.read_csv(RHO_DATASET)
    class_overall=pd.read_csv(CLASS_OVERALL)
    class_dataset=pd.read_csv(CLASS_DATASET)

    if lock["lock_schema"]!="mrbi_selected_profile_confirmation_lock_v1":
        raise AssertionError("Unexpected profile-lock schema")
    evidence=lock["development_evidence"]
    if evidence["main_raw_sha256"]!=EXPECTED_RAW_SHA:
        raise AssertionError("Development raw hash changed")
    if evidence["selected_profiles_sha256"]!=EXPECTED_SELECTED_SHA:
        raise AssertionError("Selected-profile hash changed")
    if tuple(evidence["datasets"])!=DATASETS:
        raise AssertionError("Dataset set/order changed")
    if tuple(lock["confirmation_plan"]["seeds"])!=tuple(range(40,50)):
        raise AssertionError("Confirmation seed set changed")

    methods={row["dataset"]:row["method"] for row in lock["selected_profiles"]}
    if tuple(conf["dataset"])!=DATASETS or len(conf)!=9:
        raise AssertionError("Confirmation dataset summary is incomplete")
    if dict(zip(conf["dataset"],conf["selected_method"]))!=methods:
        raise AssertionError("Confirmation methods differ from fixed profile map")

    dz=conf["delta_selected_zero"].to_numpy(float)
    dm=conf["delta_selected_multistart5"].to_numpy(float)
    close("selected MRBI mean BA",conf["selected_mrbi"].mean(),
          summary["overall"]["selected_mrbi_ba_mean"])
    close("Zero mean BA",conf["zero"].mean(),
          summary["overall"]["zero_ba_mean"])
    close("MRBI-Zero mean",dz.mean(),
          summary["overall"]["delta_selected_zero_mean"])
    close("MRBI-multistart mean",dm.mean(),
          summary["overall"]["delta_selected_multistart5_mean"])

    stat,p=exact_signed_rank_greater(dz)
    close("MRBI-Zero signed-rank statistic",stat,
          summary["overall"]["wilcoxon_selected_gt_zero"]["statistic"])
    close("MRBI-Zero signed-rank p",p,
          summary["overall"]["wilcoxon_selected_gt_zero"]["pvalue"])
    stat,p=exact_signed_rank_greater(dm)
    close("MRBI-multistart signed-rank statistic",stat,
          summary["overall"]["wilcoxon_selected_gt_multistart5"]["statistic"])
    close("MRBI-multistart signed-rank p",p,
          summary["overall"]["wilcoxon_selected_gt_multistart5"]["pvalue"])

    if tuple(rho_overall["rho"].astype(float))!=RHOS:
        raise AssertionError("Spectral-radius grid changed")
    if len(rho_dataset)!=len(RHOS)*len(DATASETS):
        raise AssertionError("Spectral-radius dataset table is incomplete")
    for rho in RHOS:
        sub=rho_dataset[np.isclose(rho_dataset["rho"],rho)]
        if tuple(sub["dataset"])!=DATASETS:
            raise AssertionError(f"Dataset order changed at rho={rho}")
        if dict(zip(sub["dataset"],sub["selected_method"]))!=methods:
            raise AssertionError(f"Fixed profile map changed at rho={rho}")

    anchor=rho_dataset[np.isclose(rho_dataset["rho"],2.0)].reset_index(drop=True)
    for col_rho,col_conf in (
        ("zero_ba","zero"),
        ("selected_mrbi_ba","selected_mrbi"),
        ("delta_selected_zero","delta_selected_zero"),
        ("zero_success","zero_success"),
        ("selected_success","selected_success"),
    ):
        if not np.allclose(
            anchor[col_rho].to_numpy(float),
            conf[col_conf].to_numpy(float),
            atol=TOL,rtol=0.0,
        ):
            raise AssertionError(f"rho=2.0 anchor mismatch in {col_rho}")

    rho2=rho_overall[np.isclose(rho_overall["rho"],2.0)].iloc[0]
    close("rho=2.0 aggregate delta",rho2["delta_mean"],
          summary["overall"]["delta_selected_zero_mean"])

    if tuple(class_overall["readout"])!=READOUTS:
        raise AssertionError("Classical readout set/order changed")
    if len(class_dataset)!=len(READOUTS)*len(DATASETS):
        raise AssertionError("Classical-readout dataset table is incomplete")
    for readout in READOUTS:
        sub=class_dataset[class_dataset["readout"]==readout]
        if tuple(sub["dataset"])!=DATASETS:
            raise AssertionError(f"Dataset order changed for {readout}")
        if dict(zip(sub["dataset"],sub["selected_method"]))!=methods:
            raise AssertionError(f"Fixed profile map changed for {readout}")
        row=class_overall[class_overall["readout"]==readout].iloc[0]
        close(
            f"{readout} aggregate delta",
            sub["delta_selected_zero"].mean(),
            row["delta_mean"],
        )

    print("Final reproducibility checks passed.")


if __name__=="__main__":
    main()
