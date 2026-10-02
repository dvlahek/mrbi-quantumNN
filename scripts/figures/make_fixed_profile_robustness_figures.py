"""Generate publication-ready figures for fixed-profile robustness checks.

Inputs are the code-generated summaries produced by:
  - summarize_fixed_profile_rho_sensitivity.py
  - summarize_fixed_profile_classical_readouts.py
  - results/final/confirmation_summary.json

No experimental values are hard-coded in this script. The rho=2.0 classification
mean is checked against the canonical confirmation summary before plotting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]

DEFAULT_RHO_BASE=ROOT/"outputs"/"fixed_profile_rho_sensitivity_v1"
DEFAULT_CLASSICAL_BASE=ROOT/"outputs"/"fixed_profile_classical_readouts_v1"
DEFAULT_CONFIRMATION=ROOT/"results"/"final"/"confirmation_summary.json"
DEFAULT_OUT=ROOT/"outputs"/"fixed_profile_robustness_figures"

RHO_VALUES=(1.20,1.60,2.00,2.25)
DATASET_ORDER=(
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
READOUT_ORDER=("logreg","svm_rbf","mlp","rf","gb")
READOUT_LABELS={
    "qnn":"QNN",
    "logreg":"Logistic regression",
    "svm_rbf":"RBF SVM",
    "mlp":"MLP",
    "rf":"Random forest",
    "gb":"Gradient boosting",
}
DATASET_LABELS={
    "breast_cancer":"Breast cancer",
    "wine_binary":"Wine binary",
    "wine_0_vs_2":"Wine 0 vs 2",
    "wine_1_vs_2":"Wine 1 vs 2",
    "digits_1_vs_7":"Digits 1 vs 7",
    "digits_2_vs_7":"Digits 2 vs 7",
    "digits_3_vs_8":"Digits 3 vs 8",
    "digits_4_vs_9":"Digits 4 vs 9",
    "digits_5_vs_6":"Digits 5 vs 6",
}


def bootstrap_mean_ci(values,seed,n_boot=20000):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or len(values)==0:
        raise ValueError("bootstrap values must be a non-empty vector")
    rng=np.random.default_rng(seed)
    draws=rng.choice(
        values,
        size=(n_boot,len(values)),
        replace=True,
    ).mean(axis=1)
    return (
        float(np.quantile(draws,0.025)),
        float(np.quantile(draws,0.975)),
    )


def save_figure(fig,out_dir,stem,dpi=300):
    out_dir.mkdir(parents=True,exist_ok=True)
    png=out_dir/f"{stem}.png"
    pdf=out_dir/f"{stem}.pdf"
    fig.savefig(png,dpi=dpi,bbox_inches="tight")
    fig.savefig(pdf,bbox_inches="tight")
    return png,pdf


def require_columns(frame,columns,name):
    missing=[c for c in columns if c not in frame.columns]
    if missing:
        raise SystemExit(f"{name}: missing columns {missing}")


def load_inputs(rho_base,classical_base,confirmation_path):
    rho_dataset=pd.read_csv(rho_base/"rho_dataset_summary.csv")
    rho_overall=pd.read_csv(rho_base/"rho_overall_summary.csv")
    classical_dataset=pd.read_csv(
        classical_base/"classical_dataset_summary.csv"
    )
    classical_overall=pd.read_csv(
        classical_base/"classical_overall_summary.csv"
    )
    confirmation=json.loads(
        confirmation_path.read_text(encoding="utf-8")
    )

    require_columns(
        rho_dataset,
        (
            "rho","dataset","delta_selected_zero",
            "zero_success","selected_success",
        ),
        "rho_dataset_summary",
    )
    require_columns(
        rho_overall,
        (
            "rho","delta_mean","delta_bootstrap95_low",
            "delta_bootstrap95_high",
        ),
        "rho_overall_summary",
    )
    require_columns(
        classical_dataset,
        ("readout","dataset","delta_selected_zero"),
        "classical_dataset_summary",
    )
    require_columns(
        classical_overall,
        (
            "readout","delta_mean","delta_bootstrap95_low",
            "delta_bootstrap95_high",
            "positive_datasets","neutral_datasets","negative_datasets",
        ),
        "classical_overall_summary",
    )

    rhos=tuple(float(x) for x in rho_overall["rho"])
    if rhos!=RHO_VALUES:
        raise SystemExit(f"Unexpected rho order: {rhos}")

    datasets=tuple(
        rho_dataset.loc[
            np.isclose(rho_dataset["rho"],RHO_VALUES[0]),"dataset"
        ].tolist()
    )
    if datasets!=DATASET_ORDER:
        raise SystemExit(f"Unexpected dataset order: {datasets}")

    canonical=float(
        confirmation["overall"]["delta_selected_zero_mean"]
    )
    anchor=float(
        rho_overall.loc[
            np.isclose(rho_overall["rho"],2.0),"delta_mean"
        ].iloc[0]
    )
    if not np.isclose(anchor,canonical,atol=1e-12,rtol=0.0):
        raise SystemExit(
            "rho=2.0 classification mean does not match confirmation: "
            f"{anchor} != {canonical}"
        )

    return (
        rho_dataset,
        rho_overall,
        classical_dataset,
        classical_overall,
        confirmation,
    )


def make_rho_main_figure(rho_dataset,rho_overall,confirmation,out_dir):
    x=rho_overall["rho"].to_numpy(dtype=float)

    ba=rho_overall["delta_mean"].to_numpy(dtype=float)*100.0
    ba_lo=rho_overall["delta_bootstrap95_low"].to_numpy(dtype=float)*100.0
    ba_hi=rho_overall["delta_bootstrap95_high"].to_numpy(dtype=float)*100.0

    # Use the canonical confirmation interval at rho=2.0 so the same anchor
    # is reported consistently in the main confirmation table and this figure.
    anchor_idx=int(np.where(np.isclose(x,2.0))[0][0])
    anchor_ci=confirmation["overall"]["delta_selected_zero_bootstrap95_dataset_ci"]
    ba_lo[anchor_idx]=float(anchor_ci[0])*100.0
    ba_hi[anchor_idx]=float(anchor_ci[1])*100.0

    ba_err=np.vstack((ba-ba_lo,ba_hi-ba))

    solver_mean=[]
    solver_lo=[]
    solver_hi=[]
    for idx,rho in enumerate(RHO_VALUES):
        sub=rho_dataset.loc[
            np.isclose(rho_dataset["rho"],rho)
        ]
        gain=(
            sub["selected_success"].to_numpy(dtype=float)
            -sub["zero_success"].to_numpy(dtype=float)
        )
        lo,hi=bootstrap_mean_ci(gain,seed=20261300+idx)
        solver_mean.append(float(np.mean(gain))*100.0)
        solver_lo.append(lo*100.0)
        solver_hi.append(hi*100.0)

    solver_mean=np.asarray(solver_mean)
    solver_lo=np.asarray(solver_lo)
    solver_hi=np.asarray(solver_hi)
    solver_err=np.vstack(
        (solver_mean-solver_lo,solver_hi-solver_mean)
    )

    fig,axes=plt.subplots(1,2,figsize=(10.2,4.0))

    ax=axes[0]
    ax.errorbar(x,ba,yerr=ba_err,marker="o",capsize=4,linewidth=1.5)
    ax.axhline(0.0,linewidth=1)
    ax.axvline(2.0,linestyle="--",linewidth=1)
    ax.set_xticks(x)
    ax.set_xlabel(r"Spectral radius $\rho(W)$")
    ax.set_ylabel("MRBI - Zero balanced accuracy (pp)")
    ax.set_title("(a) Classification effect")

    ax=axes[1]
    ax.errorbar(
        x,
        solver_mean,
        yerr=solver_err,
        marker="o",
        capsize=4,
        linewidth=1.5,
    )
    ax.axhline(0.0,linewidth=1)
    ax.axvline(2.0,linestyle="--",linewidth=1)
    ax.set_xticks(x)
    ax.set_xlabel(r"Spectral radius $\rho(W)$")
    ax.set_ylabel("MRBI - Zero solve success (pp)")
    ax.set_title("(b) Solver-success effect")

    fig.tight_layout()
    paths=save_figure(
        fig,out_dir,"rho_sensitivity_fixed_profiles"
    )
    plt.close(fig)
    return paths


def make_rho_dataset_figure(rho_dataset,out_dir):
    fig,ax=plt.subplots(figsize=(7.4,4.8))

    for dataset in DATASET_ORDER:
        sub=rho_dataset.loc[
            rho_dataset["dataset"]==dataset
        ].sort_values("rho")
        if tuple(float(x) for x in sub["rho"])!=RHO_VALUES:
            raise SystemExit(
                f"Incomplete rho series for dataset {dataset}"
            )
        ax.plot(
            sub["rho"],
            sub["delta_selected_zero"]*100.0,
            marker="o",
            markersize=4.5,
            linewidth=1.0,
            label=DATASET_LABELS[dataset],
        )

    ax.axhline(0.0,linewidth=1)
    ax.axvline(2.0,linestyle="--",linewidth=1)
    ax.set_xticks(RHO_VALUES)
    ax.set_xlabel(r"Spectral radius $\rho(W)$")
    ax.set_ylabel("MRBI - Zero balanced accuracy (pp)")
    ax.legend(
        loc="best",
        fontsize=8,
        ncol=2,
        frameon=False,
    )
    fig.tight_layout()
    paths=save_figure(
        fig,out_dir,"rho_sensitivity_by_dataset"
    )
    plt.close(fig)
    return paths


def make_readout_figure(
    classical_overall,
    confirmation,
    out_dir,
):
    rows=[]

    qnn=confirmation["overall"]
    qnn_ci=qnn["delta_selected_zero_bootstrap95_dataset_ci"]
    rows.append({
        "readout":"qnn",
        "mean":float(qnn["delta_selected_zero_mean"]),
        "low":float(qnn_ci[0]),
        "high":float(qnn_ci[1]),
        "positive":int(qnn["delta_selected_zero_positive_datasets"]),
        "neutral":int(qnn["delta_selected_zero_neutral_datasets"]),
        "negative":int(qnn["delta_selected_zero_negative_datasets"]),
    })

    by_name=classical_overall.set_index("readout")
    for name in READOUT_ORDER:
        if name not in by_name.index:
            raise SystemExit(f"Missing readout summary: {name}")
        r=by_name.loc[name]
        rows.append({
            "readout":name,
            "mean":float(r["delta_mean"]),
            "low":float(r["delta_bootstrap95_low"]),
            "high":float(r["delta_bootstrap95_high"]),
            "positive":int(r["positive_datasets"]),
            "neutral":int(r["neutral_datasets"]),
            "negative":int(r["negative_datasets"]),
        })

    labels=[
        (
            f"{READOUT_LABELS[r['readout']]}  "
            f"({r['positive']}/{r['neutral']}/{r['negative']})"
        )
        for r in rows
    ]
    means=np.asarray([r["mean"] for r in rows])*100.0
    lows=np.asarray([r["low"] for r in rows])*100.0
    highs=np.asarray([r["high"] for r in rows])*100.0
    xerr=np.vstack((means-lows,highs-means))

    y=np.arange(len(rows))[::-1]

    fig,ax=plt.subplots(figsize=(7.7,4.3))
    ax.errorbar(
        means,
        y,
        xerr=xerr,
        fmt="o",
        capsize=4,
        linewidth=1.4,
    )
    ax.axvline(0.0,linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("MRBI - Zero balanced accuracy (pp)")
    fig.tight_layout()

    paths=save_figure(
        fig,out_dir,"classical_readout_fixed_profiles"
    )
    plt.close(fig)
    return paths


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rho-base",
        type=Path,
        default=DEFAULT_RHO_BASE,
    )
    parser.add_argument(
        "--classical-base",
        type=Path,
        default=DEFAULT_CLASSICAL_BASE,
    )
    parser.add_argument(
        "--confirmation",
        type=Path,
        default=DEFAULT_CONFIRMATION,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
    )
    args=parser.parse_args()

    (
        rho_dataset,
        rho_overall,
        _classical_dataset,
        classical_overall,
        confirmation,
    )=load_inputs(
        args.rho_base.resolve(),
        args.classical_base.resolve(),
        args.confirmation.resolve(),
    )

    paths=[]
    paths.extend(
        make_rho_main_figure(
            rho_dataset,rho_overall,confirmation,args.out_dir.resolve()
        )
    )
    paths.extend(
        make_rho_dataset_figure(
            rho_dataset,args.out_dir.resolve()
        )
    )
    paths.extend(
        make_readout_figure(
            classical_overall,
            confirmation,
            args.out_dir.resolve(),
        )
    )

    print("FIXED_PROFILE_ROBUSTNESS_FIGURES_OK")
    for path in paths:
        print(path)


if __name__=="__main__":
    main()
