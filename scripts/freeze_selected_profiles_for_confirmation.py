"""Freeze the corrected-continuation development selection before confirmation.

Inputs are the completed corrected-continuation campaign on development seeds 0-4.
This script:
  1) validates all 9 x 5 jobs and implementation version;
  2) deterministically recomputes the best QNN MRBI method per dataset;
  3) checks it against selected_profiles.csv when that file is present;
  4) records SHA-256 hashes of the development evidence;
  5) writes one lock-candidate JSON for Git commit BEFORE seeds 40-49 run.

It does not load or run confirmation seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_BASE=ROOT/"outputs"/"continuation_v1"
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
DEV_SEEDS=(0,1,2,3,4)
VERSION="mrbi_continuation_v1"
EXPECTED_METHODS_PER_JOB=97
CONFIRMATION_SEEDS=tuple(range(40,50))


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def validate_raw(data:pd.DataFrame):
    required={
        "dataset","seed","method","readout",
        "implementation_version","balanced_accuracy",
        "n_qubits","latent_dim","spectral_radius",
        "input_scale","max_samples_per_class",
    }
    missing=required-set(data.columns)
    if missing:
        raise SystemExit(f"Missing corrected raw columns: {sorted(missing)}")
    if data.duplicated(["dataset","seed","method"]).any():
        raise SystemExit("Duplicate dataset/seed/method rows in corrected raw file")
    if not data["implementation_version"].eq(VERSION).all():
        raise SystemExit("Implementation-version drift in corrected raw file")
    if set(data["dataset"])!=set(DATASETS):
        raise SystemExit("Corrected raw file does not contain exactly nine datasets")
    if set(map(int,data["seed"].unique()))!=set(DEV_SEEDS):
        raise SystemExit("Development seed set must be exactly 0-4")

    for dataset in DATASETS:
        for seed in DEV_SEEDS:
            group=data[
                (data["dataset"]==dataset)&
                (data["seed"].astype(int)==seed)
            ]
            if len(group)!=EXPECTED_METHODS_PER_JOB:
                raise SystemExit(
                    f"{dataset} seed={seed}: expected "
                    f"{EXPECTED_METHODS_PER_JOB} rows, got {len(group)}"
                )
            if group["method"].nunique()!=EXPECTED_METHODS_PER_JOB:
                raise SystemExit(
                    f"{dataset} seed={seed}: duplicate/missing method names"
                )

    fixed={
        "n_qubits":4,
        "latent_dim":16,
        "spectral_radius":2.0,
        "input_scale":1.1,
        "max_samples_per_class":80,
    }
    for key,value in fixed.items():
        vals=set(data[key].dropna().tolist())
        if vals!={value}:
            raise SystemExit(f"Development configuration drift for {key}: {vals}")


def selected_mapping(data:pd.DataFrame):
    qnn=data[data["readout"]=="qnn"].copy()
    rows=[]
    for dataset in DATASETS:
        group=qnn[qnn["dataset"]==dataset]
        scores=group.groupby("method",sort=True)["balanced_accuracy"].mean()
        for required in ("pca_qnn","implicit_zero_qnn"):
            if required not in scores:
                raise SystemExit(f"{dataset}: missing {required}")
        candidates=scores.drop(index=["pca_qnn","implicit_zero_qnn"])
        if candidates.empty:
            raise SystemExit(f"{dataset}: no MRBI QNN candidates")

        # Match the already documented development selection exactly:
        # pandas Series.idxmax() on the grouped five-seed means.  Do not
        # manufacture a tolerance-based tie because near-equal floating-point
        # values can round identically in CSV output without being exact ties.
        best_method=str(candidates.idxmax())
        best_score=float(candidates.loc[best_method])
        exact_max=float(candidates.max())
        tied=sorted(
            str(method) for method,value in candidates.items()
            if float(value)==exact_max
        )
        zero=float(scores["implicit_zero_qnn"])
        pca=float(scores["pca_qnn"])
        rows.append({
            "dataset":dataset,
            "selected_method":best_method,
            "development_mean_ba":best_score,
            "development_zero_ba":zero,
            "development_pca_ba":pca,
            "development_delta_zero":best_score-zero,
            "n_development_seeds":5,
            "tie_count_at_exact_max":len(tied),
            "tie_methods":tied,
        })
    return rows


def validate_existing_selected(path:Path,rows):
    if not path.exists():
        return None
    selected=pd.read_csv(path)
    required={"dataset","best_method","n_seeds"}
    if not required.issubset(selected.columns):
        raise SystemExit(
            f"Existing selected_profiles.csv missing {sorted(required-set(selected.columns))}"
        )
    got={
        str(r.dataset):str(r.best_method)
        for r in selected.itertuples(index=False)
    }
    expected={r["dataset"]:r["selected_method"] for r in rows}
    if got!=expected:
        raise SystemExit(
            "Existing selected_profiles.csv disagrees with deterministic "
            "recomputation from main_raw.csv"
        )
    if not selected["n_seeds"].eq(5).all():
        raise SystemExit("selected_profiles.csv must use five development seeds")
    return sha256(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,default=DEFAULT_BASE)
    parser.add_argument(
        "--out",type=Path,
        default=ROOT/"outputs"/"selected_profile_confirmation_lock_candidate.json",
    )
    args=parser.parse_args()

    base=args.base.resolve()
    raw=base/"main_raw.csv"
    selected=base/"selected_profiles.csv"
    if not raw.exists():
        raise SystemExit(f"Missing corrected campaign raw file: {raw}")

    data=pd.read_csv(raw)
    validate_raw(data)
    rows=selected_mapping(data)
    selected_hash=validate_existing_selected(selected,rows)

    payload={
        "lock_schema":"mrbi_selected_profile_confirmation_lock_v1",
        "development_evidence":{
            "implementation_version":VERSION,
            "datasets":list(DATASETS),
            "seeds":list(DEV_SEEDS),
            "n_jobs":45,
            "expected_methods_per_job":EXPECTED_METHODS_PER_JOB,
            "main_raw_sha256":sha256(raw),
            "selected_profiles_sha256":selected_hash,
        },
        "selection_rule":{
            "metric":"mean balanced_accuracy across development seeds",
            "candidate_surface":"QNN MRBI methods excluding pca_qnn and implicit_zero_qnn",
            "per_dataset":True,
            "tie_break":"pandas Series.idxmax() on grouped five-seed means, matching summarize_continuation.py",
        },
        "selected_profiles":rows,
        "confirmation_plan":{
            "seeds":list(CONFIRMATION_SEEDS),
            "datasets":list(DATASETS),
            "n_dataset_seed_pairs":90,
            "qnn_epochs":60,
            "n_qubits":4,
            "latent_dim":16,
            "spectral_radius":2.0,
            "input_scale":1.1,
            "max_samples_per_class":80,
            "primary_comparison":"selected MRBI-QNN minus Zero-QNN",
            "secondary_references":[
                "PCA-QNN",
                "random-5 multistart QNN with identical readout initialization",
            ],
            "no_confirmation_result_dependent_reselection":True,
        },
        "provenance_note":(
            "The selected profile map is derived only from the completed "
            "corrected-continuation development campaign on seeds 0-4. "
            "Commit this JSON before running any confirmation seed 40-49."
        ),
    }

    args.out.parent.mkdir(parents=True,exist_ok=True)
    raw_json=json.dumps(payload,indent=2,sort_keys=True)+"\n"
    args.out.write_text(raw_json,encoding="utf-8")
    digest=hashlib.sha256(raw_json.encode("utf-8")).hexdigest()

    print("SELECTED_PROFILE_LOCK_CANDIDATE_OK",flush=True)
    print("LOCK_CANDIDATE",args.out,flush=True)
    print("LOCK_CANDIDATE_SHA256",digest,flush=True)
    for row in rows:
        print(
            f"{row['dataset']}: {row['selected_method']} "
            f"dev_BA={row['development_mean_ba']:.6f} "
            f"delta_zero={row['development_delta_zero']:+.6f}",
            flush=True,
        )


if __name__=="__main__":
    main()
