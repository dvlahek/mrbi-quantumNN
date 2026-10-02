"""Run fixed-profile classical readouts from the rho=2.0 feature cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
sys.path.insert(0,str(ROOT/"scripts"))

import main_qnn_benchmark as bench
import run_locked_selected_profile_confirmation as confirm

PLAN="fixed_profile_classical_readouts_v1"
CACHE_BASE=ROOT/"outputs"/"fixed_profile_rho_sensitivity_v1"/"rho_2p00"
DEFAULT_OUT=ROOT/"outputs"/PLAN
READOUTS=("logreg","svm_rbf","mlp","rf","gb")


def git_commit():
    return subprocess.run(
        ["git","rev-parse","HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def file_sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def run_readout(name,Xtr,ytr,Xte,yte,seed):
    return bench.READOUTS[name](Xtr,ytr,Xte,yte,seed)


def run_dataset(dataset,seed,lock):
    cache=CACHE_BASE/f"seed{seed}"/"features"/f"{dataset}.npz"
    if not cache.exists():
        raise SystemExit(
            f"Missing rho=2.0 feature cache: {cache}. "
            "Run the spectral-radius anchor first."
        )
    with np.load(cache,allow_pickle=False) as data:
        arrays={k:np.asarray(data[k]) for k in data.files}

    required={
        "Xtr_pca","Xte_pca","Ztr_zero","Zte_zero",
        "Ztr_selected","Zte_selected","ytr","yte",
    }
    if set(arrays)!=required:
        raise SystemExit(
            f"Unexpected feature cache schema for {dataset}: "
            f"{sorted(arrays)}"
        )

    ytr=arrays["ytr"].astype(int)
    yte=arrays["yte"].astype(int)
    reps={
        "pca":(arrays["Xtr_pca"],arrays["Xte_pca"]),
        "zero":(arrays["Ztr_zero"],arrays["Zte_zero"]),
        "selected_mrbi":(
            arrays["Ztr_selected"],arrays["Zte_selected"]
        ),
    }

    selected=next(
        r for r in lock["selected_profiles"] if r["dataset"]==dataset
    )

    out={}
    for readout in READOUTS:
        out[readout]={}
        for rep,(Xtr,Xte) in reps.items():
            out[readout][rep]=run_readout(
                readout,Xtr,ytr,Xte,yte,seed
            )
        z=float(out[readout]["zero"]["balanced_accuracy"])
        m=float(out[readout]["selected_mrbi"]["balanced_accuracy"])
        out[readout]["delta_selected_zero"]=m-z

    return {
        "dataset":dataset,
        "seed":seed,
        "selected_method":selected["method"],
        "feature_cache":str(cache.relative_to(ROOT)),
        "feature_cache_sha256":file_sha256(cache),
        "readouts":out,
    }


def main():
    parser=__import__("argparse").ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run",action="store_true")
    mode.add_argument("--run",action="store_true")
    parser.add_argument("--seed",type=int,required=True)
    args=parser.parse_args()

    lock=confirm.load_lock()
    if args.seed not in lock["confirmation_plan"]["seeds"]:
        raise SystemExit("Seed is not in confirmation seeds 40-49")

    if args.dry_run:
        missing=[]
        for dataset in lock["development_evidence"]["datasets"]:
            p=CACHE_BASE/f"seed{args.seed}"/"features"/f"{dataset}.npz"
            if not p.exists():
                missing.append(dataset)
        print(
            "FIXED_PROFILE_CLASSICAL_DRY_RUN_OK "
            f"seed={args.seed} readouts={','.join(READOUTS)} "
            f"missing_anchor_caches={len(missing)} "
            "profile_reselection=false hyperparameter_tuning=false",
            flush=True,
        )
        if missing:
            print("MISSING_CACHES",",".join(missing),flush=True)
        return

    rows=[]
    for dataset in lock["development_evidence"]["datasets"]:
        print(
            f"CLASSICAL_START seed={args.seed} dataset={dataset}",
            flush=True,
        )
        r=run_dataset(dataset,args.seed,lock)
        rows.append(r)
        print(
            "CLASSICAL_DONE "
            f"dataset={dataset} "
            +" ".join(
                f"{name}={r['readouts'][name]['delta_selected_zero']:+.4f}"
                for name in READOUTS
            ),
            flush=True,
        )

    payload={
        "plan":PLAN,
        "git_commit":git_commit(),
        "seed":args.seed,
        "rho":2.0,
        "readouts":list(READOUTS),
        "datasets":lock["development_evidence"]["datasets"],
        "no_profile_reselection":True,
        "no_readout_hyperparameter_tuning":True,
        "dataset_results":rows,
        "analysis_role":"secondary descriptive readout check",
    }
    out_dir=DEFAULT_OUT/f"seed{args.seed}"
    out_dir.mkdir(parents=True,exist_ok=True)
    path=out_dir/"result.json"
    confirm.atomic_write(
        path,json.dumps(payload,indent=2,allow_nan=False)+"\n"
    )
    print(f"FIXED_PROFILE_CLASSICAL_OK {path}",flush=True)


if __name__=="__main__":
    main()
