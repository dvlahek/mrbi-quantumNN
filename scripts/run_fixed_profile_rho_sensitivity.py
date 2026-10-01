"""Run the fixed-profile spectral-radius sensitivity protocol."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
sys.path.insert(0,str(ROOT/"scripts"))

import main_qnn_benchmark as bench
import run_locked_selected_profile_confirmation as confirm
from fixed_profile_robustness_common import (
    ANCHOR_RHO,
    RHO_VALUES,
    layer_at_rho,
    make_base_layer,
    prepare_dataset,
    rho_tag,
    selected_success_rate,
)

PLAN="fixed_profile_rho_sensitivity_v1"
DEFAULT_OUT=ROOT/"outputs"/PLAN
DEFAULT_CONFIRM=ROOT/"outputs"/"locked_selected_profile_confirmation_v1"
ANCHOR_TOL=1e-12


def git_commit():
    return subprocess.run(
        ["git","rev-parse","HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def array_sha256(arr):
    x=np.ascontiguousarray(np.asarray(arr))
    h=hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(str(x.shape).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def _metric_close(a,b,tol=ANCHOR_TOL):
    a=float(a)
    b=float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isclose(a,b,atol=tol,rtol=0.0))


def load_confirmation_reference(base,seed,dataset):
    path=base/f"seed{seed}"/"locked_result.json"
    if not path.exists():
        raise SystemExit(
            f"Missing confirmation anchor result: {path}. "
            "rho=2.0 sensitivity must be checked against the existing confirmation."
        )
    payload=json.loads(path.read_text(encoding="utf-8"))
    matches=[r for r in payload.get("dataset_results",[]) if r["dataset"]==dataset]
    if len(matches)!=1:
        raise SystemExit(f"Expected one confirmation row for {dataset} seed={seed}")
    return matches[0],path


def verify_anchor(result,reference):
    if result["selected_method"]!=reference["selected_method"]:
        raise AssertionError("Selected method differs from confirmation")
    if result["base_layer_seed"]!=reference["layer_seed"]:
        raise AssertionError("Layer seed differs from confirmation")
    if result["qnn_seed"]!=reference["qnn_seed"]:
        raise AssertionError("QNN seed differs from confirmation")

    diffs={}
    for rep in ("zero","selected_mrbi"):
        for key in ("accuracy","balanced_accuracy","f1","roc_auc"):
            a=result["results"][rep][key]
            b=reference["results"][rep][key]
            if not _metric_close(a,b):
                diffs[f"{rep}.{key}"]=[a,b]

    z_a=float(result["solver"]["zero_test"]["zero_success_rate"])
    z_b=float(reference["solver"]["zero_test"]["zero_success_rate"])
    if not _metric_close(z_a,z_b):
        diffs["zero_success_rate"]=[z_a,z_b]

    s_a=selected_success_rate(
        result["selected_kind"],result["solver"]["selected_test"]
    )
    ref_stats=reference["solver"]["selected_test"]
    s_b=selected_success_rate(reference["selected_kind"],ref_stats)
    if not _metric_close(s_a,s_b):
        diffs["selected_success_rate"]=[s_a,s_b]

    if diffs:
        raise AssertionError(
            "rho=2.0 regression anchor differs from confirmation: "
            +json.dumps(diffs,sort_keys=True)
        )
    return {
        "passed":True,
        "tolerance":ANCHOR_TOL,
        "metrics_checked":[
            "accuracy","balanced_accuracy","f1","roc_auc",
            "zero_success_rate","selected_success_rate",
        ],
    }


def save_anchor_cache(path,Xtr_p,Xte_p,Ztr_zero,Zte_zero,Ztr_sel,Zte_sel,ytr,yte):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            Xtr_pca=np.asarray(Xtr_p,dtype=np.float64),
            Xte_pca=np.asarray(Xte_p,dtype=np.float64),
            Ztr_zero=np.asarray(Ztr_zero,dtype=np.float64),
            Zte_zero=np.asarray(Zte_zero,dtype=np.float64),
            Ztr_selected=np.asarray(Ztr_sel,dtype=np.float64),
            Zte_selected=np.asarray(Zte_sel,dtype=np.float64),
            ytr=np.asarray(ytr,dtype=np.int64),
            yte=np.asarray(yte,dtype=np.int64),
        )
    tmp.replace(path)


def run_dataset(dataset,seed,rho,lock,confirm_base,out_dir):
    cfg,Xtr_p,Xte_p,ytr,yte=prepare_dataset(dataset,seed,lock)
    row,cfg_sel=confirm.selected_cfg(dataset,replace(cfg,spectral_radius=rho),lock)

    base_layer,base_seed=make_base_layer(Xtr_p,cfg)
    layer,base_rho,actual_rho=layer_at_rho(base_layer,rho)

    t0=time.perf_counter()
    Ztr_zero,st_tr_zero=bench.solve_zero_features(Xtr_p,layer,cfg)
    Zte_zero,st_te_zero=bench.solve_zero_features(Xte_p,layer,cfg)
    zero_time=float(time.perf_counter()-t0)

    t0=time.perf_counter()
    Ztr_sel,Zte_sel,st_tr_sel,st_te_sel=confirm.solve_selected(
        Xtr_p,Xte_p,layer,cfg_sel,row,
    )
    selected_time=float(time.perf_counter()-t0)

    Ztr_zero,Zte_zero=bench.standardize_pair(Ztr_zero,Zte_zero)
    Ztr_sel,Zte_sel=bench.standardize_pair(Ztr_sel,Zte_sel)

    results={
        "zero":confirm.qnn_metrics(Ztr_zero,ytr,Zte_zero,yte,cfg),
        "selected_mrbi":confirm.qnn_metrics(Ztr_sel,ytr,Zte_sel,yte,cfg),
    }
    z=float(results["zero"]["balanced_accuracy"])
    m=float(results["selected_mrbi"]["balanced_accuracy"])

    result={
        "dataset":dataset,
        "seed":seed,
        "rho_target":float(rho),
        "rho_base_actual":float(base_rho),
        "rho_actual":float(actual_rho),
        "base_layer_seed":int(base_seed),
        "qnn_seed":int(cfg.seed+777),
        "selected_method":row["method"],
        "selected_kind":row["kind"],
        "selected_profile":row["profile"],
        "selected_hybrid":row["hybrid"],
        "operator_sha256":{
            "W_base":array_sha256(base_layer.W),
            "U":array_sha256(base_layer.U),
            "b":array_sha256(base_layer.b),
            "W_target":array_sha256(layer.W),
        },
        "results":results,
        "paired":{
            "delta_selected_zero":m-z,
        },
        "solver":{
            "zero_train":st_tr_zero,
            "zero_test":st_te_zero,
            "selected_train":st_tr_sel,
            "selected_test":st_te_sel,
            "feature_time_zero_sec":zero_time,
            "feature_time_selected_sec":selected_time,
        },
    }

    if float(rho)==ANCHOR_RHO:
        reference,ref_path=load_confirmation_reference(
            confirm_base,seed,dataset
        )
        result["anchor_regression"]=verify_anchor(result,reference)
        result["anchor_regression"]["reference"]=str(ref_path)

        cache_path=out_dir/"features"/f"{dataset}.npz"
        save_anchor_cache(
            cache_path,
            Xtr_p,Xte_p,
            Ztr_zero,Zte_zero,
            Ztr_sel,Zte_sel,
            ytr,yte,
        )
        result["feature_cache"]=str(cache_path.relative_to(ROOT))

    return result


def dry_run(seed,rho,lock):
    if seed not in lock["confirmation_plan"]["seeds"]:
        raise SystemExit("Seed outside fixed confirmation set")
    if float(rho) not in RHO_VALUES:
        raise SystemExit(f"rho must be one of {RHO_VALUES}")
    cfg=confirm.make_cfg(seed,lock)
    base_seed=cfg.seed+1000+cfg.latent_dim+int(100*ANCHOR_RHO)
    print(
        "FIXED_PROFILE_RHO_DRY_RUN_OK "
        f"seed={seed} rho={float(rho):.2f} datasets=9 "
        f"base_seed={base_seed} anchor_rho={ANCHOR_RHO:.2f} "
        "profile_reselection=false target_rho_in_seed=false",
        flush=True,
    )


def main():
    parser=__import__("argparse").ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run",action="store_true")
    mode.add_argument("--run",action="store_true")
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--rho",type=float,required=True)
    parser.add_argument(
        "--confirmation-base",
        type=Path,
        default=DEFAULT_CONFIRM,
    )
    args=parser.parse_args()

    lock=confirm.load_lock()
    rho=float(args.rho)
    if args.seed not in lock["confirmation_plan"]["seeds"]:
        raise SystemExit("Seed is not in confirmation seeds 40-49")
    if rho not in RHO_VALUES:
        raise SystemExit(f"rho must be one of {RHO_VALUES}")
    if args.dry_run:
        dry_run(args.seed,rho,lock)
        return
    if not bench.HAS_QNN:
        raise SystemExit(f"QNN dependency unavailable: {bench.QNN_IMPORT_ERROR}")

    commit=git_commit()
    out_dir=DEFAULT_OUT/f"rho_{rho_tag(rho)}"/f"seed{args.seed}"
    out_dir.mkdir(parents=True,exist_ok=True)

    checkpoint=out_dir/"checkpoint.json"
    rows=[]
    if checkpoint.exists():
        old=json.loads(checkpoint.read_text(encoding="utf-8"))
        if (
            old.get("git_commit")!=commit
            or int(old.get("seed",-1))!=args.seed
            or float(old.get("rho",-1))!=rho
        ):
            raise SystemExit("Checkpoint provenance mismatch")
        rows=old.get("dataset_results",[])
    completed={r["dataset"] for r in rows}

    for dataset in lock["development_evidence"]["datasets"]:
        if dataset in completed:
            print(f"SKIP dataset={dataset}",flush=True)
            continue
        print(
            f"RHO_START seed={args.seed} rho={rho:.2f} dataset={dataset}",
            flush=True,
        )
        result=run_dataset(
            dataset,args.seed,rho,lock,
            args.confirmation_base.resolve(),out_dir,
        )
        rows.append(result)
        confirm.atomic_write(
            checkpoint,
            json.dumps({
                "plan":PLAN,
                "git_commit":commit,
                "seed":args.seed,
                "rho":rho,
                "dataset_results":rows,
            },indent=2,allow_nan=False)+"\n",
        )
        print(
            f"RHO_DONE dataset={dataset} "
            f"zero={result['results']['zero']['balanced_accuracy']:.4f} "
            f"mrbi={result['results']['selected_mrbi']['balanced_accuracy']:.4f} "
            f"delta={result['paired']['delta_selected_zero']:+.4f}",
            flush=True,
        )

    if len(rows)!=9:
        raise RuntimeError("Incomplete nine-dataset sensitivity job")

    payload={
        "plan":PLAN,
        "git_commit":commit,
        "seed":args.seed,
        "rho":rho,
        "rhos_fixed":list(RHO_VALUES),
        "datasets":lock["development_evidence"]["datasets"],
        "selection_map":{
            r["dataset"]:r["method"] for r in lock["selected_profiles"]
        },
        "dataset_results":rows,
        "analysis_role":"secondary descriptive sensitivity",
    }
    path=out_dir/"result.json"
    confirm.atomic_write(
        path,json.dumps(payload,indent=2,allow_nan=False)+"\n"
    )
    print(f"FIXED_PROFILE_RHO_OK {path}",flush=True)


if __name__=="__main__":
    main()
