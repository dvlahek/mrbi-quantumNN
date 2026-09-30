"""Locked fixed-policy MRBI-QNN benchmark for manuscript confirmation.

This experiment is deliberately frozen before seeds 20-29 are inspected.

Primary comparison:
    fixed MRBI = full_balanced + aggressive hybrid
    vs zero initialization
Secondary controls:
    five random starts + zero candidate
    direct PCA-QNN

All implicit readouts within a dataset/seed use the same QNN architecture,
identical initial state, optimizer settings, and minibatch order. Random
streams use explicit integer offsets; Python hash() is never used.

The benchmark uses the nine manuscript tasks, new split seeds 20-29, and
rho(W)=2.25 as a predeclared hard-regime stress test. It does not select a
profile, cap, dataset, seed, or classifier according to these outcomes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import main_qnn_benchmark as bench
import mrbi

PLAN="locked_fixed_mrbi_full_balanced_aggressive_rho225_v1"
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
FROZEN_SEEDS=tuple(range(20,30))
SPECTRAL_RADIUS=2.25
MAX_PER_CLASS=80
N_RANDOM_STARTS=5
RANDOM_START_RADIUS=1.0
QNN_EPOCHS=50
QNN_LR=0.01
QNN_BATCH=16
N_QUBITS=4
LATENT_DIM=16
PROFILE_NAME="full_balanced"
HYBRID_NAME="aggressive"
DEFAULT_OUT=ROOT/"outputs"/PLAN


def atomic_write(path:Path,text:str):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(text,encoding="utf-8")
    tmp.replace(path)


def git_commit():
    return subprocess.run(
        ["git","rev-parse","HEAD"],cwd=ROOT,
        capture_output=True,text=True,check=True,
    ).stdout.strip()


def make_cfg(seed:int):
    cfg=bench.ExperimentConfig(
        seed=seed,
        test_size=0.30,
        n_qubits=N_QUBITS,
        pca_dim=N_QUBITS,
        latent_dim=LATENT_DIM,
        max_samples_per_class=MAX_PER_CLASS,
        spectral_radius=SPECTRAL_RADIUS,
        input_scale=1.10,
        bias_scale=0.10,
        hard_layer=True,
        run_qnn=True,
        qnn_epochs=QNN_EPOCHS,
        qnn_lr=QNN_LR,
        qnn_batch_size=QNN_BATCH,
        qnn_layers=2,
    )
    profile=next(
        p for p in bench.get_mrbi_profiles("hard_article")
        if p.name==PROFILE_NAME
    )
    hybrid=next(
        p for p in bench.get_hybrid_profiles("hard_article")
        if p.name==HYBRID_NAME
    )
    cfg=bench.apply_mrbi_profile(cfg,profile)
    cfg=bench.apply_hybrid_profile(cfg,hybrid)
    if cfg.profile_name!=PROFILE_NAME or cfg.hybrid_name!=HYBRID_NAME:
        raise RuntimeError("Frozen MRBI policy drift")
    return cfg


def state_checksum(state):
    h=hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode("utf-8"))
        h.update(state[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def choose_multistart(zero,other_results):
    candidates=[zero]+list(other_results)
    successful=[
        r for r in candidates
        if r.success and np.isfinite(r.residual)
        and np.all(np.isfinite(r.z_star))
    ]
    pool=successful if successful else [
        r for r in candidates
        if np.isfinite(r.residual) and np.all(np.isfinite(r.z_star))
    ]
    if not pool:
        return zero
    return min(
        pool,
        key=lambda r:(
            float(r.residual),
            int(r.nfev) if int(r.nfev)>=0 else 10**9,
        ),
    )


def solve_paired_features(X,layer,cfg,mrbi_seed:int,multistart_seed:int):
    """Solve zero, fixed MRBI, and 5-start multistart on identical samples."""
    F,J=mrbi.make_residual_and_jacobian(layer)
    root_cfg=bench.make_root_cfg(cfg)
    mrbi_cfg=bench.make_mrbi_cfg(cfg)
    hybrid_cfg=bench.make_hybrid_cfg(cfg)
    rng_mrbi=np.random.default_rng(mrbi_seed)
    rng_ms=np.random.default_rng(multistart_seed)

    Z={"zero":[],"mrbi":[],"multistart":[]}
    rows=[]
    for position,x in enumerate(np.asarray(X,dtype=np.float64)):
        hybrid=mrbi.hybrid_mrbi_solve_tanh_layer(
            layer,x,
            mrbi_cfg=mrbi_cfg,
            hybrid_cfg=hybrid_cfg,
            root_cfg=root_cfg,
            rng=rng_mrbi,
        )
        zero=hybrid.zero_result
        fixed=hybrid.final_result

        random_results=[]
        random_runtime=0.0
        for _ in range(N_RANDOM_STARTS):
            z0=rng_ms.uniform(
                -RANDOM_START_RADIUS,RANDOM_START_RADIUS,size=layer.d
            )
            result=mrbi.solve_root(F,J,x,z0,cfg=root_cfg)
            random_results.append(result)
            random_runtime+=float(result.runtime_sec)
        multi=choose_multistart(zero,random_results)

        Z["zero"].append(np.asarray(zero.z_star,dtype=np.float64))
        Z["mrbi"].append(np.asarray(fixed.z_star,dtype=np.float64))
        Z["multistart"].append(np.asarray(multi.z_star,dtype=np.float64))

        rows.append({
            "position":position,
            "zero_success":bool(zero.success),
            "mrbi_success":bool(fixed.success),
            "multistart_success":bool(multi.success),
            "mrbi_triggered":bool(hybrid.trigger_reason!="none"),
            "mrbi_used":bool(hybrid.used_mrbi),
            "mrbi_rescue":bool((not zero.success) and fixed.success),
            "multistart_rescue":bool((not zero.success) and multi.success),
            "zero_residual":float(zero.residual),
            "mrbi_residual":float(fixed.residual),
            "multistart_residual":float(multi.residual),
            "zero_nfev":int(zero.nfev),
            "mrbi_final_nfev":int(fixed.nfev),
            "multistart_total_nfev":int(max(0,zero.nfev)+sum(
                max(0,r.nfev) for r in random_results
            )),
            "mrbi_runtime_sec":float(hybrid.total_runtime_sec or 0.0),
            "multistart_runtime_sec":float(zero.runtime_sec+random_runtime),
            "mrbi_candidate_objective_calls":(
                None if hybrid.mrbi_candidate is None
                else int(hybrid.mrbi_candidate.n_objective_calls)
            ),
            "mrbi_trigger_reason":str(hybrid.trigger_reason),
        })

    arrays={key:np.asarray(value,dtype=np.float64) for key,value in Z.items()}
    stats={}
    for method in ("zero","mrbi","multistart"):
        success=np.asarray([r[f"{method}_success"] for r in rows],dtype=bool)
        residual=np.asarray([r[f"{method}_residual"] for r in rows],dtype=float)
        stats[f"{method}_success_rate"]=float(success.mean())
        stats[f"{method}_mean_residual"]=float(residual.mean())
        stats[f"{method}_median_residual"]=float(np.median(residual))
    stats.update({
        "mrbi_rescues":int(sum(r["mrbi_rescue"] for r in rows)),
        "multistart_rescues":int(sum(r["multistart_rescue"] for r in rows)),
        "mrbi_trigger_rate":float(np.mean([r["mrbi_triggered"] for r in rows])),
        "mrbi_use_rate":float(np.mean([r["mrbi_used"] for r in rows])),
        "mrbi_mean_runtime_sec":float(np.mean([r["mrbi_runtime_sec"] for r in rows])),
        "multistart_mean_runtime_sec":float(np.mean([
            r["multistart_runtime_sec"] for r in rows
        ])),
        "multistart_mean_total_nfev":float(np.mean([
            r["multistart_total_nfev"] for r in rows
        ])),
    })
    objective=[
        r["mrbi_candidate_objective_calls"] for r in rows
        if r["mrbi_candidate_objective_calls"] is not None
    ]
    stats["mrbi_mean_candidate_objective_calls"]=(
        float(np.mean(objective)) if objective else 0.0
    )
    return arrays,stats,rows


def standardize_from_train(train,test):
    scaler=StandardScaler()
    return (
        scaler.fit_transform(train).astype(np.float64),
        scaler.transform(test).astype(np.float64),
    )


def train_qnn(Xtr,ytr,Xte,yte,cfg,seed:int,initial_state=None):
    if not bench.HAS_QNN:
        raise RuntimeError(f"QNN unavailable: {bench.QNN_IMPORT_ERROR}")
    torch=bench.torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    model=bench.TorchQNN(
        Xtr.shape[1],cfg.n_qubits,cfg.qnn_layers,seed
    )
    if initial_state is not None:
        model.load_state_dict(initial_state,strict=True)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    checksum=state_checksum(initial)

    xt=torch.as_tensor(Xtr,dtype=torch.float32)
    yt=torch.as_tensor(ytr,dtype=torch.long)
    xv=torch.as_tensor(Xte,dtype=torch.float32)
    opt=torch.optim.Adam(model.parameters(),lr=cfg.qnn_lr)
    loss_fn=torch.nn.CrossEntropyLoss()
    rng=np.random.default_rng(seed)
    t0=time.perf_counter()
    for _ in range(cfg.qnn_epochs):
        order=rng.permutation(len(xt))
        for start in range(0,len(order),cfg.qnn_batch_size):
            ids=order[start:start+cfg.qnn_batch_size]
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(xt[ids]),yt[ids])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite QNN loss")
            loss.backward()
            opt.step()
    runtime=time.perf_counter()-t0
    with torch.no_grad():
        logits=model(xv)
        probabilities=torch.softmax(logits,dim=1).cpu().numpy()
    predictions=np.argmax(probabilities,axis=1)
    metrics=bench.compute_metrics(yte,predictions,probabilities[:,1])
    return {
        **metrics,
        "train_time_sec":float(runtime),
        "predictions":predictions.tolist(),
        "probability_class1":probabilities[:,1].tolist(),
        "confusion_matrix":confusion_matrix(
            yte,predictions,labels=[0,1]
        ).tolist(),
        "initial_state_sha256":checksum,
    },initial


def run_dataset(dataset:str,seed:int,dataset_index:int):
    cfg=make_cfg(seed)
    X,y=bench.load_dataset(dataset,cfg)
    train_ids,test_ids=train_test_split(
        np.arange(len(y)),
        test_size=cfg.test_size,
        stratify=y,
        random_state=seed,
    )
    Xtr_raw,Xte_raw=X[train_ids],X[test_ids]
    ytr,yte=np.asarray(y)[train_ids],np.asarray(y)[test_ids]
    Xtr_pca,Xte_pca=bench.preprocess_to_pca(
        Xtr_raw,Xte_raw,cfg.pca_dim,seed
    )
    layer_seed=1000+seed+cfg.n_qubits
    layer=bench.make_random_implicit_layer(
        Xtr_pca.shape[1],cfg.latent_dim,cfg,layer_seed
    )

    stream_base=100000+1000*seed+20*dataset_index
    train_features,train_stats,train_audit=solve_paired_features(
        Xtr_pca,layer,cfg,
        mrbi_seed=stream_base+1,
        multistart_seed=stream_base+2,
    )
    test_features,test_stats,test_audit=solve_paired_features(
        Xte_pca,layer,cfg,
        mrbi_seed=stream_base+3,
        multistart_seed=stream_base+4,
    )

    prepared={}
    for method in ("zero","mrbi","multistart"):
        prepared[method]=standardize_from_train(
            train_features[method],test_features[method]
        )

    qnn_seed=70000+seed
    implicit_base=bench.TorchQNN(
        LATENT_DIM,N_QUBITS,cfg.qnn_layers,qnn_seed
    )
    implicit_state={
        k:v.detach().clone()
        for k,v in implicit_base.state_dict().items()
    }
    implicit_sha=state_checksum(implicit_state)

    results={}
    for method in ("zero","mrbi","multistart"):
        Xtr_method,Xte_method=prepared[method]
        metrics,initial=train_qnn(
            Xtr_method,ytr,Xte_method,yte,cfg,qnn_seed,
            initial_state=implicit_state,
        )
        if metrics["initial_state_sha256"]!=implicit_sha:
            raise RuntimeError("Implicit QNN initialization mismatch")
        results[method]=metrics

    pca_metrics,_=train_qnn(
        Xtr_pca,ytr,Xte_pca,yte,cfg,qnn_seed,
        initial_state=None,
    )
    results["pca"]=pca_metrics

    z=results["zero"]["balanced_accuracy"]
    m=results["mrbi"]["balanced_accuracy"]
    s=results["multistart"]["balanced_accuracy"]
    p=results["pca"]["balanced_accuracy"]
    paired={
        "delta_mrbi_zero":float(m-z),
        "delta_multistart_zero":float(s-z),
        "delta_mrbi_multistart":float(m-s),
        "delta_mrbi_pca":float(m-p),
    }

    for i,row in enumerate(test_audit):
        row["true_label"]=int(yte[i])
        for method in ("zero","mrbi","multistart","pca"):
            row[f"{method}_prediction"]=int(results[method]["predictions"][i])
        row["mrbi_vs_zero_prediction_changed"]=bool(
            row["mrbi_prediction"]!=row["zero_prediction"]
        )

    return {
        "dataset":dataset,
        "seed":seed,
        "spectral_radius":SPECTRAL_RADIUS,
        "n_train":int(len(ytr)),
        "n_test":int(len(yte)),
        "class_counts_train":np.bincount(ytr,minlength=2).tolist(),
        "class_counts_test":np.bincount(yte,minlength=2).tolist(),
        "layer_seed":layer_seed,
        "qnn_seed":qnn_seed,
        "implicit_initial_state_sha256":implicit_sha,
        "results":results,
        "paired":paired,
        "train_solver_stats":train_stats,
        "test_solver_stats":test_stats,
        "test_sample_audit":test_audit,
    }


def summarize(dataset_results):
    keys=(
        "delta_mrbi_zero",
        "delta_multistart_zero",
        "delta_mrbi_multistart",
        "delta_mrbi_pca",
    )
    out={}
    for key in keys:
        values=np.asarray([d["paired"][key] for d in dataset_results],dtype=float)
        out[key]={
            "mean":float(values.mean()),
            "median":float(np.median(values)),
            "positive":int(np.sum(values>1e-12)),
            "neutral":int(np.sum(np.abs(values)<=1e-12)),
            "negative":int(np.sum(values< -1e-12)),
        }
    for method in ("pca","zero","mrbi","multistart"):
        values=np.asarray([
            d["results"][method]["balanced_accuracy"]
            for d in dataset_results
        ],dtype=float)
        out[f"{method}_balanced_accuracy_mean"]=float(values.mean())
    out["test_zero_success_rate_mean"]=float(np.mean([
        d["test_solver_stats"]["zero_success_rate"] for d in dataset_results
    ]))
    out["test_mrbi_success_rate_mean"]=float(np.mean([
        d["test_solver_stats"]["mrbi_success_rate"] for d in dataset_results
    ]))
    out["test_multistart_success_rate_mean"]=float(np.mean([
        d["test_solver_stats"]["multistart_success_rate"] for d in dataset_results
    ]))
    out["test_mrbi_rescues_total"]=int(sum(
        d["test_solver_stats"]["mrbi_rescues"] for d in dataset_results
    ))
    out["test_multistart_rescues_total"]=int(sum(
        d["test_solver_stats"]["multistart_rescues"] for d in dataset_results
    ))
    return out


def dry_run(seed:int):
    cfg=make_cfg(seed)
    assert seed in FROZEN_SEEDS
    assert tuple(DATASETS)==DATASETS and len(DATASETS)==9
    assert cfg.profile_name==PROFILE_NAME=="full_balanced"
    assert cfg.hybrid_name==HYBRID_NAME=="aggressive"
    assert cfg.spectral_radius==SPECTRAL_RADIUS==2.25
    assert cfg.qnn_epochs==50 and cfg.qnn_lr==0.01
    assert cfg.max_samples_per_class==80
    assert N_RANDOM_STARTS==5 and RANDOM_START_RADIUS==1.0
    print(
        "LOCKED_FIXED_MRBI_DRY_RUN_OK "
        f"seed={seed} datasets=9 rho=2.25 profile=full_balanced "
        "hybrid=aggressive multistart=random5+zero qnn_epochs=50 "
        "qnn_lr=0.01 no_data_loaded",
        flush=True,
    )


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run",action="store_true")
    mode.add_argument("--run",action="store_true")
    parser.add_argument("--seed",type=int,choices=list(FROZEN_SEEDS),required=True)
    parser.add_argument("--out-dir",type=Path,default=None)
    args=parser.parse_args()

    if args.dry_run:
        dry_run(args.seed)
        return

    if not bench.HAS_QNN:
        raise SystemExit(
            "Locked benchmark requires torch and PennyLane in the full QNN venv"
        )
    commit=git_commit()
    out_dir=(
        args.out_dir.resolve()
        if args.out_dir is not None
        else (DEFAULT_OUT/f"seed{args.seed}").resolve()
    )
    out_dir.mkdir(parents=True,exist_ok=True)
    manifest={
        "plan":PLAN,
        "git_commit":commit,
        "seed":args.seed,
        "frozen_seeds":list(FROZEN_SEEDS),
        "datasets":list(DATASETS),
        "primary_method":{
            "profile":PROFILE_NAME,
            "hybrid":HYBRID_NAME,
            "spectral_radius":SPECTRAL_RADIUS,
        },
        "controls":{
            "zero":True,
            "random_multistart":{
                "n_random_starts":N_RANDOM_STARTS,
                "includes_zero_candidate":True,
                "radius":RANDOM_START_RADIUS,
            },
            "pca_qnn":True,
        },
        "qnn":{
            "epochs":QNN_EPOCHS,
            "lr":QNN_LR,
            "batch_size":QNN_BATCH,
            "n_qubits":N_QUBITS,
            "latent_dim":LATENT_DIM,
            "implicit_methods_share_exact_initial_state":True,
            "implicit_methods_share_minibatch_order":True,
        },
        "selection_rule":"No result-dependent dataset/profile/seed selection",
        "scope":"new splits of the nine manuscript tasks; hard-regime confirmation, not external-dataset validation",
    }
    manifest_path=out_dir/"manifest.json"
    manifest_raw=json.dumps(manifest,indent=2,sort_keys=True)+"\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8")!=manifest_raw:
        raise SystemExit("Existing manifest differs; use a new --out-dir")
    atomic_write(manifest_path,manifest_raw)

    dataset_results=[]
    checkpoint=out_dir/"checkpoint.json"
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("git_commit")!=commit or saved.get("seed")!=args.seed:
            raise SystemExit("Checkpoint provenance mismatch")
        dataset_results=saved.get("datasets",[])
    completed={d["dataset"] for d in dataset_results}

    for index,dataset in enumerate(DATASETS):
        if dataset in completed:
            print(f"SKIP completed dataset={dataset}",flush=True)
            continue
        print(
            f"LOCKED_START seed={args.seed} dataset={dataset} "
            f"rho={SPECTRAL_RADIUS}",
            flush=True,
        )
        result=run_dataset(dataset,args.seed,index)
        dataset_results.append(result)
        payload={
            "git_commit":commit,
            "seed":args.seed,
            "datasets":dataset_results,
        }
        atomic_write(
            checkpoint,
            json.dumps(payload,indent=2,allow_nan=False)+"\n",
        )
        p=result["paired"]
        print(
            f"LOCKED_DONE dataset={dataset} "
            f"zero={result['results']['zero']['balanced_accuracy']:.4f} "
            f"mrbi={result['results']['mrbi']['balanced_accuracy']:.4f} "
            f"ms5={result['results']['multistart']['balanced_accuracy']:.4f} "
            f"pca={result['results']['pca']['balanced_accuracy']:.4f} "
            f"d_mz={p['delta_mrbi_zero']:+.4f} "
            f"d_mms={p['delta_mrbi_multistart']:+.4f}",
            flush=True,
        )

    final={
        **manifest,
        "dataset_results":dataset_results,
        "seed_summary":summarize(dataset_results),
    }
    if len(dataset_results)!=len(DATASETS):
        raise RuntimeError("Incomplete locked dataset set")
    path=out_dir/"locked_result.json"
    raw=json.dumps(final,indent=2,allow_nan=False)+"\n"
    if path.exists() and path.read_text(encoding="utf-8")!=raw:
        raise SystemExit("Existing final result differs; use a new --out-dir")
    atomic_write(path,raw)
    print(f"LOCKED_FIXED_MRBI_OK {path}",flush=True)


if __name__=="__main__":
    main()
