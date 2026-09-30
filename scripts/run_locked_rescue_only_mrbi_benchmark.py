"""Locked rescue-only MRBI-QNN benchmark.

This experiment is frozen before seeds 30-39 are inspected.

Primary method
--------------
Zero solve first. Only if the zero-initialized solve is not a strict success,
run one fixed MRBI candidate construction (full_balanced) and a final HYBR
solve. Accept MRBI only when that rescue solve is a strict success. Otherwise
keep the terminal state returned by the original zero solve.

Primary comparison:
    MRBI-rescue-only vs Zero
Secondary solver-aware control:
    random-5-rescue-only vs Zero and vs MRBI-rescue-only
Reference:
    direct PCA-QNN

For zero-success samples, the raw implicit feature vectors are exactly identical
for Zero, MRBI-rescue-only, and random-5-rescue-only. All implicit QNN readouts
within one dataset/seed use identical initial weights and minibatch order.
"""
from __future__ import annotations

import argparse
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

PLAN="locked_rescue_only_mrbi_full_balanced_rho225_v1"
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
FROZEN_SEEDS=tuple(range(30,40))
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
    cfg=bench.apply_mrbi_profile(cfg,profile)
    if cfg.profile_name!=PROFILE_NAME:
        raise RuntimeError("Frozen MRBI profile drift")
    return cfg


def state_checksum(state):
    h=hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode("utf-8"))
        h.update(state[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def best_successful(results):
    successful=[
        r for r in results
        if r.success and np.isfinite(r.residual)
        and np.all(np.isfinite(r.z_star))
    ]
    if not successful:
        return None
    return min(
        successful,
        key=lambda r:(
            float(r.residual),
            int(r.nfev) if int(r.nfev)>=0 else 10**9,
        ),
    )


def solve_rescue_features(
    X,layer,cfg,mrbi_seed:int,multistart_seed:int,
):
    """One shared zero solve, then rescue methods only on zero failure."""
    F,J=mrbi.make_residual_and_jacobian(layer)
    root_cfg=bench.make_root_cfg(cfg)
    mrbi_cfg=bench.make_mrbi_cfg(cfg)
    rng_mrbi=np.random.default_rng(mrbi_seed)
    rng_ms=np.random.default_rng(multistart_seed)
    zero_init=np.zeros(layer.d,dtype=np.float64)

    Z={"zero":[],"mrbi_rescue":[],"multistart_rescue":[]}
    rows=[]
    for position,x in enumerate(np.asarray(X,dtype=np.float64)):
        zero=mrbi.solve_root(F,J,x,zero_init,cfg=root_cfg)
        zero_time=float(zero.runtime_sec)

        mrbi_final=zero
        mrbi_attempted=False
        mrbi_successful_rescue=False
        mrbi_candidate_calls=0
        mrbi_extra_time=0.0

        ms_final=zero
        ms_attempted=False
        ms_successful_rescue=False
        ms_extra_time=0.0
        ms_total_nfev=max(0,int(zero.nfev))

        if not zero.success:
            # Fixed MRBI rescue: candidate construction + one final HYBR solve.
            mrbi_attempted=True
            t0=time.perf_counter()
            opt=mrbi.MRBIOptimizer(
                F=F,J=J,x=x,dim=layer.d,
                config=mrbi_cfg,rng=rng_mrbi,
            )
            candidate=opt.optimize()
            det=mrbi.solve_root(
                F,J,x,candidate.z_init,cfg=root_cfg,
            )
            mrbi_extra_time=float(time.perf_counter()-t0)
            mrbi_candidate_calls=int(candidate.n_objective_calls)
            if det.success:
                mrbi_final=det
                mrbi_successful_rescue=True

            # Budgeted random rescue: five random starts, only after zero failure.
            ms_attempted=True
            random_results=[]
            t0=time.perf_counter()
            for _ in range(N_RANDOM_STARTS):
                start=rng_ms.uniform(
                    -RANDOM_START_RADIUS,RANDOM_START_RADIUS,size=layer.d
                )
                result=mrbi.solve_root(F,J,x,start,cfg=root_cfg)
                random_results.append(result)
                ms_total_nfev+=max(0,int(result.nfev))
            ms_extra_time=float(time.perf_counter()-t0)
            best=best_successful(random_results)
            if best is not None:
                ms_final=best
                ms_successful_rescue=True

        # Raw-feature identity is mandatory whenever Zero succeeded.
        if zero.success:
            if not np.array_equal(zero.z_star,mrbi_final.z_star):
                raise RuntimeError("MRBI rescue changed a zero-success feature")
            if not np.array_equal(zero.z_star,ms_final.z_star):
                raise RuntimeError("Multistart rescue changed a zero-success feature")

        Z["zero"].append(np.asarray(zero.z_star,dtype=np.float64))
        Z["mrbi_rescue"].append(np.asarray(mrbi_final.z_star,dtype=np.float64))
        Z["multistart_rescue"].append(np.asarray(ms_final.z_star,dtype=np.float64))

        rows.append({
            "position":position,
            "zero_success":bool(zero.success),
            "mrbi_rescue_success":bool(mrbi_final.success),
            "multistart_rescue_success":bool(ms_final.success),
            "mrbi_attempted":bool(mrbi_attempted),
            "multistart_attempted":bool(ms_attempted),
            "mrbi_rescued":bool(mrbi_successful_rescue),
            "multistart_rescued":bool(ms_successful_rescue),
            "zero_residual":float(zero.residual),
            "mrbi_rescue_residual":float(mrbi_final.residual),
            "multistart_rescue_residual":float(ms_final.residual),
            "zero_nfev":int(zero.nfev),
            "mrbi_rescue_final_nfev":int(mrbi_final.nfev),
            "multistart_rescue_total_nfev":int(ms_total_nfev),
            "zero_runtime_sec":zero_time,
            "mrbi_extra_runtime_sec":mrbi_extra_time,
            "multistart_extra_runtime_sec":ms_extra_time,
            "mrbi_total_runtime_sec":zero_time+mrbi_extra_time,
            "multistart_total_runtime_sec":zero_time+ms_extra_time,
            "mrbi_candidate_objective_calls":mrbi_candidate_calls,
        })

    arrays={key:np.asarray(value,dtype=np.float64) for key,value in Z.items()}
    stats={}
    for method,prefix in (
        ("zero","zero"),
        ("mrbi_rescue","mrbi_rescue"),
        ("multistart_rescue","multistart_rescue"),
    ):
        success=np.asarray([r[f"{prefix}_success"] for r in rows],dtype=bool)
        residual=np.asarray([r[f"{prefix}_residual"] for r in rows],dtype=float)
        stats[f"{method}_success_rate"]=float(success.mean())
        stats[f"{method}_mean_residual"]=float(residual.mean())
        stats[f"{method}_median_residual"]=float(np.median(residual))
    stats.update({
        "zero_failure_rate":float(np.mean([not r["zero_success"] for r in rows])),
        "mrbi_attempt_rate":float(np.mean([r["mrbi_attempted"] for r in rows])),
        "multistart_attempt_rate":float(np.mean([
            r["multistart_attempted"] for r in rows
        ])),
        "mrbi_rescues":int(sum(r["mrbi_rescued"] for r in rows)),
        "multistart_rescues":int(sum(r["multistart_rescued"] for r in rows)),
        "mrbi_rescue_fraction_of_zero_failures":(
            float(np.mean([
                r["mrbi_rescued"]
                for r in rows if not r["zero_success"]
            ]))
            if any(not r["zero_success"] for r in rows) else 0.0
        ),
        "multistart_rescue_fraction_of_zero_failures":(
            float(np.mean([
                r["multistart_rescued"]
                for r in rows if not r["zero_success"]
            ]))
            if any(not r["zero_success"] for r in rows) else 0.0
        ),
        "zero_mean_runtime_sec":float(np.mean([
            r["zero_runtime_sec"] for r in rows
        ])),
        "mrbi_mean_total_runtime_sec":float(np.mean([
            r["mrbi_total_runtime_sec"] for r in rows
        ])),
        "multistart_mean_total_runtime_sec":float(np.mean([
            r["multistart_total_runtime_sec"] for r in rows
        ])),
        "mrbi_mean_extra_runtime_when_attempted_sec":(
            float(np.mean([
                r["mrbi_extra_runtime_sec"]
                for r in rows if r["mrbi_attempted"]
            ]))
            if any(r["mrbi_attempted"] for r in rows) else 0.0
        ),
        "multistart_mean_extra_runtime_when_attempted_sec":(
            float(np.mean([
                r["multistart_extra_runtime_sec"]
                for r in rows if r["multistart_attempted"]
            ]))
            if any(r["multistart_attempted"] for r in rows) else 0.0
        ),
        "mrbi_mean_candidate_objective_calls_when_attempted":(
            float(np.mean([
                r["mrbi_candidate_objective_calls"]
                for r in rows if r["mrbi_attempted"]
            ]))
            if any(r["mrbi_attempted"] for r in rows) else 0.0
        ),
    })
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
    }


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

    stream_base=200000+1000*seed+20*dataset_index
    train_features,train_stats,train_audit=solve_rescue_features(
        Xtr_pca,layer,cfg,
        mrbi_seed=stream_base+1,
        multistart_seed=stream_base+2,
    )
    test_features,test_stats,test_audit=solve_rescue_features(
        Xte_pca,layer,cfg,
        mrbi_seed=stream_base+3,
        multistart_seed=stream_base+4,
    )

    prepared={}
    for method in ("zero","mrbi_rescue","multistart_rescue"):
        prepared[method]=standardize_from_train(
            train_features[method],test_features[method]
        )

    qnn_seed=80000+seed
    implicit_base=bench.TorchQNN(
        LATENT_DIM,N_QUBITS,cfg.qnn_layers,qnn_seed
    )
    implicit_state={
        k:v.detach().clone()
        for k,v in implicit_base.state_dict().items()
    }
    implicit_sha=state_checksum(implicit_state)

    results={}
    for method in ("zero","mrbi_rescue","multistart_rescue"):
        Xtr_method,Xte_method=prepared[method]
        metrics=train_qnn(
            Xtr_method,ytr,Xte_method,yte,cfg,qnn_seed,
            initial_state=implicit_state,
        )
        if metrics["initial_state_sha256"]!=implicit_sha:
            raise RuntimeError("Implicit QNN initialization mismatch")
        results[method]=metrics

    results["pca"]=train_qnn(
        Xtr_pca,ytr,Xte_pca,yte,cfg,qnn_seed,
        initial_state=None,
    )

    z=results["zero"]["balanced_accuracy"]
    m=results["mrbi_rescue"]["balanced_accuracy"]
    s=results["multistart_rescue"]["balanced_accuracy"]
    p=results["pca"]["balanced_accuracy"]
    paired={
        "delta_mrbi_rescue_zero":float(m-z),
        "delta_multistart_rescue_zero":float(s-z),
        "delta_mrbi_rescue_multistart_rescue":float(m-s),
        "delta_mrbi_rescue_pca":float(m-p),
    }

    for i,row in enumerate(test_audit):
        row["true_label"]=int(yte[i])
        for method in ("zero","mrbi_rescue","multistart_rescue","pca"):
            row[f"{method}_prediction"]=int(results[method]["predictions"][i])
        row["mrbi_changed_prediction_vs_zero"]=bool(
            row["mrbi_rescue_prediction"]!=row["zero_prediction"]
        )
        row["multistart_changed_prediction_vs_zero"]=bool(
            row["multistart_rescue_prediction"]!=row["zero_prediction"]
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
    out={}
    for key in (
        "delta_mrbi_rescue_zero",
        "delta_multistart_rescue_zero",
        "delta_mrbi_rescue_multistart_rescue",
        "delta_mrbi_rescue_pca",
    ):
        values=np.asarray([d["paired"][key] for d in dataset_results],dtype=float)
        out[key]={
            "mean":float(values.mean()),
            "median":float(np.median(values)),
            "positive":int(np.sum(values>1e-12)),
            "neutral":int(np.sum(np.abs(values)<=1e-12)),
            "negative":int(np.sum(values< -1e-12)),
        }
    for method in ("pca","zero","mrbi_rescue","multistart_rescue"):
        values=np.asarray([
            d["results"][method]["balanced_accuracy"]
            for d in dataset_results
        ],dtype=float)
        out[f"{method}_balanced_accuracy_mean"]=float(values.mean())
    for method in ("zero","mrbi_rescue","multistart_rescue"):
        out[f"test_{method}_success_rate_mean"]=float(np.mean([
            d["test_solver_stats"][f"{method}_success_rate"]
            for d in dataset_results
        ]))
    out["test_mrbi_rescues_total"]=int(sum(
        d["test_solver_stats"]["mrbi_rescues"] for d in dataset_results
    ))
    out["test_multistart_rescues_total"]=int(sum(
        d["test_solver_stats"]["multistart_rescues"]
        for d in dataset_results
    ))
    return out


def dry_run(seed:int):
    cfg=make_cfg(seed)
    assert seed in FROZEN_SEEDS
    assert len(DATASETS)==9
    assert cfg.profile_name==PROFILE_NAME=="full_balanced"
    assert cfg.spectral_radius==SPECTRAL_RADIUS==2.25
    assert cfg.qnn_epochs==50 and cfg.qnn_lr==0.01
    assert cfg.max_samples_per_class==80
    assert N_RANDOM_STARTS==5 and RANDOM_START_RADIUS==1.0
    print(
        "LOCKED_RESCUE_ONLY_DRY_RUN_OK "
        f"seed={seed} datasets=9 rho=2.25 profile=full_balanced "
        "trigger=zero_strict_failure acceptance=strict_rescue_success "
        "random_control=5starts_on_zero_failure qnn_epochs=50 qnn_lr=0.01 "
        "no_data_loaded",
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
            "trigger":"zero strict solve failure only",
            "acceptance":"MRBI strict solve success only",
            "fallback":"original zero terminal state",
            "spectral_radius":SPECTRAL_RADIUS,
        },
        "controls":{
            "zero":True,
            "random5_rescue_only":{
                "n_random_starts":N_RANDOM_STARTS,
                "trigger":"zero strict solve failure only",
                "acceptance":"best strict successful random solve",
                "fallback":"original zero terminal state",
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
        "selection_rule":"No result-dependent profile/dataset/seed/threshold selection on seeds 30-39",
        "provenance_note":"Rescue-only rule was proposed after partial seeds 20-24 of the prior locked aggressive-policy benchmark; seeds 20-29 are not used as confirmation for this rule.",
        "scope":"new splits of the nine manuscript tasks; hard-regime prospective development confirmation, not external-dataset validation",
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
            f"RESCUE_LOCKED_START seed={args.seed} dataset={dataset} "
            f"rho={SPECTRAL_RADIUS}",
            flush=True,
        )
        result=run_dataset(dataset,args.seed,index)
        dataset_results.append(result)
        atomic_write(
            checkpoint,
            json.dumps({
                "git_commit":commit,
                "seed":args.seed,
                "datasets":dataset_results,
            },indent=2,allow_nan=False)+"\n",
        )
        p=result["paired"]
        print(
            f"RESCUE_LOCKED_DONE dataset={dataset} "
            f"zero={result['results']['zero']['balanced_accuracy']:.4f} "
            f"mrbi_rescue={result['results']['mrbi_rescue']['balanced_accuracy']:.4f} "
            f"ms5_rescue={result['results']['multistart_rescue']['balanced_accuracy']:.4f} "
            f"pca={result['results']['pca']['balanced_accuracy']:.4f} "
            f"d_mz={p['delta_mrbi_rescue_zero']:+.4f} "
            f"d_mms={p['delta_mrbi_rescue_multistart_rescue']:+.4f}",
            flush=True,
        )

    if len(dataset_results)!=len(DATASETS):
        raise RuntimeError("Incomplete locked dataset set")
    final={
        **manifest,
        "dataset_results":dataset_results,
        "seed_summary":summarize(dataset_results),
    }
    path=out_dir/"locked_result.json"
    raw=json.dumps(final,indent=2,allow_nan=False)+"\n"
    if path.exists() and path.read_text(encoding="utf-8")!=raw:
        raise SystemExit("Existing final result differs; use a new --out-dir")
    atomic_write(path,raw)
    print(f"LOCKED_RESCUE_ONLY_OK {path}",flush=True)


if __name__=="__main__":
    main()
