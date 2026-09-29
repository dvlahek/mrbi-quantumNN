"""Final external-source PCA–MRBI fusion/QNN controls, fully frozen.

Three independent UCI datasets (banknote, ionosphere, sonar), five
new seeds 25--29, six fixed QNN methods. Original zero and MRBI
root attempts are shared exactly by the three implicit-fusion arms.
Dataset download is separate: this runner uses only SHA-frozen files.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
sys.path.insert(0,str(ROOT/"scripts"))
import main_qnn_benchmark as bench
import multiscale_stage_audit as audit
import run_implicit_fusion_qnn as fusion
import download_external_fusion_data as external

DATASETS=external.DATASETS
SEEDS=(25,26,27,28,29)
METHODS=(
    "pca_qnn",
    "pca_padded_qnn",
    "fusion_zero_qnn",
    "fusion_mrbi_qnn",
    "fusion_certified_mrbi_qnn",
    "fusion_permuted_certified_qnn",
)
PLAN="external_fusion_qnn_three_sources_seeds25to29_v1"
DEFAULT_OUT=ROOT/"outputs"/"external_fusion_qnn_v1"


def label_blind_derangement(n,seed):
    """Sattolo's shuffle: every latent row comes from another input.

    The pseudorandom permutation depends on only n and the frozen seed,
    not the class label, fitted QNN, checkpoint flag or test metrics.
    """
    if n<2:
        raise ValueError("Permutation control needs at least two samples")
    rng=np.random.default_rng(seed)
    order=np.arange(n)
    for i in range(n-1,0,-1):
        j=int(rng.integers(0,i))
        order[i],order[j]=order[j],order[i]
    if not np.array_equal(np.sort(order),np.arange(n)):
        raise RuntimeError("Permutation is not a bijection")
    if np.any(order==np.arange(n)):
        raise RuntimeError("Permutation did not break all row pairings")
    return order


def make_control_features(Xtr_p,Xte_p,train_z,test_z,seed):
    """Construct the same 20D QNN input width for every fusion control."""
    result=fusion.build_fusion_features(Xtr_p,Xte_p,train_z,test_z)
    tr0=result["pca_qnn"][0]
    te0=result["pca_qnn"][1]
    n_tr,n_te=len(tr0),len(te0)
    result["pca_padded_qnn"]=(
        np.column_stack([tr0,np.zeros((n_tr,16),dtype=np.float64)]),
        np.column_stack([te0,np.zeros((n_te,16),dtype=np.float64)]),
    )
    p_tr=label_blind_derangement(n_tr,seed+41007)
    p_te=label_blind_derangement(n_te,seed+72013)
    z_tr=result["fusion_certified_mrbi_qnn"][0][:,4:]
    z_te=result["fusion_certified_mrbi_qnn"][1][:,4:]
    result["fusion_permuted_certified_qnn"]=(
        np.column_stack([tr0,z_tr[p_tr]]),
        np.column_stack([te0,z_te[p_te]]),
    )
    if set(result)!=set(METHODS):
        raise RuntimeError("Incomplete prespecified external fusion methods")
    for name,(Xtr,Xte) in result.items():
        width=4 if name=="pca_qnn" else 20
        if (Xtr.shape!=(n_tr,width) or Xte.shape!=(n_te,width)
                or not np.all(np.isfinite(Xtr))
                or not np.all(np.isfinite(Xte))):
            raise RuntimeError(f"Invalid fixed QNN representation: {name}")
        if name!="pca_qnn" and (
            not np.array_equal(Xtr[:,:4],tr0) or
            not np.array_equal(Xte[:,:4],te0)
        ):
            raise RuntimeError("Fusion control changed original PCA information")
    if (not np.array_equal(
            np.sort(result["fusion_permuted_certified_qnn"][0][:,4:],
                    axis=0),
            np.sort(z_tr,axis=0),
        ) or not np.array_equal(
            np.sort(result["fusion_permuted_certified_qnn"][1][:,4:],
                    axis=0),
            np.sort(z_te,axis=0),
        )):
        raise RuntimeError("Permutation control changed latent marginal values")
    hashes={
        "train_permutation_sha256":fusion.sha_arrays(
            p_tr.astype(np.float64)
        ),
        "test_permutation_sha256":fusion.sha_arrays(
            p_te.astype(np.float64)
        ),
    }
    return result,hashes


def evaluate_qnns(dataset,seed,cfg,feature_arrays,ytr,yte,
                  train_stats,test_stats):
    """Six fixed readouts; three implicit sources have paired diagnostics."""
    if not bench.HAS_QNN:
        raise RuntimeError("Torch/PennyLane missing in full Ryzen QNN venv")
    rows=[]
    for method in METHODS:
        Xtr,Xte=feature_arrays[method]
        if (Xtr.shape[0]!=len(ytr) or Xte.shape[0]!=len(yte)
                or Xtr.shape[1]!=Xte.shape[1]
                or not np.all(np.isfinite(Xtr))
                or not np.all(np.isfinite(Xte))):
            raise RuntimeError(f"Invalid readout feature shape {method}")
        print(
            f"[{dataset} seed={seed}] QNN {method}: "
            f"{len(ytr)} train / {len(yte)} test",flush=True,
        )
        measure=bench.train_qnn(Xtr,ytr,Xte,yte,cfg,seed+777)
        if any(
            not np.isfinite(float(measure[col]))
            for col in ("accuracy","balanced_accuracy","f1","roc_auc")
        ):
            raise RuntimeError(f"Nonfinite QNN result for {method}")
        sttr=train_stats[train_stats.method.eq(method)]
        stte=test_stats[test_stats.method.eq(method)]
        if method in ("pca_qnn","pca_padded_qnn"):
            if len(sttr) or len(stte):
                raise RuntimeError("Nonimplicit control has solver records")
            work={
                "feature_test_success":np.nan,
                "feature_test_rescues":np.nan,
                "feature_mean_test_algorithm_F":0.0,
                "feature_mean_test_algorithm_J":0.0,
                "feature_mean_test_root_calls":0.0,
                "feature_mean_test_optimizer_calls":0.0,
                "feature_wall_sec":0.0,
            }
        elif method=="fusion_permuted_certified_qnn":
            if len(sttr) or len(stte):
                raise RuntimeError("Permutation control must have no aligned solver records")
            source_train=train_stats[
                train_stats.method.eq("fusion_certified_mrbi_qnn")
            ]
            source_test=test_stats[
                test_stats.method.eq("fusion_certified_mrbi_qnn")
            ]
            if len(source_train)!=len(ytr) or len(source_test)!=len(yte):
                raise RuntimeError("Missing source for permutation control")
            # Original-root success for input x does not describe
            # the deliberately *different* permuted latent z(x').
            work={
                "feature_test_success":np.nan,
                "feature_test_rescues":np.nan,
                "feature_mean_test_algorithm_F":float(
                    source_test.algorithm_F_calls.mean()
                ),
                "feature_mean_test_algorithm_J":float(
                    source_test.algorithm_J_calls.mean()
                ),
                "feature_mean_test_root_calls":float(
                    source_test.root_calls.mean()
                ),
                "feature_mean_test_optimizer_calls":float(
                    source_test.optimizer_objective_calls.mean()
                ),
                "feature_wall_sec":float(
                    source_train.total_wall_sec.sum()+
                    source_test.total_wall_sec.sum()
                ),
            }
        else:
            if len(sttr)!=len(ytr) or len(stte)!=len(yte):
                raise RuntimeError("Incomplete implicit feature diagnostics")
            work={
                "feature_test_success":float(stte.accepted_success.mean()),
                "feature_test_rescues":int((
                    (stte.zero_success==0)&
                    (stte.checkpoint_success==1)
                ).sum()),
                "feature_mean_test_algorithm_F":float(
                    stte.algorithm_F_calls.mean()
                ),
                "feature_mean_test_algorithm_J":float(
                    stte.algorithm_J_calls.mean()
                ),
                "feature_mean_test_root_calls":float(
                    stte.root_calls.mean()
                ),
                "feature_mean_test_optimizer_calls":float(
                    stte.optimizer_objective_calls.mean()
                ),
                "feature_wall_sec":float(
                    sttr.total_wall_sec.sum()+stte.total_wall_sec.sum()
                ),
            }
        rows.append({
            "plan":PLAN,"dataset":dataset,"seed":seed,
            "method":method,"readout":"qnn",
            "n_qubits":cfg.n_qubits,
            "qnn_layers":cfg.qnn_layers,
            "qnn_epochs":cfg.qnn_epochs,
            "qnn_seed":seed+777,
            "qnn_input_dim":Xtr.shape[1],
            "n_train":len(ytr),"n_test":len(yte),
            "latent_dim":cfg.latent_dim,
            "profile":"full_balanced_single_stage",
            **measure,**work,
        })
    return pd.DataFrame(rows)


def run_job(dataset,seed,sources):
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError("External QNN tasks/seeds frozen to three UCI sources and 25--29")
    cfg=replace(audit.config_for(seed),run_qnn=True)
    if (cfg.n_qubits!=4 or cfg.qnn_layers!=2 or cfg.qnn_epochs!=60
            or cfg.pca_dim!=4 or cfg.latent_dim!=16
            or cfg.max_samples_per_class!=80 or cfg.mc_samples!=10
            or cfg.spectral_radius!=2.0):
        raise RuntimeError("Frozen original fusion QNN configuration changed")
    if not bench.HAS_QNN:
        raise RuntimeError("Activate original full Ryzen QNN virtual environment")
    X0,y0,source_sha=sources[dataset]
    X,y=bench.balanced_subsample(
        X0,y0,cfg.max_samples_per_class,cfg.seed
    )
    if set(np.unique(y))!={0,1} or len(X)!=160:
        raise RuntimeError("External source selection should be 80 per class")
    Xtr_raw,Xte_raw,ytr,yte=train_test_split(
        X,y,test_size=cfg.test_size,stratify=y,random_state=seed
    )
    Xtr_p,Xte_p=bench.preprocess_to_pca(
        Xtr_raw,Xte_raw,cfg.pca_dim,seed
    )
    layer=bench.make_random_implicit_layer(
        dx=Xtr_p.shape[1],d=cfg.latent_dim,cfg=cfg,
        seed=seed+1000+cfg.latent_dim+int(100*cfg.spectral_radius),
    )
    print(f"[{dataset} seed={seed}] Generating frozen training roots",flush=True)
    train_z,train_stats=fusion.numerical_features(
        Xtr_p,layer,cfg,seed,"train",0
    )
    train_z,train_stats=fusion.certify_features(train_z,train_stats)
    print(f"[{dataset} seed={seed}] Generating frozen test roots",flush=True)
    test_z,test_stats=fusion.numerical_features(
        Xte_p,layer,cfg,seed,"test",len(Xtr_p)
    )
    test_z,test_stats=fusion.certify_features(test_z,test_stats)
    features,perm_hashes=make_control_features(
        Xtr_p,Xte_p,train_z,test_z,seed
    )
    metrics=evaluate_qnns(
        dataset,seed,cfg,features,ytr,yte,train_stats,test_stats
    )
    records=pd.concat(
        [train_stats,test_stats],ignore_index=True
    )
    records.insert(0,"plan",PLAN)
    records.insert(1,"dataset",dataset)
    records.insert(2,"seed",seed)
    if (set(metrics.method)!=set(METHODS)
            or metrics.duplicated(["dataset","seed","method"]).any()
            or records.duplicated([
                "dataset","seed","split","input_index","method"
            ]).any()
            or len(records)!=3*(len(ytr)+len(yte))):
        raise RuntimeError("Incomplete external fusion paired QNN job")
    info={
        "plan":PLAN,"dataset":dataset,"seed":seed,
        "source_raw_sha256":source_sha,
        "config":asdict(cfg),
        "root_config":asdict(bench.make_root_cfg(cfg)),
        "pca_features_and_labels_sha256":fusion.sha_arrays(
            Xtr_p,Xte_p,ytr.astype(float),yte.astype(float)
        ),
        "implicit_layer_sha256":fusion.sha_arrays(
            layer.W,layer.U,layer.b
        ),
        "qnn_seed_rule":"seed+777; reset independently for all six methods",
        "permutation_rule":"Sattolo derangement, independent train/test, using seed+41007 and seed+72013; no labels",
        "certificate_policy":"Original strict root success only; failed MRBI candidate falls back to original zero feature",
        "same_source_split_and_implicit_layer_all_methods":True,
        **perm_hashes,
    }
    return metrics,records,info


def job_paths(out,dataset,seed):
    return fusion.paths_for(out,dataset,seed)


def collect(out,jobs):
    parts,feature_parts=[],[]
    for dataset,seed in jobs:
        m,f,p=job_paths(out,dataset,seed)
        if m.is_file() and f.is_file() and p.is_file():
            parts.append(pd.read_csv(m))
            feature_parts.append(pd.read_csv(f))
    if not parts:
        print("No external fusion jobs yet.",flush=True)
        return
    metrics=pd.concat(parts,ignore_index=True)
    features=pd.concat(feature_parts,ignore_index=True)
    if (metrics.duplicated(["dataset","seed","method"]).any()
            or len(metrics)!=len(parts)*len(METHODS)
            or features.duplicated([
                "dataset","seed","split","input_index","method"
            ]).any()):
        raise RuntimeError("External fusion aggregate coverage mismatch")
    metrics.to_csv(out/"qnn_raw.csv",index=False)
    features.to_csv(out/"feature_raw.csv",index=False)
    contrasts=(
        ("delta_certified_minus_zero_ba",
         "fusion_certified_mrbi_qnn","fusion_zero_qnn"),
        ("delta_certified_minus_padded_ba",
         "fusion_certified_mrbi_qnn","pca_padded_qnn"),
        ("delta_certified_minus_permuted_ba",
         "fusion_certified_mrbi_qnn","fusion_permuted_certified_qnn"),
        ("delta_certified_minus_ungated_ba",
         "fusion_certified_mrbi_qnn","fusion_mrbi_qnn"),
        ("delta_ungated_minus_zero_ba",
         "fusion_mrbi_qnn","fusion_zero_qnn"),
        ("delta_certified_minus_pca_ba",
         "fusion_certified_mrbi_qnn","pca_qnn"),
    )
    rows=[]
    for (dataset,seed),group in metrics.groupby(
        ["dataset","seed"],sort=True
    ):
        if set(group.method)!=set(METHODS):
            raise RuntimeError(f"Missing external fusion arm: {dataset} seed {seed}")
        row={"dataset":dataset,"seed":int(seed)}
        for _,metric in group.iterrows():
            row[metric["method"]+"_ba"]=float(metric["balanced_accuracy"])
        for key,a,b in contrasts:
            row[key]=row[a+"_ba"]-row[b+"_ba"]
        rows.append(row)
    paired=pd.DataFrame(rows)
    paired.to_csv(out/"paired_job_results.csv",index=False)
    agg={"n_seeds":("seed","nunique")}
    for method in METHODS:
        agg["mean_"+method+"_ba"]=(method+"_ba","mean")
    for key,_,_ in contrasts:
        agg["mean_"+key]=(key,"mean")
    datasets=paired.groupby("dataset",as_index=False).agg(**agg)
    datasets.to_csv(out/"dataset_summary.csv",index=False)
    summary={
        "plan":PLAN,"completed_jobs":len(parts),
        "requested_jobs":len(jobs),
        "n_qnn_models":len(metrics),
        "source_datasets":list(DATASETS),
        "prespecified_primary":"paired fusion_certified_mrbi_qnn minus fusion_zero_qnn BA, three novel UCI sources, seeds 25--29",
        "prespecified_secondary":[
            "certified fusion minus equally-wide zero-padded PCA",
            "certified fusion minus row-deranged certified latent PCA fusion",
            "certified fusion minus ungated fusion",
            "certified fusion minus 4D PCA (descriptive; different input projection)",
            "strict root success and F/J work as diagnostics",
        ],
        "means_by_method":{
            method:{
                "balanced_accuracy":float(
                    metrics.loc[metrics.method.eq(method),
                                "balanced_accuracy"].mean()
                ),
                "training_time_sec":float(
                    metrics.loc[metrics.method.eq(method),
                                "train_time_sec"].mean()
                ),
            } for method in METHODS
        },
        "mean_paired_deltas":{
            key:float(paired[key].mean())
            for key,_,_ in contrasts
        },
        "limitations":"Only three genuinely new source families; repeated seeds on each family are not independent source datasets; PCA comparator has 4D rather than 20D input; batch-level permutation is a label-blind diagnostic control; classical QNN simulation.",
    }
    fusion.atomic_write(
        out/"qnn_summary.json",
        json.dumps(summary,indent=2,allow_nan=False)+"\n",
    )
    print(f"Collected {len(parts)}/{len(jobs)} external jobs, "
          f"{len(metrics)} QNN readouts.",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,default=external.DEFAULT_DIR)
    p.add_argument("--datasets",nargs="+",choices=DATASETS,
                   default=list(DATASETS))
    p.add_argument("--seeds",nargs="+",type=int,
                   default=list(SEEDS))
    p.add_argument("--max-new-jobs",type=int,default=None)
    p.add_argument("--out-dir",type=Path,default=DEFAULT_OUT)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--collect-only",action="store_true")
    args=p.parse_args()
    if (not args.datasets
            or len(set(args.datasets))!=len(args.datasets)
            or not args.seeds
            or len(set(args.seeds))!=len(args.seeds)
            or not set(args.seeds).issubset(SEEDS)
            or args.max_new_jobs is not None
            and args.max_new_jobs<1):
        raise SystemExit("Use unique frozen source datasets and seeds 25--29")
    jobs=[(ds,seed) for ds in args.datasets for seed in args.seeds]
    if args.dry_run:
        print(f"EXTERNAL_FUSION_DRY_RUN_OK {len(jobs)} jobs, "
              f"{len(jobs)*len(METHODS)} QNNs. "
              f"No datasets downloaded, opened or evaluated.",flush=True)
        return
    if not bench.HAS_QNN:
        raise SystemExit("Missing torch/PennyLane; activate full Ryzen QNN venv")
    # Fail before any result is written if data files or provenance drift.
    sources,data_manifest=external.load_sources(
        args.data_dir,requested=args.datasets
    )
    head=subprocess.run(
        ["git","rev-parse","HEAD"],cwd=ROOT,
        check=True,capture_output=True,text=True,
    ).stdout.strip()
    package_names=(
        "numpy","scipy","pandas","scikit-learn","torch","pennylane"
    )
    manifest={
        "plan":PLAN,"repo_sha":head,
        "python":sys.version,"platform":platform.platform(),
        "packages":{
            name:version(name) for name in package_names
        },
        "datasets":args.datasets,"seeds":args.seeds,
        "data_sources":{
            name:data_manifest["datasets"][name]
            for name in args.datasets
        },
        "qnn_config":asdict(replace(
            audit.config_for(25),run_qnn=True
        )),
        "methods":list(METHODS),
        "sigma":0.70,"smoothing_weight":0.75,
        "optimizer_caps":{
            "objective_calls":480,"optimizer_F_calls":5760
        },
        "sampling":"Fixed 80/class, stratified 70/30; train-only PCA and latent scaling.",
        "permutation":"Sattolo train/test independently by fixed seed offsets; label-blind diagnostic control",
        "pre_registered":"No source choice, method, sigma, QNN config or subgroup selection after outcomes",
        "quantum_note":"CPU classical PennyLane simulation; no quantum advantage claim.",
    }
    manifest=json.loads(json.dumps(manifest,allow_nan=False))
    out=args.out_dir.resolve()
    environment_file=out/"environment.json"
    if environment_file.is_file():
        if json.loads(
            environment_file.read_text(encoding="utf-8")
        )!=manifest:
            raise SystemExit(
                "External QNN code/config/data changed; use a new output directory"
            )
    else:
        out.mkdir(parents=True,exist_ok=True)
        fusion.atomic_write(
            environment_file,
            json.dumps(manifest,indent=2,allow_nan=False)+"\n",
        )
    pending=[]
    for dataset,seed in jobs:
        m,f,record_path=job_paths(out,dataset,seed)
        state=[path.is_file() for path in (m,f,record_path)]
        if all(state):
            provenance=json.loads(record_path.read_text(encoding="utf-8"))
            metrics=pd.read_csv(m)
            features=pd.read_csv(f)
            if (provenance.get("plan")!=PLAN
                    or provenance.get("repo_sha")!=head
                    or provenance.get("dataset")!=dataset
                    or provenance.get("seed")!=seed
                    or provenance.get("source_raw_sha256")!=
                       sources[dataset][2]
                    or provenance.get("metrics_csv_sha256")!=
                       hashlib.sha256(m.read_bytes()).hexdigest()
                    or provenance.get("features_csv_sha256")!=
                       hashlib.sha256(f.read_bytes()).hexdigest()
                    or len(metrics)!=len(METHODS)
                    or set(metrics.method)!=set(METHODS)
                    or metrics.duplicated([
                        "dataset","seed","method"
                    ]).any()
                    or features.duplicated([
                        "dataset","seed","split","input_index","method"
                    ]).any()
                    or len(features)!=3*int(
                        (metrics.n_train+metrics.n_test).iloc[0]
                    )):
                raise SystemExit(
                    f"Foreign or corrupt external QNN job {dataset} seed {seed}"
                )
        elif any(state):
            raise SystemExit(
                f"Partial job {dataset} seed {seed}: inspect before retry"
            )
        else:
            pending.append((dataset,seed))
    print(f"Frozen external fusion: {len(jobs)} jobs, "
          f"{len(jobs)-len(pending)} complete, "
          f"{len(pending)} pending.",flush=True)
    if not args.collect_only:
        for dataset,seed in pending[:args.max_new_jobs]:
            print(f"Starting {dataset} seed={seed}",flush=True)
            metrics,features,info=run_job(
                dataset,seed,sources
            )
            metrics_csv=metrics.to_csv(index=False)
            features_csv=features.to_csv(index=False)
            m,f,record_path=job_paths(out,dataset,seed)
            m.parent.mkdir(parents=True,exist_ok=True)
            fusion.atomic_write(m,metrics_csv)
            fusion.atomic_write(f,features_csv)
            info.update(
                plan=PLAN,dataset=dataset,seed=seed,
                repo_sha=head,
                metrics_csv_sha256=hashlib.sha256(
                    metrics_csv.encode("utf-8")
                ).hexdigest(),
                features_csv_sha256=hashlib.sha256(
                    features_csv.encode("utf-8")
                ).hexdigest(),
            )
            fusion.atomic_write(
                record_path,
                json.dumps(info,indent=2,allow_nan=False)+"\n",
            )
            print(
                f"Finished {dataset} seed={seed}: "
                f"{len(metrics)} QNN methods.",flush=True,
            )
    collect(out,jobs)


if __name__=="__main__":
    main()
