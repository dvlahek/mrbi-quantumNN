"""Frozen PCA-plus-implicit fusion QNN experiment on seeds 20--24.

The same four-qubit TorchQNN sees [PCA(x), standardize(z(x))].
Four fixed arms isolate implicit features and a strict, label-free
MRBI certificate: PCA-QNN, PCA+Zero, PCA+MRBI, PCA+certified MRBI.
No best-per-dataset selection and no quantum-advantage claim.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import main_qnn_benchmark as bench
import mrbi
import multiscale_stage_audit as audit
import run_heldout_single_stage as single

DATASETS = audit.DATASETS
SEEDS = (20, 21, 22, 23, 24)
METHODS = (
    "pca_qnn", "fusion_zero_qnn",
    "fusion_mrbi_qnn", "fusion_certified_mrbi_qnn",
)
ARMS = ("smoothed_coarse",)
PLAN = "implicit_fusion_qnn_seeds20to24_v1"
DEFAULT_OUT = ROOT / "outputs" / "implicit_fusion_qnn_v1"


def sha_arrays(*arrays):
    return single.sha_arrays(*arrays)


def atomic_write(path, content):
    single.atomic_write(path, content)


def numerical_features(inputs, layer, cfg, seed, split, offset):
    """Use the exact frozen single-stage numerical policy on every input.

    Each paired arm shares the same zero-root result and Gaussian probes.
    Train and test use disjoint probe indices; split labels are unused.
    """
    root_cfg = bench.make_root_cfg(cfg)
    outputs = {"zero": [], **{arm: [] for arm in ARMS}}
    records = []
    for j, x in enumerate(inputs):
        index = offset+j
        start = time.perf_counter()
        zero_F, zero_J, baseline = single.counted_functions(layer)
        zero, zf, zj = single.measured_root(
            zero_F, zero_J, baseline, x,
            np.zeros(layer.d, dtype=np.float64), root_cfg,
        )
        if zero.success:
            zero_smin = np.nan
        else:
            zero_smin, _ = mrbi.jacobian_health(
                zero_J, x, zero.z_star
            )
        baseline_wall = time.perf_counter()-start
        outputs["zero"].append(np.asarray(zero.z_star, dtype=np.float64))
        records.append({
            "split": split, "input_index": j, "method": "fusion_zero_qnn",
            "zero_success": int(zero.success),
            "accepted_success": int(zero.success),
            "checkpoint_success": 0, "used_mrbi": 0,
            "candidate_attempted": 0,
            "accepted_residual": float(zero.residual),
            "algorithm_F_calls": int(zf),
            "algorithm_J_calls": int(zj),
            "total_F_calls": int(zf),
            "total_J_calls": int(zj),
            "optimizer_objective_calls": 0,
            "root_calls": 1,
            "budget_hit": 0,
            "total_wall_sec": float(baseline_wall),
            "probe_sha256": "",
        })
        hashes = set()
        for arm in ARMS:
            stage, result, z_star = single.run_arm(
                layer, x, cfg, root_cfg, zero, zero_smin,
                baseline, zf, zj, baseline_wall,
                seed, index, arm, 480, 5760,
                return_solution=True,
            )
            outputs[arm].append(np.asarray(z_star, dtype=np.float64))
            if stage is not None:
                if (stage["sigma"] != single.ARM_SPECS[arm][0]
                        or stage["smooth_weight"] != single.ARM_SPECS[arm][1]
                        or stage["stage"] != 1
                        or stage["objective_calls_stage"] > 480
                        or stage["residual_evals_stage"] > 5760):
                    raise RuntimeError("Frozen single-stage optimizer changed")
                hashes.add(stage["probe_sha256"])
            records.append({
                "split": split, "input_index": j,
                "method": "fusion_mrbi_qnn",
                **{field: result[field] for field in (
                    "zero_success", "accepted_success",
                    "checkpoint_success", "used_mrbi",
                    "candidate_attempted", "accepted_residual",
                    "algorithm_F_calls", "algorithm_J_calls",
                    "total_F_calls", "total_J_calls",
                    "optimizer_objective_calls", "root_calls",
                    "budget_hit", "total_wall_sec", "probe_sha256",
                )},
            })
        if len(hashes) > 1:
            raise RuntimeError("Unexpected multiple probe hashes")
    result = {name: np.asarray(rows, dtype=np.float64)
              for name, rows in outputs.items()}
    for name, z in result.items():
        if z.shape != (len(inputs), layer.d) or not np.all(np.isfinite(z)):
            raise RuntimeError(f"Invalid numerical feature array: {name}")
    return result, pd.DataFrame(records)



def certify_features(outputs, stats):
    """Choose MRBI state only when its strict root solve succeeds.

    No label, test metric, residual-ranking threshold or extra solve
    enters this fixed gate. Root failure means the original zero
    feature is reused exactly. All candidate-computation work is kept.
    """
    if set(outputs) != {"zero", "smoothed_coarse"}:
        raise RuntimeError("Unexpected latent source in frozen fusion plan")
    n=len(outputs["zero"])
    if n<1 or outputs["smoothed_coarse"].shape != outputs["zero"].shape:
        raise RuntimeError("Missing or incompatible numerical features")
    if len(stats)!=2*n or stats.duplicated(
        ["split","input_index","method"]
    ).any():
        raise RuntimeError("Incomplete source feature diagnostics")
    zero=(stats[stats.method.eq("fusion_zero_qnn")]
          .sort_values("input_index").reset_index(drop=True))
    coarse=(stats[stats.method.eq("fusion_mrbi_qnn")]
            .sort_values("input_index").reset_index(drop=True))
    if (len(zero)!=n or len(coarse)!=n
            or not np.array_equal(
                zero.input_index.to_numpy(),np.arange(n)
            ) or not np.array_equal(
                coarse.input_index.to_numpy(),np.arange(n)
            ) or not np.array_equal(
                zero.zero_success.to_numpy(),
                coarse.zero_success.to_numpy(),
            ) or ((coarse.zero_success.eq(1)) &
                  (coarse.checkpoint_success.eq(1))).any()
            or ((coarse.checkpoint_success.eq(1)) &
                (coarse.accepted_success.ne(1) |
                 coarse.used_mrbi.ne(1))).any()):
        raise RuntimeError("Invalid MRBI/zero paired certificate inputs")
    passed=coarse.checkpoint_success.eq(1).to_numpy()
    cert=np.where(passed[:,None],
                  outputs["smoothed_coarse"],outputs["zero"]).copy()
    if (not np.array_equal(
            cert[passed],outputs["smoothed_coarse"][passed]
        ) or not np.array_equal(
            cert[~passed],outputs["zero"][~passed]
        )):
        raise RuntimeError("Certified gate changed a frozen source root")
    record=coarse.copy()
    record["method"]="fusion_certified_mrbi_qnn"
    record["accepted_success"]=(
        (coarse.zero_success.eq(1) |
         coarse.checkpoint_success.eq(1)).astype(int)
    )
    record["used_mrbi"]=coarse.checkpoint_success.astype(int)
    record["accepted_residual"]=np.where(
        passed,coarse.accepted_residual.to_numpy(),
        zero.accepted_residual.to_numpy(),
    )
    record["certified_zero_fallback"]=(
        coarse.zero_success.eq(0) &
        coarse.checkpoint_success.eq(0)
    ).astype(int)
    combined=pd.concat([stats,record],ignore_index=True)
    if (len(combined)!=3*n or
            combined.duplicated(["split","input_index","method"]).any()):
        raise RuntimeError("Invalid certified feature coverage")
    return {**outputs,"certified_coarse":cert},combined

def evaluate_qnns(dataset, seed, cfg, features, ytr, yte,
                  train_stats, test_stats):
    """Evaluate *fixed* readouts with identical training/randomization rules."""
    if not bench.HAS_QNN:
        raise RuntimeError(
            "QNN dependencies missing. Activate the full Ryzen venv with "
            "torch and pennylane; never report NaN placeholders as results."
        )
    rows = []
    for method in METHODS:
        if method not in features:
            raise RuntimeError(f"Required prespecified method missing: {method}")
        Xtr, Xte = features[method]
        if (not np.all(np.isfinite(Xtr)) or not np.all(np.isfinite(Xte))
                or Xtr.shape[0] != len(ytr) or Xte.shape[0] != len(yte)
                or Xtr.shape[1] != Xte.shape[1]):
            raise RuntimeError(f"Invalid QNN input representation: {method}")
        print(f"[{dataset} seed={seed}] QNN {method}: "
              f"{len(ytr)} train / {len(yte)} test", flush=True)
        # train_qnn internally resets torch, NumPy and minibatch RNG
        # using this same seed on each method.
        measures = bench.train_qnn(
            Xtr, ytr, Xte, yte, cfg, seed+777
        )
        for key in ("accuracy","balanced_accuracy","f1","roc_auc"):
            if not np.isfinite(float(measures[key])):
                raise RuntimeError(
                    f"Non-finite QNN metric {method}/{key}: {measures[key]}"
                )
        st_tr = train_stats[train_stats.method.eq(method)]
        st_te = test_stats[test_stats.method.eq(method)]
        if method == "pca_qnn":
            if len(st_tr) or len(st_te):
                raise RuntimeError("PCA must have no implicit solver records")
            costs = {
                "feature_train_n": len(ytr), "feature_test_n": len(yte),
                "feature_test_success": np.nan,
                "feature_test_rescues": np.nan,
                "feature_mean_test_algorithm_F": 0.0,
                "feature_mean_test_algorithm_J": 0.0,
                "feature_mean_test_root_calls": 0.0,
                "feature_mean_test_optimizer_calls": 0.0,
                "feature_wall_sec": 0.0,
            }
        else:
            if len(st_tr) != len(ytr) or len(st_te) != len(yte):
                raise RuntimeError(f"Incomplete method features: {method}")
            costs = {
                "feature_train_n": len(st_tr),
                "feature_test_n": len(st_te),
                "feature_test_success": float(st_te.accepted_success.mean()),
                "feature_test_rescues": int(
                    ((st_te.zero_success == 0)&
                     (st_te.checkpoint_success == 1)).sum()
                ),
                "feature_mean_test_algorithm_F": float(
                    st_te.algorithm_F_calls.mean()
                ),
                "feature_mean_test_algorithm_J": float(
                    st_te.algorithm_J_calls.mean()
                ),
                "feature_mean_test_root_calls": float(
                    st_te.root_calls.mean()
                ),
                "feature_mean_test_optimizer_calls": float(
                    st_te.optimizer_objective_calls.mean()
                ),
                "feature_wall_sec": float(
                    st_tr.total_wall_sec.sum()+st_te.total_wall_sec.sum()
                ),
            }
        rows.append({
            "plan": PLAN, "dataset": dataset, "seed": seed,
            "method": method, "readout": "qnn",
            "n_qubits": cfg.n_qubits, "qnn_layers": cfg.qnn_layers,
            "qnn_epochs": cfg.qnn_epochs,
            "qnn_seed": seed+777,
            "n_train": len(ytr), "n_test": len(yte),
            "latent_dim": cfg.latent_dim,
            "qnn_input_dim": Xtr.shape[1],
            "representation": "pca" if method=="pca_qnn" else "pca_plus_latent",
            "profile": "full_balanced_single_stage",
            **measures, **costs,
        })
    return pd.DataFrame(rows)


def build_fusion_features(Xtr_p,Xte_p,train_z,test_z):
    """Paired PCA+implicit representation with train-only latent scaling.

    The PCA coordinates were already fitted using training data in
    preprocess_to_pca. They are copied unchanged, not refitted for
    each arm. The same scaler family applies to each latent source
    and is fitted to that arm's training states only.
    """
    if (Xtr_p.ndim!=2 or Xte_p.ndim!=2 or
            Xtr_p.shape[1]!=4 or Xte_p.shape[1]!=4):
        raise RuntimeError("Expected four preprocessed PCA coordinates")
    feature_arrays={"pca_qnn":(Xtr_p.copy(),Xte_p.copy())}
    for source,method in (
        ("zero","fusion_zero_qnn"),
        ("smoothed_coarse","fusion_mrbi_qnn"),
        ("certified_coarse","fusion_certified_mrbi_qnn"),
    ):
        ztr,zte=bench.standardize_pair(
            train_z[source],test_z[source]
        )
        feature_arrays[method]=(
            np.concatenate([Xtr_p,ztr],axis=1),
            np.concatenate([Xte_p,zte],axis=1),
        )
        if (feature_arrays[method][0].shape!=(len(Xtr_p),20)
                or feature_arrays[method][1].shape!=(len(Xte_p),20)
                or not np.array_equal(
                    feature_arrays[method][0][:,:4],Xtr_p
                ) or not np.array_equal(
                    feature_arrays[method][1][:,:4],Xte_p
                ) or not np.all(np.isfinite(
                    feature_arrays[method][0]
                )) or not np.all(np.isfinite(
                    feature_arrays[method][1]
                ))):
            raise RuntimeError("PCA-plus-latent QNN encoding changed")
    return feature_arrays

def run_job(dataset, seed):
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError("Fusion QNN experiment frozen to nine tasks, seeds 20--24")
    cfg = replace(audit.config_for(seed), run_qnn=True)
    if (cfg.qnn_epochs != 60 or cfg.n_qubits != 4
            or cfg.latent_dim != 16 or cfg.qnn_layers != 2
            or cfg.mc_samples != 10):
        raise RuntimeError("Frozen full campaign QNN configuration changed")
    if not bench.HAS_QNN:
        raise RuntimeError(
            "Missing torch/PennyLane. Activate the Ryzen full-campaign venv."
        )
    X, y = bench.load_dataset(dataset, cfg)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=seed
    )
    Xtr_p, Xte_p = bench.preprocess_to_pca(
        Xtr_raw, Xte_raw, cfg.pca_dim, seed
    )
    layer = bench.make_random_implicit_layer(
        dx=Xtr_p.shape[1], d=cfg.latent_dim, cfg=cfg,
        seed=seed+1000+cfg.latent_dim+int(100*cfg.spectral_radius),
    )
    print(f"[{dataset} seed={seed}] Generating frozen train features",
          flush=True)
    train_z, train_stats = numerical_features(
        Xtr_p, layer, cfg, seed, "train", 0
    )
    train_z, train_stats = certify_features(train_z, train_stats)
    print(f"[{dataset} seed={seed}] Generating frozen test features",
          flush=True)
    test_z, test_stats = numerical_features(
        Xte_p, layer, cfg, seed, "test", len(Xtr_p)
    )
    test_z, test_stats = certify_features(test_z, test_stats)
    feature_arrays=build_fusion_features(
        Xtr_p,Xte_p,train_z,test_z
    )
    job_metrics = evaluate_qnns(
        dataset, seed, cfg, feature_arrays, ytr, yte,
        train_stats, test_stats,
    )
    features = pd.concat(
        [train_stats,test_stats], ignore_index=True
    )
    features.insert(0, "plan", PLAN)
    features.insert(1, "dataset", dataset)
    features.insert(2, "seed", seed)
    if (features.duplicated(
            ["split","input_index","method"]
        ).any() or set(job_metrics.method) != set(METHODS)
            or not job_metrics.balanced_accuracy.between(0,1).all()):
        raise RuntimeError("Job output coverage/metric checks failed")
    job_provenance = {
        "plan": PLAN, "dataset": dataset, "seed": seed,
        "config": asdict(cfg),
        "root_config": asdict(bench.make_root_cfg(cfg)),
        "training_and_test_pca_labels_sha256": sha_arrays(
            Xtr_p, Xte_p, ytr.astype(float), yte.astype(float)
        ),
        "implicit_layer_sha256": sha_arrays(
            layer.W, layer.U, layer.b
        ),
        "probe_seed_rule": "seed + 2000 + 1000003 * (train_index or ntrain + test_index)",
        "certificate_policy": "Strict MRBI root success only; otherwise keep zero-root latent feature and preserve F/J cost",
        "fusion_policy": "Concatenate untouched 4D PCA with train-only standardized 16D latent; input 20D for all fusion QNNs",
        "qnn_seed_rule": "seed + 777, independently reset for each method",
        "same_data_split_and_layer_all_methods": True,
    }
    return job_metrics, features, job_provenance


def paths_for(out, dataset, seed):
    prefix=out/"jobs"/f"{dataset}_seed{seed}"
    return tuple(
        Path(str(prefix)+suffix) for suffix in (
            "_metrics.csv", "_features.csv", "_manifest.json"
        )
    )


def load_complete_jobs(out, jobs):
    metrics, feats = [], []
    for ds, seed in jobs:
        m,f,p=paths_for(out,ds,seed)
        if all(path.is_file() for path in (m,f,p)):
            metrics.append(pd.read_csv(m))
            feats.append(pd.read_csv(f))
    return metrics, feats


def collect(out,jobs):
    chunks,fchunks=load_complete_jobs(out,jobs)
    if not chunks:
        print("No completed frozen fusion QNN jobs.",flush=True)
        return
    metrics=pd.concat(chunks,ignore_index=True)
    features=pd.concat(fchunks,ignore_index=True)
    if (len(metrics)!=len(chunks)*len(METHODS)
            or metrics.duplicated(
                ["dataset","seed","method"]
            ).any()
            or features.duplicated(
                ["dataset","seed","split","input_index","method"]
            ).any()):
        raise RuntimeError("Invalid aggregate fusion coverage")
    for (dataset,seed),group in metrics.groupby(
        ["dataset","seed"],sort=True
    ):
        if set(group.method)!=set(METHODS):
            raise RuntimeError(f"Missing method in {dataset} seed {seed}")
    metrics.to_csv(out/"qnn_raw.csv",index=False)
    features.to_csv(out/"feature_raw.csv",index=False)
    rows=[]
    contrasts=(
        ("delta_certified_minus_zero_ba",
         "fusion_certified_mrbi_qnn","fusion_zero_qnn"),
        ("delta_certified_minus_ungated_ba",
         "fusion_certified_mrbi_qnn","fusion_mrbi_qnn"),
        ("delta_ungated_minus_zero_ba",
         "fusion_mrbi_qnn","fusion_zero_qnn"),
        ("delta_certified_minus_pca_ba",
         "fusion_certified_mrbi_qnn","pca_qnn"),
    )
    for (dataset,seed),group in metrics.groupby(
        ["dataset","seed"],sort=True
    ):
        row={"dataset":dataset,"seed":int(seed)}
        for _,record in group.iterrows():
            name=record["method"]
            row[f"{name}_ba"]=float(record["balanced_accuracy"])
            row[f"{name}_test_success"]=float(
                record["feature_test_success"]
            )
        for key,a,b in contrasts:
            row[key]=row[f"{a}_ba"]-row[f"{b}_ba"]
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
        "plan":PLAN,"completed_jobs":len(chunks),
        "requested_jobs":len(jobs),
        "n_qnn_models":len(metrics),
        "prespecified_primary":"paired fusion_certified_mrbi_qnn minus fusion_zero_qnn balanced accuracy, seeds 20--24",
        "prespecified_secondary":[
            "fusion certified minus fusion ungated balanced accuracy",
            "fusion ungated minus fusion zero balanced accuracy",
            "fusion certified minus PCA balanced accuracy (different input dimension; descriptive)",
            "strict root success, feature and QNN runtime, F/J work",
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
        "limitations":"Nine previously explored datasets with new seeds; fusion controls have matched 20D QNN inputs, PCA control has 4D input. All QNNs run on a classical simulator.",
    }
    atomic_write(
        out/"qnn_summary.json",
        json.dumps(summary,indent=2,allow_nan=False)+"\n",
    )
    print(f"Collected {len(chunks)}/{len(jobs)} jobs and "
          f"{len(metrics)} fusion QNN results.",flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets",nargs="+",choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--seeds",nargs="+",type=int,
                        default=list(SEEDS))
    parser.add_argument("--max-new-jobs",type=int,default=None)
    parser.add_argument("--out-dir",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--collect-only",action="store_true")
    args=parser.parse_args()
    if (not args.datasets or len(set(args.datasets))!=len(args.datasets)
            or not args.seeds or len(set(args.seeds))!=len(args.seeds)
            or not set(args.seeds).issubset(SEEDS)
            or args.max_new_jobs is not None and args.max_new_jobs<1):
        raise SystemExit("Frozen fusion QNN: use unique datasets/seeds 20--24")
    jobs=[(ds,seed) for ds in args.datasets for seed in args.seeds]
    current_sha=subprocess.run(
        ["git","rev-parse","HEAD"],cwd=ROOT,check=True,
        capture_output=True,text=True
    ).stdout.strip()
    packages=["numpy","scipy","pandas","scikit-learn"]
    if not args.dry_run:
        if not bench.HAS_QNN:
            raise SystemExit(
                "Missing torch and PennyLane: activate the full Ryzen venv"
            )
        packages.extend(["torch","pennylane"])
    manifest={
        "plan":PLAN,"repo_sha":current_sha,
        "python":sys.version,"platform":platform.platform(),
        "packages":{name:version(name) for name in packages},
        "datasets":args.datasets,"seeds":args.seeds,
        "qnn_config":asdict(replace(
            audit.config_for(20),run_qnn=True
        )),
        "arms":list(METHODS),
        "sigma": {"coarse":0.70},
        "smoothing_weight":0.75,
        "optimizer_caps":{
            "objective_calls":480,"residual_F_calls":5760,
        },
        "sampling":"Full benchmark train and test split; no subsampling of either split beyond fixed max 80 per class",
        "no_profile_search":True,
        "certificate_policy":"Strict root success only. On unsuccessful MRBI root reuse the exact zero feature. Train-only fitted scaling, test labels never enter selection.",
        "source_note":"Fresh seeds 20--24, same nine dataset families; fusion specified after seed-15--19 certified QNN results.",
        "fusion_encoding":"Original PCA 4 coordinates concatenated to train-standardized 16D implicit root; all fusion arms input dimension 20; PCA baseline input dimension 4.",
        "qnn_note":"Classical simulator; PCA input width 4 versus matched fusion width 20; no quantum advantage claim.",
    }
    manifest=json.loads(json.dumps(manifest,allow_nan=False))
    out=args.out_dir.resolve()
    env_path=out/"environment.json"
    if env_path.is_file():
        if json.loads(env_path.read_text(encoding="utf-8"))!=manifest:
            raise SystemExit(
                "QNN environment/plan changed: use a new output directory"
            )
    elif not args.dry_run:
        out.mkdir(parents=True,exist_ok=True)
        atomic_write(
            env_path,
            json.dumps(manifest,indent=2,allow_nan=False)+"\n",
        )
    pending=[]
    for ds,seed in jobs:
        m,f,p=paths_for(out,ds,seed)
        state=[z.is_file() for z in (m,f,p)]
        if all(state):
            record=json.loads(p.read_text(encoding="utf-8"))
            metrics=pd.read_csv(m)
            features=pd.read_csv(f)
            if (record.get("plan")!=PLAN
                    or record.get("repo_sha")!=current_sha
                    or record.get("dataset")!=ds
                    or record.get("seed")!=seed
                    or record.get("metrics_csv_sha256")!=hashlib.sha256(
                        m.read_bytes()
                    ).hexdigest()
                    or record.get("features_csv_sha256")!=hashlib.sha256(
                        f.read_bytes()
                    ).hexdigest()
                    or len(metrics)!=len(METHODS)
                    or set(metrics.method)!=set(METHODS)
                    or metrics.duplicated(
                        ["dataset","seed","method"]
                    ).any()
                    or features.duplicated(
                        ["dataset","seed","split","input_index","method"]
                    ).any()
                    or len(features)!=3*int(
                        (metrics.n_train+metrics.n_test).iloc[0]
                    )):
                raise SystemExit(f"Corrupt/foreign QNN job: {ds} seed {seed}")
        elif any(state):
            raise SystemExit(
                f"Partial QNN job {ds} seed {seed}: inspect before retry"
            )
        else:
            pending.append((ds,seed))
    print(f"Frozen QNN: {len(jobs)} jobs, "
          f"{len(jobs)-len(pending)} complete, "
          f"{len(pending)} pending.",flush=True)
    if args.dry_run:
        return
    if not args.collect_only:
        for ds,seed in pending[:args.max_new_jobs]:
            print(f"Starting {ds} seed={seed}",flush=True)
            metrics, features, info=run_job(ds,seed)
            metrics_text=metrics.to_csv(index=False)
            features_text=features.to_csv(index=False)
            m,f,p=paths_for(out,ds,seed)
            m.parent.mkdir(parents=True,exist_ok=True)
            atomic_write(m,metrics_text)
            atomic_write(f,features_text)
            info.update(
                plan=PLAN,dataset=ds,seed=seed,
                repo_sha=current_sha,
                metrics_csv_sha256=hashlib.sha256(
                    metrics_text.encode("utf-8")
                ).hexdigest(),
                features_csv_sha256=hashlib.sha256(
                    features_text.encode("utf-8")
                ).hexdigest(),
            )
            atomic_write(
                p,json.dumps(info,indent=2,allow_nan=False)+"\n",
            )
            print(f"Finished {ds} seed={seed}: "
                  f"{len(metrics)} QNN methods.",flush=True)
    collect(out,jobs)


if __name__=="__main__":
    main()
