"""Frozen four-method QNN experiment on new seeds 10--14.

Primary: one-stage smoothed-coarse QNN vs one-stage smoothed-fine
QNN, with zero-QNN and PCA-QNN as fixed baselines. All quantum
circuits are classically simulated; the output cannot establish
quantum advantage. Source seeds 0--9 must not be used to select
profiles or re-estimate a best-per-dataset variant.
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
SEEDS = (10, 11, 12, 13, 14)
METHODS = ("pca_qnn", "zero_qnn", "smoothed_coarse_qnn",
           "smoothed_fine_qnn")
ARMS = ("smoothed_coarse", "smoothed_fine")
PLAN = "frozen_one_stage_qnn_seeds10to14_v1"
DEFAULT_OUT = ROOT / "outputs" / "frozen_one_stage_qnn_v1"


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
            "split": split, "input_index": j, "method": "zero_qnn",
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
                "method": f"{arm}_qnn",
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
            raise RuntimeError("Coarse and fine do not share exact probes")
    result = {name: np.asarray(rows, dtype=np.float64)
              for name, rows in outputs.items()}
    for name, z in result.items():
        if z.shape != (len(inputs), layer.d) or not np.all(np.isfinite(z)):
            raise RuntimeError(f"Invalid numerical feature array: {name}")
    return result, pd.DataFrame(records)


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
            "profile": "full_balanced_single_stage",
            **measures, **costs,
        })
    return pd.DataFrame(rows)


def run_job(dataset, seed):
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError("QNN experiment frozen to nine tasks, seeds 10--14")
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
    print(f"[{dataset} seed={seed}] Generating frozen test features",
          flush=True)
    test_z, test_stats = numerical_features(
        Xte_p, layer, cfg, seed, "test", len(Xtr_p)
    )
    feature_arrays = {"pca_qnn": (Xtr_p, Xte_p)}
    for source, method in (
        ("zero", "zero_qnn"),
        ("smoothed_coarse", "smoothed_coarse_qnn"),
        ("smoothed_fine", "smoothed_fine_qnn"),
    ):
        ztr, zte = bench.standardize_pair(
            train_z[source], test_z[source]
        )
        feature_arrays[method] = (ztr, zte)
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


def collect(out, jobs):
    chunks, fchunks=load_complete_jobs(out,jobs)
    if not chunks:
        print("No completed frozen QNN jobs.",flush=True)
        return
    metrics=pd.concat(chunks,ignore_index=True)
    features=pd.concat(fchunks,ignore_index=True)
    if (metrics.duplicated(["dataset","seed","method"]).any()
            or features.duplicated(
                ["dataset","seed","split","input_index","method"]
            ).any()
            or len(metrics)!=len(chunks)*len(METHODS)):
        raise RuntimeError("Aggregate QNN keys/coverage invalid")
    metrics.to_csv(out/"qnn_raw.csv",index=False)
    features.to_csv(out/"feature_raw.csv",index=False)
    rows=[]
    for (dataset, seed), group in metrics.groupby(
            ["dataset","seed"],sort=True
    ):
        if set(group.method)!=set(METHODS):
            raise RuntimeError("Missing required method in job")
        row={"dataset":dataset,"seed":int(seed)}
        for _,record in group.iterrows():
            name=record["method"]
            row[f"{name}_ba"]=float(record["balanced_accuracy"])
            row[f"{name}_test_success"]=float(
                record["feature_test_success"]
            )
        row["delta_coarse_minus_fine_ba"]=(
            row["smoothed_coarse_qnn_ba"]-row["smoothed_fine_qnn_ba"]
        )
        row["delta_coarse_minus_zero_ba"]=(
            row["smoothed_coarse_qnn_ba"]-row["zero_qnn_ba"]
        )
        row["delta_coarse_minus_pca_ba"]=(
            row["smoothed_coarse_qnn_ba"]-row["pca_qnn_ba"]
        )
        rows.append(row)
    paired=pd.DataFrame(rows)
    paired.to_csv(out/"paired_job_results.csv",index=False)
    datasets=paired.groupby("dataset",as_index=False).agg(
        n_seeds=("seed","nunique"),
        mean_coarse_minus_fine_ba=("delta_coarse_minus_fine_ba","mean"),
        mean_coarse_minus_zero_ba=("delta_coarse_minus_zero_ba","mean"),
        mean_coarse_minus_pca_ba=("delta_coarse_minus_pca_ba","mean"),
        mean_coarse_ba=("smoothed_coarse_qnn_ba","mean"),
        mean_fine_ba=("smoothed_fine_qnn_ba","mean"),
        mean_zero_ba=("zero_qnn_ba","mean"),
        mean_pca_ba=("pca_qnn_ba","mean"),
    )
    datasets.to_csv(out/"dataset_summary.csv",index=False)
    summary={
        "plan":PLAN,"completed_jobs":len(chunks),
        "requested_jobs":len(jobs),
        "n_qnn_models":len(metrics),
        "prespecified_primary":"paired smoothed_coarse_qnn minus smoothed_fine_qnn balanced accuracy",
        "prespecified_secondary":[
            "paired smoothed_coarse_qnn minus zero_qnn balanced accuracy",
            "paired smoothed_coarse_qnn minus pca_qnn balanced accuracy",
            "paired smoothed coarse-minus-fine test root success and measured F/J work",
        ],
        "means_by_method":{
            m:{
                "balanced_accuracy":float(
                    metrics.loc[metrics.method.eq(m),"balanced_accuracy"].mean()
                ),
                "training_time_sec":float(
                    metrics.loc[metrics.method.eq(m),"train_time_sec"].mean()
                ),
            } for m in METHODS
        },
        "mean_paired_deltas":{
            col:float(paired[col].mean()) for col in (
                "delta_coarse_minus_fine_ba",
                "delta_coarse_minus_zero_ba",
                "delta_coarse_minus_pca_ba",
            )
        },
        "interpretation":"Same nine datasets; new seeds 10--14, no independent dataset validation or quantum advantage claim.",
    }
    atomic_write(
        out/"qnn_summary.json",
        json.dumps(summary,indent=2,allow_nan=False)+"\n",
    )
    print(f"Collected {len(chunks)}/{len(jobs)} jobs "
          f"and {len(metrics)} QNN results.",flush=True)


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
        raise SystemExit("Frozen QNN: use unique datasets/seeds 10--14")
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
            audit.config_for(10),run_qnn=True
        )),
        "arms":list(METHODS),
        "sigma": {"coarse":0.70,"fine":0.02},
        "smoothing_weight":0.75,
        "optimizer_caps":{
            "objective_calls":480,"residual_F_calls":5760,
        },
        "sampling":"Full benchmark train and test split; no subsampling of either split beyond fixed max 80 per class",
        "no_profile_search":True,
        "source_note":"Fresh seeds 10--14 independently generated; prior full raw covers seeds 0--4 only",
        "qnn_note":"Classical simulator; input-projection shape differs for PCA (4) versus implicit features (16).",
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
                    or len(metrics)!=4
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
