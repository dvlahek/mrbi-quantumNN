"""Four-arm, budget-capped numerical pilot of smoothed-residual MRBI.

No QNN training. The two-by-two design isolates smoothed-vs-plain
residual and descending-vs-repeated-final scale under the same common
antithetic probes. All arms have identical *caps*; realized work is
reported and need not be equal if a solver converges early.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import main_qnn_benchmark as bench
import mrbi
from mrbi_homotopy import SmoothedHomotopyOptimizer
import multiscale_stage_audit as audit
import run_final_sigma_ablation as reference_runner

DATASETS = audit.DATASETS
SEEDS = audit.SEEDS
ARMS = (
    "smoothed_continuation", "smoothed_final_sigma",
    "plain_continuation", "plain_final_sigma",
)
PLAN = "smoothed_residual_pilot_full_reference_v1"
OUT_DEFAULT = ROOT / "outputs" / "smoothed_residual_pilot_v1"


def input_run(layer, x, cfg, roots_cfg, rngs, obj_cap, residual_cap):
    F, J = mrbi.make_residual_and_jacobian(layer)
    zero = mrbi.solve_root(F, J, x, np.zeros(layer.d), cfg=roots_cfg)
    zero_smin, _ = mrbi.jacobian_health(J, x, zero.z_star)
    profiles, stage_rows = {}, []
    for arm in ARMS:
        repeat = arm.endswith("final_sigma")
        smoothed = arm.startswith("smoothed")
        weights = (
            SmoothedHomotopyOptimizer.WEIGHTS if smoothed
            else (0.0, 0.0, 0.0, 0.0, 0.0)
        )
        opt = SmoothedHomotopyOptimizer(
            F, J, x, layer.d, config=replace(
                bench.make_mrbi_cfg(cfg), repeat_final_sigma=repeat
            ), rng=rngs[arm],
            objective_cap_per_stage=obj_cap,
            residual_cap_per_stage=residual_cap,
            weights=weights,
        )
        candidate = opt.optimize()
        stages = opt.stage_records
        if len(stages) != 5:
            raise RuntimeError("Missing an optimizer stage")
        stage_roots = []
        for s in stages:
            z = np.asarray(json.loads(s["candidate_json"]), dtype=float)
            solved = mrbi.solve_root(F, J, x, z, cfg=roots_cfg)
            stage_roots.append(solved)
            record = dict(s)
            del record["candidate_json"]
            record.update(
                arm=arm, stage_root_success=bool(solved.success),
                stage_root_residual=float(solved.residual),
                stage_root_nfev=int(solved.nfev),
            )
            stage_rows.append(record)
        result = stage_roots[-1]
        if not np.array_equal(
            np.asarray(json.loads(stages[-1]["candidate_json"])),
            candidate.z_init,
        ):
            raise RuntimeError("The final diagnostic candidate differs from the optimized candidate")
        smin, _ = mrbi.jacobian_health(J, x, result.z_star)
        used = audit.forced_accept(zero, result, zero_smin, smin)
        accepted = result if used else zero
        obj_actual = sum(int(s["objective_calls_stage"]) for s in stages)
        res_actual = sum(int(s["residual_evals_stage"]) for s in stages)
        if obj_actual + 1 != candidate.n_objective_calls:
            raise RuntimeError("Unexpected objective evaluation accounting")
        profiles[arm] = {
            "candidate_success": int(result.success),
            "accepted_success": int(accepted.success),
            "candidate_root_residual": float(result.residual),
            "accepted_root_residual": float(accepted.residual),
            "used_mrbi": int(used),
            "objective_calls_optimizer": obj_actual,
            "residual_evals_optimizer": res_actual,
            "jacobian_evals_optimizer": sum(
                int(s["jacobian_evals_stage"]) for s in stages
            ),
            "budget_hit_stages": sum(int(s["budget_hit"]) for s in stages),
            "candidate": candidate.z_init,
            "candidate_root": result.z_star,
        }
    hashes = [stage_rows[i*5]["probe_sha256"] for i in range(len(ARMS))]
    if len(set(hashes)) != 1:
        raise RuntimeError("Paired arms did not share the same antithetic probe sample")
    if any(len({stage_rows[i*5+j]["probe_sha256"] for j in range(5)}) != 1
           for i in range(len(ARMS))):
        raise RuntimeError("Probes changed inside one arm")
    pair = {
        "zero_success": int(zero.success),
        "zero_residual": float(zero.residual),
        "same_probe_sha256": hashes[0],
    }
    for arm, p in profiles.items():
        for field, value in p.items():
            if field not in ("candidate", "candidate_root"):
                pair[f"{arm}_{field}"] = value
    primary = profiles["smoothed_continuation"]
    final = profiles["smoothed_final_sigma"]
    plain = profiles["plain_continuation"]
    pair.update({
        "delta_smoothed_cont_minus_final_candidate_success":
            primary["candidate_success"]-final["candidate_success"],
        "delta_smoothed_cont_minus_final_accepted_success":
            primary["accepted_success"]-final["accepted_success"],
        "delta_smoothed_cont_minus_final_candidate_residual":
            primary["candidate_root_residual"]-final["candidate_root_residual"],
        "delta_smoothed_cont_minus_plain_cont_candidate_success":
            primary["candidate_success"]-plain["candidate_success"],
        "smoothed_cont_minus_final_candidate_distance": float(
            np.linalg.norm(primary["candidate"]-final["candidate"])
        ),
        "smoothed_cont_minus_plain_cont_candidate_distance": float(
            np.linalg.norm(primary["candidate"]-plain["candidate"])
        ),
    })
    return stage_rows, pair


def run_job(dataset, seed, n, obj_cap, residual_cap):
    cfg = audit.config_for(seed)
    X, y = bench.load_dataset(dataset, cfg)
    Xtr, Xte, _ytr, _yte = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=seed
    )
    Xtr_p, _Xte_p = bench.preprocess_to_pca(Xtr, Xte, cfg.pca_dim, seed)
    if n > len(Xtr_p):
        raise ValueError(f"Requested {n} inputs; only {len(Xtr_p)} are available")
    layer = bench.make_random_implicit_layer(
        dx=Xtr_p.shape[1], d=cfg.latent_dim, cfg=cfg,
        seed=seed+1000+cfg.latent_dim+int(100*cfg.spectral_radius),
    )
    rngs = {arm: np.random.default_rng(seed+2000) for arm in ARMS}
    stages, pairs = [], []
    root_cfg = bench.make_root_cfg(cfg)
    for index, x in enumerate(Xtr_p[:n]):
        sample_stages, sample_pair = input_run(
            layer, x, cfg, root_cfg, rngs, obj_cap, residual_cap
        )
        for row in sample_stages:
            row.update(dataset=dataset, seed=seed,
                       train_index=index, plan=PLAN)
        sample_pair.update(dataset=dataset, seed=seed,
                           train_index=index, plan=PLAN)
        stages.extend(sample_stages)
        pairs.append(sample_pair)
    return pd.DataFrame(stages), pd.DataFrame(pairs)


def job_paths(out, ds, seed):
    stem = out / "jobs" / f"{ds}_seed{seed}"
    return (Path(str(stem)+"_stages.csv"),
            Path(str(stem)+"_pairs.csv"),
            Path(str(stem)+"_config.json"))


def collect(out, jobs, n):
    stage_parts, pair_parts = [], []
    for ds, seed in jobs:
        stage_path, pair_path, config_path = job_paths(out, ds, seed)
        if all(p.is_file() for p in (stage_path, pair_path, config_path)):
            stage_parts.append(pd.read_csv(stage_path))
            pair_parts.append(pd.read_csv(pair_path))
    if not pair_parts:
        print("No complete pilot jobs to collect.", flush=True)
        return
    stages = pd.concat(stage_parts, ignore_index=True)
    pairs = pd.concat(pair_parts, ignore_index=True)
    if (len(stages) != len(pair_parts)*n*len(ARMS)*5
            or len(pairs) != len(pair_parts)*n):
        raise SystemExit("Aggregated pilot row count mismatch")
    stages.to_csv(out/"stage_raw.csv", index=False)
    pairs.to_csv(out/"paired_sample_results.csv", index=False)
    stage_summary = stages.groupby(
        ["dataset", "arm", "stage"], as_index=False
    ).agg(
        mean_residual=("residual_norm", "mean"),
        mean_smoothed_residual=("smoothed_residual_norm", "mean"),
        mean_step=("candidate_step_norm", "mean"),
        mean_objective_calls=("objective_calls_stage", "mean"),
        mean_residual_evals=("residual_evals_stage", "mean"),
        stage_root_success=("stage_root_success", "mean"),
        budget_hit_rate=("budget_hit", "mean"),
    )
    stage_summary.to_csv(out/"stage_summary.csv", index=False)
    metric = [
        f"{arm}_{field}" for arm in ARMS for field in (
            "candidate_success", "accepted_success",
            "objective_calls_optimizer", "residual_evals_optimizer",
            "budget_hit_stages",
        )
    ]
    aggregate = {
        col: (col, "mean") for col in metric
    }
    aggregate.update(
        samples=("train_index", "count"),
        seeds=("seed", "nunique"),
        mean_delta_success=(
            "delta_smoothed_cont_minus_final_candidate_success", "mean"
        ),
    )
    datasets = pairs.groupby("dataset", as_index=False).agg(**aggregate)
    datasets.to_csv(out/"dataset_summary.csv", index=False)
    print(
        f"Collected {len(pair_parts)}/{len(jobs)} jobs, "
        f"{len(pairs)} paired inputs. No QNN trained.", flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-raw", required=True, type=Path)
    parser.add_argument("--reference-env", nargs="+", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--samples-per-job", type=int, default=8)
    parser.add_argument("--objective-cap-per-stage", type=int, default=480)
    parser.add_argument("--residual-cap-per-stage", type=int, default=5760)
    parser.add_argument("--max-new-jobs", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or any(s < 0 for s in args.seeds)
            or len(set(args.datasets)) != len(args.datasets)
            or args.samples_per_job < 1
            or args.objective_cap_per_stage < 18
            or args.residual_cap_per_stage < 216
            or args.max_new_jobs is not None and args.max_new_jobs < 1):
        raise SystemExit("Invalid pilot datasets, seeds or budget.")
    jobs = [(ds, seed) for ds in args.datasets for seed in args.seeds]
    ref_path = args.reference_raw.resolve()
    sha, design = reference_runner.source_reference(ref_path, jobs)
    if design != "full_corrected_continuation_v1":
        raise SystemExit("The reference must be the full corrected-continuation campaign")
    env_paths = [p.resolve() for p in args.reference_env]
    if len(set(env_paths)) != len(env_paths) or any(not p.is_file() for p in env_paths):
        raise SystemExit("Duplicate/missing reference shard environment file")
    envs = [json.loads(p.read_text(encoding="utf-8")) for p in env_paths]
    packages = reference_runner.snapshot()["packages"]
    commits = {e.get("git_commit") for e in envs}
    if (len(commits) != 1 or not next(iter(commits))
            or any(e.get("packages") != packages
                   or e.get("implementation_version") != bench.IMPLEMENTATION_VERSION
                   for e in envs)):
        raise SystemExit("Source SHA or Python package versions differ")
    covered = set()
    for e in envs:
        for ds in e["datasets"]:
            for seed in e["seeds"]:
                key = (ds, int(seed))
                if key in covered:
                    raise SystemExit("Overlapping source shard manifests")
                covered.add(key)
    if not set(jobs).issubset(covered):
        raise SystemExit("Reference shards do not cover the requested jobs")

    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    manifest = {
        "design": PLAN, "source_raw_sha256": sha,
        "source_git_commit": next(iter(commits)),
        "source_environments": [str(p) for p in env_paths],
        "pilot_git_commit": git, "packages": packages,
        "datasets": args.datasets, "seeds": args.seeds,
        "samples_per_job": args.samples_per_job,
        "objective_cap_per_stage": args.objective_cap_per_stage,
        "residual_cap_per_stage": args.residual_cap_per_stage,
        "weights": list(SmoothedHomotopyOptimizer.WEIGHTS),
        "arms": list(ARMS), "no_qnn_training": True,
        "sampling": "First N training rows after benchmark split/PCA",
        "budget_note": "Equal objective and primitive-residual ceilings, not equal realized evaluation counts.",
    }
    out = args.out_dir.resolve()
    env_path = out/"environment.json"
    if env_path.is_file():
        if json.loads(env_path.read_text(encoding="utf-8")) != manifest:
            raise SystemExit("Pilot provenance changed; use a fresh output directory")
    elif not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        env_path.write_text(json.dumps(manifest, indent=2)+"\n",
                            encoding="utf-8")
    pending = []
    for ds, seed in jobs:
        stage, pair, meta = job_paths(out, ds, seed)
        found = [p.is_file() for p in (stage, pair, meta)]
        if all(found):
            record = json.loads(meta.read_text(encoding="utf-8"))
            if (record.get("pilot_git_commit") != git
                    or record.get("source_raw_sha256") != sha
                    or len(pd.read_csv(stage)) != args.samples_per_job*len(ARMS)*5
                    or len(pd.read_csv(pair)) != args.samples_per_job):
                raise SystemExit(f"Invalid completed pilot job: {ds} seed {seed}")
        elif any(found):
            raise SystemExit(f"Partial pilot job {ds} seed {seed}; inspect before retrying")
        else:
            pending.append((ds, seed))
    print(f"Homotopy pilot: {len(jobs)} jobs, "
          f"{len(jobs)-len(pending)} complete, {len(pending)} pending.",
          flush=True)
    if args.dry_run:
        return
    if not args.collect_only:
        for ds, seed in pending[:args.max_new_jobs]:
            print(f"Starting {ds} seed={seed}", flush=True)
            stage_df, pair_df = run_job(
                ds, seed, args.samples_per_job,
                args.objective_cap_per_stage,
                args.residual_cap_per_stage,
            )
            stage_path, pair_path, meta_path = job_paths(out, ds, seed)
            stage_path.parent.mkdir(parents=True, exist_ok=True)
            stage_df.to_csv(stage_path, index=False)
            pair_df.to_csv(pair_path, index=False)
            meta_path.write_text(json.dumps({
                "dataset": ds, "seed": seed,
                "pilot_git_commit": git, "source_raw_sha256": sha,
            }, indent=2)+"\n", encoding="utf-8")
            print(f"Finished {ds} seed={seed}: {len(pair_df)} paired inputs.",
                  flush=True)
    collect(out, jobs, args.samples_per_job)


if __name__ == "__main__":
    main()
