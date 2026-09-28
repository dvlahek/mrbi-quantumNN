"""Paired, solver-aware stage-one checkpoint policy for smoothed MRBI.

Exploratory numerical development: no QNN, no use of test labels. The
policy checks a candidate immediately after stage one if zero-root failed
and stops if that checkpoint root converges. Exactly the same policy is
applied to the three controls. All root F/J evaluations are counted.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import mrbi
import main_qnn_benchmark as bench
from mrbi_homotopy import SmoothedHomotopyOptimizer
import multiscale_stage_audit as audit
import run_final_sigma_ablation as source_runner

ARMS = (
    "smoothed_continuation", "smoothed_final_sigma",
    "plain_continuation", "plain_final_sigma",
)
DATASETS = audit.DATASETS
SEEDS = audit.SEEDS
PLAN = "stage1_checkpoint_smoothed_full_reference_v1"
DEFAULT_OUT = ROOT / "outputs" / "checkpoint_homotopy_v1"


class CheckpointSuccess(Exception):
    """End MRBI optimization after a successful stage-one root solve."""

    def __init__(self, z):
        super().__init__("Successful stage-one checkpoint")
        self.z = np.asarray(z, dtype=np.float64).copy()


def counted_functions(layer):
    """Return numerically identical F/J callables with primitive counters."""
    totals = {"F": 0, "J": 0}
    F, J = mrbi.make_residual_and_jacobian(layer)

    def tracked_F(z, x):
        totals["F"] += 1
        return F(z, x)

    def tracked_J(z, x):
        totals["J"] += 1
        return J(z, x)

    return tracked_F, tracked_J, totals


def root_from(F, J, totals, x, z, config):
    before_f, before_j = totals["F"], totals["J"]
    outcome = mrbi.solve_root(F, J, x, z, cfg=config)
    return outcome, int(totals["F"]-before_f), int(totals["J"]-before_j)


def run_arm(layer, x, cfg, root_cfg, zero, zero_f, zero_j, seed, index,
            arm, objective_cap, residual_cap):
    """A fixed policy: zero solve, stage-one root check, then fallback if needed."""
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    start_sec = time.perf_counter()
    if zero.success:
        return [], {
            "arm": arm, "zero_success": 1, "candidate_attempted": 0,
            "checkpoint_attempted": 0, "checkpoint_success": 0,
            "final_root_attempted": 0, "final_root_success": 0,
            "accepted_success": 1, "accepted_residual": float(zero.residual),
            "used_mrbi": 0, "early_exit": 0,
            "optimizer_objective_calls": 0,
            "optimizer_residual_evals": 0,
            "optimizer_jacobian_evals": 0,
            "optimizer_stages": 0,
            "root_attempts": 1,
            "root_F_calls": int(zero_f), "root_J_calls": int(zero_j),
            "extra_root_F_calls": 0, "extra_root_J_calls": 0,
            "total_F_calls": int(zero_f), "total_J_calls": int(zero_j),
            "budget_hit_stages": 0,
            "candidate_root_residual": np.nan,
            "checkpoint_candidate_json": "",
            "total_wall_sec": float(zero.runtime_sec),
        }

    F, J, counts = counted_functions(layer)
    local = replace(
        bench.make_mrbi_cfg(cfg), repeat_final_sigma=arm.endswith("final_sigma")
    )
    weights = (SmoothedHomotopyOptimizer.WEIGHTS if arm.startswith("smoothed")
               else (0.0,) * 5)
    # Per-input seeds prevent one arm's early exit from shifting later samples.
    rng = np.random.default_rng(int(seed) + 2000 + 1000003 * int(index))
    opt = SmoothedHomotopyOptimizer(
        F, J, x, layer.d, config=local, rng=rng,
        objective_cap_per_stage=objective_cap,
        residual_cap_per_stage=residual_cap,
        weights=weights,
    )
    stage_fn = opt._optimize_single_scale
    checkpoint_root = None
    checkpoint_f = checkpoint_j = 0

    def checkpoint_stage(z0, sigma, maxiter, *, detector_override=None):
        nonlocal checkpoint_root, checkpoint_f, checkpoint_j
        z = stage_fn(z0, sigma, maxiter,
                     detector_override=detector_override)
        if len(opt.stage_records) == 1:
            checkpoint_root, checkpoint_f, checkpoint_j = root_from(
                F, J, counts, x, z, root_cfg
            )
            if checkpoint_root.success:
                raise CheckpointSuccess(z)
        return z

    opt._optimize_single_scale = checkpoint_stage
    early = False
    final_root = None
    final_f = final_j = 0
    try:
        candidate = opt.optimize()
        z_final = candidate.z_init
        final_root, final_f, final_j = root_from(
            F, J, counts, x, z_final, root_cfg
        )
        accepted_root = final_root
    except CheckpointSuccess as stopped:
        early = True
        z_final = stopped.z
        accepted_root = checkpoint_root

    if checkpoint_root is None or len(opt.stage_records) not in (1, 5):
        raise RuntimeError("Checkpoint instrumentation did not capture stage one")
    if early != bool(checkpoint_root.success):
        raise RuntimeError("Failed to stop at a successful checkpoint")
    if not early and final_root is None:
        raise RuntimeError("Missing final root after checkpoint failure")

    checkpoint_probe = opt.stage_records[0]["probe_sha256"]
    checkpoint_candidate = opt.stage_records[0]["candidate_json"]
    records = []
    for stage in opt.stage_records:
        row = dict(stage)
        row.pop("candidate_json", None)
        row.update(
            arm=arm,
            stage_root_checked=bool(int(stage["stage"]) == 1 or (
                not early and int(stage["stage"]) == 5
            )),
            stage_root_success=(int(checkpoint_root.success)
                if int(stage["stage"]) == 1 else
                int(final_root.success) if final_root is not None
                   and int(stage["stage"]) == 5 else -1),
        )
        records.append(row)
    final_smin, _ = mrbi.jacobian_health(J, x, accepted_root.z_star)
    zero_smin, _ = mrbi.jacobian_health(J, x, zero.z_star)
    used = audit.forced_accept(zero, accepted_root, zero_smin, final_smin)
    accepted = accepted_root if used else zero
    optimizer_calls = int(opt.objective_calls)
    root_calls = int(2 + (0 if early else 1))
    optimizer_r = int(opt.optimizer_residual_evaluations)
    optimizer_j = int(opt.optimizer_jacobian_evaluations)
    if optimizer_calls != sum(r["objective_calls_stage"] for r in opt.stage_records) + (0 if early else 1):
        raise RuntimeError("Objective call totals are inconsistent")
    info = {
        "arm": arm, "zero_success": int(zero.success),
        "candidate_attempted": 1,
        "checkpoint_attempted": 1,
        "checkpoint_success": int(checkpoint_root.success),
        "final_root_attempted": int(not early),
        "final_root_success": int(final_root.success) if final_root else 0,
        "accepted_success": int(accepted.success),
        "accepted_residual": float(accepted.residual),
        "used_mrbi": int(used), "early_exit": int(early),
        "optimizer_objective_calls": optimizer_calls,
        "optimizer_residual_evals": optimizer_r,
        "optimizer_jacobian_evals": optimizer_j,
        "optimizer_stages": len(opt.stage_records),
        "root_attempts": root_calls,
        "root_F_calls": int(zero_f+checkpoint_f+final_f),
        "root_J_calls": int(zero_j+checkpoint_j+final_j),
        "extra_root_F_calls": int(checkpoint_f+final_f),
        "extra_root_J_calls": int(checkpoint_j+final_j),
        "total_F_calls": int(zero_f+counts["F"]),
        "total_J_calls": int(zero_j+counts["J"]),
        "budget_hit_stages": sum(int(r["budget_hit"]) for r in opt.stage_records),
        "candidate_root_residual": float(accepted_root.residual),
        "checkpoint_candidate_json": checkpoint_candidate,
        "probe_sha256": checkpoint_probe,
        "total_wall_sec": float(zero.runtime_sec + time.perf_counter()-start_sec),
    }
    return records, info


def run_job(dataset, seed, n, objective_cap, residual_cap):
    cfg = audit.config_for(seed)
    X, y = bench.load_dataset(dataset, cfg)
    Xtr, Xte, _ytr, _yte = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=seed
    )
    Xtr_p, _Xte_p = bench.preprocess_to_pca(Xtr, Xte, cfg.pca_dim, seed)
    if n > len(Xtr_p):
        raise ValueError(f"Only {len(Xtr_p)} training inputs available")
    layer = bench.make_random_implicit_layer(
        dx=Xtr_p.shape[1], d=cfg.latent_dim, cfg=cfg,
        seed=seed+1000+cfg.latent_dim+int(100*cfg.spectral_radius),
    )
    root_cfg = bench.make_root_cfg(cfg)
    stages, samples = [], []
    for index, x in enumerate(Xtr_p[:n]):
        zF, zJ, zero_counts = counted_functions(layer)
        zero, zero_f, zero_j = root_from(
            zF, zJ, zero_counts, x, np.zeros(layer.d), root_cfg
        )
        row = {
            "dataset": dataset, "seed": seed, "train_index": index,
            "plan": PLAN, "zero_success": int(zero.success),
        }
        first_probes = set()
        for arm in ARMS:
            detail, outcome = run_arm(
                layer, x, cfg, root_cfg, zero, zero_f, zero_j,
                seed, index, arm, objective_cap, residual_cap,
            )
            if detail:
                first_probes.add(detail[0]["probe_sha256"])
            for item in detail:
                item.update(
                    dataset=dataset, seed=seed,
                    train_index=index, plan=PLAN,
                )
                stages.append(item)
            for key, val in outcome.items():
                if key != "arm":
                    row[f"{arm}_{key}"] = val
        if len(first_probes) > 1:
            raise RuntimeError("Checkpoint arms did not use the same probes")
        samples.append(row)
    if not stages:
        stages = pd.DataFrame(columns=[
            "dataset", "seed", "train_index", "plan", "arm", "stage",
            "stage_root_checked", "stage_root_success", "probe_sha256",
        ])
    else:
        stages = pd.DataFrame(stages)
    return stages, pd.DataFrame(samples)


def job_paths(out, ds, seed):
    stem = out / "jobs" / f"{ds}_seed{seed}"
    return (Path(str(stem)+"_stages.csv"),
            Path(str(stem)+"_pairs.csv"),
            Path(str(stem)+"_config.json"))


def collect(out, jobs, n):
    stage_parts, pair_parts = [], []
    for ds, seed in jobs:
        stage, pair, meta = job_paths(out, ds, seed)
        if all(x.is_file() for x in (stage, pair, meta)):
            stage_parts.append(pd.read_csv(stage))
            pair_parts.append(pd.read_csv(pair))
    if not pair_parts:
        print("No complete checkpoint jobs collected", flush=True)
        return
    pairs = pd.concat(pair_parts, ignore_index=True)
    stages = pd.concat(stage_parts, ignore_index=True)
    if len(pairs) != n*len(pair_parts):
        raise RuntimeError("Incomplete paired-sample rows")
    pairs.to_csv(out/"paired_sample_results.csv", index=False)
    stages.to_csv(out/"stage_raw.csv", index=False)
    rows = []
    for ds, sub in pairs.groupby("dataset", sort=True):
        zeros = sub.zero_success.eq(0)
        item = {
            "dataset": ds, "n_inputs": len(sub), "n_zero_failed": int(zeros.sum()),
            "n_seeds": sub.seed.nunique(),
        }
        for arm in ARMS:
            item[f"{arm}_accepted_success"] = int(
                sub[f"{arm}_accepted_success"].sum()
            )
            item[f"{arm}_checkpoint_rescue"] = int(
                (sub[f"{arm}_checkpoint_success"].eq(1)&zeros).sum()
            )
            for metric in ("optimizer_objective_calls", "total_F_calls",
                           "total_J_calls", "root_attempts", "total_wall_sec"):
                item[f"{arm}_mean_{metric}"] = float(
                    sub[f"{arm}_{metric}"].mean()
                )
        rows.append(item)
    pd.DataFrame(rows).to_csv(out/"dataset_summary.csv", index=False)
    summary = {
        "plan": PLAN, "n_complete_jobs": len(pair_parts), "n_paired_inputs": len(pairs),
        "n_zero_fail": int(pairs.zero_success.eq(0).sum()),
        "note": "Exploratory stage-one checkpoint; four matched policies and fully counted zero/checkpoint/final root calls. Equal per-stage caps, not equal realized evaluations.",
        "arms": {
            arm: {
                "accepted_success": int(pairs[f"{arm}_accepted_success"].sum()),
                "checkpoint_rescue_zero_failed": int(
                    ((pairs.zero_success.eq(0))&
                    pairs[f"{arm}_checkpoint_success"].eq(1)).sum()
                ),
                "mean_optimizer_objective_calls": float(
                    pairs[f"{arm}_optimizer_objective_calls"].mean()
                ),
                "mean_total_F_calls": float(pairs[f"{arm}_total_F_calls"].mean()),
                "mean_total_J_calls": float(pairs[f"{arm}_total_J_calls"].mean()),
                "mean_root_attempts": float(pairs[f"{arm}_root_attempts"].mean()),
                "mean_total_wall_sec": float(pairs[f"{arm}_total_wall_sec"].mean()),
            } for arm in ARMS
        },
    }
    (out/"checkpoint_statistics.json").write_text(
        json.dumps(summary, indent=2)+"\n", encoding="utf-8"
    )
    print(f"Collected {len(pair_parts)}/{len(jobs)} jobs, "
          f"{len(pairs)} inputs, {summary['n_zero_fail']} zero-solve failures.",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-raw", required=True, type=Path)
    parser.add_argument("--reference-env", required=True, type=Path, nargs="+")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--samples-per-job", type=int, default=8)
    parser.add_argument("--objective-cap-per-stage", type=int, default=480)
    parser.add_argument("--residual-cap-per-stage", type=int, default=5760)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-new-jobs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    if (not args.seeds or len(args.seeds) != len(set(args.seeds))
            or len(args.datasets) != len(set(args.datasets))
            or any(seed < 0 for seed in args.seeds)
            or args.samples_per_job < 1
            or args.max_new_jobs is not None and args.max_new_jobs < 1):
        raise SystemExit("Invalid datasets/seeds/sample count/job limit")
    jobs = [(ds, seed) for ds in args.datasets for seed in args.seeds]
    ref = args.reference_raw.resolve()
    digest, source = source_runner.source_reference(ref, jobs)
    if source != "full_corrected_continuation_v1":
        raise SystemExit("Only full corrected-continuation source supported")
    env_paths = [path.resolve() for path in args.reference_env]
    if (len(set(env_paths)) != len(env_paths)
            or any(not e.is_file() for e in env_paths)):
        raise SystemExit("Missing or duplicate source environment")
    envs = [json.loads(path.read_text(encoding="utf-8")) for path in env_paths]
    commits = {item.get("git_commit") for item in envs}
    packages = source_runner.snapshot()["packages"]
    if (len(commits) != 1 or not next(iter(commits))
            or any(item.get("packages") != packages
                   or item.get("implementation_version") != bench.IMPLEMENTATION_VERSION
                   for item in envs)):
        raise SystemExit("Source Git commit or package versions mismatch")
    covered = set()
    for item in envs:
        for ds in item["datasets"]:
            for seed in item["seeds"]:
                key = (ds, int(seed))
                if key in covered:
                    raise SystemExit("Overlapping source shard manifests")
                covered.add(key)
    if not set(jobs).issubset(covered):
        raise SystemExit("Source shard manifests do not cover requested jobs")
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, cwd=ROOT,
        capture_output=True, text=True,
    ).stdout.strip()
    manifest = {
        "plan": PLAN, "source_raw_sha256": digest,
        "source_git_commit": next(iter(commits)),
        "source_environments": [str(e) for e in env_paths],
        "checkpoint_git_commit": git,
        "packages": packages, "datasets": args.datasets,
        "seeds": args.seeds, "samples_per_job": args.samples_per_job,
        "objective_cap_per_stage": args.objective_cap_per_stage,
        "residual_cap_per_stage": args.residual_cap_per_stage,
        "sampling": "First N training samples from full benchmark split/PCA",
        "probe_seed_rule": "seed + 2000 + 1000003 * train_index, identical for all four arms",
        "policy": "Zero solve; if failed, stage-one root check; early accept on strict root success; otherwise complete remaining stages and root check",
        "no_qnn_training": True,
    }
    out = args.out_dir.resolve()
    out_manifest = out/"environment.json"
    if out_manifest.is_file():
        if json.loads(out_manifest.read_text(encoding="utf-8")) != manifest:
            raise SystemExit("Checkpoint provenance differs; use a new output dir")
    elif not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        out_manifest.write_text(json.dumps(manifest, indent=2)+"\n",
                                encoding="utf-8")
    pending = []
    for ds, seed in jobs:
        stage, pair, cfg = job_paths(out, ds, seed)
        present = [x.is_file() for x in (stage, pair, cfg)]
        if all(present):
            meta = json.loads(cfg.read_text(encoding="utf-8"))
            if (meta.get("checkpoint_git_commit") != git
                    or meta.get("source_raw_sha256") != digest
                    or len(pd.read_csv(pair)) != args.samples_per_job):
                raise SystemExit(f"Incomplete/foreign checkpoint job: {ds} seed={seed}")
        elif any(present):
            raise SystemExit(f"Partial checkpoint job: {ds} seed={seed}")
        else:
            pending.append((ds, seed))
    print(f"Checkpoint pilot: {len(jobs)} jobs, "
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
            stage, pair, meta = job_paths(out, ds, seed)
            stage.parent.mkdir(parents=True, exist_ok=True)
            stage_df.to_csv(stage, index=False)
            pair_df.to_csv(pair, index=False)
            meta.write_text(json.dumps({
                "dataset": ds, "seed": seed,
                "checkpoint_git_commit": git,
                "source_raw_sha256": digest,
            }, indent=2)+"\n", encoding="utf-8")
            print(f"Finished {ds} seed={seed}: {len(pair_df)} inputs, "
                  f"{len(stage_df)} stage rows.", flush=True)
    collect(out, jobs, args.samples_per_job)


if __name__ == "__main__":
    main()
