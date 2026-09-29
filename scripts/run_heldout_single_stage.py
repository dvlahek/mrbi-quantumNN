"""Prespecified, label-free single-stage MRBI check on fresh seeds 5--9.

Four paired arms: smoothed/plain residual x coarse/fine sigma. Run zero
root first; if it fails, run ONE capped L-BFGS-B stage and ONE root
solve from that stage's candidate. No later continuation, QNN or labels
in candidate selection. The old full-campaign raw file is not used:
it has no rows for seeds 5--9.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
import mrbi
import main_qnn_benchmark as bench
from mrbi_homotopy import SmoothedHomotopyOptimizer
import multiscale_stage_audit as stage_audit

DATASETS = stage_audit.DATASETS
SEEDS = (5, 6, 7, 8, 9)  # Frozen before looking at these outcomes.
ARMS = (
    "smoothed_coarse", "smoothed_fine", "plain_coarse", "plain_fine"
)
ARM_SPECS = {
    "smoothed_coarse": (0.70, 0.75),
    "smoothed_fine": (0.02, 0.75),
    "plain_coarse": (0.70, 0.0),
    "plain_fine": (0.02, 0.0),
}
PLAN = "heldout_single_stage_sigma_070_vs_002_v1"
DEFAULT_OUT = ROOT / "outputs" / "heldout_single_stage_v1"
STAGE_FIELDS = [
    "dataset", "seed", "train_index", "plan", "arm", "stage",
    "sigma", "smooth_weight", "objective_calls_stage",
    "residual_evals_stage", "jacobian_evals_stage",
    "candidate_step_norm", "residual_norm",
    "smoothed_residual_norm", "weighted_residual_term",
    "jacobian_sigma_min", "optimizer_success", "optimizer_nit",
    "budget_hit", "probe_sha256", "stage_root_success",
    "stage_root_residual", "stage_root_F_calls", "stage_root_J_calls",
]


def sha_arrays(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array, dtype=np.float64)
        h.update(json.dumps(a.shape).encode("ascii"))
        h.update(a.tobytes())
    return h.hexdigest()


def counted_functions(layer):
    F, J = mrbi.make_residual_and_jacobian(layer)
    count = {"F": 0, "J": 0}

    def f(z, x):
        count["F"] += 1
        return F(z, x)

    def j(z, x):
        count["J"] += 1
        return J(z, x)

    return f, j, count


def measured_root(F, J, count, x, z, root_cfg):
    f0, j0 = count["F"], count["J"]
    outcome = mrbi.solve_root(F, J, x, z, cfg=root_cfg)
    return outcome, count["F"]-f0, count["J"]-j0


def run_arm(layer, x, cfg, root_cfg, zero, zero_smin, baseline,
            zero_root_f, zero_root_j, base_wall_sec, seed, index,
            arm, objective_cap, residual_cap, *, return_solution=False):
    """One stage and root, or zero-only if zero has already succeeded."""
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    sigma, weight = ARM_SPECS[arm]
    base_f, base_j = baseline["F"], baseline["J"]
    if zero.success:
        result = {
            "zero_success": 1, "candidate_attempted": 0,
            "checkpoint_success": 0, "accepted_success": 1,
            "used_mrbi": 0, "zero_residual": float(zero.residual),
            "candidate_root_residual": np.nan,
            "accepted_residual": float(zero.residual),
            "optimizer_objective_calls": 0,
            "optimizer_F_calls": 0, "optimizer_J_calls": 0,
            "root_calls": 1, "zero_root_F_calls": int(zero_root_f),
            "zero_root_J_calls": int(zero_root_j),
            "checkpoint_root_F_calls": 0, "checkpoint_root_J_calls": 0,
            "total_F_calls": int(base_f), "total_J_calls": int(base_j),
            "algorithm_F_calls": int(base_f),
            "algorithm_J_calls": int(base_j),
            "diagnostic_F_calls": 0, "diagnostic_J_calls": 0,
            "budget_hit": 0, "total_wall_sec": float(base_wall_sec),
            "probe_sha256": "",
        }
        return (None, result, np.asarray(zero.z_star).copy()) if return_solution else (None, result)

    start = time.perf_counter()
    F, J, count = counted_functions(layer)
    opt_cfg = bench.make_mrbi_cfg(cfg)
    rng = np.random.default_rng(int(seed) + 2000 + 1000003 * int(index))
    opt = SmoothedHomotopyOptimizer(
        F, J, x, layer.d, config=opt_cfg, rng=rng,
        objective_cap_per_stage=objective_cap,
        residual_cap_per_stage=residual_cap,
        weights=(weight, 0.0, 0.0, 0.0, 0.0),
    )
    candidate = opt._optimize_single_scale(
        np.zeros(layer.d, dtype=np.float64), sigma,
        int(opt_cfg.maxiter_per_scale), detector_override=True,
    )
    if len(opt.stage_records) != 1 or opt.objective_calls < 1:
        raise RuntimeError("A single stage was not completed")
    stage = dict(opt.stage_records[0])
    if (stage["sigma"] != sigma or stage["smooth_weight"] != weight
            or not np.array_equal(
                np.asarray(json.loads(stage["candidate_json"])),
                candidate,
            )):
        raise RuntimeError("Stage candidate or frozen arm specification changed")
    root, checkpoint_f, checkpoint_j = measured_root(
        F, J, count, x, candidate, root_cfg
    )
    final_smin, _ = mrbi.jacobian_health(J, x, root.z_star)
    use = stage_audit.forced_accept(zero, root, zero_smin, final_smin)
    accepted = root if use else zero

    # Exact primitive call accounting. The inherited stage logger adds
    # K+1 diagnostic F calls and one J call, which must not be mistaken
    # for deployable optimizer/root work.
    diagnostic_f = 1 + int(opt_cfg.mc_samples)
    diagnostic_j = 1
    expected_f = (1 + stage["residual_evals_stage"]
                  + diagnostic_f + checkpoint_f)
    expected_j = (stage["jacobian_evals_stage"]
                  + diagnostic_j + checkpoint_j + 1)
    if (count["F"] != expected_f or count["J"] != expected_j
            or stage["residual_evals_stage"] !=
            stage["objective_calls_stage"]*(2+opt_cfg.mc_samples)
            or stage["objective_calls_stage"] > objective_cap
            or stage["residual_evals_stage"] > residual_cap):
        raise RuntimeError(
            f"Primitive call/budget mismatch in {arm}: observed {count}, "
            f"expected F={expected_f}, J={expected_j}"
        )
    elapsed = time.perf_counter() - start
    record = {
        "zero_success": 0, "candidate_attempted": 1,
        "checkpoint_success": int(root.success),
        "accepted_success": int(accepted.success),
        "used_mrbi": int(use), "zero_residual": float(zero.residual),
        "candidate_root_residual": float(root.residual),
        "accepted_residual": float(accepted.residual),
        "optimizer_objective_calls": int(opt.objective_calls),
        "optimizer_F_calls": int(stage["residual_evals_stage"]),
        "optimizer_J_calls": int(stage["jacobian_evals_stage"]),
        "root_calls": 2, "zero_root_F_calls": int(zero_root_f),
        "zero_root_J_calls": int(zero_root_j),
        "checkpoint_root_F_calls": int(checkpoint_f),
        "checkpoint_root_J_calls": int(checkpoint_j),
        "total_F_calls": int(base_f+count["F"]),
        "total_J_calls": int(base_j+count["J"]),
        "algorithm_F_calls": int(base_f+count["F"]-diagnostic_f),
        "algorithm_J_calls": int(base_j+count["J"]-diagnostic_j),
        "diagnostic_F_calls": diagnostic_f,
        "diagnostic_J_calls": diagnostic_j,
        "budget_hit": int(stage["budget_hit"]),
        "total_wall_sec": float(base_wall_sec+elapsed),
        "probe_sha256": str(stage["probe_sha256"]),
    }
    stage.pop("candidate_json")
    stage.update(
        arm=arm, stage_root_success=int(root.success),
        stage_root_residual=float(root.residual),
        stage_root_F_calls=int(checkpoint_f),
        stage_root_J_calls=int(checkpoint_j),
    )
    return (stage, record, np.asarray(accepted.z_star).copy()) if return_solution else (stage, record)


def run_job(dataset, seed, n, objective_cap, residual_cap):
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError("Only frozen datasets and new seeds 5--9 are allowed")
    cfg = stage_audit.config_for(seed)
    X, y = bench.load_dataset(dataset, cfg)
    Xtr, Xte, _ytr, _yte = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=seed,
    )
    Xtr_p, _Xte_p = bench.preprocess_to_pca(Xtr, Xte, cfg.pca_dim, seed)
    if n < 1 or n > len(Xtr_p):
        raise ValueError("Invalid training-input count")
    layer = bench.make_random_implicit_layer(
        dx=Xtr_p.shape[1], d=cfg.latent_dim, cfg=cfg,
        seed=seed+1000+cfg.latent_dim+int(100*cfg.spectral_radius),
    )
    root_cfg = bench.make_root_cfg(cfg)
    samples, stages = [], []
    for index, x in enumerate(Xtr_p[:n]):
        start = time.perf_counter()
        zero_F, zero_J, baseline = counted_functions(layer)
        zero, z_f, z_j = measured_root(
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
        row = {
            "dataset": dataset, "seed": seed,
            "train_index": index, "plan": PLAN,
            "zero_success": int(zero.success),
        }
        hashes = set()
        for arm in ARMS:
            stage, outcome = run_arm(
                layer, x, cfg, root_cfg, zero, zero_smin, baseline,
                z_f, z_j, baseline_wall, seed, index, arm,
                objective_cap, residual_cap,
            )
            if stage is not None:
                hashes.add(stage["probe_sha256"])
                stage.update(dataset=dataset, seed=seed,
                             train_index=index, plan=PLAN)
                stages.append(stage)
            row.update({f"{arm}_{key}": value
                        for key, value in outcome.items()})
        if len(hashes) > 1:
            raise RuntimeError("The four arms did not share exact probes")
        row["delta_smoothed_coarse_minus_fine_success"] = (
            row["smoothed_coarse_accepted_success"]
            - row["smoothed_fine_accepted_success"]
        )
        row["delta_smoothed_coarse_minus_plain_coarse_success"] = (
            row["smoothed_coarse_accepted_success"]
            - row["plain_coarse_accepted_success"]
        )
        row["delta_smoothed_coarse_minus_fine_algorithm_F"] = (
            row["smoothed_coarse_algorithm_F_calls"]
            - row["smoothed_fine_algorithm_F_calls"]
        )
        samples.append(row)
    meta = {
        "dataset": dataset, "seed": seed,
        "n_samples": n, "config": asdict(cfg),
        "root_cfg": asdict(root_cfg),
        "training_input_sha256": sha_arrays(Xtr_p[:n]),
        "implicit_layer_sha256": sha_arrays(layer.W, layer.U, layer.b),
        "probe_seed_rule": "seed + 2000 + 1000003 * train_index",
    }
    return pd.DataFrame(stages, columns=STAGE_FIELDS), pd.DataFrame(samples), meta


def paths_for(out, dataset, seed):
    prefix = out / "jobs" / f"{dataset}_seed{seed}"
    return tuple(Path(str(prefix)+suffix) for suffix in
                 ("_stages.csv", "_pairs.csv", "_manifest.json"))


def atomic_write(path, content):
    tmp = path.with_name(path.name+".partial")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def collect(out, jobs, n):
    stage_parts, pair_parts = [], []
    for ds, seed in jobs:
        s, p, m = paths_for(out, ds, seed)
        if all(path.is_file() for path in (s, p, m)):
            stage_parts.append(pd.read_csv(s))
            pair_parts.append(pd.read_csv(p))
    if not pair_parts:
        print("No completed held-out jobs to collect.", flush=True)
        return
    pairs = pd.concat(pair_parts, ignore_index=True)
    stages = pd.concat(stage_parts, ignore_index=True)
    if (len(pairs) != len(pair_parts)*n
            or pairs.duplicated(["dataset","seed","train_index"]).any()
            or (not stages.empty and stages.duplicated(
                ["dataset","seed","train_index","arm","stage"]
            ).any())):
        raise RuntimeError("Invalid held-out aggregate coverage")
    pairs.to_csv(out/"paired_sample_results.csv", index=False)
    stages.to_csv(out/"stage_raw.csv", index=False)
    rows = []
    for ds, sub in pairs.groupby("dataset", sort=True):
        zero_fail = sub.zero_success.eq(0)
        item = {
            "dataset": ds, "n_inputs": len(sub),
            "n_seeds": sub.seed.nunique(),
            "n_zero_fail": int(zero_fail.sum()),
        }
        for arm in ARMS:
            for field in ("accepted_success", "checkpoint_success"):
                item[f"{arm}_{field}_count"] = int(
                    sub[f"{arm}_{field}"].sum()
                )
            for field in ("algorithm_F_calls", "algorithm_J_calls",
                          "total_wall_sec", "optimizer_objective_calls"):
                item[f"{arm}_mean_{field}"] = float(
                    sub[f"{arm}_{field}"].mean()
                )
        rows.append(item)
    pd.DataFrame(rows).to_csv(out/"dataset_summary.csv", index=False)
    overall = {
        "plan": PLAN, "completed_jobs": len(pair_parts),
        "requested_jobs": len(jobs), "n_inputs": len(pairs),
        "zero_success": int(pairs.zero_success.sum()),
        "zero_fail": int(pairs.zero_success.eq(0).sum()),
        "primary_contrast": "smoothed_coarse minus smoothed_fine",
        "secondary_contrast": "smoothed_coarse minus plain_coarse",
        "arms": {},
        "cost": "Primitive F/J counts excluding stage diagnostics; raw all-in counts also in paired CSV. Same caps, unequal realized calls possible.",
        "inference": "Fresh seeds 5--9 on previously explored datasets; dataset-seed pairing, not independent per-input observations.",
    }
    for arm in ARMS:
        overall["arms"][arm] = {
            "accepted_success": int(pairs[f"{arm}_accepted_success"].sum()),
            "rescues_after_zero_failure": int(
                ((pairs.zero_success.eq(0))&
                 pairs[f"{arm}_checkpoint_success"].eq(1)).sum()
            ),
            "mean_algorithm_F_calls": float(
                pairs[f"{arm}_algorithm_F_calls"].mean()
            ),
            "mean_algorithm_J_calls": float(
                pairs[f"{arm}_algorithm_J_calls"].mean()
            ),
            "mean_total_wall_sec": float(
                pairs[f"{arm}_total_wall_sec"].mean()
            ),
            "mean_optimizer_objective_calls": float(
                pairs[f"{arm}_optimizer_objective_calls"].mean()
            ),
        }
    (out/"validation_statistics.json").write_text(
        json.dumps(overall, indent=2, allow_nan=False)+"\n",
        encoding="utf-8",
    )
    print(
        f"Collected {len(pair_parts)}/{len(jobs)} jobs, "
        f"{len(pairs)} paired inputs, {overall['zero_fail']} zero failures.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--samples-per-job", type=int, default=8)
    parser.add_argument("--objective-cap-per-stage", type=int, default=480)
    parser.add_argument("--residual-cap-per-stage", type=int, default=5760)
    parser.add_argument("--max-new-jobs", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    if (not args.datasets or len(set(args.datasets)) != len(args.datasets)
            or not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or not set(args.seeds).issubset(SEEDS)
            or args.samples_per_job != 8
            or args.objective_cap_per_stage != 480
            or args.residual_cap_per_stage != 5760
            or (args.max_new_jobs is not None and args.max_new_jobs < 1)):
        raise SystemExit("Held-out plan is frozen: 8 inputs, seeds 5--9, caps 480/5760")
    jobs = [(ds, seed) for ds in args.datasets for seed in args.seeds]
    current_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    manifest = {
        "plan": PLAN, "repo_sha": current_sha,
        "python": sys.version, "platform": platform.platform(),
        "packages": {p:version(p) for p in
                     ("numpy","scipy","pandas","scikit-learn")},
        "datasets": args.datasets, "seeds": args.seeds,
        "samples_per_job": 8, "objective_cap_per_stage": 480,
        "residual_cap_per_stage": 5760,
        "arm_specs": ARM_SPECS,
        "base_config": asdict(stage_audit.config_for(5)),
        "root_config": asdict(bench.make_root_cfg(
            stage_audit.config_for(5)
        )),
        "sampling": "First eight training rows after original seeded split and PCA",
        "provenance": "Independent run for seeds 5--9: no full-campaign reference CSV",
        "no_qnn_training": True,
    }
    # JSON round-trip canonicalizes dataclass tuple fields and ARM_SPECS:
    # manifest comparison must work after a completed job is resumed.
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    out = args.out_dir.resolve()
    global_manifest = out/"environment.json"
    if global_manifest.is_file():
        if json.loads(global_manifest.read_text(encoding="utf-8")) != manifest:
            raise SystemExit("Run provenance changed; choose a new output directory")
    elif not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        atomic_write(
            global_manifest,
            json.dumps(manifest, indent=2, allow_nan=False)+"\n",
        )
    pending = []
    for ds, seed in jobs:
        s, p, m = paths_for(out, ds, seed)
        present = [z.is_file() for z in (s,p,m)]
        if all(present):
            record = json.loads(m.read_text(encoding="utf-8"))
            table = pd.read_csv(p)
            stage = pd.read_csv(s)
            if (record.get("repo_sha") != current_sha
                    or record.get("plan") != PLAN
                    or record.get("dataset") != ds
                    or record.get("seed") != seed
                    or record.get("stage_csv_sha256") != hashlib.sha256(s.read_bytes()).hexdigest()
                    or record.get("pair_csv_sha256") != hashlib.sha256(p.read_bytes()).hexdigest()
                    or len(table) != 8
                    or len(stage) != 4*int(table.zero_success.eq(0).sum())
                    or table.duplicated(
                        ["dataset","seed","train_index"]
                    ).any()
                    or not set(table.train_index).issubset(set(range(8)))
                    or stage.duplicated(
                        ["dataset","seed","train_index","arm","stage"]
                    ).any()):
                raise SystemExit(f"Completed job does not validate: {ds} seed {seed}")
        elif any(present):
            raise SystemExit(
                f"Partial job {ds} seed {seed}; inspect files before retrying"
            )
        else:
            pending.append((ds,seed))
    print(f"Fresh-seed single-stage: {len(jobs)} jobs, "
          f"{len(jobs)-len(pending)} complete, "
          f"{len(pending)} pending.", flush=True)
    if args.dry_run:
        return
    if not args.collect_only:
        for ds, seed in pending[:args.max_new_jobs]:
            print(f"Starting {ds} seed={seed}", flush=True)
            stages, pairs, meta = run_job(
                ds, seed, 8, 480, 5760
            )
            s,p,m = paths_for(out, ds, seed)
            s.parent.mkdir(parents=True, exist_ok=True)
            stage_text = stages.to_csv(index=False)
            pair_text = pairs.to_csv(index=False)
            atomic_write(s, stage_text)
            atomic_write(p, pair_text)
            meta.update(
                repo_sha=current_sha, plan=PLAN,
                stage_csv_sha256=hashlib.sha256(stage_text.encode("utf-8")).hexdigest(),
                pair_csv_sha256=hashlib.sha256(pair_text.encode("utf-8")).hexdigest(),
            )
            atomic_write(m, json.dumps(
                meta, indent=2, allow_nan=False
            )+"\n")
            print(f"Finished {ds} seed={seed}: "
                  f"{len(pairs)} inputs, {len(stages)} stage records.",
                  flush=True)
    collect(out, jobs, 8)


if __name__ == "__main__":
    main()
