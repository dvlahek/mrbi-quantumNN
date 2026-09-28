"""Label-free stage audit of full_balanced MRBI against repeated final sigma.

Run against the completed full 97-method reference. This is a trajectory
diagnostic on the first N training samples after the original seeded split,
not a new QNN benchmark or a confirmatory accuracy experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
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
import run_final_sigma_ablation as source_runner

DATASETS = (
    "breast_cancer", "wine_binary", "wine_0_vs_2", "wine_1_vs_2",
    "digits_1_vs_7", "digits_2_vs_7", "digits_3_vs_8",
    "digits_4_vs_9", "digits_5_vs_6",
)
SEEDS = tuple(range(5))
ARMS = ("continuation", "repeated_final_sigma")
PLAN = "full_multiscale_stage_audit_v1"
STAGE_COUNT = 5


def config_for(seed):
    base = bench.ExperimentConfig(
        seed=seed, n_qubits=4, pca_dim=4, latent_dim=16,
        max_samples_per_class=80, spectral_radius=2.0, input_scale=1.1,
        hard_layer=True, run_qnn=False, qnn_epochs=60, qnn_layers=2,
    )
    profile = next(p for p in bench.get_mrbi_profiles("hard_quick")
                   if p.name == "full_balanced")
    return bench.apply_mrbi_profile(base, profile)


def terms(opt, z, sigma):
    """Extra audit evaluations do not increment the optimization budget."""
    if float(sigma) not in opt._probe_cache:
        raise RuntimeError("Unexpected uninitialized sigma probe cache")
    r = float(np.linalg.norm(opt.residual(z)))
    n = float(np.linalg.norm(opt.newton_proxy(z)))
    ratio = float(opt.monte_carlo_ratio(z, sigma))
    singular = np.linalg.svd(opt.jacobian(z), compute_uv=False)
    smin, smax = float(np.min(singular)), float(np.max(singular))
    parts = {
        "residual_term": opt.cfg.alpha * r,
        "newton_term": opt.cfg.beta * n,
        "detector_term": opt.cfg.gamma * ratio,
        "zero_penalty": opt.cfg.lambda_zero * float(np.linalg.norm(z)),
        "relative_penalty": opt.cfg.lambda_relative
            * max(0.0, r - opt.zero_residual_norm),
    }
    return {
        "residual_norm": r, "newton_proxy_norm": n, "detector_ratio": ratio,
        "jacobian_sigma_min": smin,
        "jacobian_condition": smax / smin if smin > 0.0 else float("inf"),
        "stage_objective": float(sum(parts.values())), **parts,
    }


def trace_arm(F, J, x, d, cfg, root_cfg, rng, arm):
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    config = replace(cfg, repeat_final_sigma=(arm == ARMS[1]))
    opt = mrbi.MRBIOptimizer(F, J, x, d, config=config, rng=rng)
    original_single, original_minimize = opt._optimize_single_scale, mrbi.minimize
    reports, stages, roots = [], [], []

    def capture_minimize(*args, **kwargs):
        result = original_minimize(*args, **kwargs)
        reports.append({
            "optimizer_success": bool(result.success),
            "optimizer_nit": int(getattr(result, "nit", -1)),
            "optimizer_nfev": int(getattr(result, "nfev", -1)),
            "optimizer_status": int(getattr(result, "status", -1)),
        })
        return result

    def capture_stage(z0, sigma, maxiter, *, detector_override=None):
        before = opt.objective_calls
        z = original_single(z0, sigma, maxiter,
                            detector_override=detector_override)
        used = opt.objective_calls - before
        index = len(stages)
        if len(reports) != index + 1:
            raise RuntimeError("Optimizer stage was not captured")
        metrics = terms(opt, z, float(sigma))
        root = mrbi.solve_root(F, J, x, z, cfg=root_cfg)
        roots.append(root)
        stages.append({
            "arm": arm, "stage": index + 1, "sigma": float(sigma),
            "maxiter": int(maxiter), "objective_calls_stage": int(used),
            "objective_calls_cumulative": int(opt.objective_calls),
            "candidate_step_norm": float(np.linalg.norm(np.asarray(z)-z0)),
            "candidate_json": json.dumps(np.asarray(z, dtype=float).tolist()),
            "stage_root_success": bool(root.success),
            "stage_root_residual": float(root.residual),
            "stage_root_nfev": int(root.nfev),
            "stage_root_json": json.dumps(root.z_star.tolist()),
            **reports[index], **metrics,
        })
        return z

    opt._optimize_single_scale = capture_stage
    # Audit is sequential; temporarily intercept the module's actual SciPy call.
    mrbi.minimize = capture_minimize
    try:
        candidate = opt.optimize()
    finally:
        mrbi.minimize = original_minimize
    if len(stages) != STAGE_COUNT:
        raise RuntimeError(f"Expected {STAGE_COUNT} stages, got {len(stages)}")
    if not np.array_equal(
        np.asarray(json.loads(stages[-1]["candidate_json"])), candidate.z_init
    ):
        raise RuntimeError("Audit changed the final candidate")
    if opt.objective_calls != candidate.n_objective_calls:
        raise RuntimeError("Audit objective call count changed")
    return stages, roots, candidate


def forced_accept(zero, det, s_zero, s_det):
    """Mirror the exact forced-acceptance rule in the full QNN benchmark."""
    return bool(
        (not zero.success and det.success)
        or (np.isfinite(det.residual) and det.residual < zero.residual)
        or (np.isfinite(det.residual) and np.isfinite(zero.residual)
            and det.residual <= 1.10 * zero.residual and s_det > s_zero)
    )


def audit_input(layer, x, cfg, rng_by_arm):
    F, J = mrbi.make_residual_and_jacobian(layer)
    root_cfg, mrbi_cfg = bench.make_root_cfg(cfg), bench.make_mrbi_cfg(cfg)
    zero = mrbi.solve_root(F, J, x, np.zeros(layer.d), cfg=root_cfg)
    s_zero, _ = mrbi.jacobian_health(J, x, zero.z_star)
    stage_rows, outcomes = [], {}
    for arm in ARMS:
        stages, stage_roots, candidate = trace_arm(
            F, J, x, layer.d, mrbi_cfg, root_cfg, rng_by_arm[arm], arm
        )
        stage_rows.extend(stages)
        det = stage_roots[-1]
        s_det, _ = mrbi.jacobian_health(J, x, det.z_star)
        accept = forced_accept(zero, det, s_zero, s_det)
        final = det if accept else zero
        outcomes[arm] = {
            "candidate": candidate.z_init,
            "root": det.z_star,
            "candidate_success": bool(det.success),
            "accepted_success": bool(final.success),
            "accepted_residual": float(final.residual),
            "used_mrbi": accept,
            "objective_calls": candidate.n_objective_calls,
        }
    cont, control = (outcomes[arm] for arm in ARMS)
    pair = {
        "zero_success": bool(zero.success),
        "zero_residual": float(zero.residual),
        "candidate_distance": float(np.linalg.norm(cont["candidate"]-control["candidate"])),
        "final_root_distance": float(np.linalg.norm(cont["root"]-control["root"])),
        "distinct_converged_roots": bool(
            cont["candidate_success"] and control["candidate_success"]
            and np.linalg.norm(cont["root"]-control["root"]) > 1e-5
        ),
        "delta_candidate_success": int(cont["candidate_success"])-int(control["candidate_success"]),
        "delta_accepted_success": int(cont["accepted_success"])-int(control["accepted_success"]),
        "delta_accepted_residual": cont["accepted_residual"]-control["accepted_residual"],
        "objective_calls_cont": int(cont["objective_calls"]),
        "objective_calls_final": int(control["objective_calls"]),
        "continuation_used_mrbi": cont["used_mrbi"],
        "final_sigma_used_mrbi": control["used_mrbi"],
    }
    return stage_rows, pair


def run_job(dataset, seed, sample_count):
    cfg = config_for(seed)
    X, y = bench.load_dataset(dataset, cfg)
    Xtr, _Xte, _ytr, _yte = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=seed
    )
    Xtr_p, _Xte_p = bench.preprocess_to_pca(Xtr, _Xte, cfg.pca_dim, seed)
    if sample_count > len(Xtr_p):
        raise ValueError(f"Only {len(Xtr_p)} training inputs available")
    layer = bench.make_random_implicit_layer(
        dx=Xtr_p.shape[1], d=cfg.latent_dim, cfg=cfg,
        seed=seed + 1000 + cfg.latent_dim + int(100*cfg.spectral_radius),
    )
    rng = {arm: np.random.default_rng(seed+2000) for arm in ARMS}
    stages, pairs = [], []
    for sample_index, x in enumerate(Xtr_p[:sample_count]):
        part, pair = audit_input(layer, x, cfg, rng)
        for row in part:
            row.update(dataset=dataset, seed=seed,
                       train_index=sample_index, plan=PLAN)
        pair.update(dataset=dataset, seed=seed,
                    train_index=sample_index, plan=PLAN)
        stages.extend(part)
        pairs.append(pair)
    return pd.DataFrame(stages), pd.DataFrame(pairs)


def job_paths(out, ds, seed):
    prefix = out / "jobs" / f"{ds}_seed{seed}"
    return (Path(str(prefix)+"_stages.csv"),
            Path(str(prefix)+"_pairs.csv"),
            Path(str(prefix)+"_config.json"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-raw", type=Path, required=True)
    p.add_argument("--reference-env", nargs="+", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--samples-per-job", type=int, default=8)
    p.add_argument("--max-new-jobs", type=int, default=None)
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "outputs" / "multiscale_stage_audit_v1")
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if (args.samples_per_job < 1 or args.max_new_jobs is not None
            and args.max_new_jobs < 1):
        raise SystemExit("Invalid audit sample count or job limit")
    if (not args.seeds or len(args.seeds) != len(set(args.seeds))
            or len(args.datasets) != len(set(args.datasets))
            or any(seed < 0 for seed in args.seeds)):
        raise SystemExit("Duplicate/invalid datasets or seeds")
    jobs = [(ds, seed) for ds in args.datasets for seed in args.seeds]
    reference = args.reference_raw.resolve()
    ref_sha, source_plan = source_runner.source_reference(reference, jobs)
    if source_plan != "full_corrected_continuation_v1":
        raise SystemExit("The source must be the full corrected-continuation campaign")
    envs = [e.resolve() for e in args.reference_env]
    if any(not e.is_file() for e in envs):
        raise SystemExit("A source shard environment.json is missing")
    sources = [json.loads(e.read_text(encoding="utf-8")) for e in envs]
    commits = {v.get("git_commit") for v in sources}
    packages = source_runner.snapshot()["packages"]
    if (len(commits) != 1 or not next(iter(commits))
            or any(v.get("packages") != packages for v in sources)):
        raise SystemExit("Source Git commit/packages differ; use the full-campaign venv")
    covered = set()
    for v in sources:
        if v.get("implementation_version") != bench.IMPLEMENTATION_VERSION:
            raise SystemExit("Source implementation version mismatch")
        for ds in v["datasets"]:
            for seed in v["seeds"]:
                key = (ds, int(seed))
                if key in covered:
                    raise SystemExit("Overlapping source shard environments")
                covered.add(key)
    if not set(jobs).issubset(covered):
        raise SystemExit("Source shard environments do not cover requested jobs")

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.strip()
    manifest = {
        "plan": PLAN, "source_raw_sha256": ref_sha,
        "source_git_commit": next(iter(commits)),
        "source_environments": [str(e) for e in envs],
        "audit_git_commit": git, "packages": packages,
        "datasets": args.datasets, "seeds": args.seeds,
        "samples_per_job": args.samples_per_job,
        "sampling_rule": "First N training rows after original seeded split and PCA",
        "profile": "full_balanced", "arms": list(ARMS),
        "note": "Audit-only probes/diagnostic roots add work but do not alter optimization objective calls.",
        "qnn_training": False,
    }
    out = args.out_dir.resolve()
    output_manifest = out / "environment.json"
    if output_manifest.exists():
        if json.loads(output_manifest.read_text(encoding="utf-8")) != manifest:
            raise SystemExit("Audit provenance changed; use a fresh output directory")
    elif not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        output_manifest.write_text(json.dumps(manifest, indent=2)+"\n",
                                   encoding="utf-8")
    pending = []
    for ds, seed in jobs:
        stage, pair, config = job_paths(out, ds, seed)
        exists = [path.is_file() for path in (stage, pair, config)]
        if all(exists):
            meta = json.loads(config.read_text(encoding="utf-8"))
            if (meta.get("source_raw_sha256") != ref_sha
                    or meta.get("audit_git_commit") != git
                    or meta.get("samples_per_job") != args.samples_per_job
                    or len(pd.read_csv(stage)) != args.samples_per_job * len(ARMS) * STAGE_COUNT
                    or len(pd.read_csv(pair)) != args.samples_per_job):
                raise SystemExit(f"Invalid completed audit job: {ds} seed={seed}")
        elif any(exists):
            raise SystemExit(f"Partial audit job for {ds} seed={seed}; inspect before retrying")
        else:
            pending.append((ds, seed))
    print(f"Stage audit: {len(jobs)} jobs, {len(jobs)-len(pending)} complete, "
          f"{len(pending)} pending.", flush=True)
    if args.dry_run:
        return
    if not args.collect_only:
        for ds, seed in pending[:args.max_new_jobs]:
            print(f"Starting stage audit: {ds} seed={seed}", flush=True)
            stages, pairs = run_job(ds, seed, args.samples_per_job)
            stage, pair, config = job_paths(out, ds, seed)
            stage.parent.mkdir(parents=True, exist_ok=True)
            stages.to_csv(stage, index=False)
            pairs.to_csv(pair, index=False)
            config.write_text(json.dumps({
                "dataset": ds, "seed": seed,
                "samples_per_job": args.samples_per_job,
                "source_raw_sha256": ref_sha, "audit_git_commit": git,
            }, indent=2)+"\n", encoding="utf-8")
            print(f"Finished {ds} seed={seed}: {len(stages)} stages, "
                  f"{len(pairs)} paired inputs.", flush=True)
    stage_parts, pair_parts = [], []
    for ds, seed in jobs:
        stage, pair, config = job_paths(out, ds, seed)
        if all(path.is_file() for path in (stage, pair, config)):
            stage_parts.append(pd.read_csv(stage))
            pair_parts.append(pd.read_csv(pair))
    if pair_parts:
        stage_all = pd.concat(stage_parts, ignore_index=True)
        pairs_all = pd.concat(pair_parts, ignore_index=True)
        stage_all.to_csv(out / "stage_raw.csv", index=False)
        pairs_all.to_csv(out / "paired_sample_results.csv", index=False)
        summary = stage_all.groupby(["dataset", "arm", "stage"], as_index=False).agg(
            mean_residual=("residual_norm", "mean"),
            mean_detector_term=("detector_term", "mean"),
            mean_stage_objective=("stage_objective", "mean"),
            mean_objective_calls=("objective_calls_stage", "mean"),
            stage_root_success_rate=("stage_root_success", "mean"),
            mean_candidate_step=("candidate_step_norm", "mean"),
        )
        summary.to_csv(out / "stage_summary.csv", index=False)
        print(f"Collected {len(pair_parts)}/{len(jobs)} complete audit jobs.",
              flush=True)


if __name__ == "__main__":
    main()
