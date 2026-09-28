"""Resumable, budget-aware final-sigma ablation against saved core QNN rows.

Runs only the final-sigma control (one QNN per dataset/seed). The continuation
arm and baselines are read from the previously completed core main_raw.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "final_sigma_ablation_job.py"
DATASETS = (
    "breast_cancer", "wine_binary", "wine_0_vs_2", "wine_1_vs_2",
    "digits_1_vs_7", "digits_2_vs_7", "digits_3_vs_8",
    "digits_4_vs_9", "digits_5_vs_6",
)
SEEDS = (0, 1, 2, 3, 4)
PLAN = "final_sigma_repeated_vs_continuation_v1"
SOURCE_PLAN = "core_fixed_after_pilot_v1"
FULL_SOURCE_PLAN = "full_corrected_continuation_v1"
VERSION = "mrbi_continuation_v1"
SOURCE_METHOD = "forced_full_balanced_qnn"
CONTROL_METHOD = "final_sigma_repeated_full_balanced_qnn"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-raw", required=True, type=Path)
    parser.add_argument("--reference-env", type=Path, nargs="+", default=None,
                        help="Core environment JSON, or all three full-campaign shard environments.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "outputs" / "final_sigma_ablation_v1")
    parser.add_argument("--max-new-jobs", type=int, default=None)
    parser.add_argument("--max-wall-hours", type=float, default=None)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def paths(out, dataset, seed):
    name = out / "jobs" / f"{dataset}_seed{seed}"
    return (name.with_name(name.name + "_raw.csv"),
            name.with_name(name.name + "_config.json"),
            name.with_suffix(".log"))


def valid_job(path, dataset, seed):
    if not path.is_file():
        return False
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    required = {"dataset", "seed", "method", "readout", "implementation_version",
                "ablation_design", "ablation_arm", "balanced_accuracy",
                "stat_test_forced_mean_objective_calls",
                "stat_test_forced_success_rate"}
    if len(df) != 1 or not required.issubset(df.columns):
        return False
    row = df.iloc[0]
    return bool(
        row["dataset"] == dataset and int(row["seed"]) == seed
        and row["method"] == CONTROL_METHOD and row["readout"] == "qnn"
        and row["implementation_version"] == VERSION
        and row["ablation_design"] == PLAN
        and row["ablation_arm"] == "final_sigma_repeated"
        and pd.notna(row["balanced_accuracy"])
        and pd.notna(row["stat_test_forced_mean_objective_calls"])
        and pd.notna(row["stat_test_forced_success_rate"])
    )


def source_reference(path, jobs):
    """Validate the complete reference for the requested paired jobs."""
    if not path.is_file():
        raise SystemExit(f"Reference raw not found: {path}")
    df = pd.read_csv(path)
    required = {"dataset", "seed", "method", "readout",
                "implementation_version", "balanced_accuracy"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Reference missing columns: {sorted(required-set(df.columns))}")
    if not df["implementation_version"].eq(VERSION).all():
        raise SystemExit("Reference must use the corrected-continuation implementation.")
    if "campaign_design" in df.columns:
        if not df["campaign_design"].eq(SOURCE_PLAN).all():
            raise SystemExit("Unexpected reference campaign design.")
        source_plan, expected_methods, expected_qnn = SOURCE_PLAN, 25, 8
    else:
        source_plan, expected_methods, expected_qnn = FULL_SOURCE_PLAN, 97, 32
    if df.duplicated(["dataset", "seed", "method"]).any():
        raise SystemExit("Duplicate dataset/seed/method in reference.")
    for ds, seed in jobs:
        group = df[(df["dataset"] == ds) & (df["seed"] == seed)]
        if (len(group) != expected_methods or
                group["method"].nunique() != expected_methods or
                group["readout"].eq("qnn").sum() != expected_qnn):
            raise SystemExit(
                f"Incomplete {source_plan} reference for {ds} seed={seed}: "
                f"expected {expected_methods} methods, {expected_qnn} QNN rows."
            )
    ref = df[df["method"] == SOURCE_METHOD]
    if ref.duplicated(["dataset", "seed"]).any():
        raise SystemExit("Duplicate forced-full-balanced reference rows")
    present = set(zip(ref["dataset"], ref["seed"]))
    absent = set(jobs) - present
    if absent:
        raise SystemExit(f"Missing reference dataset/seed jobs: {sorted(absent)}")
    if pd.to_numeric(ref["balanced_accuracy"], errors="coerce").isna().any():
        raise SystemExit("Reference has missing QNN accuracy")
    return hashlib.sha256(path.read_bytes()).hexdigest(), source_plan


def snapshot():
    packages = {}
    for name in ("numpy", "scipy", "pandas", "scikit-learn", "torch", "pennylane"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         text=True, capture_output=True, check=False)
    return {
        "implementation_version": VERSION, "ablation_design": PLAN,
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "python": sys.version, "platform": platform.platform(),
        "packages": packages,
    }


def main():
    opts = parse_args()
    if len(set(opts.datasets)) != len(opts.datasets) or len(set(opts.seeds)) != len(opts.seeds):
        raise SystemExit("Duplicate datasets or seeds are not allowed")
    if not opts.seeds or any(seed < 0 for seed in opts.seeds):
        raise SystemExit("Seeds must be nonnegative")
    if opts.max_new_jobs is not None and opts.max_new_jobs < 1:
        raise SystemExit("--max-new-jobs must be positive")
    if opts.max_wall_hours is not None and opts.max_wall_hours <= 0:
        raise SystemExit("--max-wall-hours must be positive")
    out = opts.out_dir.resolve()
    ref_path = opts.reference_raw.resolve()
    jobs = [(ds, seed) for ds in opts.datasets for seed in opts.seeds]
    reference_sha, source_plan = source_reference(ref_path, jobs)
    env_paths = [
        path.resolve() for path in (
            opts.reference_env or [ref_path.with_name("environment.json")]
        )
    ]
    if len(set(env_paths)) != len(env_paths):
        raise SystemExit("Duplicate source environment paths.")
    for path in env_paths:
        if not path.is_file():
            raise SystemExit(f"Reference environment missing: {path}")
    reference_envs = [
        json.loads(path.read_text(encoding="utf-8")) for path in env_paths
    ]
    for reference_env in reference_envs:
        if reference_env.get("implementation_version") != VERSION:
            raise SystemExit("Reference environment implementation version differs.")
        if source_plan == SOURCE_PLAN:
            if reference_env.get("campaign_design") != SOURCE_PLAN:
                raise SystemExit("Expected the core campaign environment.")
        elif reference_env.get("campaign_design") not in (None, FULL_SOURCE_PLAN):
            raise SystemExit("Expected the full corrected-continuation campaign environment.")

    source_commits = {item.get("git_commit") for item in reference_envs}
    if len(source_commits) != 1 or not next(iter(source_commits)):
        raise SystemExit("Reference shards have different or missing Git commits.")
    source_packages = reference_envs[0].get("packages")
    if not source_packages or any(
        item.get("packages") != source_packages for item in reference_envs
    ):
        raise SystemExit("Reference shards use different or missing package versions.")
    if source_plan == FULL_SOURCE_PLAN:
        # A merged full raw CSV does not have a combined environment file.
        # Require the provenance JSON of every shard covering requested jobs.
        source_jobs = set()
        for item in reference_envs:
            shard_datasets, shard_seeds = item.get("datasets"), item.get("seeds")
            if not shard_datasets or not shard_seeds:
                raise SystemExit("Full reference shard lacks its dataset/seed manifest.")
            shard_jobs = {(ds, int(seed)) for ds in shard_datasets for seed in shard_seeds}
            if source_jobs & shard_jobs:
                raise SystemExit("Overlapping full-campaign source shard manifests.")
            source_jobs.update(shard_jobs)
        if not set(jobs).issubset(source_jobs):
            raise SystemExit("Missing source shard environment for a requested job.")

    env = snapshot()
    if env["packages"] != source_packages:
        raise SystemExit(
            "Package versions differ from the reference. "
            "Activate the venv that produced the full campaign."
        )
    env["source_campaign_design"] = source_plan
    env["source_raw_sha256"] = reference_sha
    env["source_git_commit"] = next(iter(source_commits))
    env["source_environments"] = [str(path) for path in env_paths]
    env["reference_raw"] = str(ref_path)
    env["datasets"], env["seeds"] = opts.datasets, opts.seeds

    done, pending = [], []
    for ds, seed in jobs:
        raw, config, log = paths(out, ds, seed)
        if valid_job(raw, ds, seed) and config.is_file():
            done.append((ds, seed))
        else:
            if raw.exists():
                raise SystemExit(f"Incomplete output: {raw}. Inspect or move it before resuming.")
            pending.append((ds, seed))
    print(f"Ablation: {len(jobs)} paired jobs, {len(done)} complete, {len(pending)} pending.",
          flush=True)
    if opts.verify_only or opts.dry_run:
        for ds, seed in pending:
            print(f"  pending {ds} seed={seed}")
        return

    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "environment.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        for key in ("ablation_design", "implementation_version", "git_commit",
                    "packages", "source_raw_sha256", "source_git_commit",
                    "source_campaign_design", "source_environments"):
            if old.get(key) != env.get(key):
                raise SystemExit(f"Campaign environment changed: {key}; use another output directory.")
    else:
        manifest.write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                text=True, capture_output=True, check=False)
        if freeze.returncode:
            raise SystemExit("Cannot record pip freeze")
        (out / "environment.freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    started = time.monotonic()
    completed_now = 0
    for ds, seed in pending:
        if opts.collect_only:
            break
        if opts.max_new_jobs is not None and completed_now >= opts.max_new_jobs:
            break
        elapsed = (time.monotonic()-started)/3600.0
        if opts.max_wall_hours is not None and elapsed >= opts.max_wall_hours-0.25:
            print(f"Time budget reached after {elapsed:.2f} hours", flush=True)
            break
        raw, config, log = paths(out, ds, seed)
        raw.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-u", str(RUNNER), "--dataset", ds, "--seed", str(seed),
            "--reference-raw", str(ref_path), "--out-raw", str(raw),
            "--out-config", str(config),
        ]
        print(f"Starting final-sigma control: {ds} seed={seed}", flush=True)
        start_job = time.perf_counter()
        with log.open("w", encoding="utf-8") as output:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=output,
                                    stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:
                    try:
                        proc.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        print(f"  Running {ds} seed={seed}: "
                              f"{(time.perf_counter()-start_job)/60:.1f} min; {log}",
                              flush=True)
            except KeyboardInterrupt:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise
        if proc.returncode or not valid_job(raw, ds, seed):
            raise SystemExit(f"Job failed; inspect {log}")
        completed_now += 1
        print(f"Finished {ds} seed={seed} in "
              f"{(time.perf_counter()-start_job)/60:.1f} min", flush=True)

    parts = []
    missing = []
    for ds, seed in jobs:
        raw, config, _log = paths(out, ds, seed)
        if valid_job(raw, ds, seed) and config.is_file():
            parts.append(pd.read_csv(raw))
        else:
            missing.append((ds, seed))
    if parts:
        pd.concat(parts, ignore_index=True).sort_values(
            ["dataset", "seed"]
        ).to_csv(out / "main_raw.csv", index=False)
    print(f"Saved {len(parts)} control jobs; remaining {len(missing)}.", flush=True)
    if opts.summarize and parts:
        complete = [ds for ds in opts.datasets if all(
            (ds, seed) not in missing for seed in SEEDS
        ) and set(SEEDS).issubset(set(opts.seeds))]
        if complete:
            cmd = [
                sys.executable, str(ROOT / "scripts" / "summarize_final_sigma_ablation.py"),
                "--control-raw", str(out / "main_raw.csv"),
                "--reference-raw", str(ref_path),
                "--out-dir", str(out),
            ]
            subprocess.run(cmd, cwd=ROOT, check=True)
        else:
            print("No complete five-seed dataset yet; raw results saved.", flush=True)
    if missing:
        print("Run the same command again to resume.", flush=True)
    else:
        print("All paired final-sigma controls complete.", flush=True)


if __name__ == "__main__":
    main()
