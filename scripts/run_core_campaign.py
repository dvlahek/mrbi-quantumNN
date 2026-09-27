"""Run the fixed eight-QNN continuation comparison as resumable jobs.

Each completed job has its own raw CSV, summary, configuration and log.
The historical CSVs under results/raw are never overwritten.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "core_qnn_job.py"
DEFAULT_DATASETS = (
    "breast_cancer", "wine_binary", "wine_0_vs_2", "wine_1_vs_2",
    "digits_1_vs_7", "digits_2_vs_7", "digits_3_vs_8",
    "digits_4_vs_9", "digits_5_vs_6",
)
VERSION = "mrbi_continuation_v1"
PLAN = "core_preregistered_v1"
EXPECTED_METHODS = 25
EXPECTED_QNN = 8


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                        choices=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "outputs" / "core_continuation_v1")
    parser.add_argument("--max-new-jobs", type=int, default=None,
                        help="Stop after this many newly completed jobs, then resume later.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Check completed files without running the QNN.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print pending jobs without starting the QNN.")
    return parser.parse_args()


def job_paths(out_dir: Path, dataset: str, seed: int):
    base = out_dir / "jobs" / f"{dataset}_seed{seed}"
    return tuple(base.with_name(base.name + ext) for ext in
                 ("_raw.csv", "_summary.csv", "_config.json", ".log"))


def valid_job(raw_path: Path, dataset: str, seed: int):
    if not raw_path.is_file():
        return False
    try:
        df = pd.read_csv(raw_path)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    required = {"dataset", "seed", "method", "readout", "implementation_version",
                "campaign_design", "balanced_accuracy"}
    if not required.issubset(df.columns):
        return False
    if len(df) != EXPECTED_METHODS or df["method"].nunique() != EXPECTED_METHODS:
        return False
    if not df["dataset"].eq(dataset).all() or not df["seed"].eq(seed).all():
        return False
    if not df["implementation_version"].eq(VERSION).all():
        return False
    if not df["campaign_design"].eq(PLAN).all():
        return False
    if df["balanced_accuracy"].isna().any():
        return False
    qnn = df[df["readout"] == "qnn"]
    return len(qnn) == EXPECTED_QNN and {"pca_qnn", "implicit_zero_qnn"}.issubset(qnn["method"].tolist())


def environment_snapshot():
    package_names = ("numpy", "scipy", "pandas", "scikit-learn", "torch", "pennylane")
    packages = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         text=True, capture_output=True, check=False)
    return {
        "implementation_version": VERSION,
        "campaign_design": PLAN,
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "datasets": None,
        "seeds": None,
    }


def main():
    opts = args_parser()
    if len(set(opts.datasets)) != len(opts.datasets) or len(set(opts.seeds)) != len(opts.seeds):
        raise SystemExit("Duplicate datasets or seeds are not allowed.")
    if not opts.seeds or any(seed < 0 for seed in opts.seeds):
        raise SystemExit("Seeds must be nonnegative integers.")
    if opts.max_new_jobs is not None and opts.max_new_jobs < 1:
        raise SystemExit("--max-new-jobs must be positive.")
    out = opts.out_dir.resolve()
    jobs = [(ds, seed) for ds in opts.datasets for seed in opts.seeds]
    completed, pending = [], []
    for ds, seed in jobs:
        raw, summ, cfg, log = job_paths(out, ds, seed)
        if valid_job(raw, ds, seed) and summ.is_file() and cfg.is_file():
            completed.append((ds, seed))
        else:
            if raw.exists():
                raise SystemExit(f"Incomplete result at {raw}; inspect or move it before continuing.")
            pending.append((ds, seed))
    print(f"Campaign: {len(jobs)} jobs, {len(completed)} complete, {len(pending)} pending.")
    for ds, seed in pending:
        print(f"  pending {ds} seed={seed}")
    if opts.verify_only or opts.dry_run:
        return

    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "environment.json"
    manifest = environment_snapshot()
    manifest["datasets"] = opts.datasets
    manifest["seeds"] = opts.seeds
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("implementation_version") != VERSION or old.get("campaign_design") != PLAN:
            raise SystemExit("The output directory belongs to another implementation or campaign design.")
        if old.get("git_commit") and manifest["git_commit"] and old["git_commit"] != manifest["git_commit"]:
            raise SystemExit("Git commit changed since the campaign began. Use another output directory.")
        if old.get("packages") != manifest["packages"]:
            raise SystemExit("Python package versions changed. Use another output directory.")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                                 encoding="utf-8")
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True,
                                capture_output=True, check=False)
        if freeze.returncode != 0:
            raise SystemExit("Could not record pip freeze; see environment.json.")
        (out / "environment.freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    done = 0
    for ds, seed in pending:
        if opts.max_new_jobs is not None and done >= opts.max_new_jobs:
            break
        raw, summ, cfg, log = job_paths(out, ds, seed)
        raw.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", str(RUNNER), "--mode", "hard_quick",
               "--datasets", ds, "--seeds", str(seed), "--qubits", "4",
               "--latent-dims", "16", "--spectral-radii", "2.0",
               "--input-scales", "1.1", "--max-samples-per-class", "80",
               "--qnn-epochs", "60", "--qnn-layers", "2",
               "--n-workers", "1", "--out-raw", str(raw),
               "--out-summary", str(summ), "--out-config", str(cfg)]
        print(f"Starting {ds} seed={seed}; log: {log}", flush=True)
        t0 = time.perf_counter()
        with log.open("w", encoding="utf-8") as output:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=output,
                                    stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:
                    try:
                        proc.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        elapsed = (time.perf_counter() - t0) / 60.0
                        print(f"  Running {ds} seed={seed}: {elapsed:.1f} min elapsed; "
                              f"details: {log}", flush=True)
            except KeyboardInterrupt:
                print(f"Interrupted {ds} seed={seed}; stopping its worker.", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise
        if proc.returncode != 0 or not valid_job(raw, ds, seed):
            print(f"Job failed or did not produce 25 valid core rows: {log}")
            raise SystemExit(proc.returncode or 1)
        print(f"Finished {ds} seed={seed} in {(time.perf_counter()-t0)/60:.1f} min.",
              flush=True)
        done += 1

    remaining = []
    parts = []
    for ds, seed in jobs:
        raw, summ, cfg, log = job_paths(out, ds, seed)
        if not valid_job(raw, ds, seed):
            remaining.append((ds, seed))
        else:
            parts.append(pd.read_csv(raw))
    if parts:
        merged = pd.concat(parts, ignore_index=True).sort_values(
            ["dataset", "seed", "method"]).reset_index(drop=True)
        merged.to_csv(out / "main_raw.csv", index=False)
    print(f"Saved {len(parts)} completed job files; remaining: {len(remaining)}.")
    if not remaining:
        print("All requested jobs complete. Run scripts/summarize_core_campaign.py.")
    else:
        print("Run the same command again to resume.")


if __name__ == "__main__":
    main()
