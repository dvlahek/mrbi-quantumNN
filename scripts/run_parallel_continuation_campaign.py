"""Run the full 97-method corrected-continuation campaign across independent CPU jobs.

Each dataset/seed is a separate process with single-threaded BLAS/PyTorch.
The nine-dataset, five-seed experiment is resumable without repeating valid
jobs. Nothing is written to the historical result directories or core campaign.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd

import run_continuation_campaign as serial

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "main_qnn_benchmark.py"
OUT = ROOT / "outputs" / "full_parallel_v1"
PLAN = "full_parallel_corrected_continuation_v1"
VERSION = serial.VERSION
DATASETS = serial.DEFAULT_DATASETS
SEEDS = (0, 1, 2, 3, 4)
EXPECTED_METHODS = serial.EXPECTED_METHODS
EXPECTED_QNN = 32
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
}


def options():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--workers", type=int, default=12,
                   help="Independent CPU processes, each with single-threaded math (default: 12).")
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--max-new-jobs", type=int, default=None,
                   help="Start no more than this many new dataset/seed jobs.")
    p.add_argument("--summarize", action="store_true",
                   help="Summarize all five-seed datasets after collecting raw CSVs.")
    p.add_argument("--collect-only", action="store_true",
                   help="Collect completed jobs without starting any new QNN process.")
    p.add_argument("--verify-only", action="store_true",
                   help="Report complete and pending jobs without changing files.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print job plan and exit without changing files.")
    return p.parse_args()


def valid_job(path: Path, dataset: str, seed: int) -> bool:
    if not serial.valid_job(path, dataset, seed):
        return False
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    qnn = frame.loc[frame["readout"] == "qnn"]
    if len(qnn) != EXPECTED_QNN or qnn["method"].nunique() != EXPECTED_QNN:
        return False
    if "balanced_accuracy" not in frame.columns:
        return False
    return bool(np.isfinite(pd.to_numeric(frame["balanced_accuracy"], errors="coerce")).all())


def command(dataset: str, seed: int, out: Path) -> list[str]:
    raw, summary, config, _log = serial.job_paths(out, dataset, seed)
    return [
        sys.executable, "-u", str(RUNNER), "--mode", "hard_quick",
        "--datasets", dataset, "--seeds", str(seed), "--qubits", "4",
        "--latent-dims", "16", "--spectral-radii", "2.0",
        "--input-scales", "1.1", "--max-samples-per-class", "80",
        "--qnn-epochs", "60", "--qnn-layers", "2",
        "--n-workers", "1", "--out-raw", str(raw),
        "--out-summary", str(summary), "--out-config", str(config),
    ]


def environment_snapshot(jobs, workers):
    snap = serial.environment_snapshot()
    snap["parallel_plan"] = PLAN
    snap["datasets"] = list(dict.fromkeys(ds for ds, _ in jobs))
    snap["seeds"] = list(dict.fromkeys(seed for _, seed in jobs))
    snap["workers"] = int(workers)
    snap["worker_math_threads"] = dict(SINGLE_THREAD_ENV)
    snap["output_note"] = (
        "One independent corrected-continuation QNN job per process; "
        "no multiprocessing inside the QNN runner."
    )
    return snap


def check_environment(out, manifest):
    target = out / "environment.json"
    if target.is_file():
        previous = json.loads(target.read_text(encoding="utf-8"))
        for key in (
            "parallel_plan", "implementation_version", "git_commit",
            "packages", "python", "datasets", "seeds", "worker_math_threads",
        ):
            if previous.get(key) != manifest.get(key):
                raise SystemExit(
                    f"The existing output directory has a different {key}. "
                    f"Keep it intact and choose another --out-dir."
                )
        # Changing --workers affects scheduling only; completed jobs stay valid.
        return
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], cwd=ROOT,
        text=True, capture_output=True, check=False,
    )
    if freeze.returncode:
        raise SystemExit("Could not capture pip freeze for reproducibility.")
    (out / "environment.freeze.txt").write_text(freeze.stdout, encoding="utf-8")


def run_one(dataset, seed, out, registry, mutex, stop_event):
    raw, summary, config, log = serial.job_paths(out, dataset, seed)
    raw.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(SINGLE_THREAD_ENV)
    env["PYTHONUNBUFFERED"] = "1"
    start = time.perf_counter()
    with log.open("w", encoding="utf-8") as output:
        if stop_event.is_set():
            return dataset, seed, None, "Not started because the run was interrupted"
        proc = subprocess.Popen(
            command(dataset, seed, out), cwd=ROOT,
            env=env, stdout=output, stderr=subprocess.STDOUT,
        )
        with mutex:
            registry[(dataset, seed)] = proc
        try:
            if stop_event.is_set() and proc.poll() is None:
                proc.terminate()
            while proc.poll() is None:
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    minutes = (time.perf_counter() - start) / 60
                    print(f"  {dataset} seed={seed}: {minutes:.1f} min running; {log}",
                          flush=True)
            code = proc.returncode
        finally:
            with mutex:
                registry.pop((dataset, seed), None)
    elapsed = (time.perf_counter() - start) / 60
    if code != 0 or not valid_job(raw, dataset, seed) or not summary.is_file() or not config.is_file():
        return dataset, seed, elapsed, f"FAILED (exit={code}); inspect {log}"
    return dataset, seed, elapsed, None


def collect(out, jobs):
    frames = []
    incomplete = []
    for dataset, seed in jobs:
        raw, summary, config, _log = serial.job_paths(out, dataset, seed)
        if valid_job(raw, dataset, seed) and summary.is_file() and config.is_file():
            frames.append(pd.read_csv(raw))
        else:
            incomplete.append((dataset, seed))
    if frames:
        data = pd.concat(frames, ignore_index=True).sort_values(
            ["dataset", "seed", "method"]
        ).reset_index(drop=True)
        if data.duplicated(["dataset", "seed", "method"]).any():
            raise SystemExit("Duplicate dataset/seed/method in collected raw CSVs.")
        data.to_csv(out / "main_raw.csv", index=False)
    print(f"Collected {len(frames)}/{len(jobs)} complete jobs; "
          f"{len(incomplete)} pending.", flush=True)
    return incomplete


def summarize(out, jobs):
    complete_datasets = [
        ds for ds in dict.fromkeys(d for d, _ in jobs)
        if all((ds, seed) in jobs for seed in SEEDS)
        and all(
            valid_job(serial.job_paths(out, ds, seed)[0], ds, seed)
            for seed in SEEDS
        )
    ]
    if not complete_datasets:
        print("No complete five-seed dataset yet; skipping summary.", flush=True)
        return
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "summarize_continuation.py"),
         "--raw", str(out / "main_raw.csv"), "--out-dir", str(out)],
        cwd=ROOT, check=True,
    )


def main():
    opts = options()
    if len(set(opts.datasets)) != len(opts.datasets) or len(set(opts.seeds)) != len(opts.seeds):
        raise SystemExit("Dataset and seed lists must not contain duplicates.")
    if not opts.seeds or any(seed not in SEEDS for seed in opts.seeds):
        raise SystemExit("Seeds must be drawn from 0, 1, 2, 3, 4.")
    if opts.workers < 1:
        raise SystemExit("--workers must be at least one.")
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    if available and opts.workers > available:
        raise SystemExit(f"Requested {opts.workers} workers but only {available} logical CPUs are available.")
    if opts.max_new_jobs is not None and opts.max_new_jobs < 1:
        raise SystemExit("--max-new-jobs must be positive.")
    out = opts.out_dir.resolve()
    jobs = [(ds, seed) for ds in opts.datasets for seed in opts.seeds]
    completed = []
    pending = []
    for dataset, seed in jobs:
        raw, summary, config, _log = serial.job_paths(out, dataset, seed)
        if valid_job(raw, dataset, seed) and summary.is_file() and config.is_file():
            completed.append((dataset, seed))
        elif raw.exists():
            raise SystemExit(
                f"Incomplete raw output exists: {raw}. Inspect and move it aside "
                "before resuming; completed jobs will not be rerun."
            )
        else:
            pending.append((dataset, seed))
    print(
        f"Full continuation campaign: {len(jobs)} jobs, "
        f"{len(completed)} complete, {len(pending)} pending; "
        f"{opts.workers} independent workers, one math thread each.",
        flush=True,
    )
    if opts.verify_only or opts.dry_run:
        for dataset, seed in pending:
            print(f"  pending {dataset} seed={seed}", flush=True)
        return

    out.mkdir(parents=True, exist_ok=True)
    manifest = environment_snapshot(jobs, opts.workers)
    check_environment(out, manifest)
    limit = len(pending) if opts.max_new_jobs is None else min(opts.max_new_jobs, len(pending))
    to_run = [] if opts.collect_only else pending[:limit]
    registry = {}
    mutex = threading.Lock()
    stop_event = threading.Event()
    failed = []
    if to_run:
        print(f"Launching up to {min(opts.workers, len(to_run))} QNN processes.", flush=True)
        pool = ThreadPoolExecutor(max_workers=opts.workers)
        inflight = {}
        queue = iter(to_run)

        def submit_next():
            try:
                dataset, seed = next(queue)
            except StopIteration:
                return False
            print(f"Starting {dataset} seed={seed}", flush=True)
            future = pool.submit(run_one, dataset, seed, out, registry, mutex, stop_event)
            inflight[future] = (dataset, seed)
            return True

        try:
            for _ in range(min(opts.workers, len(to_run))):
                submit_next()
            while inflight:
                ready, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for future in ready:
                    dataset, seed = inflight.pop(future)
                    try:
                        _dataset, _seed, minutes, failure = future.result()
                    except Exception as exc:
                        failure = f"Unexpected worker failure: {exc}"
                        minutes = None
                    if failure:
                        failed.append((dataset, seed, failure))
                        print(f"{dataset} seed={seed}: {failure}", flush=True)
                    else:
                        print(f"Finished {dataset} seed={seed} in {minutes:.1f} min.",
                              flush=True)
                if not failed:
                    for _ in ready:
                        submit_next()
            if failed:
                print(f"{len(failed)} job(s) failed; no more new jobs started.",
                      flush=True)
        except KeyboardInterrupt:
            stop_event.set()
            with mutex:
                running = list(registry.values())
            for proc in running:
                if proc.poll() is None:
                    proc.terminate()
            print("Interrupted: terminating active workers; valid jobs are saved.",
                  flush=True)
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    incomplete = collect(out, jobs)
    if opts.summarize:
        summarize(out, jobs)
    if failed:
        raise SystemExit("Inspect failed job logs above before resuming.")
    if incomplete:
        print("Run the same command again to resume incomplete jobs.", flush=True)
    else:
        print("All 45 full 97-method jobs complete.", flush=True)


if __name__ == "__main__":
    main()
