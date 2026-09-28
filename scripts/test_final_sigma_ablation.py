"""Fast source-level and numerical checks for the final-sigma ablation (no QNN)."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

import mrbi
import main_qnn_benchmark as bench
import final_sigma_ablation_job as control_job
import run_final_sigma_ablation as driver
import summarize_final_sigma_ablation as summary


def check_stage_schedule_and_rng():
    profile = next(p for p in bench.get_mrbi_profiles("hard_quick")
                   if p.name == "full_balanced")
    cfg = bench.apply_mrbi_profile(bench.ExperimentConfig(), profile)
    normal = bench.make_mrbi_cfg(cfg)
    fixed = bench.make_mrbi_cfg(replace(cfg, repeat_final_sigma=True))
    assert normal.sigmas == fixed.sigmas == (0.7, 0.25, 0.08, 0.02)
    assert normal.maxiter_per_scale == fixed.maxiter_per_scale == 60
    assert normal.refinement_iters == fixed.refinement_iters == 60
    assert normal.mc_samples == fixed.mc_samples == 10
    assert normal.repeat_final_sigma is False and fixed.repeat_final_sigma is True

    def F(z, x):
        return z - x

    def J(z, x):
        return np.eye(len(z))

    schedules = []
    caches = []
    next_draw = []
    for config in (
        replace(normal, maxiter_per_scale=2, refinement_iters=2),
        replace(fixed, maxiter_per_scale=2, refinement_iters=2),
    ):
        opt = mrbi.MRBIOptimizer(
            F, J, np.array([0.2, -0.1]), dim=2, config=config,
            rng=np.random.default_rng(1234),
        )
        calls = []
        original = opt._optimize_single_scale

        def recording(z0, sigma, maxiter, *, detector_override=None):
            calls.append((float(sigma), int(maxiter)))
            return original(
                z0, sigma, maxiter, detector_override=detector_override
            )

        opt._optimize_single_scale = recording
        opt.optimize()
        schedules.append(calls)
        caches.append({k: v.copy() for k, v in opt._probe_cache.items()})
        next_draw.append(opt.rng.normal(size=5))
    assert schedules[0] == [(0.7, 2), (0.25, 2), (0.08, 2), (0.02, 2), (0.02, 2)]
    assert schedules[1] == [(0.02, 2)] * 5
    assert caches[0].keys() == caches[1].keys()
    for k in caches[0]:
        assert np.array_equal(caches[0][k], caches[1][k]), k
    assert np.array_equal(next_draw[0], next_draw[1])
    try:
        mrbi.MRBIOptimizer(
            F, J, np.array([0.2, -0.1]), 2,
            config=replace(fixed, use_fixed_probes_per_scale=False),
            rng=np.random.default_rng(1234),
        ).optimize()
    except ValueError:
        pass
    else:
        raise AssertionError("Unfixed detector probes must not be accepted")
    print("Stage, optimizer-budget and probe-RNG checks passed.")


def check_partial_summary_and_driver():
    metrics = {
        "balanced_accuracy": 0.84,
        "stat_test_forced_success_rate": 0.8,
        "stat_test_forced_final_mean_residual": 0.0001,
        "stat_test_forced_mean_objective_calls": 120.0,
        "stat_feature_total_time_sec": 8.0,
    }
    settings = {
        "n_qubits": 4, "latent_dim": 16, "spectral_radius": 2.0,
        "input_scale": 1.1, "max_samples_per_class": 80,
    }
    reference, control = [], []
    for seed in range(5):
        ref = dict(metrics, **settings, dataset="breast_cancer", seed=seed,
                   method=driver.SOURCE_METHOD, implementation_version=driver.VERSION,
                   campaign_design=driver.SOURCE_PLAN, profile_name="full_balanced",
                   readout="qnn")
        cont = dict(metrics, **settings, dataset="breast_cancer", seed=seed,
                    method=driver.CONTROL_METHOD, implementation_version=driver.VERSION,
                    ablation_design=driver.PLAN, ablation_arm="final_sigma_repeated",
                    readout="qnn")
        ref["balanced_accuracy"] = 0.87
        ref["stat_test_forced_mean_objective_calls"] = 130.0
        control.append(cont)
        reference.append(ref)
        for index in range(24):
            dummy = dict(ref, method=f"core_dummy_{index}",
                         readout="qnn" if index < 7 else "logreg")
            reference.append(dummy)
    with TemporaryDirectory() as temp:
        root = Path(temp)
        reference_path, control_path = root / "reference.csv", root / "control.csv"
        pd.DataFrame(reference).to_csv(reference_path, index=False)
        pd.DataFrame(control).to_csv(control_path, index=False)
        reference_env = driver.snapshot()
        reference_env["campaign_design"] = driver.SOURCE_PLAN
        reference_env_path = root / "environment.json"
        reference_env_path.write_text(json.dumps(reference_env), encoding="utf-8")
        core_sha, core_plan = driver.source_reference(
            reference_path, [("breast_cancer", 0)]
        )
        assert len(core_sha) == 64 and core_plan == driver.SOURCE_PLAN
        _row, detected_plan = control_job.get_reference(reference_path, "breast_cancer", 0)
        assert detected_plan == driver.SOURCE_PLAN
        assert driver.valid_job(control_path, "breast_cancer", 0) is False
        one_path = root / "one.csv"
        pd.DataFrame(control[:1]).to_csv(one_path, index=False)
        assert driver.valid_job(one_path, "breast_cancer", 0)
        run = subprocess.run([
            sys.executable, str(ROOT / "scripts" / "run_final_sigma_ablation.py"),
            "--reference-raw", str(reference_path),
            "--reference-env", str(reference_env_path),
            "--datasets", "breast_cancer", "--seeds", "0", "--dry-run",
        ], capture_output=True, text=True, check=True)
        assert "1 pending" in run.stdout
        summary.analyse(control_path, reference_path, root)
        result = json.loads((root / "ablation_statistics.json").read_text())
        assert result["n_complete_datasets"] == 1
        assert result["n_paired_seeds"] == 5
        assert np.isclose(result["mean_dataset_delta_qnn_ba"], 0.03)

        # The user's full 97-method campaign has no campaign_design column;
        # the 45 merged jobs retain provenance in three shard environments.
        full_reference = [
            {key: val for key, val in row.items() if key != "campaign_design"}
            for row in reference
        ]
        for seed in range(5):
            model = next(row for row in full_reference
                         if row["seed"] == seed and row["method"] == driver.SOURCE_METHOD)
            for index in range(72):
                full_reference.append(dict(
                    model, method=f"full_dummy_{index}",
                    readout="qnn" if index < 24 else "logreg",
                ))
        full_ref_path = root / "full_reference.csv"
        pd.DataFrame(full_reference).to_csv(full_ref_path, index=False)
        full_sha, full_plan = driver.source_reference(
            full_ref_path, [("breast_cancer", seed) for seed in range(5)]
        )
        assert len(full_sha) == 64 and full_plan == driver.FULL_SOURCE_PLAN
        _row, detected_full = control_job.get_reference(
            full_ref_path, "breast_cancer", 0
        )
        assert detected_full == driver.FULL_SOURCE_PLAN
        full_env = dict(reference_env)
        full_env.pop("campaign_design", None)
        full_env["datasets"] = ["breast_cancer"]
        full_env["seeds"] = list(range(5))
        full_env_path = root / "full_environment.json"
        full_env_path.write_text(json.dumps(full_env), encoding="utf-8")
        full_run = subprocess.run([
            sys.executable, str(ROOT / "scripts" / "run_final_sigma_ablation.py"),
            "--reference-raw", str(full_ref_path),
            "--reference-env", str(full_env_path),
            "--datasets", "breast_cancer", "--seeds", "0", "--dry-run",
        ], capture_output=True, text=True, check=True)
        assert "1 pending" in full_run.stdout
        summary.analyse(control_path, full_ref_path, root)
        full_result = json.loads((root / "ablation_statistics.json").read_text())
        assert full_result["source_campaign_design"] == driver.FULL_SOURCE_PLAN
        assert full_result["n_paired_seeds"] == 5
        assert np.isclose(full_result["mean_dataset_delta_qnn_ba"], 0.03)
    print("Both core and full references, resumable driver and paired summary passed.")


if __name__ == "__main__":
    check_stage_schedule_and_rng()
    check_partial_summary_and_driver()
