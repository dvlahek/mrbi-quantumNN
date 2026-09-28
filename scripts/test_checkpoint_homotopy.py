"""Fast numerical and control-flow tests for the stage-one checkpoint pilot."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"experiments"))
sys.path.insert(0, str(ROOT/"scripts"))
import main_qnn_benchmark as bench
import mrbi
import multiscale_stage_audit as audit
import run_checkpoint_homotopy as pilot


def test_root_success_early_stop():
    """Synthetic control flow: stop after exactly the first root-successful stage."""
    x = np.array([0.3, -0.2], dtype=float)

    def F(z, x_):
        return np.asarray(z - x_, dtype=float)

    def J(z, x_):
        return np.eye(2)

    class Layer:
        d = 2
        residual = staticmethod(F)
        jacobian = staticmethod(J)

    # Use the exact full_balanced configuration with smaller per-stage limits.
    cfg = replace(audit.config_for(0), maxiter_per_scale=2, refinement_iters=2)
    zero = mrbi.RootResult(
        success=False, z_star=np.zeros(2), residual=0.4, nfev=2,
        runtime_sec=0.001, message="synthetic zero failure",
        solver_success_flag=False, has_finite_z=True,
        has_finite_residual=True,
    )
    count = [0]

    def synthetic_stage_root(F_, J_, x_, z_, cfg=None):
        count[0] += 1
        return mrbi.RootResult(
            success=True, z_star=x_.copy(), residual=0.0, nfev=2,
            runtime_sec=0.001, message="synthetic success",
            solver_success_flag=True, has_finite_z=True,
            has_finite_residual=True,
        )

    with patch.object(mrbi, "solve_root", side_effect=synthetic_stage_root):
        stages, result = pilot.run_arm(
            Layer(), x, cfg, bench.make_root_cfg(cfg), zero, 2, 1,
            seed=0, index=0, arm="smoothed_continuation",
            objective_cap=60, residual_cap=720,
        )
    assert count == [1]
    assert len(stages) == 1
    assert result["checkpoint_success"] == result["early_exit"] == 1
    assert result["accepted_success"] == 1
    assert result["root_attempts"] == 2  # shared zero + checkpoint
    assert result["final_root_attempted"] == 0
    assert result["optimizer_stages"] == 1
    print("Synthetic successful stage-one root triggers immediate exit.")


def test_one_real_input():
    stages, pairs = pilot.run_job("breast_cancer", 0, 1, 36, 432)
    assert len(pairs) == 1
    if int(pairs.zero_success.iloc[0]) == 0:
        assert stages.groupby("arm").stage.first().eq(1).all()
        assert len(stages.groupby("arm")) == 4
        assert stages.groupby("arm").probe_sha256.nunique().eq(1).all()
        assert stages.groupby("arm").stage.max().le(5).all()
        assert stages.objective_calls_stage.le(36).all()
        assert stages.residual_evals_stage.le(432).all()
        assert stages[stages.stage.eq(1)].probe_sha256.nunique() == 1
        for arm in pilot.ARMS:
            row = pairs.iloc[0]
            assert row[f"{arm}_root_attempts"] in (2, 3)
            assert row[f"{arm}_total_F_calls"] >= (
                row[f"{arm}_root_F_calls"]
            )
            assert row[f"{arm}_total_J_calls"] >= (
                row[f"{arm}_root_J_calls"]
            )
    assert pairs.plan.eq(pilot.PLAN).all()
    assert not any(col.startswith("qnn") for col in pairs.columns)
    print("Real input: identical probes, F/J accounting, no QNN training.")


if __name__ == "__main__":
    test_root_success_early_stop()
    test_one_real_input()
