"""Non-held-out tests for the frozen one-stage coarse/fine comparison.

The tests use synthetic equations only, never the fresh seeds 5--9.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))
import mrbi
import main_qnn_benchmark as bench
import multiscale_stage_audit as audit
import run_heldout_single_stage as check


def synthetic_case():
    layer = mrbi.ImplicitTanhLayer(
        W=np.diag([0.3, 0.4]),
        U=np.eye(2),
        b=np.array([0.01, -0.02]),
    )
    x = np.array([0.2, -0.1], dtype=float)
    cfg = audit.config_for(0)  # Development seed for configuration only.
    root_cfg = bench.make_root_cfg(cfg)
    F, J, base = check.counted_functions(layer)
    zero, zf, zj = check.measured_root(
        F, J, base, x, np.zeros(2), root_cfg,
    )
    assert zero.success, "Synthetic linear-stable zero solve should converge"
    return layer, x, cfg, root_cfg, F, J, base, zero, zf, zj


def test_zero_success_skips_all_arms():
    layer,x,cfg,root_cfg,F,J,base,zero,zf,zj = synthetic_case()
    snapshot = dict(base)
    for arm in check.ARMS:
        stage, row = check.run_arm(
            layer,x,cfg,root_cfg,zero,np.nan,base,
            zf,zj,zero.runtime_sec,0,0,arm,36,432,
        )
        assert stage is None and row["accepted_success"] == 1
        assert row["candidate_attempted"] == 0
        assert row["root_calls"] == 1
        assert row["optimizer_objective_calls"] == 0
        assert row["total_F_calls"] == base["F"]
        assert row["total_J_calls"] == base["J"]
    assert base == snapshot
    print("Zero-success gate skips MRBI in all four arms.")


def test_frozen_one_stage_and_exact_costs():
    layer,x,cfg,root_cfg,F,J,base,zero,zf,zj = synthetic_case()
    zero_smin, _ = mrbi.jacobian_health(J, x, zero.z_star)
    # Synthetic gate failure exercises the checkpoint even though this
    # intentionally easy toy system would succeed from zero.
    failed_zero = replace(zero, success=False, solver_success_flag=False)
    snapshot = dict(base)
    rows = {}
    stage_rows = {}
    for arm in check.ARMS:
        stage, row = check.run_arm(
            layer,x,cfg,root_cfg,failed_zero,zero_smin,base,
            zf,zj,zero.runtime_sec,0,0,arm,36,432,
        )
        assert stage is not None
        assert stage["stage"] == 1
        assert stage["sigma"] == check.ARM_SPECS[arm][0]
        assert stage["smooth_weight"] == check.ARM_SPECS[arm][1]
        assert 0 < stage["objective_calls_stage"] <= 36
        assert (stage["residual_evals_stage"]
                == 12 * stage["objective_calls_stage"])
        assert stage["residual_evals_stage"] <= 432
        assert row["candidate_attempted"] == 1
        assert row["root_calls"] == 2
        assert row["checkpoint_success"] == row["accepted_success"] == 1
        assert row["diagnostic_F_calls"] == 11
        assert row["diagnostic_J_calls"] == 1
        assert row["total_F_calls"] - row["algorithm_F_calls"] == 11
        assert row["total_J_calls"] - row["algorithm_J_calls"] == 1
        assert row["optimizer_objective_calls"] == stage["objective_calls_stage"]
        assert row["optimizer_F_calls"] == stage["residual_evals_stage"]
        assert row["checkpoint_root_F_calls"] == stage["stage_root_F_calls"]
        rows[arm] = row
        stage_rows[arm] = stage
    assert len({r["probe_sha256"] for r in rows.values()}) == 1
    assert base == snapshot
    assert set(check.SEEDS) == {5,6,7,8,9}
    assert check.PLAN == "heldout_single_stage_sigma_070_vs_002_v1"
    print("Four one-stage arms: common probes, exact F/J counts, frozen fresh seeds.")


def test_digest_is_deterministic():
    a=np.array([1.,2.],dtype=float)
    assert check.sha_arrays(a)==check.sha_arrays(a.copy())
    assert check.sha_arrays(a)!=check.sha_arrays(a+1)
    print("Independent raw-data/layer SHA-256 helper verified.")


if __name__ == "__main__":
    test_zero_success_skips_all_arms()
    test_frozen_one_stage_and_exact_costs()
    test_digest_is_deterministic()
