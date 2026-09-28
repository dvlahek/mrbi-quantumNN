"""Numerical invariance checks for the stage-level MRBI audit (no QNN)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))
import mrbi
import multiscale_stage_audit as audit


def test_trace_does_not_change_optimizer():
    x = np.array([0.2, -0.1], dtype=float)

    def F(z, input_):
        return np.asarray(z - input_, dtype=float)

    def J(z, input_):
        return np.eye(len(z))

    original = mrbi.MRBIConfig(
        sigmas=(0.7, 0.25, 0.08, 0.02),
        mc_samples=4, maxiter_per_scale=2, refinement_iters=2,
        gamma=0.04, beta=0.02, search_radius=1.5,
        use_fixed_probes_per_scale=True,
    )
    root_cfg = mrbi.RootSolveConfig()
    for arm in audit.ARMS:
        local = replace(
            original, repeat_final_sigma=(arm == "repeated_final_sigma")
        )
        ref_rng = np.random.default_rng(180)
        ref = mrbi.MRBIOptimizer(
            F, J, x, dim=2, config=local, rng=ref_rng
        ).optimize()
        ref_next = ref_rng.normal(size=8)
        audit_rng = np.random.default_rng(180)
        stages, roots, traced = audit.trace_arm(
            F, J, x, 2, original, root_cfg, audit_rng, arm
        )
        assert len(stages) == audit.STAGE_COUNT
        assert len(roots) == audit.STAGE_COUNT
        assert np.array_equal(ref.z_init, traced.z_init)
        assert ref.n_objective_calls == traced.n_objective_calls
        assert np.array_equal(ref_next, audit_rng.normal(size=8))
        assert stages[-1]["objective_calls_cumulative"] + 1 == traced.n_objective_calls
        expected = ([0.7, 0.25, 0.08, 0.02, 0.02]
                    if arm == "continuation" else [0.02]*5)
        assert [row["sigma"] for row in stages] == expected
        assert [row["maxiter"] for row in stages] == [2]*5
        assert all("candidate_json" in row for row in stages)
    print("Stage-level tracing preserves candidates, RNG and objective calls.")


def test_one_training_input():
    stage, pair = audit.run_job("breast_cancer", 0, 1)
    assert len(stage) == 10 and len(pair) == 1
    assert stage.groupby("arm")["stage"].nunique().eq(5).all()
    assert stage["objective_calls_stage"].gt(0).all()
    assert pair["objective_calls_cont"].gt(0).all()
    assert pair["objective_calls_final"].gt(0).all()
    assert pair["plan"].eq(audit.PLAN).all()
    print("One real training-input diagnostic completed without QNN training.")


if __name__ == "__main__":
    test_trace_does_not_change_optimizer()
    test_one_training_input()
