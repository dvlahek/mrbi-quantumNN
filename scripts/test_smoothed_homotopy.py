"""Fast deterministic checks of the opt-in smoothed-residual MRBI pilot."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))
import main_qnn_benchmark as bench
import mrbi
from mrbi_homotopy import SmoothedHomotopyOptimizer
import run_smoothed_homotopy as pilot


def toy():
    x = np.array([0.2, -0.15], dtype=float)

    def F(z, x_):
        return np.asarray(z + 0.2 * np.sin(z) - x_, dtype=float)

    def J(z, x_):
        return np.diag(1.0 + 0.2 * np.cos(z))

    cfg = replace(
        bench.make_mrbi_cfg(pilot.audit.config_for(0)),
        maxiter_per_scale=8, refinement_iters=8,
    )
    return F, J, x, cfg


def test_objective_and_probes():
    F, J, x, cfg = toy()
    obj = SmoothedHomotopyOptimizer(
        F, J, x, 2, config=cfg, rng=np.random.default_rng(17),
        objective_cap_per_stage=120, residual_cap_per_stage=1440,
    )
    u_large = obj._get_probes(0.70)
    assert np.array_equal(u_large, obj._get_probes(0.02))
    z = np.array([0.31, -0.21])
    obj._active_weight = 0.75
    smooth_val = obj.objective(z, 0.7)
    obj._active_weight = 0.0
    plain_val = obj.objective(z, 0.7)
    assert np.isfinite(smooth_val) and np.isfinite(plain_val)
    assert abs(smooth_val - plain_val) > 1e-6, (
        "A large scale must change the early objective."
    )
    original = mrbi.MRBIOptimizer(
        F, J, x, 2, config=cfg, rng=np.random.default_rng(17)
    )
    original._probe_cache[0.7] = u_large
    original_val = original.objective(z, 0.7, detector_override=True)
    assert np.isclose(plain_val, original_val, rtol=1e-12, atol=1e-12)
    assert obj.WEIGHTS[-2:] == (0.0, 0.0)
    print("Same antithetic U, changed early objective and exact original final objective.")


def test_budget_and_rng():
    F, J, x, cfg = toy()
    opts = []
    for repeat in (False, True):
        rng = np.random.default_rng(47)
        opt = SmoothedHomotopyOptimizer(
            F, J, x, 2,
            config=replace(cfg, repeat_final_sigma=repeat),
            rng=rng,
            objective_cap_per_stage=36,
            residual_cap_per_stage=432,
        )
        candidate = opt.optimize()
        states = [
            (r["sigma"], r["smooth_weight"],
             r["objective_calls_stage"], r["residual_evals_stage"],
             r["budget_hit"], r["probe_sha256"])
            for r in opt.stage_records
        ]
        assert len(states) == 5
        assert [s[0] for s in states] == (
            [0.7, 0.25, 0.08, 0.02, 0.02] if not repeat
            else [0.02] * 5
        )
        assert [s[1] for s in states] == list(opt.WEIGHTS)
        assert all(0 <= s[2] <= 36 and 0 <= s[3] <= 432 for s in states)
        assert all(s[3] == s[2]*12 for s in states)
        assert len(set(s[-1] for s in states)) == 1
        assert candidate.n_objective_calls == (
            sum(s[2] for s in states) + 1
        )
        assert opt.optimizer_residual_evaluations == (
            sum(s[3] for s in states) + 12
        )
        assert np.all(np.isfinite(candidate.z_init))
        opts.append((opt, rng.normal(size=6)))
    assert opts[0][0].stage_records[0]["probe_sha256"] == (
        opts[1][0].stage_records[0]["probe_sha256"]
    )
    assert np.array_equal(opts[0][1], opts[1][1])
    print("Optimizer stage caps, shared probes, and paired RNG streams verified.")


def test_one_real_training_input():
    stages, pairs = pilot.run_job(
        "breast_cancer", 0, 1, 36, 432
    )
    assert len(stages) == 20 and len(pairs) == 1
    assert stages.groupby("arm")["stage"].nunique().eq(5).all()
    assert stages.groupby("arm")["probe_sha256"].nunique().eq(1).all()
    assert stages["probe_sha256"].nunique() == 1
    assert stages["smooth_weight"].iloc[3] == 0.0
    assert stages["objective_calls_stage"].le(36).all()
    assert stages["residual_evals_stage"].le(432).all()
    assert pairs["plan"].eq(pilot.PLAN).all()
    assert pairs["smoothed_continuation_objective_calls_optimizer"].gt(0).all()
    print("Four-arm paired pilot completed on a real training input, no QNN.")


if __name__ == "__main__":
    test_objective_and_probes()
    test_budget_and_rng()
    test_one_real_training_input()
