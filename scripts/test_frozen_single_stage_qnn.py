"""Synthetic-only checks for the frozen QNN runner.

Never evaluate seeds 10--14 in CI. PennyLane is mocked only for
readout-wiring assertions, not for any reported scientific result.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
sys.path.insert(0,str(ROOT/"scripts"))
import mrbi
import main_qnn_benchmark as bench
import multiscale_stage_audit as audit
import run_heldout_single_stage as single
import run_frozen_single_stage_qnn as qnn


def simple_case():
    layer=mrbi.ImplicitTanhLayer(
        W=np.diag([0.2,0.3]),
        U=np.array([[1.0,0.3,-0.2,0.4],[-0.4,0.2,0.6,0.1]]),
        b=np.array([0.0,0.01]),
    )
    inputs=np.array([
        [0.1,-0.2,0.3,0.0],
        [-0.1,0.2,0.0,0.3],
    ],dtype=float)
    cfg=replace(audit.config_for(0),run_qnn=True)
    return layer, inputs, cfg


def test_frozen_one_stage_solution_contract():
    layer,inputs,cfg=simple_case()
    x=inputs[0]
    root_cfg=bench.make_root_cfg(cfg)
    F,J,baseline=single.counted_functions(layer)
    zero,zf,zj=single.measured_root(
        F,J,baseline,x,np.zeros(layer.d),root_cfg,
    )
    assert zero.success
    zmin,_=mrbi.jacobian_health(J,x,zero.z_star)
    # Force the zero gate to fail only for this control-flow test.
    # The actual root solver remains unmocked.
    failed=replace(zero,success=False,solver_success_flag=False)
    observed={}
    for arm in qnn.ARMS:
        stage,result,solution=single.run_arm(
            layer,x,cfg,root_cfg,failed,zmin,
            baseline,zf,zj,zero.runtime_sec,
            0,0,arm,36,432,return_solution=True,
        )
        assert stage is not None and stage["stage"]==1
        assert stage["sigma"]==single.ARM_SPECS[arm][0]
        assert stage["smooth_weight"]==single.ARM_SPECS[arm][1]
        assert stage["objective_calls_stage"]<=36
        assert stage["residual_evals_stage"]<=432
        assert result["root_calls"]==2
        assert result["accepted_success"]==1
        assert result["checkpoint_success"]==1
        assert np.all(np.isfinite(solution))
        assert np.linalg.norm(layer.residual(solution,x))<=1e-8
        observed[arm]=stage["probe_sha256"]
    assert len(set(observed.values()))==1
    assert qnn.SEEDS==(10,11,12,13,14)
    print("Frozen one-stage solution return, root gate, same probes and budget verified.")


def test_feature_shapes_and_qnn_seed_wiring():
    layer,inputs,cfg=simple_case()
    train,train_stats=qnn.numerical_features(
        inputs,layer,cfg,0,"train",0,
    )
    test,test_stats=qnn.numerical_features(
        inputs,layer,cfg,0,"test",len(inputs),
    )
    assert set(train)==set(test)=={"zero",*qnn.ARMS}
    assert len(train_stats)==len(test_stats)==6
    assert train_stats.zero_success.eq(1).all()
    assert train_stats.candidate_attempted.eq(0).all()
    arrays={"pca_qnn":(inputs,inputs)}
    for source,method in (
        ("zero","zero_qnn"),
        ("smoothed_coarse","smoothed_coarse_qnn"),
        ("smoothed_fine","smoothed_fine_qnn"),
    ):
        arrays[method]=bench.standardize_pair(
            train[source],test[source]
        )
    calls=[]

    def stub_train(Xtr,ytr,Xte,yte,received_cfg,seed):
        assert received_cfg.run_qnn
        calls.append((seed,Xtr.shape[1],len(Xtr),len(Xte)))
        return {
            "accuracy":0.5,"balanced_accuracy":0.5,
            "f1":0.0,"roc_auc":0.5,"train_time_sec":0.1,
        }

    with patch.object(bench,"HAS_QNN",True),patch.object(
        bench,"train_qnn",side_effect=stub_train
    ):
        results=qnn.evaluate_qnns(
            "synthetic",0,cfg,arrays,
            np.array([0,1]),np.array([0,1]),
            train_stats.assign(method=train_stats.method),
            test_stats,
        )
    assert list(results.method)==list(qnn.METHODS)
    assert len(calls)==4
    assert [p[0] for p in calls]==[777]*4
    assert [p[1] for p in calls]==[4,2,2,2]
    assert results.balanced_accuracy.eq(0.5).all()
    assert results.loc[
        results.method.eq("pca_qnn"),
        "feature_mean_test_algorithm_F",
    ].iloc[0]==0
    assert all(
        np.isfinite(x) for x in results.train_time_sec
    )
    print("Paired feature/readout shapes, fixed seeds, PCA/zero comparators verified.")


if __name__=="__main__":
    test_frozen_one_stage_solution_contract()
    test_feature_shapes_and_qnn_seed_wiring()
