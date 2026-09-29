"""Synthetic-only contract checks for the prespecified certificate-gated QNN.

No test may call run_job or load real datasets for seeds 15--19.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
sys.path.insert(0,str(ROOT/"scripts"))
import main_qnn_benchmark as bench
import multiscale_stage_audit as audit
import run_certified_gate_qnn as gate


def synthetic_outputs():
    """Four toy roots: zero success, coarse rescue, fine rescue, no rescue."""
    zero=np.array([
        [1.0,1.0], [2.0,2.0], [3.0,3.0], [4.0,4.0],
    ],dtype=float)
    coarse=np.array([
        [1.0,1.0], [20.0,20.0], [30.0,30.0], [4.0,4.0],
    ],dtype=float)
    fine=np.array([
        [1.0,1.0], [21.0,21.0], [31.0,31.0], [40.0,40.0],
    ],dtype=float)
    # The old forced-accept rule is permitted to select a failed MRBI
    # candidate with a smaller *nonzero* residual, which is exactly
    # the failure that the strict new gate must prevent.
    old=[
        ("zero_qnn", [1,0,0,0], [0,0,0,0],
         [0,0,0,0], [0.0,0.4,0.3,0.2]),
        ("smoothed_coarse_qnn", [1,1,0,0], [0,1,0,0],
         [0,1,1,0], [0.0,0.0,0.29,0.2]),
        ("smoothed_fine_qnn", [1,0,1,0], [0,0,1,0],
         [0,1,1,1], [0.0,0.35,0.0,0.19]),
    ]
    records=[]
    for method,accepted,checkpoint,used,residual in old:
        for idx in range(4):
            records.append({
                "split":"train","input_index":idx,"method":method,
                "zero_success":int(idx==0),
                "accepted_success":accepted[idx],
                "checkpoint_success":checkpoint[idx],
                "used_mrbi":used[idx],
                "accepted_residual":residual[idx],
                "candidate_attempted":int(idx!=0 and method!="zero_qnn"),
                "algorithm_F_calls":100 if method!="zero_qnn" else 10,
                "algorithm_J_calls":12 if method!="zero_qnn" else 2,
                "optimizer_objective_calls":10 if method!="zero_qnn" else 0,
                "root_calls":2 if method!="zero_qnn" else 1,
                "budget_hit":0,
                "total_wall_sec":0.1,
                "probe_sha256":"shared" if method!="zero_qnn" else "",
            })
    return {
        "zero":zero,"smoothed_coarse":coarse,"smoothed_fine":fine,
    },pd.DataFrame(records)


def test_exact_solver_certificate_gate():
    source,records=synthetic_outputs()
    result,stats=gate.certify_features(source,records)
    assert gate.SEEDS==(15,16,17,18,19)
    assert len(stats)==20
    assert np.array_equal(
        result["certified_coarse"],
        np.array([
            source["zero"][0],source["smoothed_coarse"][1],
            source["zero"][2],source["zero"][3],
        ])
    )
    assert np.array_equal(
        result["certified_fine"],
        np.array([
            source["zero"][0],source["zero"][1],
            source["smoothed_fine"][2],source["zero"][3],
        ])
    )
    assert np.array_equal(result["smoothed_coarse"],source["smoothed_coarse"])
    assert np.array_equal(result["zero"],source["zero"])
    cg=stats[stats.method.eq("certified_coarse_qnn")].sort_values("input_index")
    fg=stats[stats.method.eq("certified_fine_qnn")].sort_values("input_index")
    assert cg.used_mrbi.tolist()==[0,1,0,0]
    assert cg.accepted_success.tolist()==[1,1,0,0]
    assert cg.certified_zero_fallback.tolist()==[0,0,1,1]
    assert cg.accepted_residual.tolist()==[0.0,0.0,0.3,0.2]
    assert fg.used_mrbi.tolist()==[0,0,1,0]
    assert fg.accepted_success.tolist()==[1,0,1,0]
    assert fg.certified_zero_fallback.tolist()==[0,1,0,1]
    assert fg.accepted_residual.tolist()==[0.0,0.4,0.0,0.2]
    assert cg.algorithm_F_calls.tolist()==[100]*4
    print("Strict success gate preserves only certified roots and all work counts.")


def test_readout_matched_input_and_seed():
    source,records=synthetic_outputs()
    outputs,stats=gate.certify_features(source,records)
    stats=stats.assign(split="train")
    test=stats.assign(split="test")
    x=np.array([
        [0.1,0.2,0.3,0.4],
        [0.3,0.4,0.2,0.1],
        [0.5,0.6,0.7,0.8],
        [0.1,0.2,0.6,0.5],
    ])
    z={"pca_qnn":(x,x)}
    for source_name,method in (
        ("zero","zero_qnn"),
        ("smoothed_coarse","smoothed_coarse_qnn"),
        ("smoothed_fine","smoothed_fine_qnn"),
        ("certified_coarse","certified_coarse_qnn"),
        ("certified_fine","certified_fine_qnn"),
    ):
        z[method]=bench.standardize_pair(
            outputs[source_name],outputs[source_name]
        )
    cfg=replace(audit.config_for(0),run_qnn=True)
    observed=[]

    def stub_qnn(Xtr,ytr,Xte,yte,config,seed):
        observed.append((seed,Xtr.shape[1],len(ytr),len(yte)))
        return {"accuracy":0.5,"balanced_accuracy":0.5,
                "f1":0.5,"roc_auc":0.5,"train_time_sec":0.1}

    with patch.object(bench,"HAS_QNN",True),patch.object(
        bench,"train_qnn",side_effect=stub_qnn
    ):
        metrics=gate.evaluate_qnns(
            "synthetic",0,cfg,z,np.array([0,1,0,1]),
            np.array([0,1,0,1]),stats,test,
        )
    assert list(metrics.method)==list(gate.METHODS)
    assert len(observed)==6
    assert set(x[0] for x in observed)=={777}
    assert [x[1] for x in observed]==[4,2,2,2,2,2]
    assert metrics.loc[
        metrics.method.eq("certified_coarse_qnn"),
        "feature_mean_test_algorithm_F",
    ].iloc[0]==100.0
    print("All six readouts have fixed, matched QNN seeds and cost accounting.")


if __name__=="__main__":
    test_exact_solver_certificate_gate()
    test_readout_matched_input_and_seed()
