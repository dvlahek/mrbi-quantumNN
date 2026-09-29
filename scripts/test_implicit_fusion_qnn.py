"""Synthetic-only tests of PCA+implicit MRBI-QNN fusion.

Do not run seed-20--24 data or train QNN models in CI.
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
import run_implicit_fusion_qnn as fusion


def toy_gate_inputs():
    # Four inputs: zero success, certified coarse rescue, attempted
    # unsuccessful MRBI substitution, failed checkpoint and zero fallback.
    zero=np.tile(np.array([1.0,2.0,3.0,4.0])[:,None],(1,16))
    coarse=zero.copy()
    coarse[1,:]=21.0
    coarse[2,:]=31.0
    coarse[3,:]=41.0
    rows=[]
    for method in ("fusion_zero_qnn","fusion_mrbi_qnn"):
        for i in range(4):
            is_zero=(i==0)
            is_coarse=(method=="fusion_mrbi_qnn")
            checkpoint=int(is_coarse and i==1)
            attempt=int(is_coarse and not is_zero)
            rows.append({
                "split":"train","input_index":i,"method":method,
                "zero_success":int(is_zero),
                "checkpoint_success":checkpoint,
                "accepted_success":int(is_zero or checkpoint),
                "used_mrbi":int(is_coarse and i in (1,2,3)),
                "accepted_residual":(
                    0.0 if is_zero or checkpoint else
                    (0.26 if is_coarse else 0.38)
                ),
                "candidate_attempted":attempt,
                "algorithm_F_calls":100 if is_coarse else 10,
                "algorithm_J_calls":12 if is_coarse else 2,
                "root_calls":2 if attempt else 1,
                "optimizer_objective_calls":36 if attempt else 0,
                "total_wall_sec":0.1,
            })
    return {"zero":zero,"smoothed_coarse":coarse},pd.DataFrame(rows)


def test_strict_certification_keeps_zero_fallback():
    old,stats=toy_gate_inputs()
    result,new=fusion.certify_features(old,stats)
    assert fusion.SEEDS==(20,21,22,23,24)
    assert len(new)==3*4
    assert set(result)=={"zero","smoothed_coarse","certified_coarse"}
    assert np.array_equal(result["certified_coarse"][0],old["zero"][0])
    assert np.array_equal(
        result["certified_coarse"][1],old["smoothed_coarse"][1]
    )
    assert np.array_equal(
        result["certified_coarse"][2:],old["zero"][2:]
    )
    cert=(new[new.method.eq("fusion_certified_mrbi_qnn")]
          .sort_values("input_index"))
    assert cert.used_mrbi.tolist()==[0,1,0,0]
    assert cert.accepted_success.tolist()==[1,1,0,0]
    assert cert.certified_zero_fallback.tolist()==[0,0,1,1]
    assert cert.algorithm_F_calls.tolist()==[100]*4
    assert np.array_equal(old["zero"],result["zero"])
    assert np.array_equal(old["smoothed_coarse"],result["smoothed_coarse"])
    print("Strict root gate changes only noncertified implicit features.")


def test_fusion_preserves_pca_and_train_only_scaler():
    old,stats=toy_gate_inputs()
    result,_=fusion.certify_features(old,stats)
    xtr=np.array([
        [0.10,0.20,0.30,0.40],
        [0.20,0.30,0.40,0.50],
        [0.30,0.40,0.50,0.60],
        [0.40,0.50,0.60,0.70],
    ],dtype=float)
    xte=np.array([
        [0.60,0.20,0.50,0.10],
        [0.70,0.30,0.60,0.20],
    ],dtype=float)
    test={k:v[:2]+20.0 for k,v in result.items()}
    features=fusion.build_fusion_features(xtr,xte,result,test)
    assert set(features)==set(fusion.METHODS)
    assert features["pca_qnn"][0].shape==(4,4)
    assert features["pca_qnn"][1].shape==(2,4)
    for method in fusion.METHODS[1:]:
        tr,te=features[method]
        assert tr.shape==(4,20) and te.shape==(2,20)
        assert np.array_equal(tr[:,:4],xtr)
        assert np.array_equal(te[:,:4],xte)
        assert np.allclose(tr[:,4:].mean(axis=0),0,atol=1e-12)
        assert not np.allclose(te[:,4:].mean(axis=0),0)
    print("All fusion arms retain untouched PCA and training-fitted latent scales.")


def test_readout_seed_and_four_arm_contract():
    old,stats=fusion.certify_features(*toy_gate_inputs())
    x=np.array([
        [0.1,0.2,0.3,0.4],
        [0.2,0.3,0.4,0.5],
        [0.3,0.4,0.5,0.6],
        [0.4,0.5,0.6,0.7],
    ])
    features=fusion.build_fusion_features(x,x,old,old)
    cfg=replace(audit.config_for(0),run_qnn=True)
    calls=[]

    def mock_qnn(Xtr,ytr,Xte,yte,received_cfg,seed):
        calls.append((seed,Xtr.shape[1],len(Xtr),len(Xte)))
        assert received_cfg.qnn_epochs==60
        return {
            "accuracy":0.5,"balanced_accuracy":0.5,
            "f1":0.5,"roc_auc":0.5,"train_time_sec":0.1,
        }

    with patch.object(bench,"HAS_QNN",True),patch.object(
        bench,"train_qnn",side_effect=mock_qnn
    ):
        raw=fusion.evaluate_qnns(
            "synthetic",0,cfg,features,
            np.array([0,1,0,1]),np.array([0,1,0,1]),
            stats,stats.assign(split="test"),
        )
    assert list(raw.method)==list(fusion.METHODS)
    assert len(calls)==4
    assert [c[0] for c in calls]==[777]*4
    assert [c[1] for c in calls]==[4,20,20,20]
    assert raw.loc[
        raw.method.eq("fusion_certified_mrbi_qnn"),
        "feature_mean_test_algorithm_F",
    ].iloc[0]==100.0
    assert raw.balanced_accuracy.eq(0.5).all()
    print("Four QNN inputs, paired fusion architecture, QNN seeds and costs verified.")


if __name__=="__main__":
    test_strict_certification_keeps_zero_fallback()
    test_fusion_preserves_pca_and_train_only_scaler()
    test_readout_seed_and_four_arm_contract()
