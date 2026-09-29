"""Synthetic-only checks; never download or train external held-out QNNs."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
sys.path.insert(0,str(ROOT/"scripts"))
import main_qnn_benchmark as bench
import multiscale_stage_audit as audit
import download_external_fusion_data as external
import run_implicit_fusion_qnn as prior
import run_external_fusion_qnn as runner
from test_implicit_fusion_qnn import toy_gate_inputs


def make_valid_synthetic_source(name):
    """Construct a CSV fixture of the correct public schema, NOT UCI data."""
    spec=external.SPECS[name]
    keys=list(spec.label_map)
    lines=[]
    for i in range(spec.n_rows):
        features=[
            str(round((i+1)*(j+1)/1000,6))
            for j in range(spec.n_features)
        ]
        lines.append(",".join(features+[keys[i%2]]))
    return ("\n".join(lines)+"\n").encode("ascii")


def test_external_source_is_frozen_and_revalidated():
    assert runner.DATASETS==("banknote","ionosphere","sonar")
    assert runner.SEEDS==(25,26,27,28,29)
    assert len(runner.METHODS)==6
    with TemporaryDirectory() as tmp:
        dir=Path(tmp)
        descriptions={}
        for name,spec in external.SPECS.items():
            raw=make_valid_synthetic_source(name)
            X,y=external.parse_dataset(name,raw)
            assert X.shape==(spec.n_rows,spec.n_features)
            assert set(np.unique(y))=={0,1}
            desc=external.descriptor(name,raw)
            assert desc["sha256"]==hashlib.sha256(raw).hexdigest()
            (dir/spec.filename).write_bytes(raw)
            descriptions[name]=desc
        (dir/"data_manifest.json").write_text(
            json.dumps({
                "plan":external.PLAN,"datasets":descriptions
            }),
            encoding="utf-8",
        )
        sources,manifest=external.load_sources(dir)
        assert set(sources)==set(external.DATASETS)
        assert manifest["datasets"]==descriptions
        banknote=dir/external.SPECS["banknote"].filename
        banknote.write_bytes(banknote.read_bytes().replace(
            b"0.001,0.002,0.003,0.004",b"0.002,0.002,0.003,0.004",
            1,
        ))
        try:
            external.load_sources(dir)
        except RuntimeError as exc:
            assert "frozen manifest" in str(exc)
        else:
            raise AssertionError("Data drift should fail")
        try:
            external.parse_dataset("sonar",b"<html>404</html>")
        except ValueError:
            pass
        else:
            raise AssertionError("HTML source should fail strict CSV schema")
    print("All three UCI CSV schemas and source SHA manifest are enforced.")


def test_two_controls_are_width_matched_and_label_blind():
    old,stats=toy_gate_inputs()
    ztr,_=prior.certify_features(old,stats)
    xtr=np.array([
        [0.1,0.2,0.3,0.4],
        [0.2,0.3,0.4,0.5],
        [0.3,0.4,0.5,0.6],
        [0.4,0.5,0.6,0.7],
    ],dtype=float)
    xte=np.array([
        [0.7,0.6,0.5,0.4],
        [0.5,0.4,0.3,0.2],
    ],dtype=float)
    zte={k:v[:2]+20 for k,v in ztr.items()}
    features,hashes=runner.make_control_features(
        xtr,xte,ztr,zte,25
    )
    assert set(features)==set(runner.METHODS)
    assert features["pca_qnn"][0].shape==(4,4)
    for method in runner.METHODS[1:]:
        tr,te=features[method]
        assert tr.shape==(4,20) and te.shape==(2,20)
        assert np.array_equal(tr[:,:4],xtr)
        assert np.array_equal(te[:,:4],xte)
    assert np.all(features["pca_padded_qnn"][0][:,4:]==0)
    assert np.all(features["pca_padded_qnn"][1][:,4:]==0)
    certified=features["fusion_certified_mrbi_qnn"]
    permuted=features["fusion_permuted_certified_qnn"]
    assert not np.any(
        np.all(certified[0][:,4:]==permuted[0][:,4:],axis=1)
    )
    assert not np.any(
        np.all(certified[1][:,4:]==permuted[1][:,4:],axis=1)
    )
    assert np.array_equal(
        np.sort(certified[0][:,4:],axis=0),
        np.sort(permuted[0][:,4:],axis=0),
    )
    assert np.array_equal(
        np.sort(certified[1][:,4:],axis=0),
        np.sort(permuted[1][:,4:],axis=0),
    )
    again,again_hashes=runner.make_control_features(
        xtr,xte,ztr,zte,25
    )
    assert hashes==again_hashes
    for m in runner.METHODS:
        assert np.array_equal(features[m][0],again[m][0])
        assert np.array_equal(features[m][1],again[m][1])
    assert len(hashes["train_permutation_sha256"])==64
    assert len(hashes["test_permutation_sha256"])==64
    print("Zero-padding and label-blind derangement preserve exact 20D control width.")


def test_all_six_readouts_use_fixed_original_qnn():
    old,stats=toy_gate_inputs()
    states,trstats=prior.certify_features(old,stats)
    xtr=np.array([
        [0.1,0.2,0.3,0.4],
        [0.2,0.3,0.4,0.5],
        [0.3,0.4,0.5,0.6],
        [0.4,0.5,0.6,0.7],
    ])
    test_states={k:v.copy() for k,v in states.items()}
    feature_arrays,_=runner.make_control_features(
        xtr,xtr,states,test_states,25
    )
    cfg=replace(audit.config_for(25),run_qnn=True)
    seen=[]

    def pretend_qnn(Xtr,ytr,Xte,yte,config,seed):
        seen.append((seed,Xtr.shape[1],config.n_qubits,
                     config.qnn_layers,config.qnn_epochs))
        return {
            "accuracy":0.5,"balanced_accuracy":0.5,
            "f1":0.5,"roc_auc":0.5,"train_time_sec":0.1,
        }

    with patch.object(bench,"HAS_QNN",True),patch.object(
        bench,"train_qnn",side_effect=pretend_qnn
    ):
        metrics=runner.evaluate_qnns(
            "synthetic",25,cfg,feature_arrays,
            np.array([0,1,0,1]),np.array([0,1,0,1]),
            trstats,trstats.assign(split="test"),
        )
    assert list(metrics.method)==list(runner.METHODS)
    assert len(metrics)==6
    assert [x[0] for x in seen]==[802]*6
    assert [x[1] for x in seen]==[4,20,20,20,20,20]
    assert all(x[2:]==(4,2,60) for x in seen)
    assert metrics.loc[
        metrics.method.eq("fusion_permuted_certified_qnn"),
        "feature_test_success",
    ].isna().all()
    assert metrics.loc[
        metrics.method.eq("fusion_certified_mrbi_qnn"),
        "feature_test_rescues",
    ].iloc[0]==1
    print("Six QNN arms preserve original training and report permutation as unaligned.")


if __name__=="__main__":
    test_external_source_is_frozen_and_revalidated()
    test_two_controls_are_width_matched_and_label_blind()
    test_all_six_readouts_use_fixed_original_qnn()
