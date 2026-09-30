"""Diagnose disagreement between corrected-continuation main_raw.csv and selected_profiles.csv.

This script is read-only. It never writes or changes experiment results.
The complete corrected raw file is treated as the primary evidence; the
existing selected_profiles.csv is compared row-by-row against a deterministic
recomputation using the same selection rule documented in summarize_continuation.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_BASE=ROOT/"outputs"/"continuation_v1"
DATASETS=(
    "breast_cancer","wine_binary","wine_0_vs_2","wine_1_vs_2",
    "digits_1_vs_7","digits_2_vs_7","digits_3_vs_8",
    "digits_4_vs_9","digits_5_vs_6",
)
SEEDS={0,1,2,3,4}
VERSION="mrbi_continuation_v1"
EXPECTED_METHODS=97


def sha256(path:Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def validate_raw(df:pd.DataFrame):
    required={"dataset","seed","method","readout","implementation_version",
              "balanced_accuracy","n_qubits","latent_dim","spectral_radius",
              "input_scale","max_samples_per_class"}
    missing=required-set(df.columns)
    if missing:
        raise SystemExit(f"Missing raw columns: {sorted(missing)}")
    if not df["implementation_version"].eq(VERSION).all():
        raise SystemExit("Not the corrected-continuation implementation")
    if set(df["dataset"])!=set(DATASETS):
        raise SystemExit("Dataset set is not the complete nine-task campaign")
    if set(map(int,df["seed"].unique()))!=SEEDS:
        raise SystemExit("Seed set is not exactly 0-4")
    if df.duplicated(["dataset","seed","method"]).any():
        raise SystemExit("Duplicate dataset/seed/method rows")
    for ds in DATASETS:
        for seed in sorted(SEEDS):
            g=df[(df.dataset==ds)&(df.seed.astype(int)==seed)]
            if len(g)!=EXPECTED_METHODS or g.method.nunique()!=EXPECTED_METHODS:
                raise SystemExit(f"Incomplete {ds} seed={seed}: {len(g)} rows")


def recompute(df:pd.DataFrame):
    qnn=df[df.readout=="qnn"].copy()
    rows=[]
    for ds in DATASETS:
        g=qnn[qnn.dataset==ds]
        means=g.groupby("method",sort=True).balanced_accuracy.mean()
        candidates=means.drop(index=["pca_qnn","implicit_zero_qnn"])
        top=float(candidates.max())
        exact=sorted(m for m,v in candidates.items() if abs(float(v)-top)<=1e-15)
        # pandas idxmax() on the sorted groupby index, matching summarize_continuation.py
        idxmax_method=str(candidates.idxmax())
        rows.append({
            "dataset":ds,
            "recomputed_method":idxmax_method,
            "recomputed_ba":top,
            "tie_count":len(exact),
            "tie_methods":exact,
            "zero_ba":float(means["implicit_zero_qnn"]),
            "pca_ba":float(means["pca_qnn"]),
        })
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",type=Path,default=DEFAULT_BASE)
    args=p.parse_args()
    base=args.base.resolve()
    raw=base/"main_raw.csv"
    selected=base/"selected_profiles.csv"
    if not raw.exists():
        raise SystemExit(f"Missing {raw}")
    if not selected.exists():
        raise SystemExit(f"Missing {selected}")

    df=pd.read_csv(raw)
    validate_raw(df)
    rec=recompute(df)
    sel=pd.read_csv(selected)

    required={"dataset","best_method","best_mrbi_qnn","n_seeds"}
    if not required.issubset(sel.columns):
        raise SystemExit(f"selected_profiles.csv missing {sorted(required-set(sel.columns))}")

    by_sel={str(r.dataset):r for r in sel.itertuples(index=False)}
    print("RAW_SHA256",sha256(raw),flush=True)
    print("SELECTED_SHA256",sha256(selected),flush=True)
    print("RAW_ROWS",len(df),flush=True)
    print("RAW_DATASET_SEED_JOBS",df.groupby(["dataset","seed"]).ngroups,flush=True)
    print("SELECTED_ROWS",len(sel),flush=True)
    print("",flush=True)

    mismatches=[]
    for r in rec:
        ds=r["dataset"]
        if ds not in by_sel:
            mismatches.append({"dataset":ds,"reason":"missing_from_selected"})
            print(f"MISMATCH {ds}: missing from selected_profiles.csv",flush=True)
            continue
        old=by_sel[ds]
        old_method=str(old.best_method)
        old_ba=float(old.best_mrbi_qnn)
        same_method=(old_method==r["recomputed_method"])
        same_ba=(abs(old_ba-r["recomputed_ba"])<=5e-7)
        label="MATCH" if same_method and same_ba else "MISMATCH"
        print(
            f"{label} {ds}: selected={old_method} ({old_ba:.9f}) | "
            f"raw={r['recomputed_method']} ({r['recomputed_ba']:.9f}) | "
            f"ties={r['tie_count']} {r['tie_methods']}",
            flush=True,
        )
        if not (same_method and same_ba):
            # also report the selected method's actual mean in current raw, if present
            q=df[(df.dataset==ds)&(df.readout=="qnn")]
            means=q.groupby("method").balanced_accuracy.mean()
            current_old=float(means[old_method]) if old_method in means.index else None
            mismatches.append({
                "dataset":ds,
                "selected_method":old_method,
                "selected_file_ba":old_ba,
                "selected_method_current_raw_ba":current_old,
                "recomputed_method":r["recomputed_method"],
                "recomputed_ba":r["recomputed_ba"],
                "tie_count":r["tie_count"],
                "tie_methods":r["tie_methods"],
            })

    extra=sorted(set(map(str,sel.dataset))-set(DATASETS))
    if extra:
        print("EXTRA_SELECTED_DATASETS",extra,flush=True)

    print("",flush=True)
    print("MISMATCH_COUNT",len(mismatches),flush=True)
    print("DIAGNOSTIC_JSON",json.dumps(mismatches,sort_keys=True),flush=True)
    if mismatches or extra:
        raise SystemExit(2)
    print("SELECTED_PROFILE_RECOMPUTATION_MATCH_OK",flush=True)


if __name__=="__main__":
    main()
