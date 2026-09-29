"""Download and validate three frozen, distinct UCI source datasets.

The benchmark runner *never* downloads data or updates source files.
Use this tool once before running it; it writes a SHA-256 provenance
manifest that is then checked on every run/resume. Dataset labels
are never consulted when selecting MRBI roots or permuting features.

Sources and provenance:
https://archive.ics.uci.edu/dataset/267/banknote+authentication
https://archive.ics.uci.edu/dataset/52/ionosphere
https://archive.ics.uci.edu/dataset/151/connectionist
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from urllib.request import Request, urlopen

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_DIR=ROOT/"data"/"external_fusion"
PLAN="uci_external_fusion_three_sources_v1"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    url: str
    doi: str
    n_rows: int
    n_features: int
    label_map: dict[str,int]


SPECS={
    "banknote":DatasetSpec(
        "banknote","data_banknote_authentication.txt",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt",
        "10.24432/C55P57",1372,4,{"0":0,"1":1},
    ),
    "ionosphere":DatasetSpec(
        "ionosphere","ionosphere.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
        "10.24432/C5W01B",351,34,{"b":0,"g":1},
    ),
    "sonar":DatasetSpec(
        "sonar","sonar.all-data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data",
        "10.24432/C5T01Q",208,60,{"R":0,"M":1},
    ),
}
DATASETS=tuple(SPECS)


def parse_dataset(name,raw):
    """Fail closed on HTML/CSV schema drift; no heuristic label coercion."""
    spec=SPECS[name]
    try:
        body=raw.decode("ascii")
    except UnicodeError as exc:
        raise ValueError(f"{name}: expected ASCII CSV") from exc
    reader=csv.reader(io.StringIO(body))
    features,labels=[],[]
    for idx,row in enumerate(reader):
        if not row or not any(f.strip() for f in row):
            continue
        if len(row)!=spec.n_features+1:
            raise ValueError(
                f"{name}: row {idx+1} has {len(row)} fields, expected "
                f"{spec.n_features+1}"
            )
        key=row[-1].strip()
        if key not in spec.label_map:
            raise ValueError(f"{name}: unexpected label {key!r} on row {idx+1}")
        try:
            vec=[float(z) for z in row[:-1]]
        except ValueError as exc:
            raise ValueError(f"{name}: nonnumeric feature at row {idx+1}") from exc
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"{name}: nonfinite feature at row {idx+1}")
        features.append(vec)
        labels.append(spec.label_map[key])
    X=np.asarray(features,dtype=np.float64)
    y=np.asarray(labels,dtype=np.int64)
    if (X.shape!=(spec.n_rows,spec.n_features)
            or y.shape!=(spec.n_rows,)
            or set(np.unique(y))!={0,1}
            or min(int((y==0).sum()),int((y==1).sum()))<80):
        raise ValueError(f"{name}: wrong public source shape or label coverage")
    return X,y


def descriptor(name,raw):
    X,y=parse_dataset(name,raw)
    spec=SPECS[name]
    return {
        "name":name,"file":spec.filename,"source_url":spec.url,
        "doi":spec.doi,
        "sha256":hashlib.sha256(raw).hexdigest(),
        "byte_count":len(raw),
        "n_rows":len(X),"n_features":X.shape[1],
        "class_counts":{str(c):int((y==c).sum()) for c in (0,1)},
    }


def load_sources(directory,requested=DATASETS):
    """Read only: the full manifest must match every actual raw source."""
    directory=Path(directory)
    manifest_file=directory/"data_manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_file}; run download_external_fusion_data.py first"
        )
    manifest=json.loads(manifest_file.read_text(encoding="utf-8"))
    if (manifest.get("plan")!=PLAN
            or set(manifest.get("datasets",{}))!=set(DATASETS)):
        raise RuntimeError("Missing or foreign frozen UCI source manifest")
    result={}
    for name in requested:
        if name not in SPECS:
            raise ValueError(f"Unknown external source: {name}")
        file=directory/SPECS[name].filename
        raw=file.read_bytes()
        desc=descriptor(name,raw)
        if desc!=manifest["datasets"][name]:
            raise RuntimeError(f"{name}: source file differs from frozen manifest")
        result[name]=(*parse_dataset(name,raw),desc["sha256"])
    return result,manifest


def fetch_raw(spec):
    request=Request(
        spec.url,
        headers={"User-Agent":"MRBI-QNN research replication (UCI public dataset)"},
    )
    with urlopen(request,timeout=60) as response:
        if response.status!=200:
            raise RuntimeError(f"HTTP {response.status} fetching {spec.name}")
        raw=response.read(500_000)
    if len(raw)>250_000:
        raise RuntimeError(f"{spec.name}: unexpected public source size")
    return raw


def atomic_write(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",dir=str(path.parent),prefix=".download-",
        delete=False
    ) as tmp:
        tmp.write(raw)
        staged=Path(tmp.name)
    try:
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,default=DEFAULT_DIR)
    p.add_argument("--offline",action="store_true",
                   help="Validate already downloaded sources without network")
    args=p.parse_args()
    data_dir=args.data_dir.resolve()
    manifest_file=data_dir/"data_manifest.json"
    existing=(json.loads(manifest_file.read_text(encoding="utf-8"))
              if manifest_file.is_file() else None)
    descriptions={}
    for name,spec in SPECS.items():
        target=data_dir/spec.filename
        if target.is_file():
            raw=target.read_bytes()
            desc=descriptor(name,raw)
            if (existing is not None and
                    existing.get("datasets",{}).get(name)!=desc):
                raise SystemExit(
                    f"{name}: previously frozen source changed; refusing overwrite"
                )
            print(f"{name}: validated cached source sha256={desc['sha256']}",
                  flush=True)
        else:
            if args.offline:
                raise SystemExit(f"{name}: source missing in offline mode")
            if existing is not None:
                raise SystemExit(
                    f"{name}: frozen manifest exists but source is missing"
                )
            raw=fetch_raw(spec)
            desc=descriptor(name,raw)
            atomic_write(target,raw)
            print(f"{name}: downloaded and validated sha256={desc['sha256']}",
                  flush=True)
        descriptions[name]=desc
    new={"plan":PLAN,"datasets":descriptions}
    if existing is not None and existing!=new:
        raise SystemExit("Frozen source manifest disagrees with validated data")
    if existing is None:
        atomic_write(
            manifest_file,
            (json.dumps(new,indent=2,sort_keys=True)+"\n").encode("utf-8"),
        )
    # Recheck the same source paths under the strict run-time contract.
    load_sources(data_dir)
    print(f"UCI_EXTERNAL_SOURCE_MANIFEST_OK {manifest_file}",flush=True)


if __name__=="__main__":
    main()
