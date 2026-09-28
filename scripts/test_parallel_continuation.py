"""Fast regression checks for 12-process full-campaign scheduling (no QNN training)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_parallel_continuation_campaign as parallel


def fake_job(dataset, seed):
    methods = ["pca_qnn", "implicit_zero_qnn"]
    methods += [f"mrbi_{index:02d}_qnn" for index in range(30)]
    methods += [f"other_{index:02d}_logreg" for index in range(65)]
    assert len(methods) == 97
    return pd.DataFrame({
        "dataset": [dataset] * 97, "seed": [seed] * 97,
        "method": methods,
        "readout": ["qnn"] * 32 + ["logreg"] * 65,
        "implementation_version": [parallel.VERSION] * 97,
        "balanced_accuracy": np.linspace(0.5, 0.9, 97),
    })


def main():
    with TemporaryDirectory() as folder:
        out = Path(folder)
        dataset, seed = "breast_cancer", 0
        raw, summ, config, _log = parallel.serial.job_paths(out, dataset, seed)
        raw.parent.mkdir(parents=True)
        data = fake_job(dataset, seed)
        data.to_csv(raw, index=False)
        summ.write_text("stub summary", encoding="utf-8")
        config.write_text(json.dumps({"dataset": dataset, "seed": seed}),
                          encoding="utf-8")
        assert parallel.valid_job(raw, dataset, seed)
        assert not parallel.collect(out, [(dataset, seed)])
        combined = pd.read_csv(out / "main_raw.csv")
        assert len(combined) == 97 and combined["method"].nunique() == 97
        data.loc[0, "balanced_accuracy"] = np.nan
        data.to_csv(raw, index=False)
        assert not parallel.valid_job(raw, dataset, seed)
        data.loc[0, "balanced_accuracy"] = 0.65
        data.loc[1, "method"] = "pca_qnn"
        data.to_csv(raw, index=False)
        assert not parallel.valid_job(raw, dataset, seed)

        command = parallel.command(dataset, seed, out)
        expected = ["--mode", "hard_quick", "--n-workers", "1",
                    "--qnn-epochs", "60", "--max-samples-per-class", "80"]
        for arg in expected:
            assert arg in command, f"Missing fixed full-campaign option {arg}"
        assert "--no-qnn" not in command
        assert parallel.EXPECTED_METHODS == 97 and parallel.EXPECTED_QNN == 32
        assert all(value == "1" for value in parallel.SINGLE_THREAD_ENV.values())

        dry = subprocess.run(
            [sys.executable, str(ROOT / "scripts" /
                                 "run_parallel_continuation_campaign.py"),
             "--datasets", dataset, "--seeds", "0", "--workers", "12",
             "--out-dir", str(out / "dry"), "--dry-run"],
            text=True, capture_output=True, check=True,
            env={**os.environ, "OMP_NUM_THREADS": "1",
                 "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        assert "1 pending" in dry.stdout
        assert "12 independent workers" in dry.stdout
    print("Parallel full campaign tests passed: 97 rows, 32 QNN, "
          "12-worker plan, one math thread per process and resumable aggregation.")


if __name__ == "__main__":
    main()
