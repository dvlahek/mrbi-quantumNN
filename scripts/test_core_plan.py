"""Test the fixed core design without QNN training or MRBI feature generation."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

import core_qnn_job as core
import run_core_campaign as campaign


def main():
    methods = core.expected_methods()
    if len(methods) != 25 or len(set(methods)) != 25:
        raise AssertionError("Core design has an unexpected number of methods")
    if sum(name.endswith("_qnn") for name in methods) != 8:
        raise AssertionError("Core design must contain eight QNN methods")
    if tuple(p.name for p in core.selected_profiles()) != core.PROFILES:
        raise AssertionError("Unexpected MRBI profiles")
    if [p.name for p in core.selected_hybrid()] != [core.HYBRID]:
        raise AssertionError("Unexpected hybrid policy")

    df = pd.DataFrame({
        "dataset": ["breast_cancer"] * len(methods),
        "seed": [0] * len(methods),
        "method": methods,
        "readout": [name.rsplit("_", 1)[-1] for name in methods],
        "implementation_version": [campaign.VERSION] * len(methods),
        "campaign_design": [campaign.PLAN] * len(methods),
        "balanced_accuracy": [0.6] * len(methods),
    })
    with TemporaryDirectory() as dirname:
        path = Path(dirname) / "pilot.csv"
        df.to_csv(path, index=False)
        if not campaign.valid_job(path, "breast_cancer", 0):
            raise AssertionError("The campaign rejected its own complete core design")
        df.loc[0, "campaign_design"] = "other_plan"
        df.to_csv(path, index=False)
        if campaign.valid_job(path, "breast_cancer", 0):
            raise AssertionError("The campaign accepted a different experiment design")
    print("Core design checks passed: 25 rows, eight QNN methods and strict campaign labels.")


if __name__ == "__main__":
    main()
