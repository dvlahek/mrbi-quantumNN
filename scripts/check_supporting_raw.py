"""Check the supplied Spambase and multistart raw evidence for the NPL paper."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "supporting_raw"
SUM = ROOT / "results" / "summary_tables"


def check_close(name, value, reference, tolerance=5e-5):
    if not np.isfinite(value) or abs(float(value) - float(reference)) >= tolerance:
        raise AssertionError(f"{name}: {value} differs from {reference}")


def main():
    spam = pd.read_csv(RAW / "spambase_external_raw.csv")
    qnn = spam[(spam["dataset"] == "spambase") & (spam["readout"] == "qnn")]
    p = qnn.pivot(index="seed", columns="method", values="balanced_accuracy")
    expected = {"pca_qnn", "implicit_zero_qnn", "mrbi_light_qnn", "mrbi_balanced_qnn"}
    if set(p.columns) != expected or len(p) != 3 or p.isna().any().any():
        raise AssertionError("Spambase needs three paired seeds and four QNN methods")
    pca = p["pca_qnn"].mean()
    zero = p["implicit_zero_qnn"].mean()
    best = p[["mrbi_light_qnn", "mrbi_balanced_qnn"]].max(axis=1).mean()
    reported = pd.read_csv(SUM / "spambase_external.csv").iloc[0]
    for key, value in {"pca_qnn": pca, "zero_qnn": zero, "best_mrbi_qnn": best,
                       "delta_zero": best - zero, "delta_pca": best - pca}.items():
        check_close(f"Spambase {key}", reported[key], value)
    if int(reported["n_seeds"]) != 3:
        raise AssertionError("Spambase seed count differs")
    check_close("Spambase upper envelope", best, 0.8616666666666667)

    raw = pd.read_csv(RAW / "multistart_sanity_raw.csv")
    seed = pd.read_csv(RAW / "multistart_sanity_seed_level.csv")
    summary = pd.read_csv(RAW / "multistart_sanity_summary.csv")
    grouped = raw.groupby(["dataset", "seed", "method"])["balanced_accuracy"].mean().unstack()
    if set(grouped.columns) != {"pca", "zero", "mrbi", "multistart"} or len(grouped) != 9:
        raise AssertionError("Unexpected multistart raw rows")
    seed = seed.set_index(["dataset", "seed"]).sort_index()
    for key in ("zero", "mrbi", "multistart"):
        if not np.allclose(grouped[key].sort_index(), seed[key], atol=1e-12, rtol=0):
            raise AssertionError(f"Multistart seed-level mismatch: {key}")
    values = summary[summary["dataset"] != "Mean"].set_index("dataset")
    for dataset, row in values.iterrows():
        for key in ("zero", "mrbi", "multistart"):
            check_close(f"Multistart {dataset} {key}", row[key], seed.loc[dataset, key].mean(), 1e-10)
    avg = summary[summary["dataset"] == "Mean"].iloc[0]
    for key in ("zero", "mrbi", "multistart", "delta_mrbi_zero",
                "delta_multistart_zero", "delta_mrbi_multistart"):
        check_close(f"Multistart overall {key}", avg[key],
                    values[key].mean(), 1e-10)
    print("Supporting raw checks passed: Spambase (3 seeds), multistart (9 dataset-seed pairs).")


if __name__ == "__main__":
    main()
