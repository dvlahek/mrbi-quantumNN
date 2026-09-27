"""Compare available per-seed QNN results with the manuscript's dataset means."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
TABLE = ROOT / "results" / "summary_tables" / "main_qnn_results.csv"
DATASETS = (
    "breast_cancer", "wine_binary", "wine_0_vs_2", "wine_1_vs_2",
    "digits_1_vs_7", "digits_2_vs_7", "digits_3_vs_8",
    "digits_4_vs_9", "digits_5_vs_6",
)
EXPECTED_SEEDS = {0, 1, 2, 3, 4}
EXPECTED_METHODS_PER_SEED = 97
TOL = 0.000051  # reported values are rounded to four decimal places


def close(name, actual, expected):
    if abs(float(actual) - float(expected)) > TOL:
        raise AssertionError(f"{name}: derived {actual:.8f}, archived {expected:.8f}")


def main():
    pieces = [pd.read_csv(RAW / name) for name in
              ("article_qnn_final_raw.csv", "article_qnn_final_raw2.csv")]
    raw = pd.concat(pieces, ignore_index=True)
    if raw.duplicated(["dataset", "seed", "method"]).any():
        raise AssertionError("Duplicate dataset/seed/method rows")
    if not set(raw.dataset).issubset(DATASETS):
        raise AssertionError("Unexpected dataset in the main raw files")
    for key, value in {"n_qubits": 4, "latent_dim": 16, "pca_dim": 4,
                       "spectral_radius": 2.0, "max_samples_per_class": 80}.items():
        if not raw[key].eq(value).all():
            raise AssertionError(f"Unexpected experimental setting: {key}")
    per_seed = raw.groupby(["dataset", "seed"]).size()
    if not per_seed.eq(EXPECTED_METHODS_PER_SEED).all():
        raise AssertionError("Missing methods within a supplied dataset/seed")
    qnn = raw[raw["readout"] == "qnn"]
    table = pd.read_csv(TABLE).set_index("dataset")
    verified, incomplete = [], []
    for dataset in DATASETS:
        sub = qnn[qnn["dataset"] == dataset]
        seeds = set(sub["seed"])
        if seeds != EXPECTED_SEEDS:
            incomplete.append((dataset, sorted(EXPECTED_SEEDS - seeds)))
            continue
        methods = sub.groupby("method")["balanced_accuracy"].mean()
        if "pca_qnn" not in methods or "implicit_zero_qnn" not in methods:
            raise AssertionError(f"Missing QNN baselines for {dataset}")
        candidates = methods.drop(index=["pca_qnn", "implicit_zero_qnn"])
        if candidates.empty:
            raise AssertionError(f"No MRBI variants for {dataset}")
        pca, zero, best = (float(methods["pca_qnn"]),
                           float(methods["implicit_zero_qnn"]),
                           float(candidates.max()))
        for column, value in {"pca_qnn": pca, "zero_qnn": zero,
                              "best_mrbi_qnn": best, "delta_zero": best - zero,
                              "delta_pca": best - pca}.items():
            close(f"{dataset} {column}", value, table.loc[dataset, column])
        verified.append(dataset)
    if not verified:
        raise AssertionError("No complete dataset available for verification")
    print(f"Main QNN raw check passed: {len(verified)} complete datasets: "
          + ", ".join(verified))
    if incomplete:
        print("Not yet verified from raw: " + "; ".join(
            f"{name} (missing seeds {seeds})" for name, seeds in incomplete))
    print("Selection rule: best mean across five seeds for each dataset and method.")


if __name__ == "__main__":
    main()
