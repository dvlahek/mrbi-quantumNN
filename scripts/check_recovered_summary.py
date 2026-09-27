"""Check the recovered five-dataset summary against the NPL main table and raw CSVs."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "summary_tables" / "article_qnn_final_summary1.csv"
MAIN = RESULTS / "summary_tables" / "main_qnn_results.csv"
RAW = RESULTS / "raw"
SUMMARY_DATASETS = {
    "breast_cancer", "wine_binary", "wine_0_vs_2",
    "wine_1_vs_2", "digits_1_vs_7",
}
TOL = 0.000051


def close(label, a, b, tol=TOL):
    if abs(float(a) - float(b)) > tol:
        raise AssertionError(f"{label}: {a:.10f} != {b:.10f}")


def main():
    summary = pd.read_csv(SUMMARY)
    if len(summary) != 485 or set(summary["dataset"]) != SUMMARY_DATASETS:
        raise AssertionError("Recovered summary must contain five datasets and 485 methods")
    if summary.duplicated(["dataset", "method"]).any():
        raise AssertionError("Duplicate method in recovered summary")
    for name, expected in (("n_runs", 5), ("n_qubits", 4),
                           ("latent_dim", 16), ("spectral_radius", 2.0),
                           ("input_scale", 1.1)):
        if not summary[name].eq(expected).all():
            raise AssertionError(f"Unexpected {name} in recovered summary")
    if not summary.groupby("dataset").size().eq(97).all():
        raise AssertionError("Expected 97 method summaries per dataset")
    reported = pd.read_csv(MAIN).set_index("dataset")
    for ds, group in summary[summary["readout"] == "qnn"].groupby("dataset"):
        methods = group.set_index("method")["balanced_accuracy_mean"]
        pca = float(methods.loc["pca_qnn"])
        zero = float(methods.loc["implicit_zero_qnn"])
        candidates = methods.drop(index=["pca_qnn", "implicit_zero_qnn"])
        if candidates.empty:
            raise AssertionError(f"No MRBI candidates for {ds}")
        best = float(candidates.max())
        for key, value in {"pca_qnn": pca, "zero_qnn": zero,
                           "best_mrbi_qnn": best, "delta_zero": best-zero,
                           "delta_pca": best-pca}.items():
            close(f"{ds}: {key}", value, reported.loc[ds, key])

    raw = pd.concat([pd.read_csv(RAW / x) for x in
                     ("article_qnn_final_raw.csv", "article_qnn_final_raw2.csv")],
                    ignore_index=True)
    complete = raw.groupby(["dataset", "method"]).filter(
        lambda part: set(part["seed"]) == {0, 1, 2, 3, 4}
    )
    paired = complete.groupby(["dataset", "method"])["balanced_accuracy"].mean()
    recovered = summary.set_index(["dataset", "method"])["balanced_accuracy_mean"]
    common = paired.index.intersection(recovered.index)
    if len(common) != 194:
        raise AssertionError(f"Expected 194 raw-to-summary comparisons, got {len(common)}")
    for pair in common:
        close(f"Raw versus recovered summary {pair}", paired.loc[pair],
              recovered.loc[pair], tol=1e-10)
    print("Recovered summary verified: five dataset means, 194 matching raw method means.")
    print("Combined with the six complete raw datasets, all nine manuscript task means are checked.")
    print("Three tasks still lack complete per-seed raw CSVs.")


if __name__ == "__main__":
    main()
