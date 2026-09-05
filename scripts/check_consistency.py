from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "summary_tables"


def assert_close(name, actual, expected, tol=5e-4):
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(f"{name}: got {actual}, expected {expected}")


def main():
    main_tbl = pd.read_csv(DATA / "main_qnn_results.csv")
    mean_row = main_tbl[main_tbl["dataset"] == "Mean"].iloc[0]
    assert_close("main mean delta_zero", mean_row["delta_zero"], 0.0231)
    assert_close("main mean delta_pca", mean_row["delta_pca"], -0.0198)

    stat = pd.read_csv(DATA / "main_statistical_summary.csv")
    values = dict(zip(stat["quantity"], stat["value"]))
    assert_close("Wilcoxon p", values["One-sided Wilcoxon p-value"], 0.0039)

    pca4 = pd.read_csv(DATA / "pca4_control.csv")
    pca4_mean = pca4[pca4["dataset"] == "Mean"].iloc[0]
    assert_close("pca4 delta", pca4_mean["delta_pca4"], 0.0241)

    spam = pd.read_csv(DATA / "spambase_external.csv").iloc[0]
    assert_close("Spambase delta_zero", spam["delta_zero"], 0.0472)
    assert_close("Spambase delta_pca", spam["delta_pca"], 0.0056)

    classical = pd.read_csv(DATA / "classical_readout_check.csv")
    expected = {"LogReg", "SVM-RBF", "MLP", "RF", "GBM"}
    got = set(classical["readout"])
    if got != expected:
        raise AssertionError(f"Unexpected classical readouts: {got}")

    print("All consistency checks passed.")


if __name__ == "__main__":
    main()
