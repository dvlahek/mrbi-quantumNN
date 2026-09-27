"""Verify that regenerated LaTeX table rows terminate correctly."""
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs"
TABLES = ("table_main_qnn_results.tex", "table_pca4_control.tex",
          "table_classical_readout_check.tex", "table_spambase_external.tex")


def main():
    for name in TABLES:
        lines = (OUT / name).read_text(encoding="utf-8").splitlines()
        rows = [line for line in lines if line and not line.startswith(
            ("\\begin", "\\end", "\\hline"))]
        if len(rows) < 2 or any(not line.endswith(r"\\") for line in rows):
            raise AssertionError(f"Invalid LaTeX row termination in {name}")
    print("Generated table syntax checks passed.")


if __name__ == "__main__":
    main()
