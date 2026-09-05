from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "summary_tables"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def write(path, text):
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")


def fmt(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def signed(x):
    try:
        return f"{float(x):+.4f}"
    except Exception:
        return str(x)


def main_table():
    df = pd.read_csv(DATA / "main_qnn_results.csv")
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Dataset & PCA-QNN & Zero-QNN & Best MRBI-QNN & $\Delta_{\mathrm{zero}}$ & $\Delta_{\mathrm{PCA}}$ \\",
        r"\hline",
    ]
    for _, r in df.iterrows():
        lines.append(f"{str(r['dataset']).replace('_', ' ')} & {fmt(r['pca_qnn'])} & {fmt(r['zero_qnn'])} & {fmt(r['best_mrbi_qnn'])} & {signed(r['delta_zero'])} & {signed(r['delta_pca'])} \\")
    lines += [r"\hline", r"\end{tabular}"]
    write(OUT / "table_main_qnn_results.tex", "\n".join(lines))


def pca4_table():
    df = pd.read_csv(DATA / "pca4_control.csv")
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Dataset & PCA-QNN & Zero-QNN & Zero-pca4-QNN & Best MRBI-pca4-QNN & $\Delta_{\mathrm{pca4}}$ \\",
        r"\hline",
    ]
    for _, r in df.iterrows():
        lines.append(f"{str(r['dataset']).replace('_', ' ')} & {fmt(r['pca_qnn'])} & {fmt(r['zero_qnn'])} & {fmt(r['zero_pca4_qnn'])} & {fmt(r['best_mrbi_pca4_qnn'])} & {signed(r['delta_pca4'])} \\")
    lines += [r"\hline", r"\end{tabular}"]
    write(OUT / "table_pca4_control.tex", "\n".join(lines))


def classical_table():
    df = pd.read_csv(DATA / "classical_readout_check.csv")
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Readout & Zero & Best MRBI & Mean $\Delta_{\mathrm{zero}}$ & Median $\Delta_{\mathrm{zero}}$ & Pos./Neu./Neg. \\",
        r"\hline",
    ]
    for _, r in df.iterrows():
        lines.append(f"{r['readout']} & {fmt(r['zero'])} & {fmt(r['best_mrbi'])} & {signed(r['mean_delta_zero'])} & {signed(r['median_delta_zero'])} & {r['pos_neu_neg']} \\")
    lines += [r"\hline", r"\end{tabular}"]
    write(OUT / "table_classical_readout_check.tex", "\n".join(lines))


def spambase_table():
    r = pd.read_csv(DATA / "spambase_external.csv").iloc[0]
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Dataset & PCA-QNN & Zero-QNN & Best MRBI-QNN & $\Delta_{\mathrm{zero}}$ & $\Delta_{\mathrm{PCA}}$ \\",
        r"\hline",
        f"{r['dataset']} & {fmt(r['pca_qnn'])} & {fmt(r['zero_qnn'])} & {fmt(r['best_mrbi_qnn'])} & {signed(r['delta_zero'])} & {signed(r['delta_pca'])} \\",
        r"\hline",
        r"\end{tabular}",
    ]
    write(OUT / "table_spambase_external.tex", "\n".join(lines))


if __name__ == "__main__":
    main_table()
    pca4_table()
    classical_table()
    spambase_table()
