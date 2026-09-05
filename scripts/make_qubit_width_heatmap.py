from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "summary_tables" / "qubit_width_sensitivity_summary_for_plot.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATA)
    order = ["wine_binary", "wine_0_vs_2", "digits_4_vs_9"]
    pretty = {"wine_binary": "wine binary", "wine_0_vs_2": "wine 0 vs 2", "digits_4_vs_9": "digits 4 vs 9"}
    pivot = df.pivot_table(index="dataset", columns="n_qubits", values="delta_zero_mean", aggfunc="mean")
    pivot = pivot.loc[[d for d in order if d in pivot.index]].reindex(sorted(pivot.columns), axis=1)
    pivot.index = [pretty.get(x, x) for x in pivot.index]
    values = pivot.to_numpy(dtype=float)
    max_abs = float(np.nanmax(np.abs(values)))
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    im = ax.imshow(values, aspect="auto", norm=norm)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(r"Number of qubits, $n_q$")
    ax.set_ylabel("Dataset")
    ax.set_title(r"Qubit-width sensitivity of $\Delta_{\mathrm{zero}}$")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:+.3f}", ha="center", va="center", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.04)
    cbar.set_label(r"$\Delta_{\mathrm{zero}}$")
    fig.tight_layout()
    fig.savefig(OUT / "qubit_width_sensitivity_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "qubit_width_sensitivity_heatmap.pdf", bbox_inches="tight")
    print(f"wrote {OUT / 'qubit_width_sensitivity_heatmap.png'}")
    print(f"wrote {OUT / 'qubit_width_sensitivity_heatmap.pdf'}")


if __name__ == "__main__":
    main()
