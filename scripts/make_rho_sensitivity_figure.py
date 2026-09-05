from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "summary_tables" / "rho_sensitivity_delta_summary_for_plot.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATA)
    order = ["wine_0_vs_2", "wine_binary", "digits_4_vs_9"]
    labels = {
        "wine_0_vs_2": "wine 0 vs 2",
        "wine_binary": "wine binary",
        "digits_4_vs_9": "digits 4 vs 9",
    }
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for ds in order:
        sub = df[df["dataset"] == ds].sort_values("spectral_radius")
        if sub.empty:
            continue
        yerr = sub["delta_zero_std"] if sub["delta_zero_std"].notna().any() else None
        ax.errorbar(sub["spectral_radius"], sub["delta_zero_mean"], yerr=yerr, marker="o", linewidth=2.0, capsize=4, label=labels.get(ds, ds))
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel(r"Spectral radius $\rho(\mathbf{W})$")
    ax.set_ylabel(r"MRBI gain over zero initialization, $\Delta_{\mathrm{zero}}$")
    ax.set_title(r"Hardness sensitivity of MRBI gain")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "rho_sensitivity_delta_plot.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "rho_sensitivity_delta_plot.pdf", bbox_inches="tight")
    print(f"wrote {OUT / 'rho_sensitivity_delta_plot.png'}")
    print(f"wrote {OUT / 'rho_sensitivity_delta_plot.pdf'}")


if __name__ == "__main__":
    main()
