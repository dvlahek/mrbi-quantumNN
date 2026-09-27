"""Predefined eight-QNN comparison using the unchanged continuation solver.

The PCA readout, zero-initialized implicit features, and three MRBI profiles
are evaluated with identical splits, implicit layer, QNN epochs and QNN seeds.
Only the standard hybrid policy and forced MRBI are included for each profile.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import main_qnn_benchmark as bench

CAMPAIGN_DESIGN = "core_fixed_after_pilot_v1"
PROFILES = ("full_balanced", "qnn_oriented", "no_detector")
HYBRID = "standard"
PCA_READOUTS = ("logreg", "mlp", "svm_rbf", "qnn")
IMPLICIT_READOUTS = ("logreg", "svm_rbf", "qnn")
EXPECTED_ROWS = 25
EXPECTED_QNN = 8


def selected_profiles():
    options = {p.name: p for p in bench.get_mrbi_profiles("hard_quick")}
    if any(name not in options for name in PROFILES):
        raise RuntimeError("One of the predefined MRBI profiles is not available")
    return [options[name] for name in PROFILES]


def selected_hybrid():
    options = {p.name: p for p in bench.get_hybrid_profiles("hard_quick")}
    if HYBRID not in options:
        raise RuntimeError("The predefined hybrid policy is not available")
    return [options[HYBRID]]


def expected_methods():
    names = [f"pca_{name}" for name in PCA_READOUTS]
    names.extend(f"implicit_zero_{name}" for name in IMPLICIT_READOUTS)
    for profile in PROFILES:
        for rep in (f"forced_{profile}", f"hybrid_{HYBRID}_{profile}"):
            names.extend(f"{rep}_{name}" for name in IMPLICIT_READOUTS)
    if len(names) != EXPECTED_ROWS or len(set(names)) != EXPECTED_ROWS:
        raise RuntimeError("The core design must contain 25 unique methods")
    if sum(name.endswith("_qnn") for name in names) != EXPECTED_QNN:
        raise RuntimeError("The core design must contain eight QNN methods")
    return names


def main():
    args = bench.parse_args()
    if args.mode != "hard_quick":
        raise SystemExit("Core campaign requires --mode hard_quick.")
    settings = (
        ("datasets", args.datasets), ("seeds", args.seeds),
        ("qubits", args.qubits), ("latent_dims", args.latent_dims),
        ("spectral_radii", args.spectral_radii), ("input_scales", args.input_scales),
    )
    if any(len(values) != 1 for _, values in settings):
        raise SystemExit("The core runner accepts one dataset/seed/settings job per invocation.")
    if args.no_qnn or args.easy_layer:
        raise SystemExit("The predefined core campaign requires QNN and the hard implicit layer.")
    cfg = bench.ExperimentConfig(
        seed=int(args.seeds[0]),
        n_qubits=int(args.qubits[0]),
        pca_dim=int(args.qubits[0]),
        latent_dim=int(args.latent_dims[0]),
        max_samples_per_class=int(args.max_samples_per_class),
        spectral_radius=float(args.spectral_radii[0]),
        input_scale=float(args.input_scales[0]),
        hard_layer=True,
        run_qnn=True,
        qnn_epochs=int(args.qnn_epochs),
        qnn_layers=int(args.qnn_layers),
    )
    profiles, hybrids = selected_profiles(), selected_hybrid()
    if not (
        cfg.n_qubits == 4 and cfg.latent_dim == 16 and cfg.pca_dim == 4
        and cfg.spectral_radius == 2.0 and cfg.input_scale == 1.1
        and cfg.max_samples_per_class == 80 and cfg.qnn_epochs == 60
        and cfg.qnn_layers == 2
    ):
        raise SystemExit("The core campaign requires the fixed manuscript settings.")

    dataset = args.datasets[0]
    config = {
        "campaign_design": CAMPAIGN_DESIGN,
        "implementation_version": bench.IMPLEMENTATION_VERSION,
        "dataset": dataset,
        "seed": cfg.seed,
        "n_qubits": cfg.n_qubits,
        "latent_dim": cfg.latent_dim,
        "spectral_radius": cfg.spectral_radius,
        "input_scale": cfg.input_scale,
        "max_samples_per_class": cfg.max_samples_per_class,
        "qnn_epochs": cfg.qnn_epochs,
        "qnn_layers": cfg.qnn_layers,
        "readouts_pca": list(PCA_READOUTS),
        "readouts_implicit": list(IMPLICIT_READOUTS),
        "mrbi_profiles": [p.__dict__ for p in profiles],
        "hybrid_profiles": [p.__dict__ for p in hybrids],
        "selection_rule": "Compare fixed profiles across all datasets and seeds.",
    }
    config_path = Path(args.out_config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Core design: {CAMPAIGN_DESIGN}; dataset={dataset}, seed={cfg.seed}", flush=True)
    print("MRBI profiles: " + ", ".join(PROFILES), flush=True)

    data = bench.run_job(
        dataset, cfg, profiles, hybrids, list(PCA_READOUTS), list(IMPLICIT_READOUTS)
    )
    data["campaign_design"] = CAMPAIGN_DESIGN
    methods = set(expected_methods())
    if len(data) != EXPECTED_ROWS or set(data["method"]) != methods:
        raise RuntimeError("The core result does not contain the expected 25 methods")
    if not data["implementation_version"].eq(bench.IMPLEMENTATION_VERSION).all():
        raise RuntimeError("The continuation implementation version changed")
    if data["balanced_accuracy"].isna().any():
        raise RuntimeError("Core result has missing balanced-accuracy values")

    raw_path = Path(args.out_raw)
    summary_path = Path(args.out_summary)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(raw_path, index=False)
    bench.summarize_results(data).to_csv(summary_path, index=False)
    print(f"Saved {len(data)} methods ({EXPECTED_QNN} QNN variants) to {raw_path}", flush=True)


if __name__ == "__main__":
    main()
