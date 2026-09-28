"""One paired final-sigma control for the existing full_balanced continuation result.

No PCA, zero baseline or continuation QNN is retrained. The reference QNN row
comes from the completed fixed-core campaign for the same dataset and seed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import main_qnn_benchmark as bench

PLAN = "final_sigma_repeated_vs_continuation_v1"
SOURCE_PLAN = "core_fixed_after_pilot_v1"
METHOD = "final_sigma_repeated_full_balanced_qnn"
SOURCE_METHOD = "forced_full_balanced_qnn"
DATASETS = (
    "breast_cancer", "wine_binary", "wine_0_vs_2", "wine_1_vs_2",
    "digits_1_vs_7", "digits_2_vs_7", "digits_3_vs_8",
    "digits_4_vs_9", "digits_5_vs_6",
)


def get_reference(path: Path, dataset: str, seed: int):
    source = pd.read_csv(path)
    required = {
        "dataset", "seed", "method", "balanced_accuracy", "implementation_version",
        "campaign_design", "n_qubits", "latent_dim", "spectral_radius",
        "input_scale", "max_samples_per_class",
    }
    if not required.issubset(source.columns):
        raise ValueError(f"Reference raw missing columns: {required - set(source.columns)}")
    if not source["implementation_version"].eq(bench.IMPLEMENTATION_VERSION).all():
        raise ValueError("Reference has another implementation version")
    if not source["campaign_design"].eq(SOURCE_PLAN).all():
        raise ValueError("Reference raw is not the fixed core campaign")
    ref = source[
        (source["dataset"] == dataset) & (source["seed"] == seed)
        & (source["method"] == SOURCE_METHOD)
    ]
    if len(ref) != 1 or not np.isfinite(float(ref.iloc[0]["balanced_accuracy"])):
        raise ValueError(f"Missing or duplicate reference: {dataset} seed={seed}")
    row = ref.iloc[0]
    expected = {
        "n_qubits": 4, "latent_dim": 16, "spectral_radius": 2.0,
        "input_scale": 1.1, "max_samples_per_class": 80,
    }
    for key, value in expected.items():
        if not np.isclose(float(row[key]), float(value), rtol=0.0, atol=1e-12):
            raise ValueError(f"Reference setting mismatch: {key}")
    if "profile_name" in ref and str(row["profile_name"]) != "full_balanced":
        raise ValueError("Reference profile is not full_balanced")
    if "readout" in ref and str(row["readout"]) != "qnn":
        raise ValueError("Reference is not the QNN readout")
    return row


def experiment(dataset: str, seed: int, reference_raw: Path):
    get_reference(reference_raw, dataset, seed)
    base = bench.ExperimentConfig(
        seed=seed, n_qubits=4, pca_dim=4, latent_dim=16,
        max_samples_per_class=80, spectral_radius=2.0, input_scale=1.1,
        hard_layer=True, run_qnn=True, qnn_epochs=60, qnn_layers=2,
    )
    profile = next(
        p for p in bench.get_mrbi_profiles("hard_quick")
        if p.name == "full_balanced"
    )
    cfg = replace(bench.apply_mrbi_profile(base, profile), repeat_final_sigma=True)
    mrbi_cfg = bench.make_mrbi_cfg(cfg)
    if not (
        mrbi_cfg.repeat_final_sigma and mrbi_cfg.use_continuation
        and mrbi_cfg.use_detector and mrbi_cfg.gamma > 0
    ):
        raise RuntimeError("The control did not activate the final-sigma schedule")

    X, y = bench.load_dataset(dataset, cfg)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )
    Xtr_p, Xte_p = bench.preprocess_to_pca(Xtr, Xte, cfg.pca_dim, cfg.seed)
    layer = bench.make_random_implicit_layer(
        dx=Xtr_p.shape[1], d=cfg.latent_dim, cfg=cfg,
        seed=cfg.seed + 1000 + cfg.latent_dim + int(100 * cfg.spectral_radius),
    )
    t0 = time.perf_counter()
    Ztr, _train_stats = bench.solve_forced_mrbi_features(
        Xtr_p, layer, cfg, cfg.seed + 2000
    )
    Zte, test_stats = bench.solve_forced_mrbi_features(
        Xte_p, layer, cfg, cfg.seed + 3000
    )
    feature_time = time.perf_counter() - t0
    Ztr, Zte = bench.standardize_pair(Ztr, Zte)
    stats = {f"test_{key}": value for key, value in test_stats.items()}
    stats["feature_total_time_sec"] = feature_time
    rows = bench.evaluate_readouts(
        dataset, "final_sigma_repeated_full_balanced",
        Ztr, Zte, ytr, yte, cfg, ["qnn"], stats,
    )
    if len(rows) != 1:
        raise RuntimeError("Expected exactly one QNN result")
    result = rows[0]
    if result["method"] != METHOD or not np.isfinite(float(result["balanced_accuracy"])):
        raise RuntimeError("Invalid final-sigma QNN output")
    result["ablation_design"] = PLAN
    result["ablation_arm"] = "final_sigma_repeated"
    result["reference_method"] = SOURCE_METHOD
    result["stage_sigma_schedule"] = json.dumps([float(s) for s in cfg.sigmas])
    result["objective_sigma_schedule"] = json.dumps([float(cfg.sigmas[-1])] * len(cfg.sigmas))
    result["iteration_budget_per_stage"] = cfg.maxiter_per_scale
    result["refinement_iterations"] = cfg.refinement_iters
    config = {
        "ablation_design": PLAN,
        "implementation_version": bench.IMPLEMENTATION_VERSION,
        "source_campaign_design": SOURCE_PLAN,
        "dataset": dataset,
        "seed": seed,
        "reference_method": SOURCE_METHOD,
        "control_method": METHOD,
        "profile": asdict(profile),
        "stage_sigmas": list(cfg.sigmas),
        "optimized_sigmas": [cfg.sigmas[-1]] * len(cfg.sigmas),
        "maxiter_per_stage": cfg.maxiter_per_scale,
        "refinement_iters": cfg.refinement_iters,
        "budget_note": (
            "Same number of L-BFGS-B stages and stagewise iteration limits. "
            "Actual objective calls may differ because of optimizer convergence."
        ),
        "qnn_epochs": cfg.qnn_epochs,
        "qnn_layers": cfg.qnn_layers,
        "qnn_seed": cfg.seed + 777,
        "n_qubits": cfg.n_qubits,
        "latent_dim": cfg.latent_dim,
        "max_samples_per_class": cfg.max_samples_per_class,
        "spectral_radius": cfg.spectral_radius,
        "input_scale": cfg.input_scale,
        "reference_raw": str(reference_raw.resolve()),
    }
    return pd.DataFrame([result]), config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reference-raw", type=Path, required=True)
    parser.add_argument("--out-raw", type=Path, required=True)
    parser.add_argument("--out-config", type=Path, required=True)
    args = parser.parse_args()
    if args.seed < 0:
        raise SystemExit("The seed must be nonnegative")
    result, config = experiment(args.dataset, args.seed, args.reference_raw)
    args.out_raw.parent.mkdir(parents=True, exist_ok=True)
    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    # Write raw last so incomplete runs cannot appear finished.
    args.out_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    result.to_csv(args.out_raw, index=False)
    print(
        f"Saved control {args.dataset} seed={args.seed}: "
        f"QNN BA={result.iloc[0]['balanced_accuracy']:.6f}; "
        f"mean objective calls={result.iloc[0]['stat_test_forced_mean_objective_calls']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
