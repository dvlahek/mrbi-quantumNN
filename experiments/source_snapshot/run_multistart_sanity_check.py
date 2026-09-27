"""
Budgeted multistart sanity check for the MRBI-QNN paper.

Purpose
-------
This script compares three ways of constructing implicit equilibrium features
before the downstream readout:

1. zero initialization;
2. one fixed MRBI-hybrid configuration;
3. random multistart initialization with N random starts, selecting the
   converged feature with the smallest final residual.

The script is intended as a reviewer-facing diagnostic, not as a replacement
for the main benchmark. By default it runs a small hard-regime check on
wine_binary, wine_0_vs_2, and digits_4_vs_9.

Typical commands
----------------
Fast classical sanity check:
    python run_multistart_sanity_check.py --readout logreg --seeds 0 1 2

QNN check used for the paper discussion:
    python run_multistart_sanity_check.py --readout qnn --seeds 0 1 2 \
        --datasets wine_binary wine_0_vs_2 digits_4_vs_9 \
        --spectral-radius 2.0 --n-random-starts 5

Outputs
-------
- multistart_sanity_raw.csv
- multistart_sanity_summary.csv
- multistart_sanity_table.tex

Interpretation
--------------
If MRBI is better than or comparable to random multistart under this diagnostic,
that supports the claim that the residual-aware initialization is not only a
random restart effect. If random multistart is better, the paper should report
this as a limitation and narrow the claim accordingly.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.datasets import load_wine, load_digits, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import mrbi
except Exception:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import mrbi

try:
    import torch
    import torch.nn as nn
    import pennylane as qml
    HAS_QNN = True
except Exception as exc:
    HAS_QNN = False
    QNN_IMPORT_ERROR = repr(exc)


@dataclass
class Config:
    seed: int = 0
    n_qubits: int = 4
    pca_dim: int = 4
    latent_dim: int = 16
    max_samples_per_class: int = 80
    test_size: float = 0.30
    spectral_radius: float = 2.0
    input_scale: float = 1.1
    bias_scale: float = 0.1
    n_random_starts: int = 5
    random_start_radius: float = 1.0
    qnn_epochs: int = 50
    qnn_lr: float = 0.02
    qnn_batch_size: int = 16
    qnn_layers: int = 2
    readout: str = "qnn"


def balanced_subsample(X: np.ndarray, y: np.ndarray, max_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    keep: List[int] = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        keep.extend(idx[:max_per_class].tolist())
    keep = np.asarray(keep, dtype=int)
    rng.shuffle(keep)
    return X[keep], y[keep]


def load_binary_dataset(name: str, cfg: Config):
    name = name.lower()
    if name == "breast_cancer":
        data = load_breast_cancer()
        X = data.data.astype(np.float64)
        y = data.target.astype(int)
    elif name == "wine_binary":
        data = load_wine()
        X = data.data.astype(np.float64)
        yr = data.target.astype(int)
        mask = yr < 2
        X = X[mask]
        y = yr[mask].astype(int)
    elif name == "wine_0_vs_2":
        data = load_wine()
        X = data.data.astype(np.float64)
        yr = data.target.astype(int)
        mask = (yr == 0) | (yr == 2)
        X = X[mask]
        y = (yr[mask] == 2).astype(int)
    elif name == "wine_1_vs_2":
        data = load_wine()
        X = data.data.astype(np.float64)
        yr = data.target.astype(int)
        mask = (yr == 1) | (yr == 2)
        X = X[mask]
        y = (yr[mask] == 2).astype(int)
    elif name.startswith("digits_"):
        parts = name.split("_")
        a, b = int(parts[1]), int(parts[3])
        data = load_digits()
        X = data.data.astype(np.float64) / 16.0
        yr = data.target.astype(int)
        mask = (yr == a) | (yr == b)
        X = X[mask]
        y = (yr[mask] == b).astype(int)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return balanced_subsample(X, y, cfg.max_samples_per_class, cfg.seed)


def preprocess_to_pca(X_train, X_test, pca_dim: int, seed: int):
    n_components = min(pca_dim, X_train.shape[1], max(1, X_train.shape[0] - 1))
    pipe = make_pipeline(StandardScaler(), PCA(n_components=n_components, random_state=seed), StandardScaler())
    Xtr = pipe.fit_transform(X_train).astype(np.float64)
    Xte = pipe.transform(X_test).astype(np.float64)
    if n_components < pca_dim:
        Xtr = np.hstack([Xtr, np.zeros((Xtr.shape[0], pca_dim - n_components))])
        Xte = np.hstack([Xte, np.zeros((Xte.shape[0], pca_dim - n_components))])
    return np.tanh(Xtr), np.tanh(Xte)


def make_random_implicit_layer(dx: int, d: int, cfg: Config, seed: int):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(A)
    B = rng.normal(size=(d, d))
    R, _ = np.linalg.qr(B)
    scales = np.geomspace(1.0, 0.08, d)
    W = Q @ np.diag(scales) @ R.T
    upper = np.triu(rng.normal(size=(d, d)), k=1)
    W = W + 0.35 * upper / max(np.linalg.norm(upper, 2), 1e-12)
    radius = float(np.max(np.abs(np.linalg.eigvals(W))))
    W = W / max(radius, 1e-12) * cfg.spectral_radius
    U = rng.normal(scale=cfg.input_scale / np.sqrt(max(dx, 1)), size=(d, dx))
    b = rng.normal(scale=cfg.bias_scale, size=d)
    return mrbi.ImplicitTanhLayer(W=W, U=U, b=b)


def make_mrbi_cfg():
    return mrbi.MRBIConfig(
        sigmas=(0.50, 0.15, 0.05, 0.015),
        alpha=1.0,
        beta=0.01,
        gamma=0.03,
        lambda_zero=0.08,
        lambda_relative=1.0,
        epsilon=1e-8,
        search_radius=1.2,
        mc_samples=10,
        maxiter_per_scale=50,
        refinement_iters=50,
        use_continuation=True,
        use_refinement=True,
        use_detector=True,
        use_newton_term=True,
        n_starts=1,
        random_start_scale=0.05,
        newton_damping=1e-6,
        optimizer_ftol=1e-10,
        use_fixed_probes_per_scale=True,
        use_antithetic_sampling=True,
    )


def make_hybrid_cfg():
    return mrbi.HybridConfig(
        tau_r=1e-8,
        tau_sigma=0.05,
        accept_factor_vs_zero=1.05,
        use_relative_residual_trigger=True,
        residual_trigger_scale=0.05,
        trigger_on_solver_failure=True,
        trigger_on_bad_jacobian=True,
        tie_residual_rtol=0.02,
        prefer_better_conditioning_on_tie=True,
        accept_any_finite_success_if_zero_failed=True,
    )


def solve_zero_feature(F, J, x, dim: int):
    root_cfg = mrbi.RootSolveConfig(method="hybr", tol=1e-10, success_residual_tol=1e-8)
    return mrbi.solve_root(F, J, x, np.zeros(dim), cfg=root_cfg)


def solve_mrbi_feature(layer, x, rng):
    return mrbi.hybrid_mrbi_solve_tanh_layer(
        layer,
        x,
        mrbi_cfg=make_mrbi_cfg(),
        hybrid_cfg=make_hybrid_cfg(),
        root_cfg=mrbi.RootSolveConfig(method="hybr", tol=1e-10, success_residual_tol=1e-8),
        rng=rng,
    ).final_result


def solve_random_multistart_feature(F, J, x, dim: int, cfg: Config, rng):
    root_cfg = mrbi.RootSolveConfig(method="hybr", tol=1e-10, success_residual_tol=1e-8)
    results = []
    # Keep zero in the candidate set so the multistart baseline cannot be worse
    # by construction only because it excluded the standard initialization.
    results.append(mrbi.solve_root(F, J, x, np.zeros(dim), cfg=root_cfg))
    for _ in range(cfg.n_random_starts):
        z0 = rng.uniform(-cfg.random_start_radius, cfg.random_start_radius, size=dim)
        results.append(mrbi.solve_root(F, J, x, z0, cfg=root_cfg))
    finite = [r for r in results if np.isfinite(r.residual) and np.all(np.isfinite(r.z_star))]
    if not finite:
        return results[0]
    return min(finite, key=lambda r: (r.residual, -int(r.success), r.nfev if r.nfev >= 0 else 10**9))


def build_features(X, layer, method: str, cfg: Config, seed: int):
    F, J = mrbi.make_residual_and_jacobian(layer)
    rng = np.random.default_rng(seed)
    Z, residuals, successes, nfev = [], [], [], []
    t0 = time.perf_counter()
    for x in X:
        if method == "zero":
            res = solve_zero_feature(F, J, x, layer.d)
        elif method == "mrbi":
            res = solve_mrbi_feature(layer, x, rng)
        elif method == "multistart":
            res = solve_random_multistart_feature(F, J, x, layer.d, cfg, rng)
        else:
            raise ValueError(method)
        Z.append(res.z_star)
        residuals.append(float(res.residual))
        successes.append(int(res.success))
        nfev.append(int(res.nfev))
    elapsed = time.perf_counter() - t0
    stats = {
        f"{method}_feature_time_sec": elapsed,
        f"{method}_success_rate": float(np.mean(successes)),
        f"{method}_mean_residual": float(np.mean(residuals)),
        f"{method}_median_residual": float(np.median(residuals)),
        f"{method}_mean_nfev": float(np.mean(nfev)),
    }
    return np.asarray(Z, dtype=np.float64), stats


def compute_scores(y_true, y_pred, y_score=None):
    out = {"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)), "f1": float(f1_score(y_true, y_pred, zero_division=0))}
    if y_score is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
    return out


def train_logreg(X_train, y_train, X_test, y_test, seed: int):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs", random_state=seed))
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    runtime = time.perf_counter() - t0
    pred = clf.predict(X_test)
    score = clf.predict_proba(X_test)[:, 1]
    out = compute_scores(y_test, pred, score)
    out["readout_time_sec"] = float(runtime)
    return out


if HAS_QNN:
    class TorchQNN(nn.Module):
        def __init__(self, n_features: int, n_qubits: int, n_layers: int, seed: int):
            super().__init__()
            torch.manual_seed(seed)
            self.pre = nn.Linear(n_features, n_qubits)
            self.post = nn.Linear(n_qubits, 2)
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def circuit(inputs, weights):
                for q in range(n_qubits):
                    qml.RY(inputs[q], wires=q)
                for ell in range(n_layers):
                    for q in range(n_qubits):
                        qml.RY(weights[ell, q, 0], wires=q)
                        qml.RZ(weights[ell, q, 1], wires=q)
                    for q in range(n_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
                    if n_qubits > 2:
                        qml.CNOT(wires=[n_qubits - 1, 0])
                return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

            self.circuit = circuit
            self.q_weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 2))

        def forward(self, x):
            angles = torch.tanh(self.pre(x)) * np.pi
            outs = []
            for i in range(angles.shape[0]):
                outs.append(torch.stack(self.circuit(angles[i], self.q_weights)).float())
            return self.post(torch.stack(outs, dim=0))


def train_qnn(X_train, y_train, X_test, y_test, cfg: Config, seed: int):
    if not HAS_QNN:
        raise RuntimeError(f"QNN dependencies are not available: {QNN_IMPORT_ERROR}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    model = TorchQNN(X_train.shape[1], cfg.n_qubits, cfg.qnn_layers, seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.qnn_lr)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for _ in range(cfg.qnn_epochs):
        perm = rng.permutation(len(Xtr))
        for start in range(0, len(Xtr), cfg.qnn_batch_size):
            idx = perm[start:start + cfg.qnn_batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
    runtime = time.perf_counter() - t0
    with torch.no_grad():
        logits = model(Xte)
        probs = torch.softmax(logits, dim=1).numpy()
        pred = np.argmax(probs, axis=1)
    out = compute_scores(y_test, pred, probs[:, 1])
    out["readout_time_sec"] = float(runtime)
    return out


def evaluate_representation(name, Xtr, Xte, ytr, yte, cfg: Config, seed: int):
    if cfg.readout == "logreg":
        metrics = train_logreg(Xtr, ytr, Xte, yte, seed)
    elif cfg.readout == "qnn":
        metrics = train_qnn(Xtr, ytr, Xte, yte, cfg, seed)
    else:
        raise ValueError(f"Unknown readout: {cfg.readout}")
    metrics["method"] = name
    return metrics


def run_dataset_seed(dataset: str, seed: int, args):
    cfg = Config(
        seed=seed,
        n_qubits=args.n_qubits,
        pca_dim=args.n_qubits,
        latent_dim=args.latent_dim,
        max_samples_per_class=args.max_samples_per_class,
        spectral_radius=args.spectral_radius,
        n_random_starts=args.n_random_starts,
        random_start_radius=args.random_start_radius,
        qnn_epochs=args.qnn_epochs,
        qnn_layers=args.qnn_layers,
        qnn_lr=args.qnn_lr,
        readout=args.readout,
    )
    X, y = load_binary_dataset(dataset, cfg)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(X, y, test_size=cfg.test_size, stratify=y, random_state=seed)
    Xtr_p, Xte_p = preprocess_to_pca(Xtr_raw, Xte_raw, cfg.pca_dim, seed)

    layer = make_random_implicit_layer(Xtr_p.shape[1], cfg.latent_dim, cfg, seed + 1000 + cfg.n_qubits)

    rows = []
    rows.append(evaluate_representation("pca", Xtr_p, Xte_p, ytr, yte, cfg, seed + 700))

    feature_stats = {}
    for method in ["zero", "mrbi", "multistart"]:
        Ztr, st_tr = build_features(Xtr_p, layer, method, cfg, seed + 2000 + hash(method) % 1000)
        Zte, st_te = build_features(Xte_p, layer, method, cfg, seed + 3000 + hash(method) % 1000)
        scaler = StandardScaler()
        Ztr = scaler.fit_transform(Ztr)
        Zte = scaler.transform(Zte)
        metrics = evaluate_representation(method, Ztr, Zte, ytr, yte, cfg, seed + 800 + hash(method) % 1000)
        # Store test-set feature stats in the row.
        metrics.update({k: v for k, v in st_te.items()})
        feature_stats[method] = st_te
        rows.append(metrics)

    for row in rows:
        row.update(asdict(cfg))
        row["dataset"] = dataset
        row["seed"] = seed
    return rows


def make_summary(raw: pd.DataFrame):
    keep = raw[raw["method"].isin(["zero", "mrbi", "multistart"])].copy()
    by_ds = keep.pivot_table(index=["dataset", "seed"], columns="method", values="balanced_accuracy", aggfunc="mean").reset_index()
    by_ds["delta_mrbi_zero"] = by_ds["mrbi"] - by_ds["zero"]
    by_ds["delta_multistart_zero"] = by_ds["multistart"] - by_ds["zero"]
    by_ds["delta_mrbi_multistart"] = by_ds["mrbi"] - by_ds["multistart"]
    summary = by_ds.groupby("dataset", as_index=False).agg(
        n_seeds=("seed", "nunique"),
        zero=("zero", "mean"),
        mrbi=("mrbi", "mean"),
        multistart=("multistart", "mean"),
        delta_mrbi_zero=("delta_mrbi_zero", "mean"),
        delta_multistart_zero=("delta_multistart_zero", "mean"),
        delta_mrbi_multistart=("delta_mrbi_multistart", "mean"),
    )
    mean_row = {"dataset": "Mean", "n_seeds": summary["n_seeds"].min()}
    for c in ["zero", "mrbi", "multistart", "delta_mrbi_zero", "delta_multistart_zero", "delta_mrbi_multistart"]:
        mean_row[c] = summary[c].mean()
    summary = pd.concat([summary, pd.DataFrame([mean_row])], ignore_index=True)
    return summary, by_ds


def write_latex_table(summary: pd.DataFrame, path: Path):
    def fmt(x, signed=False):
        if pd.isna(x):
            return "--"
        return f"{x:+.4f}" if signed else f"{x:.4f}"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Random multistart sanity check. The table compares zero initialization, one fixed MRBI-hybrid configuration, and random multistart initialization with the same downstream readout. Balanced accuracy is averaged over the listed seeds.}",
        r"\label{tab:multistart-sanity}",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Dataset & Zero & MRBI & Multistart & $\Delta_{\mathrm{MRBI-zero}}$ & $\Delta_{\mathrm{MS-zero}}$ & $\Delta_{\mathrm{MRBI-MS}}$ \\",
        r"\hline",
    ]
    for _, row in summary.iterrows():
        dataset = str(row["dataset"]).replace("_", r"\_")
        lines.append(
            f"{dataset} & {fmt(row['zero'])} & {fmt(row['mrbi'])} & {fmt(row['multistart'])} & "
            f"{fmt(row['delta_mrbi_zero'], True)} & {fmt(row['delta_multistart_zero'], True)} & {fmt(row['delta_mrbi_multistart'], True)} \\\\" 
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["wine_binary", "wine_0_vs_2", "digits_4_vs_9"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--readout", choices=["qnn", "logreg"], default="qnn")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--spectral-radius", type=float, default=2.0)
    parser.add_argument("--max-samples-per-class", type=int, default=80)
    parser.add_argument("--n-random-starts", type=int, default=5)
    parser.add_argument("--random-start-radius", type=float, default=1.0)
    parser.add_argument("--qnn-epochs", type=int, default=50)
    parser.add_argument("--qnn-layers", type=int, default=2)
    parser.add_argument("--qnn-lr", type=float, default=0.02)
    parser.add_argument("--out-raw", default="multistart_sanity_raw.csv")
    parser.add_argument("--out-summary", default="multistart_sanity_summary.csv")
    parser.add_argument("--out-table", default="multistart_sanity_table.tex")
    args = parser.parse_args()

    if args.readout == "qnn" and not HAS_QNN:
        raise RuntimeError(f"QNN readout requested, but torch/pennylane imports failed: {QNN_IMPORT_ERROR}")

    print("Configuration:")
    print(json.dumps(vars(args), indent=2))

    all_rows: List[Dict] = []
    total = len(args.datasets) * len(args.seeds)
    job = 0
    for dataset in args.datasets:
        for seed in args.seeds:
            job += 1
            print(f"\n[{job}/{total}] dataset={dataset}, seed={seed}")
            rows = run_dataset_seed(dataset, seed, args)
            part = pd.DataFrame(rows)
            print(part[["dataset", "seed", "method", "balanced_accuracy", "f1", "roc_auc"]].to_string(index=False))
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(args.out_raw, index=False)

    raw = pd.DataFrame(all_rows)
    raw.to_csv(args.out_raw, index=False)
    summary, seed_level = make_summary(raw)
    summary.to_csv(args.out_summary, index=False)
    seed_level.to_csv(Path(args.out_summary).with_name("multistart_sanity_seed_level.csv"), index=False)
    write_latex_table(summary, Path(args.out_table))

    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(" -", args.out_raw)
    print(" -", args.out_summary)
    print(" - multistart_sanity_seed_level.csv")
    print(" -", args.out_table)


if __name__ == "__main__":
    main()
