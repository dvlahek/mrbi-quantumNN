"""
qnn_mrbi_article_benchmark.py

Article-oriented benchmark for simulated QNN readouts and MRBI-hybrid implicit
representations.

This script assumes that mrbi.py is in the same folder:
    import mrbi

Install:
    pip install numpy scipy scikit-learn pandas torch pennylane

Quick test:
    python qnn_mrbi_article_benchmark.py --mode quick

Stronger article-style test:
    python qnn_mrbi_article_benchmark.py --mode article --seeds 0 1 2 3 4

Larger qubit sweep, slower:
    python qnn_mrbi_article_benchmark.py --mode article --qubits 4 6 8 --seeds 0 1 2

Main additions compared with the first prototype:
    1) Fixed MLP baseline.
    2) Added SVM-RBF, RandomForest and GradientBoosting baselines.
    3) Added zero-init implicit features as a control.
    4) Added MRBI-hybrid implicit features under multiple MRBI configurations.
    5) Added MRBI ablation without detector.
    6) Added multiple seed support.
    7) Added qubit sweep.
    8) Saves raw results and mean±std summary CSV files.

Important scientific interpretation:
    This does not demonstrate quantum speedup because all QNNs are simulated on
    a classical computer. The test is about representation quality, solver-aware
    initialization, and whether QNN readouts are competitive in small-data regimes.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, asdict, replace
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import mrbi

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------------
# Optional quantum dependency
# ---------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import pennylane as qml
    HAS_QNN = True
except Exception as e:
    HAS_QNN = False
    QNN_IMPORT_ERROR = repr(e)


# ---------------------------------------------------------------------
# Config classes
# ---------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    seed: int = 42
    test_size: float = 0.30

    n_qubits: int = 4
    pca_dim: int = 4
    latent_dim: int = 8
    max_samples_per_class: int = 100

    # Implicit layer regime.
    # Values above 1 create harder non-contractive regimes.
    spectral_radius: float = 1.15
    input_scale: float = 0.70
    bias_scale: float = 0.05

    # QNN training
    qnn_epochs: int = 50
    qnn_lr: float = 0.02
    qnn_batch_size: int = 16
    qnn_layers: int = 2
    run_qnn: bool = True

    # MRBI defaults; overwritten by MRBIProfile where needed
    mrbi_profile_name: str = "balanced"
    mrbi_sigmas: Tuple[float, ...] = (0.30, 0.08, 0.02)
    mrbi_mc_samples: int = 6
    mrbi_maxiter_per_scale: int = 30
    mrbi_refinement_iters: int = 30
    mrbi_gamma: float = 0.02
    mrbi_beta: float = 0.01
    mrbi_lambda_zero: float = 0.10
    mrbi_search_radius: float = 1.0
    mrbi_use_detector: bool = True
    mrbi_use_newton_term: bool = True

    # Hybrid trigger settings
    tau_r: float = 1e-8
    tau_sigma: float = 0.05
    residual_trigger_scale: float = 0.05
    accept_factor_vs_zero: float = 1.05


@dataclass
class MRBIProfile:
    name: str
    sigmas: Tuple[float, ...]
    mc_samples: int
    maxiter_per_scale: int
    refinement_iters: int
    gamma: float
    beta: float
    lambda_zero: float
    search_radius: float
    use_detector: bool
    use_newton_term: bool
    tau_sigma: float
    residual_trigger_scale: float


def get_mrbi_profiles(mode: str) -> List[MRBIProfile]:
    """
    MRBI profile grid.

    For article mode, keep the grid interpretable:
        - light: cheap MRBI
        - balanced: default compromise
        - strong: more expensive, more stable
        - no_detector: ablation of the multiscale detector term
        - residual_only: strong ablation; mostly residual localization
    """
    profiles = [
        MRBIProfile(
            name="light",
            sigmas=(0.20, 0.05),
            mc_samples=4,
            maxiter_per_scale=15,
            refinement_iters=15,
            gamma=0.01,
            beta=0.005,
            lambda_zero=0.10,
            search_radius=0.8,
            use_detector=True,
            use_newton_term=True,
            tau_sigma=0.04,
            residual_trigger_scale=0.05,
        ),
        MRBIProfile(
            name="balanced",
            sigmas=(0.30, 0.08, 0.02),
            mc_samples=6,
            maxiter_per_scale=30,
            refinement_iters=30,
            gamma=0.02,
            beta=0.01,
            lambda_zero=0.10,
            search_radius=1.0,
            use_detector=True,
            use_newton_term=True,
            tau_sigma=0.05,
            residual_trigger_scale=0.05,
        ),
    ]

    if mode == "article":
        profiles.extend([
            MRBIProfile(
                name="strong",
                sigmas=(0.50, 0.15, 0.05, 0.015),
                mc_samples=10,
                maxiter_per_scale=50,
                refinement_iters=50,
                gamma=0.03,
                beta=0.01,
                lambda_zero=0.08,
                search_radius=1.2,
                use_detector=True,
                use_newton_term=True,
                tau_sigma=0.06,
                residual_trigger_scale=0.03,
            ),
            MRBIProfile(
                name="no_detector",
                sigmas=(0.30, 0.08, 0.02),
                mc_samples=1,
                maxiter_per_scale=30,
                refinement_iters=30,
                gamma=0.0,
                beta=0.01,
                lambda_zero=0.10,
                search_radius=1.0,
                use_detector=False,
                use_newton_term=True,
                tau_sigma=0.05,
                residual_trigger_scale=0.05,
            ),
            MRBIProfile(
                name="residual_only",
                sigmas=(0.30, 0.08, 0.02),
                mc_samples=1,
                maxiter_per_scale=30,
                refinement_iters=30,
                gamma=0.0,
                beta=0.0,
                lambda_zero=0.10,
                search_radius=1.0,
                use_detector=False,
                use_newton_term=False,
                tau_sigma=0.05,
                residual_trigger_scale=0.05,
            ),
        ])

    return profiles


def apply_profile(cfg: ExperimentConfig, profile: MRBIProfile) -> ExperimentConfig:
    return replace(
        cfg,
        mrbi_profile_name=profile.name,
        mrbi_sigmas=profile.sigmas,
        mrbi_mc_samples=profile.mc_samples,
        mrbi_maxiter_per_scale=profile.maxiter_per_scale,
        mrbi_refinement_iters=profile.refinement_iters,
        mrbi_gamma=profile.gamma,
        mrbi_beta=profile.beta,
        mrbi_lambda_zero=profile.lambda_zero,
        mrbi_search_radius=profile.search_radius,
        mrbi_use_detector=profile.use_detector,
        mrbi_use_newton_term=profile.use_newton_term,
        tau_sigma=profile.tau_sigma,
        residual_trigger_scale=profile.residual_trigger_scale,
    )


# ---------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------

def balanced_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        keep.extend(idx[:max_per_class].tolist())
    keep = np.array(keep, dtype=int)
    rng.shuffle(keep)
    return X[keep], y[keep]


def load_dataset(name: str, cfg: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray]:
    name = name.lower()

    if name == "breast_cancer":
        data = load_breast_cancer()
        X = data.data.astype(np.float64)
        y = data.target.astype(int)

    elif name == "wine_binary":
        data = load_wine()
        X = data.data.astype(np.float64)
        y_raw = data.target.astype(int)
        mask = y_raw < 2
        X = X[mask]
        y = y_raw[mask].astype(int)

    elif name == "iris_binary":
        data = load_iris()
        X = data.data.astype(np.float64)
        y_raw = data.target.astype(int)
        mask = y_raw > 0
        X = X[mask]
        y = (y_raw[mask] == 2).astype(int)

    elif name == "digits_3_vs_8":
        data = load_digits()
        X = data.data.astype(np.float64) / 16.0
        y_raw = data.target.astype(int)
        mask = (y_raw == 3) | (y_raw == 8)
        X = X[mask]
        y = (y_raw[mask] == 8).astype(int)

    elif name == "digits_1_vs_7":
        data = load_digits()
        X = data.data.astype(np.float64) / 16.0
        y_raw = data.target.astype(int)
        mask = (y_raw == 1) | (y_raw == 7)
        X = X[mask]
        y = (y_raw[mask] == 7).astype(int)

    elif name == "digits_4_vs_9":
        data = load_digits()
        X = data.data.astype(np.float64) / 16.0
        y_raw = data.target.astype(int)
        mask = (y_raw == 4) | (y_raw == 9)
        X = X[mask]
        y = (y_raw[mask] == 9).astype(int)

    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Use one of: "
            "breast_cancer, wine_binary, iris_binary, digits_3_vs_8, "
            "digits_1_vs_7, digits_4_vs_9"
        )

    X, y = balanced_subsample(X, y, cfg.max_samples_per_class, cfg.seed)
    return X, y


def preprocess_to_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    pca_dim: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_components = min(pca_dim, X_train.shape[1], max(1, X_train.shape[0] - 1))
    pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=seed),
        StandardScaler(),
    )
    X_train_p = pipe.fit_transform(X_train).astype(np.float64)
    X_test_p = pipe.transform(X_test).astype(np.float64)

    # If requested pca_dim is larger than possible, pad zeros to keep dimensions stable.
    if n_components < pca_dim:
        pad_tr = np.zeros((X_train_p.shape[0], pca_dim - n_components), dtype=np.float64)
        pad_te = np.zeros((X_test_p.shape[0], pca_dim - n_components), dtype=np.float64)
        X_train_p = np.hstack([X_train_p, pad_tr])
        X_test_p = np.hstack([X_test_p, pad_te])

    # Compress values for stable angle encoding and implicit tanh forcing.
    X_train_p = np.tanh(X_train_p)
    X_test_p = np.tanh(X_test_p)
    return X_train_p, X_test_p


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
    return out


def add_common_metadata(
    row: Dict[str, Any],
    dataset: str,
    method: str,
    cfg: ExperimentConfig,
    representation: str,
    readout: str,
) -> Dict[str, Any]:
    row.update({
        "dataset": dataset,
        "method": method,
        "representation": representation,
        "readout": readout,
        "seed": cfg.seed,
        "n_qubits": cfg.n_qubits,
        "pca_dim": cfg.pca_dim,
        "latent_dim": cfg.latent_dim,
        "spectral_radius": cfg.spectral_radius,
        "max_samples_per_class": cfg.max_samples_per_class,
        "mrbi_profile": cfg.mrbi_profile_name,
    })
    return row


# ---------------------------------------------------------------------
# Classical readouts / baselines
# ---------------------------------------------------------------------

def train_logreg(X_train, y_train, X_test, y_test, seed: int) -> Dict[str, float]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, solver="lbfgs", random_state=seed),
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    pred = clf.predict(X_test)
    score = clf.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, pred, score)
    m["train_time_sec"] = float(train_time)
    return m


def train_mlp(X_train, y_train, X_test, y_test, seed: int) -> Dict[str, float]:
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            alpha=1e-4,
            max_iter=3000,
            random_state=seed,
            early_stopping=False,
        ),
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    pred = clf.predict(X_test)
    score = clf.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, pred, score)
    m["train_time_sec"] = float(train_time)
    return m


def train_svm_rbf(X_train, y_train, X_test, y_test, seed: int) -> Dict[str, float]:
    clf = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=seed,
        ),
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    pred = clf.predict(X_test)
    score = clf.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, pred, score)
    m["train_time_sec"] = float(train_time)
    return m


def train_random_forest(X_train, y_train, X_test, y_test, seed: int) -> Dict[str, float]:
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    pred = clf.predict(X_test)
    score = clf.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, pred, score)
    m["train_time_sec"] = float(train_time)
    return m


def train_gradient_boosting(X_train, y_train, X_test, y_test, seed: int) -> Dict[str, float]:
    clf = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=seed,
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    pred = clf.predict(X_test)
    score = clf.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, pred, score)
    m["train_time_sec"] = float(train_time)
    return m


READOUTS = {
    "logreg": train_logreg,
    "mlp": train_mlp,
    "svm_rbf": train_svm_rbf,
    "rf": train_random_forest,
    "gb": train_gradient_boosting,
}


# ---------------------------------------------------------------------
# QNN model
# ---------------------------------------------------------------------

if HAS_QNN:
    class TorchQNN(nn.Module):
        def __init__(self, n_features: int, n_qubits: int, n_layers: int, seed: int):
            super().__init__()
            torch.manual_seed(seed)

            self.n_features = n_features
            self.n_qubits = n_qubits
            self.n_layers = n_layers

            self.pre = nn.Linear(n_features, n_qubits)
            self.post = nn.Linear(n_qubits, 2)

            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def circuit(inputs, weights):
                # Angle encoding
                for q in range(n_qubits):
                    qml.RY(inputs[q], wires=q)

                # Variational block
                for l in range(n_layers):
                    for q in range(n_qubits):
                        qml.RY(weights[l, q, 0], wires=q)
                        qml.RZ(weights[l, q, 1], wires=q)

                    for q in range(n_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
                    if n_qubits > 2:
                        qml.CNOT(wires=[n_qubits - 1, 0])

                return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

            self.circuit = circuit
            self.q_weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 2))

        def forward(self, x):
            angles = torch.tanh(self.pre(x)) * np.pi

            q_out_list = []
            for i in range(angles.shape[0]):
                q_vals = self.circuit(angles[i], self.q_weights)
                q_vals = torch.stack(q_vals).float()
                q_out_list.append(q_vals)

            q_out = torch.stack(q_out_list, dim=0)
            return self.post(q_out)


def train_qnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: ExperimentConfig,
    seed: int,
) -> Dict[str, float]:
    if not cfg.run_qnn:
        return {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "train_time_sec": 0.0,
            "note": "QNN disabled",
        }

    if not HAS_QNN:
        return {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "train_time_sec": 0.0,
            "note": f"Skipped QNN because imports failed: {QNN_IMPORT_ERROR}",
        }

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.long, device=device)
    Xte = torch.tensor(X_test, dtype=torch.float32, device=device)

    model = TorchQNN(
        n_features=X_train.shape[1],
        n_qubits=cfg.n_qubits,
        n_layers=cfg.qnn_layers,
        seed=seed,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.qnn_lr)
    loss_fn = nn.CrossEntropyLoss()

    n = Xtr.shape[0]
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    for epoch in range(cfg.qnn_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, cfg.qnn_batch_size):
            idx = perm[start:start + cfg.qnn_batch_size]
            xb = Xtr[idx]
            yb = ytr[idx]

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    train_time = time.perf_counter() - t0

    with torch.no_grad():
        logits = model(Xte)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = np.argmax(probs, axis=1)

    metrics = compute_metrics(y_test, pred, probs[:, 1])
    metrics["train_time_sec"] = float(train_time)
    return metrics


# ---------------------------------------------------------------------
# Implicit layer and MRBI transforms
# ---------------------------------------------------------------------

def make_random_implicit_layer(
    dx: int,
    d: int,
    cfg: ExperimentConfig,
    seed: int,
):
    rng = np.random.default_rng(seed)

    W = rng.normal(size=(d, d))
    eigvals = np.linalg.eigvals(W)
    radius = float(np.max(np.abs(eigvals)))
    W = W / max(radius, 1e-12) * cfg.spectral_radius

    U = rng.normal(scale=cfg.input_scale / np.sqrt(max(dx, 1)), size=(d, dx))
    b = rng.normal(scale=cfg.bias_scale, size=d)

    return mrbi.ImplicitTanhLayer(W=W, U=U, b=b)


def make_mrbi_configs(cfg: ExperimentConfig):
    root_cfg = mrbi.RootSolveConfig(
        method="hybr",
        tol=1e-10,
        success_residual_tol=1e-8,
    )

    mrbi_cfg = mrbi.MRBIConfig(
        sigmas=cfg.mrbi_sigmas,
        alpha=1.0,
        beta=cfg.mrbi_beta,
        gamma=cfg.mrbi_gamma,
        lambda_zero=cfg.mrbi_lambda_zero,
        lambda_relative=1.0,
        epsilon=1e-8,
        search_radius=cfg.mrbi_search_radius,
        mc_samples=cfg.mrbi_mc_samples,
        maxiter_per_scale=cfg.mrbi_maxiter_per_scale,
        refinement_iters=cfg.mrbi_refinement_iters,
        use_continuation=True,
        use_refinement=True,
        use_detector=cfg.mrbi_use_detector,
        use_newton_term=cfg.mrbi_use_newton_term,
        n_starts=1,
        random_start_scale=0.05,
        newton_damping=1e-6,
        optimizer_ftol=1e-10,
        use_fixed_probes_per_scale=True,
        use_antithetic_sampling=True,
    )

    hybrid_cfg = mrbi.HybridConfig(
        tau_r=cfg.tau_r,
        tau_sigma=cfg.tau_sigma,
        accept_factor_vs_zero=cfg.accept_factor_vs_zero,
        use_relative_residual_trigger=True,
        residual_trigger_scale=cfg.residual_trigger_scale,
        trigger_on_solver_failure=True,
        trigger_on_bad_jacobian=True,
        tie_residual_rtol=0.02,
        prefer_better_conditioning_on_tie=True,
        accept_any_finite_success_if_zero_failed=True,
    )

    return root_cfg, mrbi_cfg, hybrid_cfg


def solve_zero_features(
    X: np.ndarray,
    layer,
    cfg: ExperimentConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    F, J = mrbi.make_residual_and_jacobian(layer)
    root_cfg = mrbi.RootSolveConfig(method="hybr", tol=1e-10, success_residual_tol=1e-8)

    Z = []
    residuals = []
    nfev = []
    success = []
    runtimes = []

    z0 = np.zeros(layer.d, dtype=np.float64)
    for x in X:
        res = mrbi.solve_root(F, J, x, z0, cfg=root_cfg)
        Z.append(res.z_star)
        residuals.append(float(res.residual))
        nfev.append(int(res.nfev))
        success.append(int(res.success))
        runtimes.append(float(res.runtime_sec))

    stats = {
        "zero_success_rate": float(np.mean(success)),
        "zero_mean_residual": float(np.mean(residuals)),
        "zero_median_residual": float(np.median(residuals)),
        "zero_mean_nfev": float(np.mean(nfev)),
        "zero_mean_runtime_sec_per_sample": float(np.mean(runtimes)),
    }
    return np.asarray(Z, dtype=np.float64), stats


def solve_mrbi_hybrid_features(
    X: np.ndarray,
    layer,
    cfg: ExperimentConfig,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    root_cfg, mrbi_cfg, hybrid_cfg = make_mrbi_configs(cfg)
    rng = np.random.default_rng(seed)

    Z = []
    used_mrbi = []
    triggered = []
    success = []
    residuals = []
    nfev_zero = []
    nfev_final = []
    runtimes = []
    sigma_zero = []
    sigma_det = []
    trigger_reasons = {}

    for x in X:
        res = mrbi.hybrid_mrbi_solve_tanh_layer(
            layer,
            x,
            mrbi_cfg=mrbi_cfg,
            hybrid_cfg=hybrid_cfg,
            root_cfg=root_cfg,
            rng=rng,
        )
        Z.append(res.z_star)
        used_mrbi.append(int(res.used_mrbi))
        triggered.append(int(res.trigger_reason != "none"))
        success.append(int(res.success))
        residuals.append(float(res.residual))
        nfev_zero.append(int(res.zero_result.nfev))
        nfev_final.append(int(res.final_result.nfev))
        runtimes.append(float(res.total_runtime_sec or 0.0))
        sigma_zero.append(float(res.sigma_min_zero or 0.0))
        sigma_det.append(float(res.sigma_min_det or np.nan))
        trigger_reasons[res.trigger_reason] = trigger_reasons.get(res.trigger_reason, 0) + 1

    stats = {
        "mrbi_success_rate": float(np.mean(success)),
        "mrbi_trigger_rate": float(np.mean(triggered)),
        "mrbi_accept_rate": float(np.mean(used_mrbi)),
        "mrbi_mean_residual": float(np.mean(residuals)),
        "mrbi_median_residual": float(np.median(residuals)),
        "mrbi_mean_zero_nfev": float(np.mean(nfev_zero)),
        "mrbi_mean_final_nfev": float(np.mean(nfev_final)),
        "mrbi_mean_runtime_sec_per_sample": float(np.mean(runtimes)),
        "mrbi_mean_sigma_min_zero": float(np.nanmean(sigma_zero)),
        "mrbi_mean_sigma_min_det": float(np.nanmean(sigma_det)),
        "mrbi_trigger_reasons_json": json.dumps(trigger_reasons),
    }
    return np.asarray(Z, dtype=np.float64), stats


def standardize_features(
    Z_train: np.ndarray,
    Z_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(Z_train), scaler.transform(Z_test)


# ---------------------------------------------------------------------
# Evaluation blocks
# ---------------------------------------------------------------------

def evaluate_readouts(
    dataset_name: str,
    representation_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cfg: ExperimentConfig,
    readouts: List[str],
    extra_stats: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    rows = []
    extra_stats = extra_stats or {}

    for readout in readouts:
        if readout == "qnn":
            metrics = train_qnn(X_train, y_train, X_test, y_test, cfg, cfg.seed + 777)
        else:
            metrics = READOUTS[readout](X_train, y_train, X_test, y_test, cfg.seed)

        method = f"{representation_name}_{readout}"
        row = {**metrics, **extra_stats}
        row = add_common_metadata(
            row=row,
            dataset=dataset_name,
            method=method,
            cfg=cfg,
            representation=representation_name,
            readout=readout,
        )
        rows.append(row)

    return rows


def run_one_dataset_one_seed_one_qubit(
    dataset_name: str,
    base_cfg: ExperimentConfig,
    profiles: List[MRBIProfile],
    readouts_pca: List[str],
    readouts_implicit: List[str],
) -> pd.DataFrame:
    X, y = load_dataset(dataset_name, base_cfg)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=base_cfg.test_size,
        random_state=base_cfg.seed,
        stratify=y,
    )

    # PCA dimension is tied to n_qubits for fair QNN input size.
    X_train_p, X_test_p = preprocess_to_pca(
        X_train_raw,
        X_test_raw,
        pca_dim=base_cfg.pca_dim,
        seed=base_cfg.seed,
    )

    rows: List[Dict[str, Any]] = []

    # Classical/QNN baselines on PCA features.
    rows.extend(
        evaluate_readouts(
            dataset_name,
            "pca",
            X_train_p,
            X_test_p,
            y_train,
            y_test,
            base_cfg,
            readouts=readouts_pca,
            extra_stats={},
        )
    )

    # Same implicit layer across zero-init and MRBI profiles for this seed/qubit.
    layer = make_random_implicit_layer(
        dx=X_train_p.shape[1],
        d=base_cfg.latent_dim,
        cfg=base_cfg,
        seed=base_cfg.seed + 1000 + base_cfg.n_qubits,
    )

    # Control: implicit features solved from zero initialization only.
    t0 = time.perf_counter()
    Z_train_zero, zero_train_stats = solve_zero_features(X_train_p, layer, base_cfg)
    Z_test_zero, zero_test_stats = solve_zero_features(X_test_p, layer, base_cfg)
    zero_feature_time = float(time.perf_counter() - t0)

    Z_train_zero_s, Z_test_zero_s = standardize_features(Z_train_zero, Z_test_zero)
    zero_stats = {f"test_{k}": v for k, v in zero_test_stats.items()}
    zero_stats["feature_time_sec"] = zero_feature_time

    rows.extend(
        evaluate_readouts(
            dataset_name,
            "implicit_zero",
            Z_train_zero_s,
            Z_test_zero_s,
            y_train,
            y_test,
            base_cfg,
            readouts=readouts_implicit,
            extra_stats=zero_stats,
        )
    )

    # MRBI profiles.
    for profile in profiles:
        cfg = apply_profile(base_cfg, profile)

        t0 = time.perf_counter()
        Z_train_mrbi, mrbi_train_stats = solve_mrbi_hybrid_features(
            X_train_p,
            layer,
            cfg,
            seed=cfg.seed + 2000 + cfg.n_qubits,
        )
        Z_test_mrbi, mrbi_test_stats = solve_mrbi_hybrid_features(
            X_test_p,
            layer,
            cfg,
            seed=cfg.seed + 3000 + cfg.n_qubits,
        )
        mrbi_feature_time = float(time.perf_counter() - t0)

        Z_train_mrbi_s, Z_test_mrbi_s = standardize_features(Z_train_mrbi, Z_test_mrbi)
        mrbi_stats = {f"test_{k}": v for k, v in mrbi_test_stats.items()}
        mrbi_stats["feature_time_sec"] = mrbi_feature_time

        rows.extend(
            evaluate_readouts(
                dataset_name,
                f"mrbi_{profile.name}",
                Z_train_mrbi_s,
                Z_test_mrbi_s,
                y_train,
                y_test,
                cfg,
                readouts=readouts_implicit,
                extra_stats=mrbi_stats,
            )
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "train_time_sec",
        "feature_time_sec",
        "test_mrbi_trigger_rate",
        "test_mrbi_accept_rate",
        "test_mrbi_mean_residual",
        "test_mrbi_mean_runtime_sec_per_sample",
        "test_zero_mean_residual",
        "test_zero_mean_runtime_sec_per_sample",
    ]

    available_metrics = [c for c in metric_cols if c in results.columns]

    group_cols = [
        "dataset",
        "method",
        "representation",
        "readout",
        "n_qubits",
        "latent_dim",
        "spectral_radius",
        "mrbi_profile",
    ]

    rows = []
    grouped = results.groupby(group_cols, dropna=False)
    for keys, g in grouped:
        row = dict(zip(group_cols, keys))
        row["n_runs"] = int(len(g))
        for c in available_metrics:
            vals = pd.to_numeric(g[c], errors="coerce")
            row[f"{c}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{c}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        rows.append(row)

    summary = pd.DataFrame(rows)
    if "balanced_accuracy_mean" in summary.columns:
        summary = summary.sort_values(
            ["dataset", "n_qubits", "balanced_accuracy_mean", "roc_auc_mean"],
            ascending=[True, True, False, False],
        )
    return summary


def print_compact_leaderboard(summary: pd.DataFrame, top_k: int = 8) -> None:
    if summary.empty:
        return

    show_cols = [
        "dataset",
        "n_qubits",
        "method",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "f1_mean",
        "roc_auc_mean",
        "n_runs",
    ]
    show_cols = [c for c in show_cols if c in summary.columns]

    for (dataset, nq), g in summary.groupby(["dataset", "n_qubits"], dropna=False):
        print(f"\n--- Leaderboard: dataset={dataset}, qubits={nq} ---")
        print(g[show_cols].head(top_k).to_string(index=False))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["quick", "article"],
        default="quick",
        help="quick is fast; article runs more profiles and is more suitable for paper tables.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to run. If omitted, defaults depend on mode.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--qubits", nargs="+", type=int, default=None)
    parser.add_argument("--latent-dims", nargs="+", type=int, default=None)
    parser.add_argument("--spectral-radii", nargs="+", type=float, default=None)

    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--qnn-epochs", type=int, default=None)
    parser.add_argument("--qnn-layers", type=int, default=2)
    parser.add_argument("--no-qnn", action="store_true")

    parser.add_argument("--out-raw", type=str, default="qnn_mrbi_article_raw.csv")
    parser.add_argument("--out-summary", type=str, default="qnn_mrbi_article_summary.csv")
    parser.add_argument("--out-config", type=str, default="qnn_mrbi_article_config.json")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "quick":
        datasets = args.datasets or ["breast_cancer", "wine_binary", "iris_binary", "digits_3_vs_8"]
        seeds = args.seeds or [42]
        qubits = args.qubits or [4]
        latent_dims = args.latent_dims or [8]
        spectral_radii = args.spectral_radii or [1.15]
        max_samples_per_class = args.max_samples_per_class or 80
        qnn_epochs = args.qnn_epochs or 40
        readouts_pca = ["logreg", "mlp", "svm_rbf", "qnn"]
        readouts_implicit = ["logreg", "svm_rbf", "qnn"]

    else:
        datasets = args.datasets or [
            "breast_cancer",
            "wine_binary",
            "iris_binary",
            "digits_3_vs_8",
            "digits_1_vs_7",
            "digits_4_vs_9",
        ]
        seeds = args.seeds or [0, 1, 2, 3, 4]
        qubits = args.qubits or [4, 6]
        latent_dims = args.latent_dims or [8, 12]
        spectral_radii = args.spectral_radii or [0.90, 1.15, 1.35]
        max_samples_per_class = args.max_samples_per_class or 100
        qnn_epochs = args.qnn_epochs or 60
        readouts_pca = ["logreg", "mlp", "svm_rbf", "rf", "gb", "qnn"]
        readouts_implicit = ["logreg", "svm_rbf", "qnn"]

    if args.no_qnn:
        readouts_pca = [r for r in readouts_pca if r != "qnn"]
        readouts_implicit = [r for r in readouts_implicit if r != "qnn"]

    profiles = get_mrbi_profiles(args.mode)

    config_dump = {
        "mode": args.mode,
        "datasets": datasets,
        "seeds": seeds,
        "qubits": qubits,
        "latent_dims": latent_dims,
        "spectral_radii": spectral_radii,
        "max_samples_per_class": max_samples_per_class,
        "qnn_epochs": qnn_epochs,
        "qnn_layers": args.qnn_layers,
        "run_qnn": not args.no_qnn,
        "mrbi_profiles": [p.__dict__ for p in profiles],
        "readouts_pca": readouts_pca,
        "readouts_implicit": readouts_implicit,
    }

    print("Benchmark configuration:")
    print(json.dumps(config_dump, indent=2, default=str))

    Path(args.out_config).write_text(json.dumps(config_dump, indent=2, default=str), encoding="utf-8")

    all_results = []
    total_jobs = (
        len(datasets) * len(seeds) * len(qubits) * len(latent_dims) * len(spectral_radii)
    )
    job_id = 0

    for dataset in datasets:
        for seed in seeds:
            for n_qubits in qubits:
                for latent_dim in latent_dims:
                    for spectral_radius in spectral_radii:
                        job_id += 1
                        pca_dim = n_qubits

                        cfg = ExperimentConfig(
                            seed=seed,
                            n_qubits=n_qubits,
                            pca_dim=pca_dim,
                            latent_dim=latent_dim,
                            max_samples_per_class=max_samples_per_class,
                            spectral_radius=spectral_radius,
                            qnn_epochs=qnn_epochs,
                            qnn_layers=args.qnn_layers,
                            run_qnn=(not args.no_qnn),
                        )

                        print(
                            f"\n[{job_id}/{total_jobs}] "
                            f"dataset={dataset}, seed={seed}, qubits={n_qubits}, "
                            f"latent_dim={latent_dim}, rho={spectral_radius}"
                        )

                        df = run_one_dataset_one_seed_one_qubit(
                            dataset_name=dataset,
                            base_cfg=cfg,
                            profiles=profiles,
                            readouts_pca=readouts_pca,
                            readouts_implicit=readouts_implicit,
                        )

                        compact_cols = [
                            "dataset",
                            "method",
                            "n_qubits",
                            "latent_dim",
                            "spectral_radius",
                            "accuracy",
                            "balanced_accuracy",
                            "f1",
                            "roc_auc",
                        ]
                        print(df[compact_cols].sort_values("balanced_accuracy", ascending=False).head(10).to_string(index=False))

                        all_results.append(df)

                        # Save incrementally so long article runs are not lost.
                        raw_partial = pd.concat(all_results, ignore_index=True)
                        raw_partial.to_csv(args.out_raw, index=False)

    results = pd.concat(all_results, ignore_index=True)
    summary = summarize_results(results)

    results.to_csv(args.out_raw, index=False)
    summary.to_csv(args.out_summary, index=False)

    print(f"\nSaved raw results to: {args.out_raw}")
    print(f"Saved summary to: {args.out_summary}")
    print(f"Saved config to: {args.out_config}")

    print_compact_leaderboard(summary)


if __name__ == "__main__":
    main()
