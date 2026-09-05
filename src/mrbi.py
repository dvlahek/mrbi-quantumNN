"""
Minimal MRBI implementation for the MRBI-QNN reproducibility package.

The module implements the classical implicit tanh layer used in the paper,
a root-solver wrapper, Multiscale Residual-Based Initialization (MRBI), and a
hybrid trigger-and-accept solve. It is intentionally small and dependency-light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple
import time

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, root

Array = NDArray[np.float64]
ResidualFn = Callable[[Array, Array], Array]
JacobianFn = Callable[[Array, Array], Array]


def _arr(x) -> Array:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Expected a one-dimensional array.")
    return x


def _norm(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        return float("inf")
    return float(np.linalg.norm(x))


@dataclass
class ImplicitTanhLayer:
    """Implicit layer z* = tanh(W z* + U x + b)."""

    W: Array
    U: Array
    b: Array

    def __post_init__(self) -> None:
        self.W = np.asarray(self.W, dtype=np.float64)
        self.U = np.asarray(self.U, dtype=np.float64)
        self.b = np.asarray(self.b, dtype=np.float64)
        d = self.W.shape[0]
        if self.W.shape != (d, d):
            raise ValueError("W must be square.")
        if self.U.ndim != 2 or self.U.shape[0] != d:
            raise ValueError("U must have shape (d, input_dim).")
        if self.b.shape != (d,):
            raise ValueError("b must have shape (d,).")

    @property
    def d(self) -> int:
        return int(self.W.shape[0])

    def residual(self, z: Array, x: Array) -> Array:
        z = _arr(z)
        x = _arr(x)
        return z - np.tanh(self.W @ z + self.U @ x + self.b)

    def jacobian(self, z: Array, x: Array) -> Array:
        z = _arr(z)
        x = _arr(x)
        a = self.W @ z + self.U @ x + self.b
        g = 1.0 - np.tanh(a) ** 2
        return np.eye(self.d) - g[:, None] * self.W


def make_residual_and_jacobian(layer: ImplicitTanhLayer) -> Tuple[ResidualFn, JacobianFn]:
    return layer.residual, layer.jacobian


@dataclass
class RootSolveConfig:
    method: str = "hybr"
    tol: float = 1e-10
    success_residual_tol: float = 1e-8


@dataclass
class MRBIConfig:
    sigmas: Tuple[float, ...] = (0.5, 0.1, 0.03, 0.01)
    alpha: float = 1.0
    beta: float = 0.01
    gamma: float = 0.01
    lambda_zero: float = 0.2
    lambda_relative: float = 1.0
    epsilon: float = 1e-8
    search_radius: float = 0.5
    mc_samples: int = 16
    maxiter_per_scale: int = 80
    refinement_iters: int = 80
    use_detector: bool = True
    use_newton_term: bool = True
    use_antithetic_sampling: bool = True
    newton_damping: float = 1e-6
    optimizer_ftol: float = 1e-10


@dataclass
class HybridConfig:
    tau_r: float = 1e-8
    tau_sigma: float = 0.03
    accept_factor_vs_zero: float = 1.05
    residual_trigger_scale: float = 0.1
    tie_residual_rtol: float = 0.02


@dataclass
class RootResult:
    success: bool
    z_star: Array
    residual: float
    nfev: int
    runtime_sec: float
    solver_success_flag: bool
    message: str = ""


@dataclass
class MRBICandidate:
    z_init: Array
    objective_value: float
    last_sigma: float
    n_objective_calls: int


@dataclass
class HybridSolveResult:
    success: bool
    z_star: Array
    residual: float
    used_mrbi: bool
    trigger_reason: str
    zero_result: RootResult
    final_result: RootResult
    mrbi_candidate: Optional[MRBICandidate]
    sigma_min_zero: Optional[float]
    sigma_min_det: Optional[float]
    total_runtime_sec: float


def solve_root(F: ResidualFn, J: JacobianFn, x: Array, z0: Array, cfg: Optional[RootSolveConfig] = None) -> RootResult:
    cfg = cfg or RootSolveConfig()
    x = _arr(x)
    z0 = _arr(z0)
    t0 = time.perf_counter()

    def fun(z):
        return np.asarray(F(np.asarray(z, dtype=np.float64), x), dtype=np.float64)

    def jac(z):
        return np.asarray(J(np.asarray(z, dtype=np.float64), x), dtype=np.float64)

    try:
        out = root(fun, z0, jac=jac, method=cfg.method, tol=cfg.tol)
        z = np.asarray(out.x, dtype=np.float64)
        residual = _norm(fun(z))
        solver_success = bool(getattr(out, "success", False))
        success = bool(solver_success and np.all(np.isfinite(z)) and residual <= cfg.success_residual_tol)
        return RootResult(success, z, residual, int(getattr(out, "nfev", -1)), time.perf_counter() - t0, solver_success, str(getattr(out, "message", "")))
    except Exception as exc:
        return RootResult(False, z0.copy(), float("inf"), -1, time.perf_counter() - t0, False, repr(exc))


def jacobian_health(J: JacobianFn, x: Array, z: Array) -> Tuple[float, bool]:
    try:
        Jz = np.asarray(J(_arr(z), _arr(x)), dtype=np.float64)
        if Jz.ndim != 2 or Jz.shape[0] != Jz.shape[1] or not np.all(np.isfinite(Jz)):
            return 0.0, False
        svals = np.linalg.svd(Jz, compute_uv=False)
        return float(np.min(svals)), bool(np.all(np.isfinite(svals)))
    except Exception:
        return 0.0, False


def sigma_min_jacobian(J: JacobianFn, x: Array, z: Array) -> float:
    return jacobian_health(J, x, z)[0]


class MRBIOptimizer:
    """Constructs an MRBI candidate initialization for F(z; x)=0."""

    def __init__(self, F: ResidualFn, J: JacobianFn, x: Array, dim: int, config: Optional[MRBIConfig] = None, rng: Optional[np.random.Generator] = None):
        self.F = F
        self.J = J
        self.x = _arr(x)
        self.d = int(dim)
        self.cfg = config or MRBIConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.objective_calls = 0
        self.zero = np.zeros(self.d, dtype=np.float64)
        self.zero_residual_norm = _norm(self.F(self.zero, self.x))

    def residual(self, z: Array) -> Array:
        return np.asarray(self.F(_arr(z), self.x), dtype=np.float64)

    def jacobian(self, z: Array) -> Array:
        return np.asarray(self.J(_arr(z), self.x), dtype=np.float64)

    def _probes(self, sigma: float) -> Array:
        n = int(self.cfg.mc_samples)
        if n <= 0:
            return np.zeros((0, self.d))
        if self.cfg.use_antithetic_sampling and n >= 2:
            half = (n + 1) // 2
            u = self.rng.normal(size=(half, self.d))
            return np.concatenate([u, -u], axis=0)[:n]
        return self.rng.normal(size=(n, self.d))

    def detector_ratio(self, z: Array, sigma: float) -> float:
        if not self.cfg.use_detector or self.cfg.gamma == 0.0:
            return 0.0
        probes = self._probes(sigma)
        if len(probes) == 0:
            return 0.0
        try:
            vals = np.stack([self.residual(z + sigma * u) for u in probes], axis=0)
        except Exception:
            return 1e12
        if not np.all(np.isfinite(vals)):
            return 1e12
        P = np.mean(vals, axis=0)
        V = np.mean(vals * probes, axis=0) / max(sigma, 1e-16)
        return _norm(P) / (_norm(V) + self.cfg.epsilon)

    def newton_proxy_norm(self, z: Array) -> float:
        if not self.cfg.use_newton_term or self.cfg.beta == 0.0:
            return 0.0
        Fz = self.residual(z)
        Jz = self.jacobian(z)
        try:
            A = Jz.T @ Jz + self.cfg.newton_damping * np.eye(self.d)
            step = np.linalg.solve(A, Jz.T @ Fz)
            return _norm(step)
        except Exception:
            return 1e12

    def objective(self, z: Array, sigma: float) -> float:
        self.objective_calls += 1
        z = _arr(z)
        rz = _norm(self.residual(z))
        value = self.cfg.alpha * rz
        value += self.cfg.gamma * self.detector_ratio(z, sigma)
        value += self.cfg.beta * self.newton_proxy_norm(z)
        value += self.cfg.lambda_zero * _norm(z)
        value += self.cfg.lambda_relative * max(0.0, rz - self.zero_residual_norm)
        return float(value if np.isfinite(value) else 1e12)

    def optimize(self) -> MRBICandidate:
        z = np.zeros(self.d, dtype=np.float64)
        bounds = [(-self.cfg.search_radius, self.cfg.search_radius)] * self.d
        last_sigma = float(self.cfg.sigmas[-1])
        for sigma in self.cfg.sigmas:
            res = minimize(lambda zz: self.objective(np.asarray(zz, dtype=np.float64), float(sigma)), z, method="L-BFGS-B", bounds=bounds, options={"maxiter": self.cfg.maxiter_per_scale, "ftol": self.cfg.optimizer_ftol})
            z = np.clip(np.asarray(res.x, dtype=np.float64), -self.cfg.search_radius, self.cfg.search_radius)
            last_sigma = float(sigma)
        val = self.objective(z, last_sigma)
        return MRBICandidate(z, val, last_sigma, int(self.objective_calls))


def hybrid_mrbi_solve(F: ResidualFn, J: JacobianFn, x: Array, *, dim: int, mrbi_cfg: Optional[MRBIConfig] = None, hybrid_cfg: Optional[HybridConfig] = None, root_cfg: Optional[RootSolveConfig] = None, rng: Optional[np.random.Generator] = None) -> HybridSolveResult:
    mrbi_cfg = mrbi_cfg or MRBIConfig()
    hybrid_cfg = hybrid_cfg or HybridConfig()
    root_cfg = root_cfg or RootSolveConfig()
    rng = rng if rng is not None else np.random.default_rng()
    t0 = time.perf_counter()

    x = _arr(x)
    z0 = np.zeros(int(dim), dtype=np.float64)
    F0_norm = _norm(F(z0, x))
    tau_r_eff = max(hybrid_cfg.tau_r, hybrid_cfg.residual_trigger_scale * F0_norm)

    zero = solve_root(F, J, x, z0, root_cfg)
    s_zero, ok_zero = jacobian_health(J, x, zero.z_star)

    reasons = []
    if zero.residual > tau_r_eff:
        reasons.append("residual")
    if s_zero < hybrid_cfg.tau_sigma:
        reasons.append("conditioning")
    if not zero.solver_success_flag:
        reasons.append("solver_failure")
    if not ok_zero:
        reasons.append("bad_jacobian")

    if not reasons:
        return HybridSolveResult(zero.success, zero.z_star, zero.residual, False, "none", zero, zero, None, s_zero, None, time.perf_counter() - t0)

    opt = MRBIOptimizer(F, J, x, dim, mrbi_cfg, rng)
    cand = opt.optimize()
    det = solve_root(F, J, x, cand.z_init, root_cfg)
    s_det, _ = jacobian_health(J, x, det.z_star)

    use_mrbi = bool(det.residual <= hybrid_cfg.accept_factor_vs_zero * zero.residual)
    final = det if use_mrbi else zero
    return HybridSolveResult(final.success, final.z_star, final.residual, use_mrbi, "+".join(reasons), zero, final, cand, s_zero, s_det, time.perf_counter() - t0)


def hybrid_mrbi_solve_tanh_layer(layer: ImplicitTanhLayer, x: Array, *, mrbi_cfg: Optional[MRBIConfig] = None, hybrid_cfg: Optional[HybridConfig] = None, root_cfg: Optional[RootSolveConfig] = None, rng: Optional[np.random.Generator] = None) -> HybridSolveResult:
    F, J = make_residual_and_jacobian(layer)
    return hybrid_mrbi_solve(F, J, x, dim=layer.d, mrbi_cfg=mrbi_cfg, hybrid_cfg=hybrid_cfg, root_cfg=root_cfg, rng=rng)
