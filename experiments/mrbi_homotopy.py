"""Opt-in smoothed-residual homotopy for the MRBI-QNN numerical pilot.

The production MRBIOptimizer and historical full campaign are unchanged.
Both pilot arms use the same homotopy weights and common antithetic probes.
The only arm difference is the Gaussian scale presented to each stage.
"""
from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np
from scipy.optimize import minimize

import mrbi


class StageBudgetReached(Exception):
    """Internal signal: stop at the previous *completed* L-BFGS-B iterate."""


class SmoothedHomotopyOptimizer(mrbi.MRBIOptimizer):
    """MRBI with an early smoothed residual and original final-scale target.

    Early objective:
      alpha * [(1-w) ||F(z)|| + w ||mean_U F(z + sigma U)||]
      + the unchanged Newton, detector and regularization terms.
    The last scale and refinement have w=0.  The same antithetic U is used
    at every scale and in both paired arms, not independently sampled U_sigma.

    Stage caps are identical across arms, but actual consumed evaluations
    may differ because L-BFGS-B can converge early. Both realized objective
    calls and primitive residual/Jacobian evaluations are recorded.
    """

    WEIGHTS = (0.75, 0.50, 0.25, 0.0, 0.0)

    def __init__(
        self, F, J, x, dim, config=None, rng=None,
        *, objective_cap_per_stage=480, residual_cap_per_stage=5760,
        weights=WEIGHTS,
    ):
        super().__init__(F, J, x, dim, config=config, rng=rng)
        if (not self.cfg.use_continuation or not self.cfg.use_detector
                or not self.cfg.use_fixed_probes_per_scale or not self.cfg.use_refinement
                or self.cfg.n_starts != 1 or len(self.cfg.sigmas) != 4
                or self.cfg.mc_samples != 10):
            raise ValueError("The pilot requires fixed full_balanced with four scales and refinement.")
        if (objective_cap_per_stage < self.d + 2
                or residual_cap_per_stage < (self.d + 2) * (self.cfg.mc_samples + 2)):
            raise ValueError("Per-stage caps cannot fit one numerical gradient.")
        self.weights = tuple(float(w) for w in weights)
        if (len(self.weights) != 5 or self.weights[-2:] != (0.0, 0.0)
                or any(not np.isfinite(w) or w < 0.0 or w > 1.0
                       for w in self.weights)):
            raise ValueError("Expected five finite weights in [0,1], with original final-scale objective.")
        self.objective_cap_per_stage = int(objective_cap_per_stage)
        self.residual_cap_per_stage = int(residual_cap_per_stage)
        self._shared_U = None
        self._active_weight = 0.0
        self._stage_index = 0
        self._in_objective = False
        self._residual_evals = 0
        self._jacobian_evals = 0
        self.stage_records = []

    def _get_probes(self, sigma):
        """One common antithetic sample; changing sigma changes only sigma*U."""
        if self._shared_U is None:
            self._shared_U = self._make_probes(float(sigma))
        self._probe_cache[float(sigma)] = self._shared_U
        return self._shared_U

    def residual(self, z):
        if self._in_objective:
            self._residual_evals += 1
        return super().residual(z)

    def jacobian(self, z):
        if self._in_objective:
            self._jacobian_evals += 1
        return super().jacobian(z)

    def objective(self, z, sigma, *, detector_override=None):
        """Use the smoothed residual only at nonzero homotopy weights."""
        self.objective_calls += 1
        z = np.asarray(z, dtype=np.float64)
        self._in_objective = True
        try:
            Fz = self.residual(z)
            Fnorm = mrbi._safe_norm(Fz)
            detector = (self.cfg.use_detector if detector_override is None
                        else bool(detector_override))
            needs_probes = (self._active_weight > 0.0
                            or (detector and self.cfg.gamma != 0.0))
            if needs_probes:
                U = self._get_probes(float(sigma))
                try:
                    shifted = np.stack(
                        [self.residual(z + float(sigma) * u) for u in U], axis=0
                    )
                except Exception:
                    return 1e12
                if not np.all(np.isfinite(shifted)):
                    return 1e12
                Phat = np.mean(shifted, axis=0)
                smooth_norm = mrbi._safe_norm(Phat)
                if detector and self.cfg.gamma != 0.0:
                    What = np.mean(shifted * U, axis=0)
                    ratio = smooth_norm / (
                        mrbi._safe_norm(What) / max(float(sigma), 1e-16)
                        + float(self.cfg.epsilon)
                    )
                else:
                    ratio = 0.0
            else:
                smooth_norm, ratio = Fnorm, 0.0
            weight = self._active_weight
            residual_term = self.cfg.alpha * (
                (1.0 - weight) * Fnorm + weight * smooth_norm
            )
            if self.cfg.use_newton_term and self.cfg.beta != 0.0:
                newton_term = self.cfg.beta * mrbi._safe_norm(self.newton_proxy(z))
            else:
                newton_term = 0.0
            value = (
                residual_term + newton_term + self.cfg.gamma * ratio
                + self.cfg.lambda_zero * mrbi._safe_norm(z)
                + self.cfg.lambda_relative * max(0.0, Fnorm - self.zero_residual_norm)
            )
            return float(value if np.isfinite(value) else 1e12)
        finally:
            self._in_objective = False

    def _optimize_single_scale(self, z0, sigma, maxiter, *, detector_override=None):
        stage = self._stage_index
        if stage >= len(self.weights):
            raise RuntimeError("Unexpected extra homotopy stage.")
        weight = self.weights[stage]
        self._active_weight = weight
        self._stage_index += 1
        self._get_probes(float(sigma))
        z0 = np.clip(np.asarray(z0, dtype=np.float64),
                     -self.cfg.search_radius, self.cfg.search_radius)
        before_calls = self.objective_calls
        before_residual = self._residual_evals
        before_jacobian = self._jacobian_evals
        last_complete = z0.copy()
        budget_hit = False

        def cost(z):
            # The profile always evaluates 2 unshifted residuals (including
            # the Newton proxy) plus K shifted residuals per objective call.
            next_residuals = 2 + int(self.cfg.mc_samples)
            if (self.objective_calls - before_calls >= self.objective_cap_per_stage
                    or self._residual_evals - before_residual
                    + next_residuals > self.residual_cap_per_stage):
                raise StageBudgetReached()
            return self.objective(z, sigma, detector_override=detector_override)

        def save_completed(xk):
            nonlocal last_complete
            last_complete = np.asarray(xk, dtype=np.float64).copy()

        try:
            result = minimize(
                cost, z0, method="L-BFGS-B",
                bounds=[(-self.cfg.search_radius, self.cfg.search_radius)] * self.d,
                callback=save_completed,
                options={
                    "maxiter": int(maxiter),
                    "ftol": float(self.cfg.optimizer_ftol),
                },
            )
            z = np.asarray(result.x, dtype=np.float64)
            optimizer_success = bool(result.success)
            optimizer_nit = int(getattr(result, "nit", -1))
        except StageBudgetReached:
            z = last_complete
            budget_hit = True
            optimizer_success, optimizer_nit = False, -1
        z = np.clip(z, -self.cfg.search_radius, self.cfg.search_radius)

        # Diagnostics are outside the optimization budget and do not feed
        # back into the next stage. The next step uses only z.
        Fz = self.residual(z)
        shifted = np.stack(
            [self.residual(z + float(sigma)*u) for u in self._get_probes(float(sigma))],
            axis=0,
        )
        smooth = float(np.linalg.norm(np.mean(shifted, axis=0)))
        residual = float(np.linalg.norm(Fz))
        smin, _ = mrbi.jacobian_health(self.J, self.x, z)
        self.stage_records.append({
            "stage": stage + 1,
            "sigma": float(sigma),
            "smooth_weight": weight,
            "residual_norm": residual,
            "smoothed_residual_norm": smooth,
            "weighted_residual_term": self.cfg.alpha * (
                (1.0-weight)*residual + weight*smooth
            ),
            "candidate_step_norm": float(np.linalg.norm(z-z0)),
            "candidate_json": __import__("json").dumps(z.tolist()),
            "jacobian_sigma_min": smin,
            "objective_calls_stage": self.objective_calls-before_calls,
            "residual_evals_stage": self._residual_evals-before_residual,
            "jacobian_evals_stage": self._jacobian_evals-before_jacobian,
            "optimizer_success": optimizer_success,
            "optimizer_nit": optimizer_nit,
            "budget_hit": budget_hit,
            "probe_sha256": hashlib.sha256(
                self._shared_U.tobytes()
            ).hexdigest(),
        })
        return z

    def optimize(self):
        candidate = super().optimize()
        if len(self.stage_records) != len(self.weights):
            raise RuntimeError("Incomplete smoothed homotopy stages")
        if self._active_weight != 0.0:
            raise RuntimeError("The final objective is not the original residual")
        return candidate

    @property
    def optimizer_residual_evaluations(self):
        """F calls in optimization, including the one final candidate scoring."""
        return self._residual_evals

    @property
    def optimizer_jacobian_evaluations(self):
        return self._jacobian_evals
