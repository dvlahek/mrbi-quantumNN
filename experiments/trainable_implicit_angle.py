"""Trainable implicit angle correction. Separate from frozen benchmark scripts."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import mrbi


class CertifiedEquilibrium(torch.autograd.Function):
    """Root search is detached; backward uses the implicit function theorem."""

    @staticmethod
    def forward(ctx, x, W, U, b, use_mrbi, seed, residual_tol):
        layer = mrbi.ImplicitTanhLayer(
            W.detach().cpu().double().numpy(),
            U.detach().cpu().double().numpy(),
            b.detach().cpu().double().numpy(),
        )
        xs = x.detach().cpu().double().numpy()
        roots, accepted = [], []
        cfg = mrbi.RootSolveConfig(success_residual_tol=float(residual_tol))
        for i, xi in enumerate(xs):
            initial = np.zeros(layer.d, dtype=np.float64)
            if use_mrbi:
                # The optimizer constructs only an initial point, never a gradient path.
                optimizer = mrbi.MRBIOptimizer(
                    layer.residual, layer.jacobian, xi, layer.d,
                    config=mrbi.MRBIConfig(
                        sigmas=(0.25, 0.05), mc_samples=4,
                        maxiter_per_scale=12, refinement_iters=12,
                    ),
                    rng=np.random.default_rng(int(seed) + i),
                )
                initial = optimizer.optimize().z_init
            result = mrbi.solve_root(layer.residual, layer.jacobian, xi, initial, cfg)
            valid = bool(
                result.success and result.solver_success_flag
                and result.has_finite_z and result.has_finite_residual
                and np.isfinite(result.residual)
                and result.residual <= residual_tol
                and np.all(np.isfinite(result.z_star))
            )
            roots.append(result.z_star if valid else np.zeros(layer.d))
            accepted.append(valid)
        z = torch.as_tensor(np.stack(roots), dtype=x.dtype, device=x.device)
        mask = torch.as_tensor(accepted, dtype=torch.bool, device=x.device)
        ctx.save_for_backward(x, W, U, b, z, mask)
        return z, mask

    @staticmethod
    def backward(ctx, grad_z, grad_mask):
        x, W, U, b, z, mask = ctx.saved_tensors
        gx = torch.zeros_like(x)
        gW, gU, gb = torch.zeros_like(W), torch.zeros_like(U), torch.zeros_like(b)
        if grad_z is None:
            return gx, gW, gU, gb, None, None, None
        for i in torch.where(mask)[0].tolist():
            zi, xi = z[i], x[i]
            t = torch.tanh(W @ zi + U @ xi + b)
            d = 1 - t.square()
            J = torch.eye(W.shape[0], dtype=W.dtype, device=W.device) - d[:, None] * W
            v = torch.linalg.solve(J.T, grad_z[i])
            h = d * v
            gx[i] = U.T @ h
            gW = gW + torch.outer(h, zi)
            gU = gU + torch.outer(h, xi)
            gb = gb + h
        return gx, gW, gU, gb, None, None, None


class TrainableImplicitAngleQNN(nn.Module):
    """Wrap the original TorchQNN without changing its PCA path or circuit."""

    def __init__(self, pca_qnn, latent_dim=16, seed=0, residual_tol=1e-8):
        super().__init__()
        self.qnn = pca_qnn
        self.seed = int(seed)
        self.residual_tol = float(residual_tol)
        dx = self.qnn.pre.in_features
        d = int(latent_dim)
        self.W = nn.Parameter(0.05 * torch.eye(d))
        self.U = nn.Parameter(0.05 * torch.randn(d, dx))
        self.b = nn.Parameter(torch.zeros(d))
        self.B = nn.Linear(d, self.qnn.n_qubits, bias=False)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x, use_mrbi=False, return_mask=False):
        # alpha=0 is exactly the original QNN, including floating-point operations.
        if not self.alpha.requires_grad and self.alpha.item() == 0:
            out = self.qnn(x)
            return (out, torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)) if return_mask else out
        z, mask = CertifiedEquilibrium.apply(
            x, self.W, self.U, self.b, use_mrbi, self.seed, self.residual_tol
        )
        angles = torch.tanh(
            self.qnn.pre(x) + self.alpha * self.B(z) * mask[:, None]
        ) * np.pi
        outs = []
        for i in range(x.shape[0]):
            outs.append(torch.stack(self.qnn.circuit(angles[i], self.qnn.q_weights)).float())
        out = self.qnn.post(torch.stack(outs))
        return (out, mask) if return_mask else out
