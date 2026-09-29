"""Prototype: trainable equilibrium root coupled to the ORIGINAL PCA-QNN angles.

The implicit equation is exactly F(z;x)=z-tanh(Wz+Ux+b)=0.
Root selection is discrete and detached. Only strictly converged,
locally nonsingular roots receive an implicit-function gradient.
The original PCA-QNN is recovered exactly at alpha=0.

This is a development/proof-of-mechanism module, NOT a completed
classification benchmark. No tests or CLI need real dataset files.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import math
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
sys.path.insert(0,str(ROOT/"scripts"))
import main_qnn_benchmark as bench
import mrbi
import multiscale_stage_audit as audit
import run_heldout_single_stage as single

try:
    import torch
    from torch import nn
except ImportError:
    torch=None
    nn=None

# Defined before seeing any new model outcome. A strict root may
# still be too ill-conditioned to differentiate reliably.
JACOBIAN_MIN_SINGULAR=1e-5
SOLVER_MODES=("zero","mrbi")
COARSE_STAGE="smoothed_coarse"
OBJECTIVE_CAP=480
OPTIMIZER_F_CAP=5760


@dataclass(frozen=True)
class RootAudit:
    index: int
    mode: str
    zero_success: bool
    checkpoint_success: bool
    used_mrbi: bool
    strict_root_success: bool
    gradient_eligible: bool
    min_jacobian_singular: float
    residual: float
    objective_calls: int
    algorithm_f_calls: int
    algorithm_j_calls: int


def frozen_config(seed=0):
    """Original full_balanced hard regime; no QNN configuration changes."""
    return replace(audit.config_for(seed),run_qnn=True)


def solve_one(x,W,U,b,cfg,seed,index,mode):
    """Compute the same zero root in both modes; MRBI only on zero failure.

    MRBI root candidates are identical to the earlier fixed coarse
    single-stage policy. In contrast to the older forced acceptance,
    a failed checkpoint is never used in angle correction.
    """
    if mode not in SOLVER_MODES:
        raise ValueError(f"Expected one of {SOLVER_MODES}, got {mode!r}")
    x=np.asarray(x,dtype=np.float64)
    layer=mrbi.ImplicitTanhLayer(W=W,U=U,b=b)
    if x.shape!=(layer.U.shape[1],):
        raise ValueError("PCA input must match implicit input dimension")
    root_cfg=bench.make_root_cfg(cfg)
    start=time.perf_counter()
    F,J,counters=single.counted_functions(layer)
    zero,zf,zj=single.measured_root(
        F,J,counters,x,np.zeros(layer.d,dtype=np.float64),
        root_cfg,
    )
    base_wall=time.perf_counter()-start
    chosen=zero
    attempted=False
    checkpoint_success=False
    used_mrbi=False
    objective_calls=0
    if mode=="mrbi" and not zero.success:
        smin_zero,_=mrbi.jacobian_health(J,x,zero.z_star)
        stage,record,chosen_raw=single.run_arm(
            layer,x,cfg,root_cfg,zero,smin_zero,
            counters,zf,zj,base_wall,seed,index,
            COARSE_STAGE,OBJECTIVE_CAP,OPTIMIZER_F_CAP,
            return_solution=True,
        )
        attempted=bool(record["candidate_attempted"])
        checkpoint_success=bool(record["checkpoint_success"])
        objective_calls=int(record["optimizer_objective_calls"])
        if attempted and stage is not None:
            if (float(stage["sigma"])!=0.70
                    or float(stage["smooth_weight"])!=0.75
                    or int(stage["stage"])!=1
                    or int(stage["objective_calls_stage"])>OBJECTIVE_CAP
                    or int(stage["residual_evals_stage"])>OPTIMIZER_F_CAP):
                raise RuntimeError("Frozen coarse MRBI stage drifted")
        if checkpoint_success:
            if not bool(record["used_mrbi"]):
                raise RuntimeError("Successful checkpoint was not selected")
            z=np.asarray(chosen_raw,dtype=np.float64)
            residual=float(np.linalg.norm(layer.residual(z,x)))
            # Audit strict success once more, under the CURRENT learned F.
            if not(np.all(np.isfinite(z)) and
                   np.isfinite(residual) and
                   residual<=root_cfg.success_residual_tol):
                raise RuntimeError("Selected MRBI root failed original F")
            chosen=mrbi.RootResult(
                success=True,z_star=z,residual=residual,
                nfev=-1,runtime_sec=0.0,
                solver_success_flag=True,
                has_finite_z=True,has_finite_residual=True,
            )
            used_mrbi=True
        else:
            # Failed MRBI cannot change the representation.
            chosen=zero
        f_calls=int(record["algorithm_F_calls"])
        j_calls=int(record["algorithm_J_calls"])
    else:
        f_calls=int(counters["F"])
        j_calls=int(counters["J"])
    z=np.asarray(chosen.z_star,dtype=np.float64)
    if z.shape!=(layer.d,) or not np.all(np.isfinite(z)):
        raise RuntimeError("Nonfinite implicit state; fail closed")
    strict=bool(chosen.success)
    smin=float("nan")
    if strict:
        smin,ok=mrbi.jacobian_health(layer.jacobian,x,z)
        if not ok:
            smin=0.0
    gradient_eligible=bool(
        strict and np.isfinite(smin) and
        smin>=JACOBIAN_MIN_SINGULAR
    )
    audit_record=RootAudit(
        index=int(index),mode=mode,
        zero_success=bool(zero.success),
        checkpoint_success=checkpoint_success,
        used_mrbi=used_mrbi,
        strict_root_success=strict,
        gradient_eligible=gradient_eligible,
        min_jacobian_singular=smin,
        residual=float(chosen.residual),
        objective_calls=objective_calls,
        algorithm_f_calls=f_calls,
        algorithm_j_calls=j_calls,
    )
    return z,gradient_eligible,audit_record


def solve_batch(x,W,U,b,cfg,seed,indices,mode):
    """Detach only discrete candidate search, not the local root derivative."""
    x=np.asarray(x,dtype=np.float64)
    indices=tuple(int(i) for i in indices)
    if (x.ndim!=2 or len(x)!=len(indices) or
            len(set(indices))!=len(indices)):
        raise ValueError("Batch rows require unique, stable sample indices")
    out,mask,records=[],[],[]
    for value,index in zip(x,indices):
        z,good,record=solve_one(
            value,W,U,b,cfg,seed,index,mode
        )
        out.append(z)
        mask.append(good)
        records.append(record)
    return (
        np.stack(out),
        np.asarray(mask,dtype=np.bool_),
        records,
    )


if torch is not None:
    class CertifiedEquilibriumRoot(torch.autograd.Function):
        """IFT vector-Jacobian product at a fixed, certified root.

        For F=z-tanh(Wz+Ux+b), J=I-DW with
        D=diag(1-tanh(Wz+Ux+b)^2). Given dL/dz, solve
        J^T lambda=dL/dz and set q=D lambda, yielding
        dL/dW=q z^T; dL/dU=q x^T; dL/db=q; dL/dx=U^T q.

        The solver's discrete selection and MRBI optimization
        have no gradients. Noncertified roots contribute zero
        derivative to all four implicit-function inputs.
        """
        @staticmethod
        def forward(ctx,x,W,U,b,root,eligible):
            if (x.dtype!=torch.float64 or
                    W.dtype!=torch.float64 or
                    U.dtype!=torch.float64 or
                    b.dtype!=torch.float64 or
                    root.dtype!=torch.float64 or
                    eligible.dtype!=torch.bool):
                raise TypeError("IFT requires float64 inputs and boolean gate")
            if (x.ndim!=2 or root.ndim!=2 or
                    W.shape!=(root.shape[1],root.shape[1]) or
                    U.shape!=(root.shape[1],x.shape[1]) or
                    b.shape!=(root.shape[1],) or
                    eligible.shape!=(len(root),) or
                    len(x)!=len(root)):
                raise ValueError("Inconsistent batched implicit root shapes")
            if not all(torch.isfinite(t).all().item()
                       for t in (x,W,U,b,root)):
                raise ValueError("IFT cannot receive nonfinite tensors")
            # Eligibility must have been established against the
            # same current, unsmoothed F and its nonsingular Jacobian.
            with torch.no_grad():
                for i in torch.where(eligible)[0].tolist():
                    residual=root[i]-torch.tanh(
                        W@root[i]+U@x[i]+b
                    )
                    if torch.linalg.vector_norm(residual).item()>1e-8:
                        raise RuntimeError("IFT received a non-root")
                    g=1-torch.tanh(W@root[i]+U@x[i]+b).square()
                    J=torch.eye(
                        W.shape[0],device=W.device,dtype=W.dtype
                    )-g[:,None]*W
                    if (torch.linalg.svdvals(J).min().item()<
                            JACOBIAN_MIN_SINGULAR):
                        raise RuntimeError("IFT received singular root")
            ctx.save_for_backward(x,W,U,b,root,eligible)
            return root.clone()

        @staticmethod
        def backward(ctx,dout):
            x,W,U,b,root,eligible=ctx.saved_tensors
            dx=torch.zeros_like(x)
            dW=torch.zeros_like(W)
            dU=torch.zeros_like(U)
            db=torch.zeros_like(b)
            eye=torch.eye(W.shape[0],dtype=W.dtype,device=W.device)
            for i in torch.where(eligible)[0].tolist():
                z=root[i]
                a=W@z+U@x[i]+b
                diag=1-torch.tanh(a).square()
                J=eye-diag[:,None]*W
                multiplier=torch.linalg.solve(J.T,dout[i])
                q=diag*multiplier
                dW+=q[:,None]*z[None,:]
                dU+=q[:,None]*x[i][None,:]
                db+=q
                dx[i]=U.T@q
            return dx,dW,dU,db,None,None


    class TrainableImplicitAngleQNN(nn.Module):
        """Original 4→4 PCA circuit with an additive learned root angle."""

        def __init__(self,initial_layer,cfg,seed):
            super().__init__()
            if not bench.HAS_QNN:
                raise RuntimeError("Install PennyLane and Torch for the QNN")
            if (cfg.n_qubits!=4 or cfg.pca_dim!=4
                    or cfg.latent_dim!=16 or cfg.qnn_layers!=2):
                raise ValueError("Prototype preserves the original 4-qubit QNN")
            self.backbone=bench.TorchQNN(4,4,2,seed)
            self.W=nn.Parameter(
                torch.as_tensor(
                    initial_layer.W.copy(),dtype=torch.float64
                )
            )
            self.U=nn.Parameter(
                torch.as_tensor(
                    initial_layer.U.copy(),dtype=torch.float64
                )
            )
            self.b=nn.Parameter(
                torch.as_tensor(
                    initial_layer.b.copy(),dtype=torch.float64
                )
            )
            # Nonzero B at initialization allows alpha to learn from
            # step one. At alpha=0 all model outputs equal PCA-QNN.
            self.angle_from_z=nn.Linear(
                cfg.latent_dim,cfg.n_qubits,bias=False
            )
            self.alpha=nn.Parameter(torch.zeros((),dtype=torch.float32))

        def forward(self,x_pca,z_root,eligible):
            if (x_pca.ndim!=2 or x_pca.shape[1]!=4 or
                    z_root.shape!=(len(x_pca),16) or
                    eligible.shape!=(len(x_pca),)):
                raise ValueError("Expected PCA(4), latent(16), boolean gate")
            x32=x_pca.to(dtype=torch.float32)
            z32=z_root.to(dtype=torch.float32)
            base_angles=torch.tanh(self.backbone.pre(x32))*math.pi
            correction=(
                self.alpha *
                eligible.to(dtype=torch.float32)[:,None] *
                self.angle_from_z(z32)
            )
            angles=base_angles+correction
            encoded=[]
            for row in angles:
                vals=self.backbone.circuit(row,self.backbone.q_weights)
                encoded.append(torch.stack(vals).float())
            return self.backbone.post(torch.stack(encoded,dim=0))

        def solve_current_batch(self,x_pca,cfg,seed,indices,mode):
            x64=x_pca.detach().to(
                dtype=torch.float64,device="cpu"
            )
            if x64.requires_grad:
                raise RuntimeError("PCA inputs must be frozen for pilot")
            z,eligible,diagnostics=solve_batch(
                x64.numpy(),
                self.W.detach().cpu().numpy().copy(),
                self.U.detach().cpu().numpy().copy(),
                self.b.detach().cpu().numpy().copy(),
                cfg,seed,indices,mode,
            )
            zt=torch.as_tensor(
                z,dtype=torch.float64,device=self.W.device
            )
            gate=torch.as_tensor(
                eligible,dtype=torch.bool,device=self.W.device
            )
            # For unsuccessful solves this gate is False. The saved
            # root is an audit value, not a differentiable equilibrium.
            root=CertifiedEquilibriumRoot.apply(
                x_pca.to(dtype=torch.float64),
                self.W,self.U,self.b,zt,gate,
            )
            return root,gate,diagnostics


def original_root_config(seed=0):
    return bench.make_root_cfg(frozen_config(seed))
