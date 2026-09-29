"""Numerical and QNN contract checks for trainable implicit-angle MRBI.

Pure-Numpy source/solver tests run in the ordinary CI environment.
PyTorch implicit-function finite-difference tests run if Torch is
installed. The exact PCA-QNN identity test also requires PennyLane.
No data, QNN accuracy experiment or external test set is accessed.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.optimize import root as scipy_root

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
sys.path.insert(0,str(ROOT/"experiments"))
import main_qnn_benchmark as bench
import mrbi
import trainable_implicit_angle as prototype


def test_analytic_root_jacobian():
    W=np.array([[0.20,0.05],[-0.08,0.16]],dtype=np.float64)
    U=np.array([[0.30,-0.13],[0.21,0.24]],dtype=np.float64)
    b=np.array([0.04,-0.06],dtype=np.float64)
    x=np.array([0.20,-0.37],dtype=np.float64)
    layer=mrbi.ImplicitTanhLayer(W=W,U=U,b=b)
    solve=mrbi.solve_root(
        layer.residual,layer.jacobian,x,
        np.zeros(2,dtype=np.float64),
        cfg=mrbi.RootSolveConfig(
            method="hybr",tol=1e-12,
            success_residual_tol=1e-10,
        ),
    )
    assert solve.success and solve.residual<1e-10
    J=layer.jacobian(solve.z_star,x)
    eps=1e-6
    Jfd=np.column_stack([
        (layer.residual(
            solve.z_star+np.eye(2)[j]*eps,x
        )-layer.residual(
            solve.z_star-np.eye(2)[j]*eps,x
        ))/(2*eps)
        for j in range(2)
    ])
    assert np.max(np.abs(J-Jfd))<1e-8
    print("Analytic J=I-DW agrees with the original F finite differences.")
    return W,U,b,x,solve.z_star


def test_exact_zero_vs_mrbi_when_zero_succeeds():
    cfg=prototype.frozen_config(0)
    W=np.diag([0.15]*16)
    U=np.zeros((16,4),dtype=np.float64)
    b=np.full(16,0.1)
    x=np.array([0.1,-0.2,0.3,-0.4],dtype=np.float64)
    z0,g0,r0=prototype.solve_one(
        x,W,U,b,cfg,0,17,"zero"
    )
    z1,g1,r1=prototype.solve_one(
        x,W,U,b,cfg,0,17,"mrbi"
    )
    assert np.array_equal(z0,z1)
    assert g0 and g1
    assert r0.zero_success and r1.zero_success
    assert not r1.used_mrbi and not r1.checkpoint_success
    assert r1.objective_calls==0
    print("MRBI mode is the identical zero solve when zero converges.")


def test_ift_gradients():
    if prototype.torch is None:
        print("SKIP IFT gradcheck: Torch not installed in this environment.")
        return
    torch=prototype.torch
    W,U,b,x,z=test_analytic_root_jacobian()
    tw=torch.tensor(W,dtype=torch.float64,requires_grad=True)
    tu=torch.tensor(U,dtype=torch.float64,requires_grad=True)
    tb=torch.tensor(b,dtype=torch.float64,requires_grad=True)
    tx=torch.tensor(
        x[None,:],dtype=torch.float64,requires_grad=True
    )
    zr=torch.tensor(z[None,:],dtype=torch.float64)
    weight=torch.tensor(
        [0.77,-1.22],dtype=torch.float64
    )
    out=prototype.CertifiedEquilibriumRoot.apply(
        tx,tw,tu,tb,zr,
        torch.tensor([True],dtype=torch.bool),
    )
    (out[0]*weight).sum().backward()

    def loss_for(W_,U_,b_,x_):
        layer=mrbi.ImplicitTanhLayer(W=W_,U=U_,b=b_)
        sol=mrbi.solve_root(
            layer.residual,layer.jacobian,x_,
            z.copy(),cfg=mrbi.RootSolveConfig(
                method="hybr",tol=1e-12,
                success_residual_tol=1e-10,
            )
        )
        assert sol.success
        return float(weight.detach().numpy()@sol.z_star)

    eps=1e-6
    checks=(
        ("W",W,tw.grad,0),
        ("U",U,tu.grad,1),
        ("b",b,tb.grad,2),
        ("x",x,tx.grad[0],3),
    )
    for name,source,actual,index in checks:
        derivative=np.zeros_like(source)
        for k in np.ndindex(source.shape):
            plus=source.copy()
            minus=source.copy()
            plus[k]+=eps
            minus[k]-=eps
            state=[W,U,b,x]
            hi=state.copy()
            lo=state.copy()
            hi[index]=plus
            lo[index]=minus
            derivative[k]=(loss_for(*hi)-loss_for(*lo))/(2*eps)
        error=np.max(np.abs(
            actual.detach().numpy()-derivative
        ))
        assert error<2e-6,(name,error)
    # An unsuccessful root must not silently get an IFT gradient.
    tw.grad=None
    tu.grad=None
    tb.grad=None
    tx.grad=None
    stopped=prototype.CertifiedEquilibriumRoot.apply(
        tx,tw,tu,tb,zr,
        torch.tensor([False],dtype=torch.bool),
    )
    (stopped[0]*weight).sum().backward()
    assert torch.count_nonzero(tx.grad)==0
    assert torch.count_nonzero(tw.grad)==0
    assert torch.count_nonzero(tu.grad)==0
    assert torch.count_nonzero(tb.grad)==0
    print("IFT gradients of W, U, b and x match finite differences; failed roots stop gradients.")


def test_original_qnn_exact_at_zero_correction():
    if prototype.torch is None or not bench.HAS_QNN:
        print("SKIP PCA-QNN circuit identity: requires Torch and PennyLane.")
        return
    torch=prototype.torch
    cfg=prototype.frozen_config(0)
    layer=bench.make_random_implicit_layer(
        dx=4,d=16,cfg=cfg,seed=1600
    )
    model=prototype.TrainableImplicitAngleQNN(
        layer,cfg,seed=1401
    )
    x=torch.tensor([
        [0.15,-0.24,0.38,0.11],
        [-0.12,0.33,-0.20,0.25],
    ],dtype=torch.float32)
    z=torch.tensor(
        np.full((2,16),0.27),dtype=torch.float64
    )
    gate=torch.tensor([True,False])
    original=model.backbone(x)
    extended=model(x,z,gate)
    assert torch.equal(original,extended),(
        torch.max(torch.abs(original-extended)).item()
    )
    assert model.alpha.item()==0.0
    # Two copies of this model also share the identical 4→4
    # PCA projection and quantum circuit parameter initialization.
    twin=prototype.TrainableImplicitAngleQNN(
        layer,cfg,seed=1401
    )
    for key,value in model.state_dict().items():
        assert torch.equal(value,twin.state_dict()[key]),key
    print("Alpha=0 is bitwise the original PCA-QNN; paired initial weights match.")


def main():
    test_analytic_root_jacobian()
    test_exact_zero_vs_mrbi_when_zero_succeeds()
    test_ift_gradients()
    test_original_qnn_exact_at_zero_correction()
    print("TRAINABLE_IMPLICIT_ANGLE_PROTOTYPE_TESTS_OK")


if __name__=="__main__":
    main()
