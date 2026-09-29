"""Tiny SYNTHETIC coupled QNN/IFT pilot: not a classification benchmark.

Usage:
  python scripts/run_trainable_implicit_angle_pilot.py --dry-run
  python scripts/test_trainable_implicit_angle.py
  python scripts/run_trainable_implicit_angle_pilot.py --pilot

This intentionally does NOT load or score any previous held-out dataset.
It runs just two paired optimizer steps on eight deterministic synthetic
examples to check that learned W,U,b and QNN angles can train jointly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
sys.path.insert(0,str(ROOT/"experiments"))
import main_qnn_benchmark as bench
import trainable_implicit_angle as proto

SEED=0
DATA_SEED=8917
MODEL_SEED=6177
PLAN="synthetic_trainable_implicit_angle_two_step_v1"
DEFAULT_OUT=ROOT/"outputs"/"trainable_implicit_angle_prototype_v1"


def run_pilot():
    if proto.torch is None or not bench.HAS_QNN:
        raise SystemExit(
            "Pilot requires Torch and PennyLane. "
            "Activate ~/mrbi-qnn-full-ryzen/.venv first."
        )
    torch=proto.torch
    cfg=proto.frozen_config(SEED)
    if (cfg.n_qubits!=4 or cfg.qnn_layers!=2 or cfg.latent_dim!=16
            or cfg.spectral_radius!=2.0 or cfg.qnn_epochs!=60):
        raise RuntimeError("Prototype drifted from the original QNN")
    rng=np.random.default_rng(DATA_SEED)
    x_np=np.tanh(rng.normal(0.0,0.3,size=(8,4)))
    target_np=((x_np[:,0]-0.4*x_np[:,1]+0.25*x_np[:,2])>0).astype(int)
    if set(target_np)!={0,1}:
        raise RuntimeError("Synthetic pilot labels are accidentally one-class")
    layer=bench.make_random_implicit_layer(
        dx=4,d=16,cfg=cfg,
        seed=SEED+1000+cfg.latent_dim+
             int(100*cfg.spectral_radius),
    )
    x=torch.as_tensor(x_np,dtype=torch.float64)
    y=torch.as_tensor(target_np,dtype=torch.long)
    models={
        mode:proto.TrainableImplicitAngleQNN(
            layer,cfg,MODEL_SEED
        ) for mode in proto.SOLVER_MODES
    }
    originals=list(models.values())
    for name,value in originals[0].state_dict().items():
        if not torch.equal(value,originals[1].state_dict()[name]):
            raise RuntimeError("Zero/MRBI learned parameters not paired at start")
    outcome={}
    for mode,model in models.items():
        implicit=(model.W,model.U,model.b)
        qparams=list(model.backbone.parameters())+list(
            model.angle_from_z.parameters()
        )+[model.alpha]
        optimizer=torch.optim.Adam([
            {"params":implicit,"lr":1e-4},
            {"params":qparams,"lr":0.01},
        ])
        initial_implicit={
            name:getattr(model,name).detach().clone()
            for name in ("W","U","b")
        }
        audit=[]
        for step,indices in enumerate(((0,1,2,3),(4,5,6,7))):
            idx=list(indices)
            xb=x[idx]
            yb=y[idx]
            z,eligible,diagnostics=model.solve_current_batch(
                xb,cfg,SEED,idx,mode
            )
            if step==0:
                with torch.no_grad():
                    plain=model.backbone(xb.float())
                    adapted=model(xb,z,eligible)
                if not torch.equal(plain,adapted):
                    raise RuntimeError(
                        "Alpha=0 changed original QNN predictions"
                    )
            optimizer.zero_grad(set_to_none=True)
            logits=model(xb,z,eligible)
            loss=torch.nn.functional.cross_entropy(logits,yb)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite QNN training objective")
            loss.backward()
            for name,param in (
                ("W",model.W),("U",model.U),("b",model.b),
            ):
                if param.grad is not None and not torch.isfinite(
                    param.grad
                ).all():
                    raise RuntimeError(f"Nonfinite IFT gradient for {name}")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),max_norm=1.0
            )
            optimizer.step()
            if not all(
                torch.isfinite(param).all().item()
                for param in model.parameters()
            ):
                raise RuntimeError("Nonfinite parameter after gradient step")
            audit.append({
                "step":step,
                "strict_roots":sum(
                    int(r.strict_root_success) for r in diagnostics
                ),
                "gradient_eligible_roots":int(eligible.sum().item()),
                "mrbi_checkpoint_rescues":sum(
                    int(r.checkpoint_success) for r in diagnostics
                ),
                "mrbi_attempts":sum(
                    int(r.mode=="mrbi" and not r.zero_success)
                    for r in diagnostics
                ),
                "alpha_after":float(model.alpha.detach()),
                "loss_before_step":float(loss.detach()),
            })
            print(
                f"[{mode}] synthetic step={step+1}/2 "
                f"strict={audit[-1]['strict_roots']}/4 "
                f"IFT={audit[-1]['gradient_eligible_roots']}/4 "
                f"mrbi_rescue={audit[-1]['mrbi_checkpoint_rescues']}",
                flush=True,
            )
        deltas={
            name:float(torch.linalg.vector_norm(
                getattr(model,name).detach()-start
            )) for name,start in initial_implicit.items()
        }
        outcome[mode]={
            "steps":audit,
            "implicit_parameter_delta":deltas,
            "final_alpha":float(model.alpha.detach()),
        }
    return {
        "plan":PLAN,
        "purpose":"Synthetic gradient and QNN integration check only; NOT classification validation",
        "data":"eight deterministic synthetic PCA-like vectors; labels are a toy deterministic rule",
        "architecture":"unchanged four-qubit two-layer PCA-QNN with additive certified implicit angle",
        "frozen_original_equilibrium":"z=tanh(Wz+Ux+b), spectral radius 2.0 at initialization, original root solver and coarse stage",
        "gradients":"IFT only through strict roots with sigma_min(J)>=1e-5; MRBI and root selection detached",
        "alpha_initial":0.0,
        "same_initial_parameters":True,
        "results":outcome,
        "scientific_result":None,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    choice=p.add_mutually_exclusive_group(required=True)
    choice.add_argument("--dry-run",action="store_true")
    choice.add_argument("--pilot",action="store_true")
    p.add_argument("--out-dir",type=Path,default=DEFAULT_OUT)
    args=p.parse_args()
    if args.dry_run:
        print(
            "TRAINABLE_ANGLE_DRY_RUN_OK toy=8 "
            "paired_modes=zero,mrbi steps=2; no data or QNN loaded.",
            flush=True,
        )
        return
    sha=subprocess.run(
        ["git","rev-parse","HEAD"],cwd=ROOT,
        capture_output=True,text=True,check=True,
    ).stdout.strip()
    result=run_pilot()
    result["git_commit"]=sha
    raw=(json.dumps(result,indent=2,allow_nan=False)+"\n")
    path=args.out_dir.resolve()/"synthetic_pilot.json"
    if path.is_file():
        if path.read_text(encoding="utf-8")!=raw:
            raise SystemExit(
                "Pilot output already exists with different provenance; "
                "use a new --out-dir."
            )
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        proto.single.atomic_write(path,raw)
    print(f"SYNTHETIC_PILOT_OK {path}",flush=True)


if __name__=="__main__":
    main()
