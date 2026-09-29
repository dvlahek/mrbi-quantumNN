"""Predeclared eight-epoch DEVELOPMENT-only trainable implicit-angle diagnostic.

Uses a new stratified development split from a previously explored source family.
No external datasets, frozen campaign outputs, or held-out test sets are read.
This is a mechanism check, not independent classification validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
sys.path.insert(0,str(ROOT/"scripts"))
import main_qnn_benchmark as bench
import trainable_implicit_angle as proto

PLAN="trainable_implicit_angle_diagnostic_wine_seed0_v1"
SEED=0
MODEL_SEED=1401
TRAIN_PER_CLASS=16
VALID_PER_CLASS=8
EPOCHS=8
BATCH=8
DEFAULT_OUT=ROOT/"outputs"/"trainable_implicit_angle_diagnostic_v1"


def make_development_data():
    cfg=proto.frozen_config(SEED)
    X,y=bench.load_dataset("wine_binary",cfg)
    # Deterministic, balanced development subset. Nothing is fitted on validation.
    train_ids,valid_ids=train_test_split(
        np.arange(len(y)),train_size=TRAIN_PER_CLASS*2,
        test_size=VALID_PER_CLASS*2,stratify=y,random_state=SEED,
    )
    Xtr,Xva=bench.preprocess_to_pca(
        X[train_ids],X[valid_ids],cfg.pca_dim,SEED,
    )
    return cfg,Xtr,Xva,np.asarray(y)[train_ids],np.asarray(y)[valid_ids]


def score(model,mode,X,y,cfg,torch):
    model.eval()
    with torch.no_grad():
        x=torch.as_tensor(X,dtype=torch.float64)
        if mode=="pca":
            logits=model(x.float())
            records=[]
            correction=0.0
        else:
            z,eligible,records=model.solve_current_batch(
                x,cfg,SEED,range(10000,10000+len(x)),mode,
            )
            logits=model(x,z,eligible)
            correction=float((
                model.alpha*eligible.float()[:,None]*
                model.angle_from_z(z.float())
            ).abs().mean().item())
        predicted=logits.argmax(dim=1).cpu().numpy()
        loss=float(torch.nn.functional.cross_entropy(
            logits,torch.as_tensor(y,dtype=torch.long),
        ).item())
        return {
            "balanced_accuracy":float(balanced_accuracy_score(y,predicted)),
            "predicted_class_counts":np.bincount(predicted,minlength=2).tolist(),
            "confusion_matrix":confusion_matrix(y,predicted,labels=[0,1]).tolist(),
            "cross_entropy":loss,
            "strict_roots":sum(int(r.strict_root_success) for r in records),
            "gradient_eligible_roots":sum(int(r.gradient_eligible) for r in records),
            "mrbi_attempts":sum(int(r.mode=="mrbi" and not r.zero_success) for r in records),
            "mrbi_rescues":sum(int(r.checkpoint_success) for r in records),
            "mean_absolute_angle_correction":correction,
        }


def run():
    if proto.torch is None or not bench.HAS_QNN:
        raise SystemExit("Development pilot requires torch and PennyLane in the full Ryzen venv")
    torch=proto.torch
    cfg,Xtr,Xva,ytr,yva=make_development_data()
    layer=bench.make_random_implicit_layer(
        dx=4,d=16,cfg=cfg,
        seed=SEED+1000+cfg.latent_dim+int(100*cfg.spectral_radius),
    )
    models={
        "pca":bench.TorchQNN(4,4,2,MODEL_SEED),
        "zero":proto.TrainableImplicitAngleQNN(layer,cfg,MODEL_SEED),
        "mrbi":proto.TrainableImplicitAngleQNN(layer,cfg,MODEL_SEED),
    }
    for name,parameter in models["zero"].state_dict().items():
        if not torch.equal(parameter,models["mrbi"].state_dict()[name]):
            raise RuntimeError("Zero and MRBI initialization mismatch")
    for name,parameter in models["pca"].state_dict().items():
        if not torch.equal(parameter,models["zero"].backbone.state_dict()[name]):
            raise RuntimeError("PCA and implicit QNN initialization mismatch")
    x=torch.as_tensor(Xtr,dtype=torch.float64)
    y=torch.as_tensor(ytr,dtype=torch.long)
    results={}
    for mode,model in models.items():
        model.train()
        if mode=="pca":
            opt=torch.optim.Adam(model.parameters(),lr=0.01)
            initial=None
        else:
            opt=torch.optim.Adam([
                {"params":[model.W,model.U,model.b],"lr":1e-4},
                {"params":list(model.backbone.parameters())+
                 list(model.angle_from_z.parameters())+[model.alpha],"lr":0.01},
            ])
            initial={key:getattr(model,key).detach().clone()
                     for key in ("W","U","b")}
        history=[]
        for epoch in range(EPOCHS):
            rng=np.random.default_rng(SEED+epoch)
            order=rng.permutation(len(x))
            counts={"strict_roots":0,"eligible_roots":0,
                    "mrbi_attempts":0,"mrbi_rescues":0}
            gradient_norms={"W":[],"U":[],"b":[],"alpha":[]}
            batch_losses=[]
            for start in range(0,len(order),BATCH):
                ids=order[start:start+BATCH].tolist()
                xb=x[ids]
                if mode=="pca":
                    logits=model(xb.float())
                else:
                    z,eligible,records=model.solve_current_batch(
                        xb,cfg,SEED,ids,mode,
                    )
                    counts["strict_roots"]+=sum(int(r.strict_root_success) for r in records)
                    counts["eligible_roots"]+=int(eligible.sum().item())
                    counts["mrbi_attempts"]+=sum(int(not r.zero_success) for r in records) if mode=="mrbi" else 0
                    counts["mrbi_rescues"]+=sum(int(r.checkpoint_success) for r in records)
                    logits=model(xb,z,eligible)
                loss=torch.nn.functional.cross_entropy(logits,y[ids])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                batch_losses.append(float(loss.detach()))
                if mode!="pca":
                    for key in gradient_norms:
                        grad=getattr(model,key).grad
                        gradient_norms[key].append(
                            None if grad is None else float(torch.linalg.vector_norm(grad).item())
                        )
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step()
            # The validation split is never used for optimizer updates or selection.
            validation=score(model,mode,Xva,yva,cfg,torch)
            training=score(model,mode,Xtr,ytr,cfg,torch)
            history.append({"epoch":epoch+1,"training_root_audit":counts,
                            "training_batch_loss_mean":float(np.mean(batch_losses)),
                            "training":training,"validation":validation,
                            "implicit_gradient_norms_before_clipping":gradient_norms})
            print(f"[{mode}] development epoch={epoch+1}/{EPOCHS} "
                  f"val_BA={validation['balanced_accuracy']:.4f} "
                  f"strict={validation['strict_roots']}/{len(yva)}",flush=True)
        result={"epochs":history}
        if mode!="pca":
            result["final_alpha"]=float(model.alpha.detach())
            result["implicit_parameter_delta"]={
                key:float(torch.linalg.vector_norm(
                    getattr(model,key).detach()-value,
                ).item()) for key,value in initial.items()
            }
            # Same trained backbone with correction disabled, to measure
            # contribution of implicit angles without retraining.
            alpha=model.alpha.detach().clone()
            with torch.no_grad():
                model.alpha.zero_()
            result["validation_alpha_zero_ablation"]=score(
                model,mode,Xva,yva,cfg,torch,
            )
            with torch.no_grad():
                model.alpha.copy_(alpha)
        results[mode]=result
    return {
        "plan":PLAN,"purpose":"Eight-epoch development diagnostic; not external validation",
        "dataset":"wine_binary (previously explored source family)",
        "split":"32 train / 16 validation, stratified; PCA fit only on train",
        "seeds":{"data":SEED,"model":MODEL_SEED},
        "epochs":EPOCHS,"batch_size":BATCH,
        "methods":["pca","zero","mrbi"],
        "validation_use":"reporting only; no checkpoint selection or tuning",
        "results":results,"scientific_result":None,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run",action="store_true")
    mode.add_argument("--pilot",action="store_true")
    parser.add_argument("--out-dir",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    if args.dry_run:
        assert TRAIN_PER_CLASS==16 and VALID_PER_CLASS==8 and EPOCHS==8
        print("TRAINABLE_ANGLE_DIAGNOSTIC_DRY_RUN_OK "
              "wine_binary train=32 validation=16 epochs=8 methods=pca,zero,mrbi; "
              "no dataset loaded",flush=True)
        return
    result=run()
    result["git_commit"]=subprocess.run(
        ["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,
        text=True,check=True,
    ).stdout.strip()
    raw=json.dumps(result,indent=2,allow_nan=False)+"\n"
    path=args.out_dir.resolve()/"diagnostic_pilot.json"
    if path.exists():
        if path.read_text(encoding="utf-8")!=raw:
            raise SystemExit("Existing output has different provenance; use a new --out-dir")
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        proto.single.atomic_write(path,raw)
    print(f"TRAINABLE_ANGLE_DEVELOPMENT_OK {path}",flush=True)


if __name__=="__main__":
    main()
