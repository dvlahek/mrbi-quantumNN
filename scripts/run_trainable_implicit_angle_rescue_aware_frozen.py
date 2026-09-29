"""Prospective-development test of frozen rescue-aware angle correction.

Rule fixed from earlier seeds 1-3: apply a smooth 0.10-radian L2 cap
ONLY to certified MRBI-only rescue corrections at inference. All
nonrescued corrections remain unchanged. Run on new split seeds 4-6.
This is split-level development evidence on wine_binary, not an
independent external dataset or a confirmatory held-out test.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
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

PLAN="trainable_implicit_angle_rescue_aware_frozen_rho23_v1"
CONSISTENCY_WEIGHT=0.2
SPECTRAL_RADIUS=2.3
SEED=0
MODEL_SEED=1401
TRAIN_PER_CLASS=16
VALID_PER_CLASS=8
EPOCHS=8
BATCH=8
DEFAULT_OUT=ROOT/"outputs"/"trainable_implicit_angle_rescue_aware_frozen_rho23_v1"


def make_development_data():
    cfg=replace(proto.frozen_config(SEED),spectral_radius=SPECTRAL_RADIUS)
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
            "mrbi_rescued_positions":[i for i,r in enumerate(records) if r.checkpoint_success],
            "mean_absolute_angle_correction":correction,
        }


def frozen_paired_audit(model,X,y,cfg,torch):
    """Same trained parameters; change only root-initialization policy."""
    model.eval()
    x=torch.as_tensor(X,dtype=torch.float64)
    outputs={}
    with torch.no_grad():
        for mode in ("zero","mrbi"):
            z,eligible,records=model.solve_current_batch(
                x,cfg,SEED,range(10000,10000+len(x)),mode,
            )
            logits=model(x,z,eligible)
            prob=torch.softmax(logits,dim=1)
            outputs[mode]={
                "logits":logits.detach().cpu().numpy(),
                "probabilities":prob.detach().cpu().numpy(),
                "predictions":logits.argmax(dim=1).cpu().numpy(),
                "eligible":eligible.detach().cpu().numpy(),
                "records":records,
                "roots":z.detach().cpu().numpy(),
            }
    a,b=outputs["zero"],outputs["mrbi"]
    rows=[]
    for i in range(len(y)):
        za,zb=a["records"][i],b["records"][i]
        rows.append({
            "validation_position":i,"true_label":int(y[i]),
            "zero_prediction":int(a["predictions"][i]),
            "mrbi_prediction":int(b["predictions"][i]),
            "zero_correct":bool(a["predictions"][i]==y[i]),
            "mrbi_correct":bool(b["predictions"][i]==y[i]),
            "zero_strict":bool(za.strict_root_success),
            "mrbi_strict":bool(zb.strict_root_success),
            "mrbi_rescue":bool(zb.checkpoint_success),
            "zero_gradient_eligible":bool(a["eligible"][i]),
            "mrbi_gradient_eligible":bool(b["eligible"][i]),
            "zero_residual":float(za.residual),
            "mrbi_residual":float(zb.residual),
            "zero_logits":a["logits"][i].tolist(),
            "mrbi_logits":b["logits"][i].tolist(),
            "zero_probabilities":a["probabilities"][i].tolist(),
            "mrbi_probabilities":b["probabilities"][i].tolist(),
            "logit_l2_delta":float(np.linalg.norm(b["logits"][i]-a["logits"][i])),
            "root_l2_delta":float(np.linalg.norm(b["roots"][i]-a["roots"][i])),
        })
    return {
        "model_trained_with":None,
        "zero_balanced_accuracy":float(balanced_accuracy_score(y,a["predictions"])),
        "mrbi_balanced_accuracy":float(balanced_accuracy_score(y,b["predictions"])),
        "prediction_changes":sum(r["zero_prediction"]!=r["mrbi_prediction"] for r in rows),
        "mrbi_rescues":sum(r["mrbi_rescue"] for r in rows),
        "rows":rows,
    }



RESCUE_FIXED_CAP=0.10  # radians; fixed using previous development seeds 1-3
PROSPECTIVE_SPLIT_SEEDS=(4,5,6)


def frozen_rescue_aware_validation(model,X,y,cfg,torch):
    """Same learned weights and solved roots for unrestricted and selective cap.

    The rescue selection uses solver diagnostics only, never labels,
    logits, validation confidence, or validation losses.
    """
    import math
    model.eval()
    x=torch.as_tensor(X,dtype=torch.float64)
    targets=torch.as_tensor(y,dtype=torch.long)
    weights_before={name:value.detach().clone()
                    for name,value in model.state_dict().items()}
    with torch.no_grad():
        base_angles=torch.tanh(model.backbone.pre(x.float()))*math.pi
        by_policy={}
        for root_mode in ("zero","mrbi"):
            # Exactly one root search for both angle policies at these weights.
            z,eligible,records=model.solve_current_batch(
                x,cfg,SEED,range(10000,10000+len(x)),root_mode,
            )
            rescue_mask=torch.as_tensor([
                bool(record.checkpoint_success and record.used_mrbi and
                     not record.zero_success and record.strict_root_success and
                     record.gradient_eligible)
                for record in records
            ],dtype=torch.bool,device=eligible.device)
            if torch.any(rescue_mask & ~eligible):
                raise RuntimeError("Rescue mask contains an ineligible root")
            if root_mode=="zero" and bool(rescue_mask.any()):
                raise RuntimeError("Zero policy cannot contain MRBI rescues")
            if root_mode=="mrbi":
                expected=sum(int(record.checkpoint_success and record.gradient_eligible)
                             for record in records)
                if int(rescue_mask.sum())!=expected:
                    raise RuntimeError("Rescue diagnostics do not match selective mask")
            raw=model.alpha*eligible.float()[:,None]*model.angle_from_z(z.float())
            raw_norm=torch.linalg.vector_norm(raw,dim=1,keepdim=True)
            capped=raw*(RESCUE_FIXED_CAP/(RESCUE_FIXED_CAP+raw_norm))
            selective=torch.where(rescue_mask[:,None],capped,raw)
            if not torch.equal(selective[~rescue_mask],raw[~rescue_mask]):
                raise RuntimeError("Selective cap changed a nonrescued correction")
            if torch.any(torch.linalg.vector_norm(
                    selective[rescue_mask],dim=1)>RESCUE_FIXED_CAP+1e-6):
                raise RuntimeError("Selective cap exceeds fixed bound")

            mapping_results={}
            for name,correction in (
                ("unrestricted",raw),
                ("rescue_aware_fixed_cap",selective),
            ):
                encoded=[]
                for angles in base_angles+correction:
                    values=model.backbone.circuit(
                        angles,model.backbone.q_weights,
                    )
                    encoded.append(torch.stack(values).float())
                logits=model.backbone.post(torch.stack(encoded,dim=0))
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Nonfinite frozen inference logits")
                probabilities=torch.softmax(logits,dim=1)
                predicted=logits.argmax(1).cpu().numpy()
                per_sample_loss=torch.nn.functional.cross_entropy(
                    logits,targets,reduction="none",
                ).cpu().numpy()
                mapping_results[name]={
                    "balanced_accuracy":float(balanced_accuracy_score(y,predicted)),
                    "cross_entropy":float(np.mean(per_sample_loss)),
                    "confusion_matrix":confusion_matrix(
                        y,predicted,labels=[0,1],
                    ).tolist(),
                    "predictions":predicted.tolist(),
                    "per_sample_cross_entropy":per_sample_loss.tolist(),
                    "logits":logits.cpu().numpy().tolist(),
                    "probabilities":probabilities.cpu().numpy().tolist(),
                    "angle_correction_l2":torch.linalg.vector_norm(
                        correction,dim=1,
                    ).cpu().numpy().tolist(),
                }
            # Check exact baseline against the original model.forward path.
            direct=model(x,z,eligible)
            reference=torch.as_tensor(
                mapping_results["unrestricted"]["logits"],
                dtype=direct.dtype,device=direct.device,
            )
            if not torch.allclose(direct,reference,rtol=1e-5,atol=1e-6):
                raise RuntimeError("Reconstructed unrestricted forward drifted")
            if root_mode=="zero":
                a=mapping_results["unrestricted"]["logits"]
                b=mapping_results["rescue_aware_fixed_cap"]["logits"]
                if not np.array_equal(np.asarray(a),np.asarray(b)):
                    raise RuntimeError("Selective cap unexpectedly changed Zero output")
            # In the absence of a rescue, the output must be unchanged.
            for i in range(len(y)):
                if not bool(rescue_mask[i]):
                    a=mapping_results["unrestricted"]["logits"][i]
                    b=mapping_results["rescue_aware_fixed_cap"]["logits"][i]
                    if not np.array_equal(np.asarray(a),np.asarray(b)):
                        raise RuntimeError(
                            f"Nonrescued prediction changed at position {i}"
                        )
            by_policy[root_mode]={
                "strict_roots":sum(int(r.strict_root_success) for r in records),
                "gradient_eligible_roots":int(eligible.sum()),
                "mrbi_attempts":sum(int(not r.zero_success) for r in records)
                                if root_mode=="mrbi" else 0,
                "mrbi_rescues":sum(int(r.checkpoint_success) for r in records),
                "selectively_capped_positions":torch.where(
                    rescue_mask,
                )[0].cpu().numpy().tolist(),
                "root_residuals":[float(r.residual) for r in records],
                "root_checkpoint_success":[bool(r.checkpoint_success)
                                          for r in records],
                "root_gradient_eligible":[bool(r.gradient_eligible)
                                          for r in records],
                "mappings":mapping_results,
            }

    for name,value in model.state_dict().items():
        if not torch.equal(value,weights_before[name]):
            raise RuntimeError("Frozen evaluation modified model parameters")

    unrestricted=by_policy["mrbi"]["mappings"]["unrestricted"]
    selective=by_policy["mrbi"]["mappings"]["rescue_aware_fixed_cap"]
    zeros=by_policy["zero"]["mappings"]["unrestricted"]
    capped=set(by_policy["mrbi"]["selectively_capped_positions"])
    rows=[]
    for i in range(len(y)):
        rows.append({
            "validation_position":i,
            "true_label":int(y[i]),
            "rescue_eligible_for_cap":i in capped,
            "zero_prediction":zeros["predictions"][i],
            "mrbi_unrestricted_prediction":unrestricted["predictions"][i],
            "mrbi_selective_prediction":selective["predictions"][i],
            "unrestricted_loss":unrestricted["per_sample_cross_entropy"][i],
            "selective_loss":selective["per_sample_cross_entropy"][i],
            "unrestricted_correction_l2":unrestricted["angle_correction_l2"][i],
            "selective_correction_l2":selective["angle_correction_l2"][i],
            "logit_delta_l2":float(np.linalg.norm(
                np.asarray(selective["logits"][i])-
                np.asarray(unrestricted["logits"][i]),
            )),
        })
    return {
        "rule":"cap only gradient-eligible MRBI-only rescues at 0.10 rad L2",
        "selection_uses_labels":False,
        "weights":"final unrestricted MRBI model, frozen",
        "fixed_cap_radians_l2":RESCUE_FIXED_CAP,
        "frozen_identical_roots_per_angle_mapping":True,
        "by_root_policy":by_policy,
        "paired_rows":rows,
        "primary_metric_delta_balanced_accuracy":(
            selective["balanced_accuracy"]-unrestricted["balanced_accuracy"]
        ),
        "primary_comparison_prediction_changes":sum(
            a!=b for a,b in zip(
                unrestricted["predictions"],selective["predictions"],
            )
        ),
        "rescued_changes_corrected":sum(
            i in capped and unrestricted["predictions"][i]!=int(y[i])
            and selective["predictions"][i]==int(y[i])
            for i in range(len(y))
        ),
        "rescued_changes_harmed":sum(
            i in capped and unrestricted["predictions"][i]==int(y[i])
            and selective["predictions"][i]!=int(y[i])
            for i in range(len(y))
        ),
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
            rescued_training_positions=[]
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
                    rescued_training_positions.extend(ids[j] for j,r in enumerate(records) if r.checkpoint_success)
                    logits=model(xb,z,eligible)
                loss=torch.nn.functional.cross_entropy(logits,y[ids])
                if mode!="pca":
                    # Same-current-backbone reference, detached to prevent
                    # the consistency term from moving its own target.
                    with torch.no_grad():
                        reference_logits=model.backbone(xb.float())
                        reference_probs=torch.softmax(reference_logits,dim=1)
                    consistency=torch.nn.functional.kl_div(
                        torch.log_softmax(logits,dim=1),
                        reference_probs,
                        reduction="batchmean",
                    )
                    loss=loss+CONSISTENCY_WEIGHT*consistency
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
                            "mrbi_rescued_training_positions":rescued_training_positions,
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
    rescue_aware=frozen_rescue_aware_validation(
        models["mrbi"],Xva,yva,cfg,torch,
    )
    paired={}
    for trained_mode in ("zero","mrbi"):
        audit_result=frozen_paired_audit(models[trained_mode],Xva,yva,cfg,torch)
        audit_result["model_trained_with"]=trained_mode
        paired[trained_mode]=audit_result
    return {
        "frozen_rescue_aware_validation":rescue_aware,
        "fixed_rule_seed_source":[1,2,3],
        "prospective_split_seeds":list(PROSPECTIVE_SPLIT_SEEDS),
        "paired_frozen_validation_audit":paired,
        "plan":PLAN,"purpose":"New wine_binary splits for previously fixed rescue-aware inference rule; not independent external validation",
        "dataset":"wine_binary (previously explored source family)",
        "spectral_radius":SPECTRAL_RADIUS,
        "train_only_pca_consistency_weight":CONSISTENCY_WEIGHT,
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
    parser.add_argument("--seed",type=int,choices=list(PROSPECTIVE_SPLIT_SEEDS),required=True)
    parser.add_argument("--out-dir",type=Path,default=None)
    args=parser.parse_args()
    global SEED,MODEL_SEED
    SEED=args.seed
    MODEL_SEED=1401+SEED
    if args.dry_run:
        assert SEED in PROSPECTIVE_SPLIT_SEEDS
        assert RESCUE_FIXED_CAP==0.10 and TRAIN_PER_CLASS==16 and VALID_PER_CLASS==8 and EPOCHS==8 and SPECTRAL_RADIUS==2.3 and CONSISTENCY_WEIGHT==0.2
        print("TRAINABLE_ANGLE_RESCUE_AWARE_FROZEN_DRY_RUN_OK "
              "wine_binary rho=2.3 seeds=4,5,6 cap=0.10 rescue-only train=32 validation=16 epochs=8 methods=pca,zero,mrbi; "
              "no dataset loaded",flush=True)
        return
    result=run()
    result["plan"]=PLAN
    result["split_seed"]=SEED
    result["spectral_radius"]=SPECTRAL_RADIUS
    result["git_commit"]=subprocess.run(
        ["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,
        text=True,check=True,
    ).stdout.strip()
    raw=json.dumps(result,indent=2,allow_nan=False)+"\n"
    out_dir=args.out_dir if args.out_dir is not None else DEFAULT_OUT/f"seed{SEED}"
    path=out_dir.resolve()/"diagnostic_pilot.json"
    if path.exists():
        if path.read_text(encoding="utf-8")!=raw:
            raise SystemExit("Existing output has different provenance; use a new --out-dir")
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        proto.single.atomic_write(path,raw)
    print(f"TRAINABLE_ANGLE_RESCUE_AWARE_FROZEN_OK {path}",flush=True)


if __name__=="__main__":
    main()
