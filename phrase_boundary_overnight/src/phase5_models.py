from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .models import Normalizer, seed_everything
from .phase2_models import choose_single_threshold, evaluate_single_performance, nms_probabilities
from .phase3_models import MultiModalWindowDataset, SmallBiGRU, fit_train_normalizer
from .phase4_models import ExperimentDeadline, soft_target


def target_data(data: dict[str, dict[str, np.ndarray]], target_kind: str, radius: int = 1, neighbor: float = 0.5):
    if target_kind not in {"hard", "soft"}: raise ValueError(target_kind)
    output={}
    for piece_id,item in data.items():
        copied=dict(item);copied["labels"]=item["labels"].copy() if target_kind=="hard" else soft_target(item["labels"],item["label_mask"],radius,neighbor);output[piece_id]=copied
    return output


def loss_balance(data: dict[str, dict[str, np.ndarray]], weight_kind: str, cap: float = 10.0):
    positive=sum(float(x["labels"][x["label_mask"]>0].sum()) for x in data.values());total=sum(float((x["label_mask"]>0).sum()) for x in data.values());negative=total-positive;ratio=negative/max(positive,1e-12)
    if weight_kind=="original_capped":weight=min(ratio,cap)
    elif weight_kind=="mass_balanced":weight=ratio
    else:raise ValueError(weight_kind)
    return {"positive_mass":positive,"negative_mass":negative,"raw_ratio":ratio,"pos_weight":weight,"cap":cap,"effective_positive_coefficient":positive*weight,"effective_negative_coefficient":negative}


def apply_modality(values: torch.Tensor, modality: str) -> torch.Tensor:
    if values.shape[-1]!=25:raise ValueError(values.shape)
    if modality in {"combined","combined_moddrop"}:return values
    output=values.clone()
    if modality=="score_only":output[...,:9]=0
    elif modality=="curves_only":output[...,9:]=0
    else:raise ValueError(modality)
    return output


def apply_training_modality(values: torch.Tensor, modality: str, dropout_probability: float) -> torch.Tensor:
    if modality!="combined_moddrop":return apply_modality(values,modality)
    output=values.clone();draw=torch.rand(len(output),device=output.device);half=dropout_probability/2;output[draw<half,:9]=0;output[(draw>=half)&(draw<dropout_probability),9:]=0;return output


def raw_predictions(model, data, curve_norm: Normalizer, score_norm: Normalizer, device, modality: str):
    model.eval();output={}
    with torch.no_grad():
        for piece_id,item in sorted(data.items()):
            curves=item["curves"] if len(item["curves"]) else np.zeros((1,len(item["labels"]),9),np.float32);ids=[str(x) for x in item["performance_ids"]] or ["missing_curve"];c=torch.from_numpy(curve_norm.apply(curves).astype(np.float32)).to(device);score=score_norm.apply(item["score_phase3"]).astype(np.float32);s=torch.from_numpy(np.broadcast_to(score[None],(len(curves),*score.shape)).copy()).to(device);values=apply_modality(torch.cat([c,s],dim=-1),modality);probs=torch.sigmoid(model(values)).cpu().numpy();output[piece_id]={pid:p for pid,p in zip(ids,probs)}
    return output


def prediction_diagnostics(raw, data, threshold: float):
    flat=[];within=[];densities=[];widths=[]
    for piece_id,perfs in raw.items():
        stack=np.stack(list(perfs.values()));flat.extend(stack.ravel());within.append(float(np.mean(np.std(stack,axis=0))))
        valid=data[piece_id]["label_mask"]>0
        for probability in stack:
            nms=nms_probabilities(probability);densities.append(float(((nms>=threshold)&valid).sum()/max(valid.sum(),1)));active=(probability>=threshold)&valid;starts=np.flatnonzero(active & ~np.r_[False,active[:-1]]);ends=np.flatnonzero(active & ~np.r_[active[1:],False]);widths.extend((ends-starts+1).tolist())
    return {"probability_variance":float(np.var(flat)),"cross_performance_probability_std":float(np.mean(within)),"prediction_density":float(np.mean(densities)),"mean_raw_peak_width":float(np.mean(widths)) if widths else 0.0}


def deadline_guard(deadline: datetime, guard_minutes: float):
    remaining=(deadline-datetime.now().astimezone()).total_seconds()
    if remaining<=guard_minutes*60:raise ExperimentDeadline(f"Phase5 training stopped with {remaining:.1f}s before deadline")


def train_matched_bigru(train_data, validation_data, config: dict[str,Any], checkpoint_dir: Path, deadline: datetime, seed: int, modality: str, target_kind: str, weight_kind: str, resume: bool=True, contract_hash: str | None=None):
    seed_everything(seed);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");targeted=target_data(train_data,target_kind,int(config["targets"]["soft_radius_beats"]),float(config["targets"]["soft_neighbor_weight"]));curve_arrays=[x["curves"] if len(x["curves"]) else np.zeros((1,len(x["labels"]),9),np.float32) for x in train_data.values()];score_arrays=[x["score_phase3"] for x in train_data.values()];curve_norm,score_norm=fit_train_normalizer(curve_arrays),fit_train_normalizer(score_arrays);m=config["model"];t=config["training"];dataset=MultiModalWindowDataset(targeted,curve_norm,score_norm,int(m["window_beats"]),int(m["stride_beats"]));loader=DataLoader(dataset,batch_size=int(t["batch_size"]),shuffle=True,num_workers=0);model=SmallBiGRU(int(m["input_dimensions"]),int(m["hidden_size"]),float(m["dropout"])).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=float(t["learning_rate"]),weight_decay=float(t["weight_decay"]));balance=loss_balance(targeted,weight_kind,float(config["targets"]["original_pos_weight_cap"]));criterion=nn.BCEWithLogitsLoss(reduction="none",pos_weight=torch.tensor(balance["pos_weight"],device=device));checkpoint_dir.mkdir(parents=True,exist_ok=True);best_path,latest_path=checkpoint_dir/"best.pt",checkpoint_dir/"latest.pt";start,best_score,patience,history=0,-1.0,0,[]
    if resume and best_path.exists() and latest_path.exists():
        state=torch.load(latest_path,map_location=device,weights_only=False);model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);start=int(state["epoch"])+1;best_score=float(state["best_score"]);patience=int(state["patience"]);history=list(state["history"])
    maximum=int(t["maximum_epochs"])
    if resume and (start>=maximum or patience>=int(t["patience"])):maximum=start
    for epoch in range(start,maximum):
        deadline_guard(deadline,float(t["deadline_guard_minutes"]));model.train();losses=[]
        for batch_index,(curves,score,labels,mask,valid,work_weight) in enumerate(loader):
            if batch_index%20==0:deadline_guard(deadline,float(t["deadline_guard_minutes"]))
            curves,score,labels,mask=curves.to(device),score.to(device),labels.to(device),mask.to(device);work_weight=work_weight.to(device).view(-1,1);values=apply_training_modality(torch.cat([curves,score],dim=-1),modality,float(t["modality_dropout_probability"]));optimizer.zero_grad(set_to_none=True);logits=model(values);loss_values=criterion(logits,labels)*mask*work_weight;loss=loss_values.sum()/torch.clamp((mask*work_weight).sum(),min=1.0)
            if not torch.isfinite(loss):raise FloatingPointError("non-finite phase5 loss")
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),float(t["gradient_clip_norm"]));optimizer.step();losses.append(float(loss.detach().cpu()))
        raw=raw_predictions(model,validation_data,curve_norm,score_norm,device,modality);threshold,_=choose_single_threshold(raw,validation_data,config["evaluation"]["threshold_grid"]);_,_,summary=evaluate_single_performance(raw,validation_data,threshold);score=float(summary["macro_f1_tol1"]);history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_f1_tol1":score,"threshold":threshold})
        common={"model":model.state_dict(),"epoch":epoch,"best_score":max(best_score,score),"threshold":threshold if score>best_score else None,"history":history,"curve_normalizer_mean":curve_norm.mean,"curve_normalizer_std":curve_norm.std,"score_normalizer_mean":score_norm.mean,"score_normalizer_std":score_norm.std,"balance":balance,"modality":modality,"target_kind":target_kind,"weight_kind":weight_kind,"seed":seed,"contract_hash":contract_hash}
        if score>best_score+1e-9:best_score,patience=score,0;common["best_score"]=best_score;common["threshold"]=threshold;torch.save(common,best_path)
        else:patience+=1
        common["optimizer"]=optimizer.state_dict();common["patience"]=patience;torch.save(common,latest_path)
        if patience>=int(t["patience"]):break
    best=torch.load(best_path,map_location=device,weights_only=False);latest=torch.load(latest_path,map_location=device,weights_only=False);model.load_state_dict(best["model"]);curve_norm=Normalizer(np.asarray(best["curve_normalizer_mean"]),np.asarray(best["curve_normalizer_std"]));score_norm=Normalizer(np.asarray(best["score_normalizer_mean"]),np.asarray(best["score_normalizer_std"]));raw_val=raw_predictions(model,validation_data,curve_norm,score_norm,device,modality);raw_train=raw_predictions(model,train_data,curve_norm,score_norm,device,modality);_,_,val_summary=evaluate_single_performance(raw_val,validation_data,float(best["threshold"]));_,_,train_summary=evaluate_single_performance(raw_train,train_data,float(best["threshold"]));diag=prediction_diagnostics(raw_val,validation_data,float(best["threshold"]));info={"parameters":sum(p.numel() for p in model.parameters()),"best_epoch":int(best["epoch"]),"last_epoch":int(latest["epoch"]),"epoch_zero_meaning":"first completed training epoch","threshold":float(best["threshold"]),"balance":balance,"history":latest["history"],"best_checkpoint_history":best["history"],"validation":val_summary,"training":train_summary,"train_minus_validation_f1":float(train_summary["macro_f1_tol1"]-val_summary["macro_f1_tol1"]),**diag}
    # cuDNN RNN input gradients require training mode; parameters are not stepped.
    model.train();curves,score,labels,mask,valid,work_weight=next(iter(DataLoader(dataset,batch_size=2,shuffle=False,num_workers=0)));values=apply_modality(torch.cat([curves,score],dim=-1).to(device),modality).detach().requires_grad_(True);logits=model(values);loss=(criterion(logits,labels.to(device))*mask.to(device)).sum()/torch.clamp(mask.to(device).sum(),min=1.0);model.zero_grad(set_to_none=True);loss.backward();gradient=values.grad.detach().abs().mean(dim=(0,1)).cpu().numpy();info["input_gradient_curve_mean_abs"]=float(gradient[:9].mean());info["input_gradient_score_mean_abs"]=float(gradient[9:].mean());info["finite_gradients"]=bool(np.isfinite(gradient).all());model.eval()
    return model,curve_norm,score_norm,raw_val,info
