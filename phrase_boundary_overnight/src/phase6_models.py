from __future__ import annotations

import copy
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .models import Normalizer, seed_everything
from .phase2_models import choose_single_threshold, evaluate_single_performance
from .phase3_models import MultiModalWindowDataset, SmallBiGRU, fit_train_normalizer
from .phase4_models import ExperimentDeadline


def window_starts(length: int, window: int, stride: int) -> list[int]:
    starts = list(range(0, max(length - window + 1, 1), stride))
    final = max(0, length - window)
    if not starts or starts[-1] != final:
        starts.append(final)
    return sorted(set(starts))


def triangular_weights(window: int) -> np.ndarray:
    position = np.arange(window)
    return (1.0 + np.minimum(position, window - 1 - position)).astype(np.float32)


def sliding_predictions(model, data, curve_norm: Normalizer, score_norm: Normalizer, device, window: int = 64, stride: int = 32):
    model.eval(); output = {}
    weights = triangular_weights(window)
    with torch.no_grad():
        for piece_id, item in sorted(data.items()):
            curves = item["curves"] if len(item["curves"]) else np.zeros((1, len(item["labels"]), 9), np.float32)
            ids = [str(x) for x in item["performance_ids"]] or ["missing_curve"]
            score = score_norm.apply(item["score_phase3"]).astype(np.float32); length = len(score); starts = window_starts(length, window, stride); perfs = {}
            for performance_id, curve in zip(ids, curves):
                curve = curve_norm.apply(curve).astype(np.float32); numerator = np.zeros(length, np.float64); denominator = np.zeros(length, np.float64)
                batches=[];valid_lengths=[]
                for start in starts:
                    values=np.concatenate([curve[start:start+window],score[start:start+window]],axis=1);valid=len(values);valid_lengths.append(valid)
                    if valid<window:values=np.pad(values,((0,window-valid),(0,0)))
                    batches.append(values)
                probabilities=torch.sigmoid(model(torch.from_numpy(np.stack(batches).astype(np.float32)).to(device))).cpu().numpy()
                for start,valid,probability in zip(starts,valid_lengths,probabilities):
                    w=weights[:valid];numerator[start:start+valid]+=probability[:valid]*w;denominator[start:start+valid]+=w
                if np.any(denominator<=0):raise AssertionError(f"uncovered beat in {piece_id}")
                perfs[performance_id]=(numerator/denominator).astype(np.float32)
            output[piece_id]=perfs
    return output


def full_predictions(model, data, curve_norm: Normalizer, score_norm: Normalizer, device):
    model.eval();output={}
    with torch.no_grad():
        for piece_id,item in sorted(data.items()):
            curves=item["curves"] if len(item["curves"]) else np.zeros((1,len(item["labels"]),9),np.float32);ids=[str(x) for x in item["performance_ids"]] or ["missing_curve"]
            c=torch.from_numpy(curve_norm.apply(curves).astype(np.float32)).to(device);score=score_norm.apply(item["score_phase3"]).astype(np.float32);s=torch.from_numpy(np.broadcast_to(score[None],(len(curves),*score.shape)).copy()).to(device);prob=torch.sigmoid(model(torch.cat([c,s],dim=-1))).cpu().numpy();output[piece_id]={pid:p for pid,p in zip(ids,prob)}
    return output


def load_phase5_bigru(checkpoint: Path, device):
    state=torch.load(checkpoint,map_location=device,weights_only=False);model=SmallBiGRU(25,32,.2).to(device);model.load_state_dict(state["model"]);curve=Normalizer(np.asarray(state["curve_normalizer_mean"]),np.asarray(state["curve_normalizer_std"]));score=Normalizer(np.asarray(state["score_normalizer_mean"]),np.asarray(state["score_normalizer_std"]));return model,curve,score,state


def context_difference_rows(full, sliding, data, window: int, stride: int):
    rows=[]
    for piece_id in sorted(full):
        length=len(data[piece_id]["labels"]);starts=window_starts(length,window,stride);best_edge=np.zeros(length,int)
        for start in starts:
            valid=min(window,length-start);pos=np.arange(valid);best_edge[start:start+valid]=np.maximum(best_edge[start:start+valid],np.minimum(pos,valid-1-pos))
        for perf in sorted(full[piece_id]):
            diff=np.abs(full[piece_id][perf]-sliding[piece_id][perf]);
            for lo,hi in [(0,8),(8,16),(16,33)]:
                mask=(best_edge>=lo)&(best_edge<hi);rows.append({"piece_id":piece_id,"performance_id":perf,"beats":length,"edge_lo":lo,"edge_hi":hi,"count":int(mask.sum()),"mae":float(diff[mask].mean()) if mask.any() else math.nan,"max_abs":float(diff[mask].max()) if mask.any() else math.nan})
    return rows


def exposure_audit(data, batch_size: int, window: int, stride: int):
    dataset=MultiModalWindowDataset(data,fit_train_normalizer([x["curves"] if len(x["curves"]) else np.zeros((1,len(x["labels"]),9),np.float32) for x in data.values()]),fit_train_normalizer([x["score_phase3"] for x in data.values()]),window,stride);boundary_exposures=[];piece_rows=[]
    for piece_id,item in sorted(data.items()):
        length=len(item["labels"]);performances=max(len(item["curves"]),1);starts=window_starts(length,window,stride);cover=np.zeros(length,int)
        for start in starts:cover[start:min(start+window,length)]+=1
        valid=(item["labels"]>0)&(item["label_mask"]>0);boundary_exposures.extend((cover[valid]*performances).tolist());piece_rows.append({"piece_id":piece_id,"beats":length,"performances":performances,"windows_per_performance":len(starts),"samples_per_epoch":performances*len(starts),"sample_weight":1/performances,"weighted_sample_mass":len(starts)})
    values=np.asarray(boundary_exposures,float);return piece_rows,{"training_pieces":len(data),"dataset_windows":len(dataset),"batch_size":batch_size,"optimizer_steps_per_epoch":math.ceil(len(dataset)/batch_size),"boundary_count":len(values),"boundary_exposure_min":float(values.min()),"boundary_exposure_median":float(np.median(values)),"boundary_exposure_mean":float(values.mean()),"boundary_exposure_max":float(values.max())}


class StepSampler:
    def __init__(self,data,curve_norm,score_norm,window,stride,batch_size,mode,seed,state=None):
        self.data=data;self.curve_norm=curve_norm;self.score_norm=score_norm;self.window=window;self.stride=stride;self.batch_size=batch_size;self.mode=mode;self.pieces=sorted(data);self.rng=np.random.default_rng(seed);self.dataset=MultiModalWindowDataset(data,curve_norm,score_norm,window,stride);self.permutation=np.array([],int);self.cursor=0
        if state:self.load_state(state)
    def state(self):return {"rng":copy.deepcopy(self.rng.bit_generator.state),"permutation":self.permutation.tolist(),"cursor":self.cursor}
    def load_state(self,state):self.rng.bit_generator.state=state["rng"];self.permutation=np.asarray(state["permutation"],int);self.cursor=int(state["cursor"])
    def _old_indices(self):
        if self.cursor>=len(self.permutation):self.permutation=self.rng.permutation(len(self.dataset));self.cursor=0
        end=min(self.cursor+self.batch_size,len(self.permutation));indices=self.permutation[self.cursor:end];self.cursor=end;return indices
    def batch(self):
        if self.mode=="old_window":samples=[self.dataset[int(i)] for i in self._old_indices()]
        elif self.mode=="piece_balanced":
            samples=[]
            for _ in range(self.batch_size):
                item=self.data[str(self.rng.choice(self.pieces))];curves=item["curves"] if len(item["curves"]) else np.zeros((1,len(item["labels"]),9),np.float32);curve=curves[int(self.rng.integers(len(curves)))];starts=window_starts(len(item["labels"]),self.window,self.stride);start=int(self.rng.choice(starts));valid=min(self.window,len(item["labels"])-start);c=self.curve_norm.apply(curve[start:start+self.window]).astype(np.float32);s=self.score_norm.apply(item["score_phase3"])[start:start+self.window].astype(np.float32);y=item["labels"][start:start+self.window].astype(np.float32);m=item["label_mask"][start:start+self.window].astype(np.float32);v=np.ones(valid,np.float32)
                if valid<self.window:c=np.pad(c,((0,self.window-valid),(0,0)));s=np.pad(s,((0,self.window-valid),(0,0)));y=np.pad(y,(0,self.window-valid));m=np.pad(m,(0,self.window-valid));v=np.pad(v,(0,self.window-valid))
                samples.append((c,s,y,m,v,1.0))
        else:raise ValueError(self.mode)
        return tuple(torch.as_tensor(np.stack([sample[i] for sample in samples])) for i in range(5))+(torch.as_tensor([sample[5] for sample in samples],dtype=torch.float32),)


def positive_weight(data,cap):
    positive=sum(float(x["labels"][x["label_mask"]>0].sum()) for x in data.values());negative=sum(float((x["label_mask"]>0).sum()-x["labels"][x["label_mask"]>0].sum()) for x in data.values());return min(negative/max(positive,1.0),cap)


def validation_snapshot(model,data,curve_norm,score_norm,device,config):
    raw=sliding_predictions(model,data,curve_norm,score_norm,device,int(config["model"]["window_beats"]),int(config["model"]["stride_beats"]));threshold,_=choose_single_threshold(raw,data,config["evaluation"]["threshold_grid"]);_,_,selected=evaluate_single_performance(raw,data,threshold);_,_,fixed=evaluate_single_performance(raw,data,float(config["evaluation"]["fixed_threshold"]));return raw,threshold,selected,fixed


def train_step_budget(train_data,validation_data,config,checkpoint_dir:Path,deadline:datetime,seed:int,mode:str,weight_decay:float,resume=True,contract_hash=None):
    seed_everything(seed);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");curve_dim=int(next(iter(train_data.values()))["curves"].shape[-1]);score_dim=int(next(iter(train_data.values()))["score_phase3"].shape[-1]);input_dim=curve_dim+score_dim;curve_norm=fit_train_normalizer([x["curves"] if len(x["curves"]) else np.zeros((1,len(x["labels"]),curve_dim),np.float32) for x in train_data.values()]);score_norm=fit_train_normalizer([x["score_phase3"] for x in train_data.values()]);t=config["training"];m=config["model"];audit=exposure_audit(train_data,int(t["batch_size"]),int(m["window_beats"]),int(m["stride_beats"]))[1];max_steps=min(int(t["maximum_steps"]),int(t["old_epoch_multiples"])*int(audit["optimizer_steps_per_epoch"]));frequency=max(1,int(audit["optimizer_steps_per_epoch"])//int(t["validation_quarters_per_old_epoch"]));model=SmallBiGRU(input_dim,int(m["hidden_size"]),float(m["dropout"])).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=float(t["learning_rate"]),weight_decay=float(weight_decay));criterion=nn.BCEWithLogitsLoss(reduction="none",pos_weight=torch.tensor(positive_weight(train_data,float(t["positive_weight_cap"])),device=device));sampler=StepSampler(train_data,curve_norm,score_norm,int(m["window_beats"]),int(m["stride_beats"]),int(t["batch_size"]),mode,seed);checkpoint_dir.mkdir(parents=True,exist_ok=True);best_path,latest_path=checkpoint_dir/"best.pt",checkpoint_dir/"latest.pt";step=0;best_score=-1.0;history=[]
    if resume and latest_path.exists():
        state=torch.load(latest_path,map_location=device,weights_only=False);model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);step=int(state["step"]);best_score=float(state["best_score"]);history=list(state["history"]);sampler.load_state(state["sampler"]);torch.set_rng_state(state["torch_rng"].cpu()); 
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    def save_latest():torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,"best_score":best_score,"history":history,"sampler":sampler.state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,"contract_hash":contract_hash,"mode":mode,"seed":seed,"input_dimensions":input_dim,"curve_dimensions":curve_dim,"score_dimensions":score_dim,"curve_normalizer_mean":curve_norm.mean,"curve_normalizer_std":curve_norm.std,"score_normalizer_mean":score_norm.mean,"score_normalizer_std":score_norm.std},latest_path)
    if step==0:
        _,threshold,selected,fixed=validation_snapshot(model,validation_data,curve_norm,score_norm,device,config);history.append({"step":0,"trained":False,"validation_f1_tol1":selected["macro_f1_tol1"],"fixed_f1_tol1":fixed["macro_f1_tol1"],"pr_auc":selected["macro_pr_auc"],"threshold":threshold,"train_loss_since_validation":None})
    losses=[]
    while step<max_steps:
        remaining=(deadline-datetime.now().astimezone()).total_seconds()
        if remaining<=float(t["deadline_guard_minutes"])*60:save_latest();raise ExperimentDeadline(f"Phase6 stopped with {remaining:.1f}s remaining")
        curves,score,labels,mask,valid,weights=sampler.batch();curves,score,labels,mask,weights=curves.to(device),score.to(device),labels.to(device),mask.to(device),weights.to(device).view(-1,1);model.train();optimizer.zero_grad(set_to_none=True);logits=model(torch.cat([curves,score],dim=-1));loss_values=criterion(logits,labels)*mask*weights;loss=loss_values.sum()/torch.clamp((mask*weights).sum(),min=1.0)
        if not torch.isfinite(loss):raise FloatingPointError("non-finite Phase6 loss")
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),float(t["gradient_clip_norm"]));optimizer.step();step+=1;losses.append(float(loss.detach().cpu()))
        if step%int(t["latest_every_steps"])==0:save_latest()
        if step%frequency==0 or step==max_steps:
            raw,threshold,selected,fixed=validation_snapshot(model,validation_data,curve_norm,score_norm,device,config);record={"step":step,"trained":True,"validation_f1_tol1":selected["macro_f1_tol1"],"validation_precision_tol1":selected["macro_precision_tol1"],"validation_recall_tol1":selected["macro_recall_tol1"],"fixed_f1_tol1":fixed["macro_f1_tol1"],"pr_auc":selected["macro_pr_auc"],"threshold":threshold,"train_loss_since_validation":float(np.mean(losses))};history.append(record);losses=[]
            if record["validation_f1_tol1"]>best_score+1e-9:best_score=record["validation_f1_tol1"];torch.save({"model":model.state_dict(),"step":step,"best_score":best_score,"threshold":threshold,"history":history,"mode":mode,"seed":seed,"weight_decay":weight_decay,"input_dimensions":input_dim,"curve_dimensions":curve_dim,"score_dimensions":score_dim,"curve_normalizer_mean":curve_norm.mean,"curve_normalizer_std":curve_norm.std,"score_normalizer_mean":score_norm.mean,"score_normalizer_std":score_norm.std,"contract_hash":contract_hash},best_path)
            save_latest()
    best=torch.load(best_path,map_location=device,weights_only=False);model.load_state_dict(best["model"]);raw,threshold,selected,fixed=validation_snapshot(model,validation_data,curve_norm,score_norm,device,config);raw_train=sliding_predictions(model,train_data,curve_norm,score_norm,device,int(m["window_beats"]),int(m["stride_beats"]));_,_,train_summary=evaluate_single_performance(raw_train,train_data,threshold);return model,curve_norm,score_norm,raw,{"parameters":sum(p.numel() for p in model.parameters()),"input_dimensions":input_dim,"curve_dimensions":curve_dim,"score_dimensions":score_dim,"best_step":int(best["step"]),"maximum_steps":max_steps,"validation_frequency":frequency,"old_steps_per_epoch":audit["optimizer_steps_per_epoch"],"history":history,"selected":selected,"fixed":fixed,"training":train_summary,"train_minus_validation_f1":float(train_summary["macro_f1_tol1"]-selected["macro_f1_tol1"]),"threshold":threshold,"mode":mode,"weight_decay":weight_decay}
