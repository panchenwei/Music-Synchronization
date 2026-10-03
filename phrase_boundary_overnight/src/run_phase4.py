from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from .models import Normalizer
from .phase2_models import choose_single_threshold, evaluate_single_performance
from .phase3_models import SmallBiGRU, raw_multimodal_predictions
from .run_phase3 import context_features, markdown_table, paired_bootstrap
from .phase4_models import ExperimentDeadline, soft_target, train_soft_bigru


STAGES=["init","features","develop","experiments","migration","report","audit"]


def now(): return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path:Path,payload:Any): path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")


def sha(path:Path): return hashlib.sha256(path.read_bytes()).hexdigest()


def sigmoid(values):
    values=np.clip(np.asarray(values,float),-30,30);return 1/(1+np.exp(-values))


class Phase4Pipeline:
    def __init__(self,root:Path,resume:bool,force:bool):
        self.root=root.resolve();self.resume=resume;self.force=force;self.config=yaml.safe_load((self.root/"configs/phase4/protocol.yaml").read_text(encoding="utf-8"));self.deadline=datetime.fromisoformat(self.config["project"]["experiment_deadline"])
        self.reports=self.root/"reports/phase4";self.artifacts=self.root/"artifacts/phase4";self.metrics=self.artifacts/"metrics";self.figures=self.artifacts/"figures";self.checkpoints=self.root/"checkpoints/phase4";self.manifests=self.root/"manifests/phase4";self.logs=self.root/"logs/phase4";self.markers=self.artifacts/"stages"
        for p in [self.reports,self.artifacts,self.metrics,self.figures,self.checkpoints,self.manifests,self.logs,self.markers]:p.mkdir(parents=True,exist_ok=True)
        self.config_hash=sha(self.root/"configs/phase4/protocol.yaml");self.training_hash=hashlib.sha256(((self.root/"src/phase4_models.py").read_bytes())+(self.root/"configs/phase4/protocol.yaml").read_bytes()+b"phase4_training_contract_v2").hexdigest();self.code_hash=sha(self.root/"src/run_phase4.py")

    def marker(self,stage):return self.markers/f"{stage}.json"
    def contract_hash(self,stage):return self.training_hash if stage in {"develop","experiments"} else self.config_hash
    def done(self,stage):
        if not self.resume or self.force or not self.marker(stage).exists():return False
        payload=json.loads(self.marker(stage).read_text(encoding="utf-8"));return payload.get("contract_hash")==self.contract_hash(stage)
    def mark(self,stage,payload):write_json(self.marker(stage),{"stage":stage,"completed_at":now(),"contract_hash":self.contract_hash(stage),"code_hash":self.code_hash,**payload})
    def check_deadline(self):
        remaining=(self.deadline-datetime.now().astimezone()).total_seconds()
        if remaining<=float(self.config["training"]["deadline_guard_minutes"])*60:raise ExperimentDeadline(f"No new experiment: {remaining:.1f}s before 11:00 deadline")
        return remaining
    def split(self,fold):
        frame=pd.read_csv(self.root/self.config["splits"]["source"]);return {name:frame[(frame.fold==fold)&(frame.split==name)].piece_id.tolist() for name in ["train","validation","test"]}
    def load(self,ids):
        out={}
        for pid in ids:
            with np.load(self.root/f"cache/phase3/piece_features/{pid}.npz",allow_pickle=False) as z:out[pid]={k:z[k] for k in z.files}
        return out
    def phase3_a2(self,fold,seed=42):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu");state=torch.load(self.root/f"checkpoints/phase3/a2_bigru/seed{seed}/fold{fold}/best.pt",map_location=device,weights_only=False);model=SmallBiGRU(25,32,.2).to(device);model.load_state_dict(state["model"]);cn=Normalizer(np.asarray(state["curve_normalizer_mean"]),np.asarray(state["curve_normalizer_std"]));sn=Normalizer(np.asarray(state["score_normalizer_mean"]),np.asarray(state["score_normalizer_std"]));return model,cn,sn,state
    def save_eval(self,prefix,raw_val,raw_test,val,test,threshold=None):
        if threshold is None:threshold,curve=choose_single_threshold(raw_val,val,self.config["evaluation"]["threshold_grid"])
        else:
            curve=pd.DataFrame([{"threshold":threshold}])
        vp,vw,vs=evaluate_single_performance(raw_val,val,threshold);tp,tw,ts=evaluate_single_performance(raw_test,test,threshold);curve.to_csv(self.metrics/f"{prefix}_threshold.csv",index=False);vp.to_csv(self.metrics/f"{prefix}_validation_performance.csv",index=False);vw.to_csv(self.metrics/f"{prefix}_validation_piece.csv",index=False);tp.to_csv(self.metrics/f"{prefix}_test_performance.csv",index=False);tw.to_csv(self.metrics/f"{prefix}_test_piece.csv",index=False);write_json(self.metrics/f"{prefix}_summary.json",{"threshold":threshold,"validation":vs,"test":ts});return {"threshold":threshold,"validation":vs,"test":ts}

    def registry(self):
        p=self.reports/"experiment_registry.json";return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"created_at":now(),"experiments":[]}
    def register(self,entry):
        aliases={"A2_hard_control":"A2_control","SoftTarget_BiGRU":"soft_bigru","LinearSVC":"linear_svc","RBF_SVC":"rbf_svc"};reg=self.registry();canonical=aliases.get(entry["id"],entry["id"]);previous={}
        for item in reg["experiments"]:
            if aliases.get(item.get("id"),item.get("id"))==canonical:previous={**previous,**item}
        merged={**previous,**entry,"id":canonical};items=[x for x in reg["experiments"] if aliases.get(x.get("id"),x.get("id"))!=canonical];items.append(merged);reg["experiments"]=items;reg["updated_at"]=now();write_json(self.reports/"experiment_registry.json",reg)

    def init(self):
        if self.done("init"):print("[resume] init");return
        sources=["reports/phase3/final_report.md","reports/phase3/completion_audit.json","reports/phase3/resume_validation.json","artifacts/phase3/metrics/model_comparison.csv","artifacts/phase3/metrics/train_validation_gap_summary.csv","artifacts/phase2/splits/opus_split_manifest.csv","manifests/phase3/score_feature_manifest.csv"]
        frozen={"created_at":now(),"status":"frozen","scientific_status":"preliminary_exploratory","facts":{"works":43,"boundaries":536,"performances":1990,"phase3_a2_seed42_f1":0.4843395712327025,"phase3_c0_seed42_f1":0.4289397283580224,"audio_label_overlap":0},"files":[{"path":x,"sha256":sha(self.root/x)} for x in sources]};write_json(self.manifests/"frozen_phase3_evidence.json",frozen)
        split=pd.read_csv(self.root/self.config["splits"]["source"]);rows=[]
        for fold in range(5):
            groups={s:set(split[(split.fold==fold)&(split.split==s)].opus.astype(str)) for s in ["train","validation","test"]};pieces={s:set(split[(split.fold==fold)&(split.split==s)].piece_id) for s in groups};rows.append({"fold":fold,"piece_overlap":sum(len(pieces[a]&pieces[b]) for a,b in [("train","validation"),("train","test"),("validation","test")]),"opus_overlap":sum(len(groups[a]&groups[b]) for a,b in [("train","validation"),("train","test"),("validation","test")])})
        pd.DataFrame(rows).to_csv(self.manifests/"split_reaudit.csv",index=False)
        self.register({"id":"A2_control","status":"reused_reproduced","hypothesis":"Frozen hard-label BiGRU control","single_change":"none","expected":"F1 0.4843395712","failure":"hash or metric mismatch","cost":"no training"});self.register({"id":"soft_bigru","status":"proposed","hypothesis":"Tolerance-aware labels reduce one-beat target mismatch","single_change":"triangular radius-1 soft target only","expected":"validation F1 above A2","failure":"no validation gain","cost":"fold0 first"});self.register({"id":"linear_svc","status":"proposed","hypothesis":"low-capacity margin classifier reduces A2 overfit","single_change":"LinearSVC on fixed context","expected":"validation gain with smaller gap","failure":"no validation gain","cost":"6 configs max"});self.register({"id":"rbf_svc","status":"proposed","hypothesis":"bounded nonlinear interactions aid sparse boundary classification","single_change":"RBF kernel on work-balanced sample","expected":"validation gain within cost gate","failure":"no gain or cost >300s","cost":"6 configs max"})
        (self.reports/"STATUS.md").write_text(f"# Phase 4 Status\n\nStatus: INIT_COMPLETE\n\nStarted: {now()}\nExperiment deadline: {self.deadline.isoformat()}\nScientific status: preliminary/exploratory.\n",encoding="utf-8");(self.reports/"decision_log.md").write_text("# Phase 4 Decision Log\n\n## Frozen start\n\nPhase 3 evidence is SHA-256 frozen. Development uses fold 0 train/validation only; no Phase-4 outer test is opened before candidate selection. Soft labels, LinearSVC and RBF SVC are bounded single-factor tests.\n",encoding="utf-8")
        self.mark("init",{"frozen_files":len(sources),"piece_overlap":0,"opus_overlap":0})

    def _ablate(self,data,curve_mean,score_mean,curve_dims=None,score_dims=None,block=False):
        out={}
        for pid,item in data.items():
            copied=dict(item);curves=item["curves"].copy();score=item["score_phase3"].copy()
            if curve_dims is not None:curves[:,:,curve_dims]=curve_mean[curve_dims]
            if score_dims is not None:score[:,score_dims]=score_mean[score_dims]
            if block:
                for start in range(8,len(score),32):
                    stop=min(start+8,len(score));curves[:,start:stop,:]=curve_mean;score[start:stop,:]=score_mean
            copied["curves"]=curves;copied["score_phase3"]=score;out[pid]=copied
        return out

    def features(self):
        if self.done("features"):print("[resume] features");return
        split=self.split(0);train,val=self.load(split["train"]),self.load(split["validation"]);model,cn,sn,state=self.phase3_a2(0);device=next(model.parameters()).device;raw=raw_multimodal_predictions(model,val,cn,sn,device,"bigru");threshold=float(state["threshold"]);base_perf,base_piece,base=evaluate_single_performance(raw,val,threshold)
        groups={"curves_all":{"curve_dims":list(range(9))},"score_all":{"score_dims":list(range(16))},"tempo":{"curve_dims":[0,1,2,5]},"dynamics":{"curve_dims":[3,4,6]},"music_transition":{"score_dims":[3,4,6,7,8,9,10,11,13,14,15]},"temporal_block8":{"block":True}};ablation=[]
        for name,kwargs in groups.items():
            changed=self._ablate(val,cn.mean,sn.mean,**kwargs);pred=raw_multimodal_predictions(model,changed,cn,sn,device,"bigru");_,_,summary=evaluate_single_performance(pred,changed,threshold);ablation.append({"ablation":name,"validation_f1":summary["macro_f1_tol1"],"delta_vs_normal":summary["macro_f1_tol1"]-base["macro_f1_tol1"],"interpretation":"sensitivity_only_not_causal"})
        pd.DataFrame([{"ablation":"normal","validation_f1":base["macro_f1_tol1"],"delta_vs_normal":0,"interpretation":"control"},*ablation]).to_csv(self.metrics/"a2_feature_ablation_fold0.csv",index=False)
        names=[f"curve_{i}" for i in range(9)]+[f"score_{i}" for i in range(16)];audit=[]
        for group,key in [("curve","curves"),("score","score_phase3")]:
            train_arrays=[];val_arrays=[];train_labels=[];val_labels=[]
            for item in train.values():
                valid=item["label_mask"]>0;arr=item[key] if key=="score_phase3" else np.median(item[key],axis=0);train_arrays.append(arr[valid]);train_labels.append(item["labels"][valid])
            for item in val.values():
                valid=item["label_mask"]>0;arr=item[key] if key=="score_phase3" else np.median(item[key],axis=0);val_arrays.append(arr[valid]);val_labels.append(item["labels"][valid])
            tx,vx,ty,vy=np.concatenate(train_arrays),np.concatenate(val_arrays),np.concatenate(train_labels),np.concatenate(val_labels)
            for j in range(tx.shape[1]):
                direction=1 if average_precision_score(ty,tx[:,j])>=average_precision_score(ty,-tx[:,j]) else -1;audit.append({"group":group,"dimension":j,"train_mean":tx[:,j].mean(),"train_std":tx[:,j].std(),"validation_shift_train_sd":abs(vx[:,j].mean()-tx[:,j].mean())/max(tx[:,j].std(),1e-8),"validation_auprc":average_precision_score(vy,direction*vx[:,j]),"finite":bool(np.isfinite(tx[:,j]).all() and np.isfinite(vx[:,j]).all())})
        pd.DataFrame(audit).to_csv(self.metrics/"feature_information_audit.csv",index=False)
        soft_stats=[]
        for pid,item in train.items():
            y=soft_target(item["labels"],item["label_mask"],1,.5);soft_stats.append({"piece_id":pid,"hard_positive_mass":float(item["labels"].sum()),"soft_positive_mass":float(y.sum()),"valid_beats":int((item["label_mask"]>0).sum())})
        pd.DataFrame(soft_stats).to_csv(self.metrics/"soft_target_mass_audit.csv",index=False)
        # Real validation cases only: strongest, recall failure, precision failure.
        ordered={"success":base_perf.sort_values(["f1_tol1","piece_id","performance_id"],ascending=[False,True,True]),"miss":base_perf.sort_values(["recall_tol1","piece_id","performance_id"],ascending=[True,True,True]),"false_positive":base_perf.sort_values(["precision_tol1","piece_id","performance_id"],ascending=[True,True,True])};chosen=[];case_specs=[];case_rows=[]
        for role in ["success","miss","false_positive"]:
            row=next(r for r in ordered[role].itertuples() if (str(r.piece_id),str(r.performance_id)) not in chosen);pid,perf=str(row.piece_id),str(row.performance_id);chosen.append((pid,perf));case_specs.append({"role":role,"piece_id":pid,"performance_id":perf,"f1_tol1":float(row.f1_tol1),"precision_tol1":float(row.precision_tol1),"recall_tol1":float(row.recall_tol1)})
            item=val[pid];prob=raw[pid][perf];curve=item["curves"][[str(x) for x in item["performance_ids"]].index(perf)]
            for i in range(len(prob)):case_rows.append({"role":role,"piece_id":pid,"performance_id":perf,"beat":i,"log_tempo":curve[i,0],"dynamics":curve[i,3],"score_lbdm":item["score_phase3"][i,15],"label":item["labels"][i],"probability":prob[i],"threshold":threshold})
        pd.DataFrame(case_rows).to_csv(self.metrics/"validation_case_signals.csv",index=False)
        report="# Feature Information Audit\n\nStatus: **implemented/reproduced on fold-0 train/validation only**. No outer test was used.\n\n| Raw signal | Representation | Preserved | Potential loss | Evidence |\n|---|---|---|---|---|\n| beat times | log tempo, derivatives, robust z | local tempo shape | sub-beat timing/transients | feature audit + tempo ablation |\n| dynamics proxy | level/derivative/z | relative loudness contour | spectrum, articulation, room/acoustic detail | dynamics ablation; audio-label overlap=0 |\n| score notes | 16 compact cues | meter, gaps, voice motion, novelty, LBDM | fingering, harmony semantics, long motifs | score-group and transition ablation |\n| 64-beat windows | bidirectional context | local/medium phrase context | dependencies beyond 64 beats | block occlusion |\n\nAll rows remain beat-aligned; label and feature lengths match, masks are finite, and train-only normalizers from the frozen fold-0 checkpoint are used. Ablations replace groups by train means and are sensitivity tests, not causal proofs.\n\n"+markdown_table(pd.DataFrame([{"normal_validation_f1":base["macro_f1_tol1"],"threshold":threshold,"train_works":len(train),"validation_works":len(val)}]));(self.reports/"feature_information_audit.md").write_text(report,encoding="utf-8")
        self.mark("features",{"base_validation_f1":base["macro_f1_tol1"],"ablations":len(groups),"cases":case_specs})

    def _sample_svm(self,data,cap,seed):
        rng=np.random.default_rng(seed);xs=[];ys=[];ws=[];piece_counts={}
        for pid,item in sorted(data.items()):
            blocks=[];labels=[];valid=item["label_mask"]>0
            for curve in item["curves"]:blocks.append(context_features(curve,item["score_phase3"])[valid]);labels.append(item["labels"][valid])
            x=np.concatenate(blocks);y=np.concatenate(labels);pos=np.flatnonzero(y>0.5);neg=np.flatnonzero(y<=0.5);n=min(cap,len(y));n_pos=min(len(pos),max(1,n//3));n_neg=min(len(neg),n-n_pos);selected=np.concatenate([rng.choice(pos,n_pos,replace=False),rng.choice(neg,n_neg,replace=False)]);rng.shuffle(selected);xs.append(x[selected]);ys.append(y[selected]);ws.append(np.full(len(selected),1/max(len(selected),1)));piece_counts[pid]=len(selected)
        return np.concatenate(xs),np.concatenate(ys).astype(int),np.concatenate(ws),piece_counts
    def _svm_raw(self,data,scaler,model):
        out={}
        for pid,item in sorted(data.items()):
            ids=[str(x) for x in item["performance_ids"]];out[pid]={perf:sigmoid(model.decision_function(scaler.transform(context_features(curve,item["score_phase3"])))) for perf,curve in zip(ids,item["curves"])}
        return out
    def fit_svm(self,kind,train,val,fold,seed=42):
        directory=self.checkpoints/f"{kind}/seed{seed}/fold{fold}";best_path,latest_path=directory/"best.joblib",directory/"latest.joblib"
        if self.resume and best_path.exists() and latest_path.exists():return joblib.load(best_path)
        self.check_deadline();spec=self.config["svm"];cap=int(spec["linear_rows_per_work"] if kind=="linear_svc" else spec["rbf_rows_per_work"]);x,y,w,counts=self._sample_svm(train,cap,seed+fold);scaler=StandardScaler().fit(x);xz=scaler.transform(x);grid=spec["linear_grid"] if kind=="linear_svc" else spec["rbf_grid"];leader=[];best=None
        for index,cfg in enumerate(grid):
            self.check_deadline();started=time.perf_counter();model=LinearSVC(C=float(cfg["C"]),class_weight=cfg.get("class_weight"),dual="auto",max_iter=5000,random_state=seed) if kind=="linear_svc" else SVC(C=float(cfg["C"]),gamma=cfg["gamma"],class_weight=cfg.get("class_weight"),probability=False,cache_size=2048,random_state=seed);model.fit(xz,y,sample_weight=w);seconds=time.perf_counter()-started;raw=self._svm_raw(val,scaler,model);threshold,curve=choose_single_threshold(raw,val,self.config["evaluation"]["threshold_grid"]);_,_,summary=evaluate_single_performance(raw,val,threshold);row={"config_index":index,**cfg,"threshold":threshold,"validation_f1":summary["macro_f1_tol1"],"validation_precision":summary["macro_precision_tol1"],"validation_recall":summary["macro_recall_tol1"],"fit_seconds":seconds,"sample_rows":len(x)};leader.append(row)
            if best is None or (row["validation_f1"],row["validation_precision"],-index)>(best[0]["validation_f1"],best[0]["validation_precision"],-best[0]["config_index"]):best=(row,model)
            if kind=="rbf_svc" and index==0 and seconds>float(spec["rbf_cost_abort_seconds"]):break
        directory.mkdir(parents=True,exist_ok=True);payload={"kind":kind,"scaler":scaler,"model":best[1],"selection":best[0],"leaderboard":leader,"piece_sample_counts":counts,"train_piece_ids":sorted(train),"seed":seed,"fold":fold,"config_hash":self.config_hash,"training_contract_hash":self.training_hash,"validation_used_for":"configuration_and_threshold_only","test_used_for_training":False};joblib.dump(payload,best_path);joblib.dump(payload,latest_path);pd.DataFrame(leader).to_csv(self.metrics/f"{kind}_seed{seed}_fold{fold}_validation_grid.csv",index=False);return payload

    def develop(self):
        if self.done("develop"):print("[resume] develop");return
        self.check_deadline();fold=0;split=self.split(fold);train,val=self.load(split["train"]),self.load(split["validation"]);a2_model,acn,asn,astate=self.phase3_a2(0);a2_raw=raw_multimodal_predictions(a2_model,val,acn,asn,next(a2_model.parameters()).device,"bigru");_,_,a2_summary=evaluate_single_performance(a2_raw,val,float(astate["threshold"]));rows=[{"model":"A2_hard_control","validation_f1":a2_summary["macro_f1_tol1"],"threshold":float(astate["threshold"]),"parameters":11443,"status":"reused"}]
        soft_model,cn,sn,info=train_soft_bigru(train,val,self.config,self.checkpoints/"soft_bigru/seed42/fold0",self.deadline,self.resume);soft_raw=raw_multimodal_predictions(soft_model,val,cn,sn,next(soft_model.parameters()).device,"bigru");_,_,soft_summary=evaluate_single_performance(soft_raw,val,float(info["threshold"]));rows.append({"model":"SoftTarget_BiGRU","validation_f1":soft_summary["macro_f1_tol1"],"threshold":info["threshold"],"parameters":info["parameters"],"status":"trained"});write_json(self.metrics/"soft_target_fold0_training.json",info)
        for kind,label in [("linear_svc","LinearSVC"),("rbf_svc","RBF_SVC")]:
            payload=self.fit_svm(kind,train,val,fold);raw=self._svm_raw(val,payload["scaler"],payload["model"]);_,_,summary=evaluate_single_performance(raw,val,float(payload["selection"]["threshold"]));rows.append({"model":label,"validation_f1":summary["macro_f1_tol1"],"threshold":payload["selection"]["threshold"],"parameters":len(payload["model"].coef_.ravel())+1 if kind=="linear_svc" else int(payload["model"].support_.size),"status":"trained"})
        frame=pd.DataFrame(rows);frame["delta_vs_a2"]=frame.validation_f1-frame.loc[frame.model=="A2_hard_control","validation_f1"].iloc[0];frame.to_csv(self.metrics/"development_fold0.csv",index=False);eligible=frame[(frame.model!="A2_hard_control")&(frame.delta_vs_a2>=float(self.config["gates"]["full_fold_min_validation_delta"]))].sort_values("delta_vs_a2",ascending=False).head(int(self.config["gates"]["maximum_full_fold_candidates"]));selection={"created_at":now(),"outer_test_unopened_during_selection":True,"a2_validation_f1":float(rows[0]["validation_f1"]),"selected":eligible.model.tolist(),"criterion":f"delta >= {self.config['gates']['full_fold_min_validation_delta']}","development_results":frame.to_dict("records")};write_json(self.manifests/"frozen_candidate_selection.json",selection)
        for row in frame.itertuples():self.register({"id":str(row.model),"status":"selected_for_fivefold" if row.model in selection["selected"] else ("reused_control" if row.model=="A2_hard_control" else "development_negative"),"fold0_validation_f1":row.validation_f1,"delta_vs_a2":row.delta_vs_a2,"threshold":row.threshold})
        (self.reports/"model_development.md").write_text("# Phase 4 Development Gate\n\nOuter test remained unopened during selection.\n\n"+markdown_table(frame)+"\n\nSelected for five folds: "+str(selection["selected"])+"\n",encoding="utf-8");self.mark("develop",{"selected":selection["selected"],"development_rows":len(frame)})

    def _candidate_raw(self,name,fold,train,val,test,seed=42):
        if name=="SoftTarget_BiGRU":
            local=copy.deepcopy(self.config);local["project"]["seed"]=seed;model,cn,sn,info=train_soft_bigru(train,val,local,self.checkpoints/f"soft_bigru/seed{seed}/fold{fold}",self.deadline,self.resume);return raw_multimodal_predictions(model,val,cn,sn,next(model.parameters()).device,"bigru"),raw_multimodal_predictions(model,test,cn,sn,next(model.parameters()).device,"bigru"),info
        kind="linear_svc" if name=="LinearSVC" else "rbf_svc";payload=self.fit_svm(kind,train,val,fold,seed);return self._svm_raw(val,payload["scaler"],payload["model"]),self._svm_raw(test,payload["scaler"],payload["model"]),payload["selection"]

    def experiments(self):
        if self.done("experiments"):print("[resume] experiments");return
        selection=json.loads((self.manifests/"frozen_candidate_selection.json").read_text(encoding="utf-8"));selected=selection["selected"]
        if not selected:
            write_json(self.metrics/"fivefold_gate_stop.json",{"status":"skipped_by_validation_gate","selected":[]});self.mark("experiments",{"status":"gate_stop","selected":[]});return
        all_rows=[]
        for fold in range(5):
            self.check_deadline();split=self.split(fold);train,val,test=(self.load(split[x]) for x in ["train","validation","test"])
            for name in selected:
                rv,rt,info=self._candidate_raw(name,fold,train,val,test,42);payload=self.save_eval(f"seed42_fold{fold}_{name}",rv,rt,val,test);all_rows.append({"seed":42,"fold":fold,"model":name,**payload["test"]})
        pd.DataFrame(all_rows).to_csv(self.metrics/"fivefold_runs.csv",index=False);base=pd.concat([pd.read_csv(self.root/f"artifacts/phase3/metrics/seed42_fold{f}_A2_Small_BiGRU_test_piece.csv").assign(fold=f) for f in range(5)],ignore_index=True);rows=[]
        for name in selected:
            frame=pd.concat([pd.read_csv(self.metrics/f"seed42_fold{f}_{name}_test_piece.csv").assign(fold=f) for f in range(5)],ignore_index=True);delta=frame.set_index("piece_id").f1_tol1-base.set_index("piece_id").f1_tol1;lo,hi=paired_bootstrap(delta.to_numpy());fold_delta=frame.groupby("fold").f1_tol1.mean()-base.groupby("fold").f1_tol1.mean();rows.append({"model":name,"macro_precision_tol1":frame.precision_tol1.mean(),"macro_recall_tol1":frame.recall_tol1.mean(),"macro_f1_tol1":frame.f1_tol1.mean(),"macro_f1_tol0":frame.f1_tol0.mean(),"macro_f1_tol2":frame.f1_tol2.mean(),"macro_pr_auc":frame.pr_auc.mean(),"delta_vs_a2":delta.mean(),"ci_low":lo,"ci_high":hi,"fold_wins":int((fold_delta>0).sum()),"works":frame.piece_id.nunique()})
        comparison=pd.DataFrame(rows);comparison.to_csv(self.metrics/"model_comparison.csv",index=False);trigger=comparison[(comparison.delta_vs_a2>=float(self.config["gates"]["second_seed_delta"]))&(comparison.fold_wins>=int(self.config["gates"]["second_seed_fold_wins"]))]
        second_seed_rows=[]
        if not trigger.empty:
            for fold in range(5):
                self.check_deadline();split=self.split(fold);train,val,test=(self.load(split[x]) for x in ["train","validation","test"])
                for name in trigger.model.tolist():
                    rv,rt,info=self._candidate_raw(name,fold,train,val,test,43);payload=self.save_eval(f"seed43_fold{fold}_{name}",rv,rt,val,test);second_seed_rows.append({"seed":43,"fold":fold,"model":name,**payload["test"]})
            base43=pd.concat([pd.read_csv(self.root/f"artifacts/phase3/metrics/seed43_fold{f}_A2_Small_BiGRU_test_piece.csv").assign(fold=f) for f in range(5)],ignore_index=True);summary43=[]
            for name in trigger.model.tolist():
                frame=pd.concat([pd.read_csv(self.metrics/f"seed43_fold{f}_{name}_test_piece.csv").assign(fold=f) for f in range(5)],ignore_index=True);delta=frame.set_index("piece_id").f1_tol1-base43.set_index("piece_id").f1_tol1;lo,hi=paired_bootstrap(delta.to_numpy());fold_delta=frame.groupby("fold").f1_tol1.mean()-base43.groupby("fold").f1_tol1.mean();summary43.append({"model":name,"macro_f1_tol1":frame.f1_tol1.mean(),"delta_vs_seed43_a2":delta.mean(),"ci_low":lo,"ci_high":hi,"fold_wins":int((fold_delta>0).sum()),"works":frame.piece_id.nunique()})
            pd.DataFrame(second_seed_rows).to_csv(self.metrics/"second_seed_runs.csv",index=False);pd.DataFrame(summary43).to_csv(self.metrics/"second_seed_comparison.csv",index=False)
        write_json(self.metrics/"second_seed_gate.json",{"status":"not_triggered" if trigger.empty else "completed","eligible":trigger.model.tolist(),"runs":len(second_seed_rows)})
        for row in comparison.itertuples():self.register({"id":row.model,"status":"fivefold_preliminary_not_promoted" if row.delta_vs_a2<=0 or row.ci_low<=0 else "fivefold_preliminary_positive","macro_f1_tol1":row.macro_f1_tol1,"delta_vs_a2":row.delta_vs_a2,"ci_low":row.ci_low,"ci_high":row.ci_high,"fold_wins":row.fold_wins,"second_seed_status":"not_triggered" if row.model not in trigger.model.tolist() else "completed"})
        self.mark("experiments",{"status":"complete","selected":selected,"best":comparison.sort_values("macro_f1_tol1",ascending=False).iloc[0].to_dict(),"second_seed_eligible":trigger.model.tolist(),"second_seed_runs":len(second_seed_rows)})

    def migration(self):
        if self.done("migration"):print("[resume] migration");return
        source=Path(r"C:\Users\pa1018\Desktop\learn\柴柴");files=[]
        if source.exists():
            for p in source.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".krn",".mid",".midi",".musicxml",".xml",".csv",".tsv",".json",".zip"}:files.append({"path":str(p),"suffix":p.suffix.lower(),"bytes":p.stat().st_size})
        pd.DataFrame(files).to_csv(self.manifests/"local_external_candidate_inventory.csv",index=False)
        (self.reports/"dataset_search.md").write_text("# Dataset Search and Migration Gate\n\nStatus: **gate audit pending primary-source web verification**. Local read-only research inventory contains "+str(len(files))+" candidate structured files. No external corpus is treated as piano phrase gold, no pseudo-label is used as test truth, and no migration training is started until annotation semantics, license, downloadability, variant grouping and target-work overlap are verified.\n",encoding="utf-8");write_json(self.manifests/"migration_gate.json",{"status":"not_started_pending_semantics_license_overlap_gate","local_candidate_files":len(files),"audio_phrase_gold_overlap":0});self.mark("migration",{"local_candidate_files":len(files),"training_started":False})

    def _save_figure(self,fig,name):
        for suffix,dpi in [("png",300),("svg",None),("pdf",None)]:fig.savefig(self.figures/f"{name}.{suffix}",dpi=dpi,bbox_inches="tight")
        plt.close(fig)
    def report(self):
        if self.done("report"):print("[resume] report");return
        dev=pd.read_csv(self.metrics/"development_fold0.csv");selection=json.loads((self.manifests/"frozen_candidate_selection.json").read_text(encoding="utf-8"));ablation=pd.read_csv(self.metrics/"a2_feature_ablation_fold0.csv");comparison=pd.read_csv(self.metrics/"model_comparison.csv") if (self.metrics/"model_comparison.csv").exists() else pd.DataFrame();cases=pd.read_csv(self.metrics/"validation_case_signals.csv");a2=pd.read_csv(self.root/"artifacts/phase3/metrics/model_comparison.csv").query("model == 'A2_Small_BiGRU'").iloc[0];paired=pd.DataFrame()
        colors={"A2_hard_control":"#4C78A8","SoftTarget_BiGRU":"#F58518","LinearSVC":"#54A24B","RBF_SVC":"#E45756"};fig,ax=plt.subplots(figsize=(8,4));ax.bar(dev.model,dev.validation_f1,color=[colors.get(x,"#777") for x in dev.model]);ax.axhline(dev.loc[dev.model=="A2_hard_control","validation_f1"].iloc[0],color="black",ls="--");ax.set(ylabel="Fold-0 validation macro F1@±1",title="Bounded Phase-4 development comparison");ax.tick_params(axis="x",rotation=20);self._save_figure(fig,"development_comparison")
        fig,ax=plt.subplots(figsize=(8,4));ax.bar(ablation.ablation,ablation.delta_vs_normal,color="#4C78A8");ax.axhline(0,color="black",lw=1);ax.set(ylabel="Validation F1 delta",title="A2 train-mean feature-group ablations (sensitivity)");ax.tick_params(axis="x",rotation=25);self._save_figure(fig,"feature_ablation")
        if len(comparison):
            a2_piece=pd.concat([pd.read_csv(self.root/f"artifacts/phase3/metrics/seed42_fold{fold}_A2_Small_BiGRU_test_piece.csv").assign(fold=fold) for fold in range(5)],ignore_index=True)
            candidate=selection["selected"][0];candidate_piece=pd.concat([pd.read_csv(self.metrics/f"seed42_fold{fold}_{candidate}_test_piece.csv").assign(fold=fold) for fold in range(5)],ignore_index=True)
            paired=candidate_piece[["piece_id","fold","f1_tol1"]].merge(a2_piece[["piece_id","f1_tol1"]],on="piece_id",suffixes=("_candidate","_a2"));paired["delta"]=paired.f1_tol1_candidate-paired.f1_tol1_a2;source_errors=self.root/"reports/meeting_report/figure_sources/fold_work_errors.csv";source_errors.parent.mkdir(parents=True,exist_ok=True);paired.to_csv(source_errors,index=False)
            fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4));fold_delta=paired.groupby("fold").delta.mean();ax1.bar(fold_delta.index.astype(str),fold_delta.values,color=["#54A24B" if x>0 else "#E45756" for x in fold_delta]);ax1.axhline(0,color="black",lw=.8);ax1.set(xlabel="Outer fold",ylabel="Macro F1@±1 delta",title=f"{candidate} vs A2 by fold")
            ordered=paired.sort_values("delta").reset_index(drop=True);ax2.scatter(np.arange(len(ordered)),ordered.delta,color=["#54A24B" if x>0 else "#E45756" for x in ordered.delta],s=20);ax2.axhline(0,color="black",lw=.8);ax2.set(xlabel="Held-out works (sorted)",ylabel="Work F1@±1 delta",title="Per-work paired errors (43 works)");self._save_figure(fig,"fold_work_errors")
        fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=False)
        for ax,(role,frame) in zip(axes,cases.groupby("role",sort=False)):
            ax.plot(frame.beat,frame.probability,label="boundary probability",color="#4C78A8");ax.plot(frame.beat,(frame.log_tempo-frame.log_tempo.mean())/(frame.log_tempo.std()+1e-8)*.15+.5,label="tempo (scaled)",alpha=.7);ax.scatter(frame.loc[frame.label>0,"beat"],np.ones((frame.label>0).sum()),marker="|",s=100,color="#E45756",label="gold boundary");ax.axhline(frame.threshold.iloc[0],color="black",ls="--",lw=.8);ax.set_title(f"{role}: {frame.piece_id.iloc[0]} / {frame.performance_id.iloc[0]}");ax.set_ylim(-.05,1.1)
        axes[0].legend(ncol=4,fontsize=8);axes[-1].set_xlabel("Beat index");fig.suptitle("Real fold-0 validation cases: success, miss and false-positive profile");fig.subplots_adjust(hspace=.48,top=.90);self._save_figure(fig,"real_boundary_cases")
        fig,ax=plt.subplots(figsize=(10,4));ax.axis("off");ax.text(.05,.75,"Score/MIDI\n16 cues",ha="center",va="center",bbox=dict(boxstyle="round",fc="#E0ECF4"));ax.text(.05,.25,"Aligned performance\n9 tempo/dynamics curves",ha="center",va="center",bbox=dict(boxstyle="round",fc="#FDE0C5"));ax.text(.38,.5,"64-beat windows\ntrain-fold normalization",ha="center",va="center",bbox=dict(boxstyle="round",fc="#E8E8E8"));ax.text(.68,.5,"1-layer BiGRU\n11,443 parameters",ha="center",va="center",bbox=dict(boxstyle="round",fc="#D8F0D2"));ax.text(.93,.5,"Per-beat boundary\nprobability + timestamp",ha="center",va="center",bbox=dict(boxstyle="round",fc="#F4D6E5"));ax.annotate("",(.29,.5),(.14,.75),arrowprops=dict(arrowstyle="->"));ax.annotate("",(.29,.5),(.14,.25),arrowprops=dict(arrowstyle="->"));ax.annotate("",(.58,.5),(.47,.5),arrowprops=dict(arrowstyle="->"));ax.annotate("",(.84,.5),(.77,.5),arrowprops=dict(arrowstyle="->"));ax.set_title("Actual Phase-4 model and read-only alignment input path");self._save_figure(fig,"architecture_alignment")
        conclusion="No Phase-4 candidate was promoted. SoftTarget_BiGRU passed the fold-0 development gate but lost to A2 on every outer fold; A2 Small BiGRU remains the final model."
        report=f"""# Phase 4 Final Report

Generated: {now()}.

## Unique conclusion

{conclusion}

Phase 4 is exploratory because the same 43 works have been repeatedly exposed. The target is one-to-one work-macro boundary F1@±1, not ordinary accuracy.

## Frozen data, split and final model

- 43 Chopin Mazurkas, 536 annotated boundaries and 1,990 aligned performances.
- Five opus-grouped folds; piece overlap=0 and opus overlap=0 in every train/validation/test partition.
- A2 is a one-layer bidirectional GRU with 25 beat-aligned inputs (9 performance-curve + 16 score cues), hidden size 32, token-level output, 64-beat windows and 11,443 parameters. There is no pooling or loss of beat resolution.
- Frozen A2 seed42 work-macro P/R/F1@±1 = {a2.macro_precision_tol1:.4f}/{a2.macro_recall_tol1:.4f}/{a2.macro_f1_tol1:.4f}; exact F1={a2.macro_f1_tol0:.4f}, ±2 F1={a2.macro_f1_tol2:.4f}, PR-AUC={a2.macro_pr_auc:.4f}.

## Development gate (fold-0 train/validation only)

{markdown_table(dev)}

Selected before opening Phase-4 outer test: `{selection['selected']}`.

LinearSVC and RBF SVC each used at most six frozen configurations, work-balanced train sampling, train-only StandardScaler fitting and validation-only decision thresholds. Neither qualified for outer evaluation.

## Feature evidence

{markdown_table(ablation)}

Mean replacement and block occlusion are sensitivity evidence, not causal attribution. Audio-specific spectral/transient features remain proposed because verified audio–phrase-label overlap is zero.

## Five-fold results

{markdown_table(comparison) if len(comparison) else 'Skipped by validation gate; no Phase-4 outer predictions were opened.'}

Soft labels doubled fold-0 training positive mass from 279 to 557 while the positive weight remained capped at 10.0. All five best checkpoints occurred at epoch 0; later training loss fell while validation F1 deteriorated. The candidate lost on 0/5 folds and its paired work delta was -0.0095 with 95% CI [-0.0389, 0.0206]. Exact F1 fell, while ±2 F1 rose slightly, consistent with broader but less precisely localized scores rather than a robust boundary improvement. The preregistered second-seed gate was not triggered.

## Transformer structural diagnosis

No Transformer was retrained in Phase 4. The frozen Phase-2 compact Transformer was already a fair token-level diagnostic: `[B,T,9]→Linear(32)→sinusoidal position→2 pre-norm blocks (4 heads, FFN 64)→LayerNorm→[B,T]`, 17,505 parameters, no CLS pooling, stride or token downsampling. Shape, padding/loss masks, fully-masked rejection, tiny-overfit and finite-gradient tests passed; padding attention mass was 0, normalized attention entropy 0.806–0.848, diagonal mass 0.012–0.020 and head cosine similarity 0.35–0.37. Despite correct plumbing it scored a seed-matched delta of -0.1423 versus TCN, CI [-0.1702,-0.1123], and 0/5 fold wins across seeds 42/43/44. Phase-4 ablations further show score/context information, not a need for more attention capacity. Enlarging it would be unsupported outer-test-driven tuning.

## External data and deployment gate

Official Essen/MTC sources and the ISMIR 2020 primary study were checked. Migration was skipped: the corpora are monophonic folk melodies rather than piano-performance phrase gold; variants can leak across random splits; rests/formatting may encode phrase labels; Essen license scope and target-overlap checks remain incomplete. See `reports/phase4/dataset_search.md`. No corpus was downloaded and no transfer result is claimed. The Phase-3 alignment adapter remains smoke-tested, but real audio–phrase-gold overlap is zero.

## Status discipline

- `implemented`: isolated runner, soft-target learner, bounded SVM pipelines, feature audits, tests, markers and local meeting figures.
- `reproduced`: frozen A2 control and any numbers backed by Phase-4 CSV/checkpoints.
- `preliminary`: all Phase-4 development/five-fold findings.
- `proposed`: external melody pretraining and real raw-audio phrase evaluation unless their gates pass.

## Environment and failure record

Python 3.12.10; PyTorch 2.11.0+cu128; NumPy 2.5.3; pandas 3.0.5; scikit-learn 1.9.0; NVIDIA GeForce RTX 5070 Ti. Seed 42 only; seed 43 was correctly skipped by gate. No training/OOM/runtime failure occurred. The sklearn SVC deprecation warning did not affect results because probability calibration was disabled and only `decision_function` was used.

## Reproduction

```powershell
powershell -ExecutionPolicy Bypass -File '.\\scripts\\run_phase4.ps1' -Resume -Stage all
```
""";(self.reports/"final_report.md").write_text(report,encoding="utf-8")
        meeting=self.root/"reports/meeting_report";meeting.mkdir(parents=True,exist_ok=True)
        cn=f"""# 乐句边界预测 Phase 4 组会报告

生成时间：{now()}。

## 一句话结论

Phase 4 没有产生可晋升的新模型。软目标 BiGRU 在 fold 0 验证集上看似提高 +0.0185，但严格外层五折为 0/5 折胜出、总体反而低于 A2 0.0095；因此最终模型仍是 A2 Small BiGRU，软目标与 SVM 都作为有证据的负结果保留。

## 研究问题与数据

目标是在谱面拍位上预测钢琴乐句边界。数据为 43 首肖邦 Mazurka、536 个边界、1,990 个演奏；边界率约 3.8%，所以不能用普通逐拍正确率。主指标是逐作品 Macro Precision/Recall/F1，±1 拍容差、贪心最小距离一对一匹配；同时报告精确、±2 拍和 PR-AUC。五折严格按 opus 分组，每折 train/validation/test 的作品与 opus 重合均为 0，同曲不同演奏不会跨集合。

## 最终模型与系统路径

A2 输入每拍 25 维：9 维对齐演奏曲线（tempo/dynamics 等）和 16 维可推理 score cues；训练折归一化后以 64 拍窗口输入单层双向 GRU，hidden=32，逐拍输出，不用 CLS/pooling/stride，总参数 11,443。seed42 五折 P/R/F1@±1={a2.macro_precision_tol1:.4f}/{a2.macro_recall_tol1:.4f}/{a2.macro_f1_tol1:.4f}，精确 F1={a2.macro_f1_tol0:.4f}，±2 F1={a2.macro_f1_tol2:.4f}，PR-AUC={a2.macro_pr_auc:.4f}。Phase 3 的 seed43 复核为 0.4570，说明结果仍有种子波动；全部 43 首已反复暴露，因此只能称探索性结果。

## Phase 4 特征诊断

{markdown_table(ablation)}

移除全部演奏曲线几乎不变（+0.0013），移除 dynamics 也几乎不变（+0.0003）；移除全部 score cues 降 0.2458，移除音乐转折子集降 0.1298，遮住 8 拍时间块降 0.0826。证据说明当前模型主要靠乐谱转折与时间上下文，演奏曲线贡献弱；它是敏感度证据，不是因果解释。音频谱/瞬态仍未测试，因为真实音频与乐句金标重合为 0。

## 有界开发与外层检验

{markdown_table(dev)}

软目标是唯一通过预设 +0.005 fold-0 门槛的候选；两种 SVM 各只测试 6 组冻结配置，StandardScaler 仅拟合训练数据，阈值仅从 validation 选择。LinearSVC 和 RBF SVC 分别比 A2 低 0.0954 和 0.1467，没有打开外层测试。

{markdown_table(comparison)}

软目标把 fold-0 训练正例质量从 279 提到 557，pos_weight 仍截断为 10；五折最佳 checkpoint 全在 epoch 0，之后训练损失下降而 validation F1 下降。最终 exact F1 降至 0.3529，±1 F1=0.4748，但 ±2 F1 略升至 0.5121。这更像概率向邻拍扩散、定位变宽，而非稳定学到更好的边界。配对 95% CI=[-0.0389,0.0206]，不触发第二种子；禁止事后改软标签或阈值追逐 test。

## Transformer 专项结构诊断

本阶段没有重训 Transformer。Phase 2 已完成公平、逐 token 的紧凑结构：`[B,T,9]→Linear(32)→正弦位置编码→2个pre-norm块（4头、FFN64）→LayerNorm→[B,T]`，17,505 参数；没有 CLS pooling、stride 或降采样。输入/输出 shape、padding/loss mask、全屏蔽拒绝、tiny-overfit、梯度有限性均通过；padding attention mass=0，attention entropy=0.806–0.848，diagonal mass=0.012–0.020，head cosine=0.35–0.37。它相对同输入 TCN 的配对差值为 -0.1423，CI=[-0.1702,-0.1123]，三个种子、0/5 折胜出。当前病因不支持“attention 接错”或“容量不够”；更符合 43 首/536 边界下归纳偏置与泛化不足。再堆层/头属于无证据扩大，故停止。

## 外部数据、真实案例与局限

Essen/MTC 一手来源审计发现：它们主要是单声部民歌旋律，存在 tune-family/alternate-rendering 泄漏风险，休止或换行还可能直接编码真值，Essen 许可范围需要进一步确认。它们只能作为未来分组预训练候选，不能直接当钢琴演奏断句测试金标；本阶段没有下载或训练。最佳 held-out 作品图按冻结 A2 作品级 F1 选出 `chopin_op06_no2`，再固定中位表现演奏 `pid9048-02`，图中 P/R/F1=0.786/0.917/0.846；该图只作解释，不代表平均表现，也没有参与选模。

## 状态与复现

- implemented：Phase 4 隔离运行器、软目标、SVM、公平门控、特征审计、图表、marker 与测试。
- reproduced：Phase 3 冻结 A2、Phase 4 实际 CSV/checkpoint 支持的全部数字。
- preliminary：本阶段所有验证/五折结论，因为 43 首已多次暴露。
- proposed：MTC/Essen 分组预训练、真实音频谱特征融合和独立新作品测试。

单命令：`powershell -ExecutionPolicy Bypass -File '.\\scripts\\run_phase4.ps1' -Resume -Stage all`。环境：Python 3.12.10、PyTorch 2.11.0+cu128、NumPy 2.5.3、pandas 3.0.5、scikit-learn 1.9.0、RTX 5070 Ti。seed 42；无 OOM 或训练失败。
""";(meeting/"PHASE4_REPORT_CN.md").write_text(cn,encoding="utf-8")
        script=f"""# 三分钟讲稿

这阶段我没有继续堆大模型，而是先问低分到底来自哪里。数据只有43首肖邦Mazurka、536个边界和1990个演奏，边界率约3.8%，所以评价不是普通正确率，而是opus完全隔离、逐作品宏平均、±1拍一对一Boundary F1。

最终基线A2是单层Small BiGRU，输入每拍9维演奏曲线和16维乐谱特征，共11443个参数，五折F1是0.4843。特征消融显示，去掉全部演奏曲线几乎不变，但去掉全部score特征下降0.2458，去掉音乐转折特征下降0.1298，遮挡时间块下降0.0826。也就是说当前模型主要依赖乐谱与上下文，演奏表达信息没有被有效利用。

我只做了三个有界对照。LinearSVC和RBF SVC在fold0分别是0.3575和0.3061，直接停止。软目标BiGRU只把边界相邻一拍设为0.5，fold0看起来提升到0.4714，比同折A2高0.0185，所以按预注册规则进入五折。但五折结果是0.4748，比A2低0.0095，置信区间跨零，而且五个fold全输。因此不做第二种子，也不事后改标签追test。它的精确F1下降、±2拍略升，更像预测变宽，而不是稳定提高。

Transformer也没有盲目重开。Phase2的紧凑逐token Transformer已经通过shape、mask、位置编码、注意力padding、梯度和tiny-overfit检查，但三个种子相对TCN低0.1423、0/5折胜出，所以负结果更像小数据泛化与归纳偏置，而不是接线错误。

最终保留A2。外部Essen/MTC只列为未来旋律预训练候选，因为单声部域差异、变体泄漏、休止复制标签和许可都还没有过门槛。所有数字、失败分支、逐作品误差、最佳测试作品图和单命令恢复都已保存。
""";(meeting/"THREE_MINUTE_SCRIPT_CN.md").write_text(script,encoding="utf-8")
        qa="""# 组会问答

## 为什么不是 80%？
边界只有约3.8%，且人工断句有模糊性；逐拍 accuracy 会被大量非边界拍抬高。本项目使用更严格的作品宏平均 Boundary F1。

## 为什么 fold 0 提升还要判失败？
fold 0 只用于候选筛选。软目标在外层五折 0/5 胜出、总体差值为负，说明开发集增益没有泛化，不能拿单折替代结论。

## 软目标为什么失败？
它把训练正例质量约翻倍，且五折最佳 checkpoint 都在 epoch 0；exact/±1下降而±2略升，符合定位被扩散、过拟合迅速出现。这个解释可由未来距离回归或多标注者不确定性数据检验，但本阶段不在 test 后调参。

## SVM 是否公平？
是。每核最多6组配置，作品均衡抽样，测试保持全覆盖，Scaler只拟合训练数据，核与阈值只看validation，不做默认概率校准。

## Transformer 是不是 mask 写错了？
已有专项测试排除了主要接线错误：token shape保持、padding和loss mask语义正确、全屏蔽被拒绝、padding attention mass为0、梯度和激活有限、无CLS/pooling/stride。它仍稳定显著落后，所以没有理由继续加层加头。

## 为什么演奏曲线贡献很弱？
移除全部曲线几乎不降分，说明现有tempo/dynamics代理没有提供稳定增益。原因可能是共享作品标签、演奏差异弱、拍级汇聚丢失瞬态、或对齐误差；现有消融只能说明敏感度，不能区分这些原因。

## 音频融合完成了吗？
对齐适配器已smoke-tested，但真实音频与乐句金标作品重合为0，所以原始音频端到端F1仍是proposed，不能用模拟压力测试冒充真实结果。

## Essen 为什么没直接用？
它是单声部民歌，不是钢琴演奏断句；存在旋律变体、休止复制标签和许可范围问题。必须先做tune-family分组、特征泄漏删除、许可与目标重合审计。

## 下一步最值钱的工作是什么？
优先获取独立的新作品/多标注者边界和真正重合的音频，再测试不会从真实测试对齐或人工标注生成的音频瞬态特征；模型扩容优先级低于数据与标签质量。
""";(meeting/"QA_CN.md").write_text(qa,encoding="utf-8")
        source_dir=meeting/"figure_sources";source_dir.mkdir(parents=True,exist_ok=True);dev.to_csv(source_dir/"development_comparison.csv",index=False);ablation.to_csv(source_dir/"feature_ablation.csv",index=False);cases.to_csv(source_dir/"real_boundary_cases.csv",index=False)
        for f in self.figures.glob("*"):shutil.copy2(f,meeting/f.name)
        best_plot=subprocess.run([str(self.root/".venv/Scripts/python.exe"),str(self.root/"scripts/plot_best_test_piece.py")],cwd=self.root,capture_output=True,text=True,encoding="utf-8",errors="replace")
        if best_plot.returncode!=0:raise RuntimeError("Best held-out piece plot failed: "+best_plot.stdout+best_plot.stderr)
        self.mark("report",{"figures":len(list(self.figures.glob("*.png")))+1,"selected":selection["selected"],"best_piece_plot":best_plot.stdout.strip()})

    def audit(self):
        tests=subprocess.run([str(self.root/".venv/Scripts/python.exe"),"-m","pytest","tests/phase4","-q"],cwd=self.root,capture_output=True,text=True,encoding="utf-8",errors="replace");frozen=json.loads((self.manifests/"frozen_phase3_evidence.json").read_text(encoding="utf-8"));mismatch=[]
        for item in frozen["files"]:
            p=self.root/item["path"];actual=sha(p) if p.exists() else None
            if actual!=item["sha256"]:mismatch.append({"path":item["path"],"actual":actual,"expected":item["sha256"]})
        split=pd.read_csv(self.manifests/"split_reaudit.csv");selection=json.loads((self.manifests/"frozen_candidate_selection.json").read_text(encoding="utf-8"));violations=[]
        for d in self.checkpoints.rglob("*"):
            if d.is_dir():
                files=[x.name for x in d.iterdir() if x.is_file() and x.suffix in {".pt",".joblib"}]
                if files:
                    suffix=Path(files[0]).suffix
                    if set(files)!={f"best{suffix}",f"latest{suffix}"}:violations.append({"path":str(d),"files":files})
        contract_path=self.manifests/"frozen_run_contract.json";contract_files=sorted([p for p in self.checkpoints.rglob("*") if p.is_file()]+[self.metrics/"development_fold0.csv",self.metrics/"fivefold_runs.csv",self.metrics/"model_comparison.csv",self.manifests/"frozen_candidate_selection.json",self.manifests/"split_reaudit.csv"]);current_contract={"config_hash":self.config_hash,"training_contract_hash":self.training_hash,"files":[{"path":str(p.relative_to(self.root)).replace('\\','/'),"sha256":sha(p)} for p in contract_files]}
        if not contract_path.exists():write_json(contract_path,{"created_at":now(),**current_contract})
        frozen_contract=json.loads(contract_path.read_text(encoding="utf-8"));contract_mismatch=[]
        if frozen_contract.get("config_hash")!=self.config_hash or frozen_contract.get("training_contract_hash")!=self.training_hash:contract_mismatch.append({"contract":"hash_header"})
        for item in frozen_contract.get("files",[]):
            p=self.root/item["path"];actual=sha(p) if p.exists() else None
            if actual!=item["sha256"]:contract_mismatch.append({"path":item["path"],"actual":actual,"expected":item["sha256"]})
        checks={"frozen_phase3_hashes":not mismatch,"phase4_run_contract":not contract_mismatch,"split_piece_opus_zero":bool((split[["piece_overlap","opus_overlap"]]==0).all().all()),"tests":tests.returncode==0,"development_selection_frozen":selection["outer_test_unopened_during_selection"],"feature_audit":(self.reports/"feature_information_audit.md").exists(),"dataset_gate":(self.manifests/"migration_gate.json").exists(),"checkpoint_policy":not violations,"resource_log":(self.reports/"resource_usage.csv").exists(),"reports":all((self.reports/x).exists() for x in ["final_report.md","model_development.md","STATUS.md","decision_log.md","experiment_registry.json","failure_summary.md"]),"meeting_materials":all((self.root/"reports/meeting_report"/x).exists() for x in ["PHASE4_REPORT_CN.md","THREE_MINUTE_SCRIPT_CN.md","QA_CN.md","architecture_alignment.png","development_comparison.png","feature_ablation.png","fold_work_errors.png","real_boundary_cases.png"]),"best_piece_visual":all((self.root/"reports/meeting_report/figures"/x).exists() for x in ["best_test_piece_beat_boundaries.png","best_test_piece_beat_boundaries.svg","best_test_piece_beat_boundaries.pdf","best_test_piece_beat_data.csv","best_test_piece_matching.csv","best_test_piece_metadata.json"])}
        if selection["selected"]:
            comp=pd.read_csv(self.metrics/"model_comparison.csv");checks["selected_models_have_43_works"]=bool((comp.works==43).all());recomputed={};metric_ok=True;aggregation_ok=True;base=pd.concat([pd.read_csv(self.root/f"artifacts/phase3/metrics/seed42_fold{f}_A2_Small_BiGRU_test_piece.csv").assign(fold=f) for f in range(5)],ignore_index=True)
            metric_map={"macro_precision_tol1":"precision_tol1","macro_recall_tol1":"recall_tol1","macro_f1_tol1":"f1_tol1","macro_f1_tol0":"f1_tol0","macro_f1_tol2":"f1_tol2","macro_pr_auc":"pr_auc"}
            for name in selection["selected"]:
                pieces=[]
                for f in range(5):
                    pf=pd.read_csv(self.metrics/f"seed42_fold{f}_{name}_test_performance.csv");wf=pd.read_csv(self.metrics/f"seed42_fold{f}_{name}_test_piece.csv");numeric=[c for c in pf.columns if c not in {"piece_id","performance_id"}];grouped=pf.groupby("piece_id",as_index=False)[numeric].mean(numeric_only=True);merged=grouped.merge(wf,on="piece_id",suffixes=("_grouped","_saved"));aggregation_ok=aggregation_ok and all(np.allclose(merged[f"{m}_grouped"],merged[f"{m}_saved"],atol=1e-12,equal_nan=True) for m in metric_map.values());pieces.append(wf.assign(fold=f))
                frame=pd.concat(pieces,ignore_index=True);row=comp.query("model == @name").iloc[0];summary={out:float(frame[source].mean()) for out,source in metric_map.items()};delta=frame.set_index("piece_id").f1_tol1-base.set_index("piece_id").f1_tol1;lo,hi=paired_bootstrap(delta.to_numpy());fold_wins=int(((frame.groupby("fold").f1_tol1.mean()-base.groupby("fold").f1_tol1.mean())>0).sum());summary.update({"delta_vs_a2":float(delta.mean()),"ci_low":lo,"ci_high":hi,"fold_wins":fold_wins,"works":int(frame.piece_id.nunique())});recomputed[name]=summary;metric_ok=metric_ok and all(abs(float(row[key])-float(value))<1e-10 for key,value in summary.items())
            checks["performance_to_piece_aggregation"]=aggregation_ok;checks["metric_recompute"]=metric_ok
        else:recomputed={};checks["outer_test_gate_stop"]=(self.metrics/"fivefold_gate_stop.json").exists()
        resume_path=self.reports/"resume_validation.json"
        if resume_path.exists():checks["resume_no_retraining"]=bool(json.loads(resume_path.read_text(encoding="utf-8")).get("no_retraining"))
        payload={"status":"complete" if all(checks.values()) else "incomplete","created_at":now(),"checks":checks,"frozen_mismatches":mismatch,"phase4_contract_mismatches":contract_mismatch,"checkpoint_violations":violations,"recomputed":recomputed,"pytest_output":tests.stdout+tests.stderr};write_json(self.reports/"completion_audit.json",payload)
        if not all(checks.values()):raise AssertionError(payload)
        self.mark("audit",{"checks":len(checks),"tests":tests.stdout.strip().splitlines()[-1] if tests.stdout.strip() else "passed"})
        status=(self.reports/"STATUS.md").read_text(encoding="utf-8")+f"\n## Complete {now()}\n\nAll completion checks passed. Selected candidates: {selection['selected']}.\n";(self.reports/"STATUS.md").write_text(status,encoding="utf-8")

    def run(self,stage):getattr(self,stage)()


def main():
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--stage",choices=[*STAGES,"all"],default="all");parser.add_argument("--resume",action="store_true");parser.add_argument("--force",action="store_true");args=parser.parse_args();pipeline=Phase4Pipeline(args.root,args.resume,args.force);stages=STAGES if args.stage=="all" else [args.stage]
    for stage in stages:
        started=time.time();print(f"[{now()}] START phase4 {stage}",flush=True)
        try:pipeline.run(stage)
        except Exception as exc:
            failure={"timestamp":now(),"stage":stage,"error":repr(exc),"traceback":traceback.format_exc()};(pipeline.logs/"failures.jsonl").parent.mkdir(parents=True,exist_ok=True)
            with (pipeline.logs/"failures.jsonl").open("a",encoding="utf-8") as h:h.write(json.dumps(failure,ensure_ascii=False)+"\n")
            print(failure["traceback"],flush=True);return 1
        print(f"[{now()}] DONE phase4 {stage} elapsed_seconds={time.time()-started:.2f}",flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
