from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .phase2_models import choose_single_threshold, evaluate_single_performance, nms_probabilities
from .phase6_models import context_difference_rows, exposure_audit, full_predictions, load_phase5_bigru, sliding_predictions, train_step_budget
from .phase6_multiscale import SCALES, augment_dataset
from .run_phase3 import markdown_table, paired_bootstrap


STAGES=["init","r1","r2","select","r3","outer","external","report","audit"]


def now():return datetime.now().astimezone().isoformat(timespec="seconds")
def sha(path:Path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path:Path,payload):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=lambda x:float(x) if isinstance(x,np.generic) else str(x)),encoding="utf-8")


class Phase6Pipeline:
    def __init__(self,root:Path,resume:bool,force:bool):
        self.root=root.resolve();self.resume=resume;self.force=force;self.config=yaml.safe_load((self.root/"configs/phase6/protocol.yaml").read_text(encoding="utf-8"));self.deadline=datetime.fromisoformat(self.config["project"]["experiment_deadline"]);self.reports=self.root/"reports/phase6";self.artifacts=self.root/"artifacts/phase6";self.metrics=self.artifacts/"metrics";self.checkpoints=self.root/"checkpoints/phase6";self.manifests=self.root/"manifests/phase6";self.logs=self.root/"logs/phase6";self.markers=self.artifacts/"stages"
        for p in [self.reports,self.artifacts,self.metrics,self.checkpoints,self.manifests,self.logs,self.markers]:p.mkdir(parents=True,exist_ok=True)
        self.contract=hashlib.sha256((self.root/"configs/phase6/protocol.yaml").read_bytes()+(self.root/"src/phase6_models.py").read_bytes()+(self.root/"src/run_phase6.py").read_bytes()+b"phase6_v1").hexdigest()
    def marker(self,stage):return self.markers/f"{stage}.json"
    def done(self,stage):return self.resume and not self.force and self.marker(stage).exists() and json.loads(self.marker(stage).read_text(encoding="utf-8")).get("contract")==self.contract
    def mark(self,stage,payload):write_json(self.marker(stage),{"stage":stage,"completed_at":now(),"contract":self.contract,**payload})
    def split(self,fold):
        frame=pd.read_csv(self.root/self.config["data"]["split_source"]);return {name:frame[(frame.fold==fold)&(frame.split==name)].piece_id.tolist() for name in ["train","validation","test"]}
    def load(self,ids):
        out={}
        for pid in ids:
            with np.load(self.root/self.config["data"]["cache"]/f"{pid}.npz",allow_pickle=False) as z:out[pid]={k:z[k] for k in z.files}
        return out
    def budget(self):
        path=self.manifests/"training_budget.json";state=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"maximum_seconds":float(self.config["project"]["maximum_new_training_hours"])*3600,"completed_seconds":0.0,"runs":[],"active":None}
        if state.get("active"):
            elapsed=max(0.0,(datetime.now().astimezone()-datetime.fromisoformat(state["active"]["started_at"])).total_seconds());state["completed_seconds"]+=elapsed;state["runs"].append({**state["active"],"elapsed_seconds":elapsed,"status":"interrupted_reconciled"});state["active"]=None;write_json(path,state)
        return path,state
    def train_one(self,name,mode,fold,seed,weight_decay,train,val):
        budget_path,budget=self.budget();remaining=float(budget["maximum_seconds"])-float(budget["completed_seconds"])
        if remaining<=0:raise RuntimeError("Phase6 two-hour training budget exhausted")
        started=datetime.now().astimezone();budget["active"]={"name":name,"mode":mode,"fold":fold,"seed":seed,"started_at":started.isoformat(timespec="seconds")};write_json(budget_path,budget);effective=min(self.deadline,started+timedelta(seconds=remaining))
        try:model,cn,sn,raw,info=train_step_budget(train,val,self.config,self.checkpoints/name/f"seed{seed}"/f"fold{fold}",effective,seed,mode,weight_decay,self.resume,self.contract)
        except Exception:
            elapsed=(datetime.now().astimezone()-started).total_seconds();budget["completed_seconds"]+=elapsed;budget["runs"].append({**budget["active"],"elapsed_seconds":elapsed,"status":"failed"});budget["active"]=None;write_json(budget_path,budget);raise
        elapsed=(datetime.now().astimezone()-started).total_seconds();budget["completed_seconds"]+=elapsed;budget["runs"].append({**budget["active"],"elapsed_seconds":elapsed,"status":"complete"});budget["active"]=None;write_json(budget_path,budget);perf,piece,summary=evaluate_single_performance(raw,val,info["threshold"]);prefix=f"{name}_seed{seed}_fold{fold}";perf.to_csv(self.metrics/f"{prefix}_validation_performance.csv",index=False);piece.to_csv(self.metrics/f"{prefix}_validation_piece.csv",index=False);write_json(self.metrics/f"{prefix}_info.json",{**info,"elapsed_seconds":elapsed});return {"setting":name,"mode":mode,"fold":fold,"seed":seed,"weight_decay":weight_decay,"validation_precision_tol1":summary["macro_precision_tol1"],"validation_recall_tol1":summary["macro_recall_tol1"],"validation_f1_tol1":summary["macro_f1_tol1"],"validation_f1_tol0":summary["macro_f1_tol0"],"validation_f1_tol2":summary["macro_f1_tol2"],"validation_pr_auc":summary["macro_pr_auc"],"fixed_f1_tol1":info["fixed"]["macro_f1_tol1"],"threshold":info["threshold"],"best_step":info["best_step"],"maximum_steps":info["maximum_steps"],"validation_frequency":info["validation_frequency"],"old_steps_per_epoch":info["old_steps_per_epoch"],"train_minus_validation_f1":info["train_minus_validation_f1"],"elapsed_seconds":elapsed}
    def init(self):
        if self.done("init"):print("[resume] init");return
        prior=["reports/phase5/final_report.md","reports/phase5/completion_audit.json","reports/phase5/resume_validation.json","manifests/phase5/frozen_selection.json","artifacts/phase2/splits/opus_split_manifest.csv"];write_json(self.manifests/"frozen_phase5_evidence.json",{"created_at":now(),"files":[{"path":p,"sha256":sha(self.root/p)} for p in prior]})
        split=pd.read_csv(self.root/self.config["data"]["split_source"]);overlap=[];piece_rows=[];summaries=[]
        for fold in range(5):
            parts={s:split[(split.fold==fold)&(split.split==s)] for s in ["train","validation","test"]};overlap.append({"fold":fold,"piece_overlap":sum(len(set(parts[a].piece_id)&set(parts[b].piece_id)) for a,b in [("train","validation"),("train","test"),("validation","test")]),"opus_overlap":sum(len(set(parts[a].opus.astype(str))&set(parts[b].opus.astype(str))) for a,b in [("train","validation"),("train","test"),("validation","test")])})
        for fold in self.config["development"]["folds"]:
            rows,summary=exposure_audit(self.load(self.split(fold)["train"]),int(self.config["training"]["batch_size"]),int(self.config["model"]["window_beats"]),int(self.config["model"]["stride_beats"]));piece_rows.extend([{"fold":fold,**r} for r in rows]);summaries.append({"fold":fold,**summary})
        pd.DataFrame(overlap).to_csv(self.manifests/"split_reaudit.csv",index=False);pd.DataFrame(piece_rows).to_csv(self.metrics/"training_exposure_by_piece.csv",index=False);pd.DataFrame(summaries).to_csv(self.metrics/"training_exposure_summary.csv",index=False)
        answer="# Phase6 supervisor answers on Phase5 evidence\n\n## 1. First-epoch exposure\n\n"+markdown_table(pd.DataFrame(summaries))+"\n\nEach independent score boundary is repeated once for every performance and every overlapping window that covers its beat. Phase5 validates only after the complete epoch; therefore the earliest saved validation point cannot exclude an intra-epoch rise and fall. Evidence: `src/phase5_models.py::train_matched_bigru`, `MultiModalWindowDataset`, the Phase5 `*_info.json` histories, and the exposure CSVs above.\n\n## 2. Full versus 64-beat inference\n\nNot measured before Phase6. No numerical claim is made here; R1 will update this section from the same checkpoints.\n\n## 3. Falsified and open hypotheses\n\nFalsified: complete modality collapse (combined exceeds both from-scratch unimodal controls); simple capped-vs-mass loss weighting as a sufficient fix; soft-target widening as a promotion-worthy fix. Still open: train/inference context mismatch and optimizer-step overexposure caused by performance×overlap enumeration with only epoch-end validation. Minimal discriminating experiment: same initialization, architecture, hard labels, step budget and validation frequency; compare old window enumeration with piece→performance→window balanced sampling. Falsifier: mean delta < +.015 or a fold mean drop >.020.\n";(self.reports/"supervisor_answers_phase5.md").write_text(answer,encoding="utf-8");(self.reports/"decision_log.md").write_text("# Phase6 Decision Log\n\n## Frozen development protocol\n\nR1 uses no retraining. R2 changes only sampling under matched steps and validation opportunities. Gate is mean +0.015 and no fold below -0.020; outer test remains unopened. Soft weighting, SVM and Transformer are excluded. Absolute experiment deadline is 10:00 Asia/Shanghai.\n",encoding="utf-8");(self.reports/"STATUS.md").write_text(f"# Phase6 Status\n\nINIT_COMPLETE {now()}\nExperiment deadline: {self.deadline.isoformat()}\n",encoding="utf-8");self.mark("init",{"frozen_files":len(prior),"zero_overlap":True,"development_folds":2})
    def r1(self):
        if self.done("r1"):print("[resume] r1");return
        summaries=[];edge_rows=[];piece_rows=[]
        for fold in self.config["development"]["folds"]:
            val=self.load(self.split(fold)["validation"])
            for seed in self.config["project"]["seeds"]:
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu");checkpoint=self.root/f"checkpoints/phase5/combined_hard_cap/seed{seed}/fold{fold}/best.pt";model,cn,sn,state=load_phase5_bigru(checkpoint,device);full=full_predictions(model,val,cn,sn,device);sliding=sliding_predictions(model,val,cn,sn,device,64,32);original=float(state["threshold"]);full_thr,_=choose_single_threshold(full,val,self.config["evaluation"]["threshold_grid"]);slide_thr,_=choose_single_threshold(sliding,val,self.config["evaluation"]["threshold_grid"]);_,full_piece_original,full_original=evaluate_single_performance(full,val,original);_,slide_piece_original,slide_original=evaluate_single_performance(sliding,val,original);_,full_piece,full_selected=evaluate_single_performance(full,val,full_thr);_,slide_piece,slide_selected=evaluate_single_performance(sliding,val,slide_thr);diff=[]
                for pid in full:
                    for perf in full[pid]:diff.extend(np.abs(full[pid][perf]-sliding[pid][perf]).tolist())
                summaries.append({"fold":fold,"seed":seed,"original_threshold":original,"full_reselected_threshold":full_thr,"sliding_reselected_threshold":slide_thr,"probability_mae":float(np.mean(diff)),"probability_p95":float(np.quantile(diff,.95)),"probability_max":float(np.max(diff)),"full_f1_original_threshold":full_original["macro_f1_tol1"],"sliding_f1_original_threshold":slide_original["macro_f1_tol1"],"delta_original_threshold":slide_original["macro_f1_tol1"]-full_original["macro_f1_tol1"],"full_f1_reselected":full_selected["macro_f1_tol1"],"sliding_f1_reselected":slide_selected["macro_f1_tol1"],"delta_reselected":slide_selected["macro_f1_tol1"]-full_selected["macro_f1_tol1"]});edge_rows.extend([{"fold":fold,"seed":seed,**r} for r in context_difference_rows(full,sliding,val,64,32)]);merged=full_piece[["piece_id","beats","f1_tol1"]].rename(columns={"f1_tol1":"full_f1"}).merge(slide_piece[["piece_id","f1_tol1"]].rename(columns={"f1_tol1":"sliding_f1"}),on="piece_id");merged["delta"]=merged.sliding_f1-merged.full_f1;piece_rows.extend([{"fold":fold,"seed":seed,**r} for r in merged.to_dict("records")])
        summary=pd.DataFrame(summaries);summary.to_csv(self.metrics/"r1_context_summary.csv",index=False);pd.DataFrame(edge_rows).to_csv(self.metrics/"r1_edge_differences.csv",index=False);pieces=pd.DataFrame(piece_rows);pieces.to_csv(self.metrics/"r1_piece_differences.csv",index=False);answer=self.reports/"supervisor_answers_phase5.md";text=answer.read_text(encoding="utf-8");text=text.replace("Not measured before Phase6. No numerical claim is made here; R1 will update this section from the same checkpoints.",f"Measured in R1 on the same four Phase5 checkpoints. Mean probability MAE={summary.probability_mae.mean():.6f}, p95={summary.probability_p95.mean():.6f}; sliding-minus-full F1 at the original threshold={summary.delta_original_threshold.mean():+.6f}, after separate validation threshold selection={summary.delta_reselected.mean():+.6f}. Edge-bin and per-piece/length evidence are in `r1_edge_differences.csv` and `r1_piece_differences.csv`; no test set was read.");answer.write_text(text,encoding="utf-8");self.mark("r1",{"runs":len(summary),"mean_probability_mae":summary.probability_mae.mean(),"mean_delta_reselected":summary.delta_reselected.mean()})
    def r2(self):
        if self.done("r2"):print("[resume] r2");return
        rows=[]
        for fold in self.config["development"]["folds"]:
            split=self.split(fold);train,val=self.load(split["train"]),self.load(split["validation"])
            for seed in self.config["project"]["seeds"]:
                for name,mode in [("old_step_control","old_window"),("piece_balanced","piece_balanced")]:rows.append(self.train_one(name,mode,int(fold),int(seed),float(self.config["training"]["weight_decay"]),train,val))
        pd.DataFrame(rows).to_csv(self.metrics/"r2_sampling_results.csv",index=False);self.mark("r2",{"runs":len(rows)})
    def gate(self,frame):
        control=frame[frame.setting=="old_step_control"][["fold","seed","validation_f1_tol1"]].rename(columns={"validation_f1_tol1":"control_f1"});merged=frame.merge(control,on=["fold","seed"]);merged["delta"]=merged.validation_f1_tol1-merged.control_f1;rows=[]
        for name,g in merged[merged.setting!="old_step_control"].groupby("setting"):
            fd=g.groupby("fold").delta.mean();rows.append({"setting":name,"mean_f1":g.validation_f1_tol1.mean(),"mean_delta":g.delta.mean(),"fold0_delta":fd.get(0,np.nan),"fold1_delta":fd.get(1,np.nan),"minimum_fold_delta":fd.min(),"eligible":bool(g.delta.mean()>=float(self.config["gate"]["mean_delta_vs_matched_control"]) and fd.min()>=-float(self.config["gate"]["maximum_single_fold_drop"]))})
        return pd.DataFrame(rows).sort_values(["eligible","mean_delta"],ascending=[False,False])
    def select(self):
        if self.done("select"):print("[resume] select");return
        frame=pd.read_csv(self.metrics/"r2_sampling_results.csv");gate=self.gate(frame);correction=False
        balanced=frame[frame.setting=="piece_balanced"]
        if not gate.eligible.any() and balanced.train_minus_validation_f1.mean()>=float(self.config["conditional_regularization"]["trigger_mean_train_validation_gap"]):
            correction=True;rows=[]
            for fold in self.config["development"]["folds"]:
                split=self.split(fold);train,val=self.load(split["train"]),self.load(split["validation"])
                for seed in self.config["project"]["seeds"]:rows.append(self.train_one("piece_balanced_wd10","piece_balanced",int(fold),int(seed),float(self.config["conditional_regularization"]["weight_decay"]),train,val))
            frame=pd.concat([frame,pd.DataFrame(rows)],ignore_index=True);frame.to_csv(self.metrics/"r2_sampling_results.csv",index=False);gate=self.gate(frame)
        gate.to_csv(self.metrics/"development_gate.csv",index=False);eligible=gate[gate.eligible].head(1);write_json(self.manifests/"frozen_selection.json",{"created_at":now(),"outer_test_unopened":True,"selected":eligible.setting.tolist(),"correction_run":correction,"gate":gate.to_dict("records")});self.mark("select",{"selected":eligible.setting.tolist(),"correction_run":correction})
    def r3(self):
        if self.done("r3"):print("[resume] r3");return
        selection=json.loads((self.manifests/"frozen_selection.json").read_text(encoding="utf-8"))
        if not selection["selected"]:
            write_json(self.metrics/"r3_gate_stop.json",{"status":"no_r2_control_selected"});self.mark("r3",{"status":"gate_stop"});return
        rows=[]
        for fold in self.config["development"]["folds"]:
            split=self.split(fold);base_train,base_val=self.load(split["train"]),self.load(split["validation"])
            for seed in self.config["project"]["seeds"]:
                for name,group in [("r3_tempo_multiscale","tempo"),("r3_dynamics_multiscale","dynamics"),("r3_both_multiscale","both")]:
                    train,val=augment_dataset(base_train,group),augment_dataset(base_val,group)
                    row=self.train_one(name,"piece_balanced",int(fold),int(seed),float(self.config["training"]["weight_decay"]),train,val);row["feature_group"]=group;rows.append(row)
        frame=pd.DataFrame(rows);frame.to_csv(self.metrics/"r3_multiscale_results.csv",index=False)
        control=pd.read_csv(self.metrics/"r2_sampling_results.csv");control=control[control.setting=="piece_balanced"][["fold","seed","validation_f1_tol1"]].rename(columns={"validation_f1_tol1":"control_f1"});comparison=frame.merge(control,on=["fold","seed"]);comparison["delta_vs_r2_control"]=comparison.validation_f1_tol1-comparison.control_f1;comparison.to_csv(self.metrics/"r3_multiscale_comparison.csv",index=False)
        sample=augment_dataset(self.load(self.split(0)["train"][:1]),"both");item=next(iter(sample.values()));quality=item["phase6_multiscale_quality"]
        write_json(self.metrics/"r3_feature_audit.json",{"feature_names":[str(x) for x in item["curve_feature_names"][-6:]],"scales":list(SCALES),"input_shape_example":list(item["curves"].shape),"quality_shape_example":list(quality.shape),"finite":bool(np.isfinite(item["curves"]).all()),"quality_valid_fraction":float(quality.mean()),"offline_future_context":True,"dynamics_is_proxy_not_audio_energy":True,"normalizer_fit":"training split only","outer_use":"none; post-selection diagnostic is non-promotable"})
        self.mark("r3",{"status":"complete","runs":len(frame),"best_mean_delta":float(comparison.groupby("setting").delta_vs_r2_control.mean().max()),"outer_use":"none"})
    def outer(self):
        if self.done("outer"):print("[resume] outer");return
        selection=json.loads((self.manifests/"frozen_selection.json").read_text(encoding="utf-8"));selected=selection["selected"]
        if not selected:write_json(self.metrics/"outer_gate_stop.json",{"status":"no_candidate","selected":[]});self.mark("outer",{"status":"gate_stop"});return
        name=selected[0];mode="piece_balanced" if name.startswith("piece_balanced") else "old_window";wd=float(self.config["conditional_regularization"]["weight_decay"] if name.endswith("wd10") else self.config["training"]["weight_decay"]);rows=[];pieces=[]
        for fold in range(5):
            split=self.split(fold);train,val,test=(self.load(split[x]) for x in ["train","validation","test"]);self.train_one(name,mode,fold,42,wd,train,val);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model,cn,sn,state=load_phase5_bigru(self.checkpoints/name/"seed42"/f"fold{fold}/best.pt",device);raw=sliding_predictions(model,test,cn,sn,device,64,32);perf,piece,summary=evaluate_single_performance(raw,test,float(state["threshold"]));piece["fold"]=fold;piece.to_csv(self.metrics/f"{name}_seed42_fold{fold}_test_piece.csv",index=False);rows.append({"fold":fold,**summary});pieces.append(piece)
        pd.DataFrame(rows).to_csv(self.metrics/"outer_runs.csv",index=False);candidate=pd.concat(pieces);base=pd.concat([pd.read_csv(self.root/f"artifacts/phase3/metrics/seed42_fold{f}_A2_Small_BiGRU_test_piece.csv").assign(fold=f) for f in range(5)]);delta=candidate.set_index("piece_id").f1_tol1-base.set_index("piece_id").f1_tol1;lo,hi=paired_bootstrap(delta.to_numpy());write_json(self.metrics/"outer_comparison.json",{"setting":name,"macro_f1_tol1":candidate.f1_tol1.mean(),"delta_vs_a2":delta.mean(),"ci_low":lo,"ci_high":hi,"works":candidate.piece_id.nunique()});self.mark("outer",{"status":"complete","selected":name})
    def external(self):
        if self.done("external"):print("[resume] external");return
        audit_path=self.artifacts/"external/mtc_ann_sample_audit.json"
        if audit_path.exists():
            audit=json.loads(audit_path.read_text(encoding="utf-8"));self.mark("external",{"status":audit["status"],"collection":audit["collection"],"items":audit["items"],"events":audit["events"],"phrase_boundaries":audit["phrase_boundaries"],"families":audit["unique_tune_families"],"training_performed":audit["training_performed"]})
        else:
            if not (self.reports/"external_data_audit.md").exists():(self.reports/"external_data_audit.md").write_text("# External data audit\n\nOfficial sample was not available. No data downloaded or trained.\n",encoding="utf-8")
            self.mark("external",{"status":"no_sample","training_performed":False})
    def report(self):
        if self.done("report"):print("[resume] report");return
        r1=pd.read_csv(self.metrics/"r1_context_summary.csv");r2=pd.read_csv(self.metrics/"r2_sampling_results.csv");r3=pd.read_csv(self.metrics/"r3_multiscale_comparison.csv");gate=pd.read_csv(self.metrics/"development_gate.csv");selection=json.loads((self.manifests/"frozen_selection.json").read_text(encoding="utf-8"));outer=json.loads((self.metrics/"outer_comparison.json").read_text(encoding="utf-8"));budget=json.loads((self.manifests/"training_budget.json").read_text(encoding="utf-8"));summary=r2.groupby("setting").agg(precision=("validation_precision_tol1","mean"),recall=("validation_recall_tol1","mean"),f1=("validation_f1_tol1","mean"),pr_auc=("validation_pr_auc","mean"),fixed_f1=("fixed_f1_tol1","mean"),gap=("train_minus_validation_f1","mean"),best_step=("best_step","mean")).reset_index();summary.to_csv(self.metrics/"r2_summary.csv",index=False);r3summary=r3.groupby("setting").agg(precision=("validation_precision_tol1","mean"),recall=("validation_recall_tol1","mean"),f1=("validation_f1_tol1","mean"),pr_auc=("validation_pr_auc","mean"),delta=("delta_vs_r2_control","mean"),gap=("train_minus_validation_f1","mean")).reset_index();r3summary.to_csv(self.metrics/"r3_summary.csv",index=False)
        text=f"""# Phase6 最终报告：上下文、采样与多尺度表示闭环

生成时间：{now()}。研究状态：**preliminary / exploratory**；不把同一 43 首作品上的迭代开发写成确认性结论。

## 结论

R1 否证了“训练 64 拍而整首推理”是主要低分原因。R2 的作品→演奏→窗口均衡采样是唯一通过预冻结开发门槛的改变：开发 F1@±1 从 {summary.loc[summary.setting=='old_step_control','f1'].iloc[0]:.4f} 提高到 {summary.loc[summary.setting=='piece_balanced','f1'].iloc[0]:.4f}，增量 {gate.iloc[0].mean_delta:+.4f}。冻结后 seed42 五折得到 P/R/F1@±1=0.4415/0.6030/{outer['macro_f1_tol1']:.4f}，相对 Phase3 A2 为 {outer['delta_vs_a2']:+.4f}，作品配对 95% CI [{outer['ci_low']:.4f}, {outer['ci_high']:.4f}]。CI 跨 0，因此只算有希望的初步结果；当前确认保留模型仍是 Phase3 A2，不宣称稳定超越。

## 数据、标签与零泄漏

- 43 首肖邦玛祖卡、1,990 个演奏、14,153 个监督 beat、536 个边界；结构独立单位是 43 首作品，不是 1,990 个演奏。
- 五折按 opus 分组，`split_reaudit.csv` 的每折 piece overlap=0、opus overlap=0；同一作品的所有演奏只在一个 split。
- 输入与标签逐 beat 一一对应；窗口只重叠，不 stride/pooling 降采样。loss mask 与 padding validity 分离。
- 单演奏先评价，再按作品宏平均；主容差 ±1 beat，另报 exact 和 ±2。一对一匹配禁止一个预测重复命中多个边界。

## R1：上下文诊断（无重训）

同四个 Phase5 checkpoint 比较整首与 64-beat/stride-32 三角权重滑窗。概率 MAE={r1.probability_mae.mean():.6f}，原阈值 F1 差={r1.delta_original_threshold.mean():+.6f}，各自只在 validation 重选阈值后差={r1.delta_reselected.mean():+.6f}。概率差在 0–8/8–16/16–33 拍离窗边区域依次约 0.000694/0.001822/0.004160，反而不是边缘最大；长度与增量相关约 −0.048。故该假设被本地证据否证。

## R2：等步数采样诊断

{markdown_table(summary)}

控制与候选使用相同初始化、hard labels、Small BiGRU、最多两个旧 epoch 的 optimizer steps，并按旧 epoch 四等分验证；step0 只作未训练诊断。作品均衡对四个 fold×seed 单元均为正，fold0/fold1 平均增量分别 {gate.iloc[0].fold0_delta:+.4f}/{gate.iloc[0].fold1_delta:+.4f}，通过 mean≥+0.015 且任一 fold 不低于−0.020 的冻结门槛。没有触发 weight-decay×10，也没有做搜索。

## R3：多尺度表示（冻结后诊断，不可晋级）

{markdown_table(r3summary)}

在 tempo、dynamics proxy 上分别加入 4/8/16 拍前后段均值绝对差。tempo-only 最好，四个单元均为正，均值 +0.0137；但低于原 +0.015 门槛。dynamics-only +0.0098；合并 +0.0090，且有一个单元 −0.0007。输入从 25 维变为单组 28、双组 31，参数从 11,443 变为 12,025/12,607。结论是“多尺度 tempo 有小而一致的验证信号”，不是新最终模型；不继续堆特征、不打开第二个 outer。

## 五折 outer 与失败案例

冻结候选 piece-balanced 的五折 F1@±1 为 0.5140/0.4519/0.4747/0.4988/0.5468；相对 A2 为 −0.0045/+0.0233/+0.0160/−0.0025/+0.0158，即 3/5 folds 胜。43 首中 25 胜、1 平、17 负。最差退化包括 op59 no1 −0.1386、op41 no4 −0.0753、op06 no3 −0.0659；最大改善包括 op41 no3 +0.2703、op07 no1 +0.1028。模型仍偏高 recall、较低 precision，且作品异质性明显。

## 为什么“正确率”不高

这里不是普通 accuracy：边界仅约占 beat 的 3.8%，全预测非边界会得到很高 accuracy 却没有研究价值。主要证据是有效独立样本只有 43 首/536 边界、每个结构标签被许多同作品演奏与重叠窗口反复曝光、硬边界存在音乐学歧义、不同作品分布变化，以及公开 tempo/dynamics 曲线在真实部署中依赖自动对齐。R2 提升证明重复曝光/作品贡献失衡确实是可修复病因之一；R3 未过门槛说明仅扩大局部曲线尺度不是充分修复。

文献只作为动机：van Kranenburg 的旋律边界研究强调局部节奏线索；FMP novelty 教程说明 kernel/scale 会改变可见结构尺度；Sagawa 等说明组鲁棒训练需配合正则，不能把 GroupDRO 当自动修复；MS-TCN 的平滑针对连续动作段，而本项目目标是稀疏尖峰，不能直接照搬。对应一手入口：

- https://program.ismir2020.net/static/final_papers/226.pdf
- https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S4_NoveltySegmentation.html
- https://arxiv.org/abs/1911.08731
- https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html

## Transformer 专项结论

本轮没有再训练 Transformer。Phase2 已有可复核的 17,505 参数紧凑 Transformer：逐 token 输出、正弦位置编码、正确 padding/loss mask、无 CLS/pooling/stride，tiny-overfit=1.0，attention/梯度有限；在相同作品级五折及 seeds42/43/44 下相对 TCN 的作品配对差 −0.1423，95% CI [−0.1702,−0.1123]，0/5 fold wins。现有证据不指向 mask、位置编码或分辨率 bug，继续加层/头违反受控诊断原则。因此 Transformer 保留为 reproduced negative result，未强行扩大。

## 外部数据与真实推理边界

官方 MTC-ANN 2.0.1 sample smoke：10 首、495 事件、50 个 phrase endpoints、8 个 tune families，family metadata 10/10；50 个边界中 15 个落在休止事件，故未来 transfer 必须 family 分组并做 rest-cue 泄漏消融。本轮没有外部训练。完整 Zenodo sequence 下载两次返回 504，保存失败证据但未当数据使用。

当前曲线来自公开 MazurkaBL beat-synchronous tempo/dynamics，不含测试 phrase labels。真正未见 raw-WAV 演奏仍需自动 score–audio alignment 前端；本地真实 audio–phrase-gold 重合为 0，未达到扩展门槛，所以音频融合与真实对齐鲁棒性仍为 proposed/skipped，而不是失败或已实现。

## 状态纪律

- **implemented**：R1 上下文对照、R2 等步数采样、R3 三组多尺度消融、预算/checkpoint/resume、MTC sample parser 与审计。
- **reproduced**：本文 CSV/checkpoint 支持的 Phase6 数值、冻结 Phase3 A2、Phase2 Transformer 负结果。
- **preliminary**：piece-balanced 五折 +0.0099 与 R3 validation 小增益；CI 或门槛不支持确认性提升。
- **proposed**：新作品确认集、自动对齐 raw-WAV 端到端评估、family-safe 外部预训练、真实音频融合。

## 资源、失败与复现

Phase6 实际训练 {budget['completed_seconds']/60:.2f} 分钟/上限 120 分钟；单进程，seeds 42/43，无 OOM、无超参搜索。两次工程故障均已记录：缺 `tabulate` 后改用内置 Markdown formatter；恢复 GPU checkpoint 时 RNG ByteTensor 设备错误，在读取 outer test 前修复并恢复。未购买额度、未兑换 reset。

单命令恢复/复现：

```powershell
powershell -ExecutionPolicy Bypass -File '.\\scripts\\run_phase6.ps1' -Stage all -Resume
```

外部样本审计单命令：

```powershell
.\\.venv\\Scripts\\python.exe .\\scripts\\audit_mtc_sample.py
```

详细证据见 `supervisor_answers_phase5.md`、`external_data_audit.md`、`decision_log.md`、`completion_audit.json`、`training_budget.json` 和所有 metrics CSV。
""";(self.reports/"final_report.md").write_text(text,encoding="utf-8");self.mark("report",{"selected":selection["selected"],"outer_f1":outer["macro_f1_tol1"],"outer_delta":outer["delta_vs_a2"],"r3_best_delta":float(r3summary.delta.max())})
    def audit(self):
        tests=subprocess.run([str(self.root/".venv/Scripts/python.exe"),"-m","pytest","-q"],cwd=self.root,capture_output=True,text=True,encoding="utf-8",errors="replace");frozen=json.loads((self.manifests/"frozen_phase5_evidence.json").read_text(encoding="utf-8"));mismatch=[x["path"] for x in frozen["files"] if not (self.root/x["path"]).exists() or sha(self.root/x["path"])!=x["sha256"]];overlap=pd.read_csv(self.manifests/"split_reaudit.csv");selection=json.loads((self.manifests/"frozen_selection.json").read_text(encoding="utf-8"));violations=[str(d) for d in self.checkpoints.rglob("fold*") if d.is_dir() and set(p.name for p in d.glob("*.pt"))!={"best.pt","latest.pt"}];budget=json.loads((self.manifests/"training_budget.json").read_text(encoding="utf-8"));external=json.loads((self.artifacts/"external/mtc_ann_sample_audit.json").read_text(encoding="utf-8"));checks={"phase5_hashes":not mismatch,"zero_overlap":bool((overlap[["piece_overlap","opus_overlap"]]==0).all().all()),"all_tests":tests.returncode==0,"outer_unopened_during_selection":selection["outer_test_unopened"],"checkpoint_policy":not violations,"r3_complete":self.marker("r3").exists(),"external_smoke":external["status"]=="smoke_pass" and not external["training_performed"],"training_budget":budget["completed_seconds"]<=budget["maximum_seconds"] and budget["active"] is None,"reports":all((self.reports/x).exists() for x in ["final_report.md","external_data_audit.md","supervisor_answers_phase5.md","decision_log.md"]) };write_json(self.reports/"completion_audit.json",{"status":"complete" if all(checks.values()) else "incomplete","created_at":now(),"checks":checks,"mismatches":mismatch,"checkpoint_violations":violations,"training_seconds":budget["completed_seconds"],"pytest":tests.stdout+tests.stderr});
        if not all(checks.values()):raise AssertionError(checks)
        self.mark("audit",{"checks":len(checks)});(self.reports/"STATUS.md").write_text((self.reports/"STATUS.md").read_text(encoding="utf-8")+f"\nCOMPLETE {now()} selected={selection['selected']}\n",encoding="utf-8")
    def run(self,stage):getattr(self,stage)()


def main():
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--stage",choices=[*STAGES,"all"],default="all");p.add_argument("--resume",action="store_true");p.add_argument("--force",action="store_true");a=p.parse_args();runner=Phase6Pipeline(a.root,a.resume,a.force)
    for stage in STAGES if a.stage=="all" else [a.stage]:
        started=time.time();print(f"[{now()}] START phase6 {stage}",flush=True)
        try:runner.run(stage)
        except Exception as exc:
            failure={"timestamp":now(),"stage":stage,"error":repr(exc),"traceback":traceback.format_exc()};
            with (runner.logs/"failures.jsonl").open("a",encoding="utf-8") as h:h.write(json.dumps(failure,ensure_ascii=False)+"\n")
            print(failure["traceback"],flush=True);return 1
        print(f"[{now()}] DONE phase6 {stage} elapsed={time.time()-started:.2f}",flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
