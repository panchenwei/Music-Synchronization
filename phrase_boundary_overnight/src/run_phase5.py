from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from .phase2_models import evaluate_single_performance
from .phase5_models import ExperimentDeadline, loss_balance, raw_predictions, target_data, train_matched_bigru
from .run_phase3 import markdown_table, paired_bootstrap

STAGES=["init","develop","select","outer","report","audit"]


def now():return datetime.now().astimezone().isoformat(timespec="seconds")
def sha(path:Path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path:Path,payload):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=lambda x:float(x) if isinstance(x,np.generic) else str(x)),encoding="utf-8")


class Phase5Pipeline:
    def __init__(self,root:Path,resume:bool,force:bool):
        self.root=root.resolve();self.resume=resume;self.force=force;self.config=yaml.safe_load((self.root/"configs/phase5/protocol.yaml").read_text(encoding="utf-8"));self.deadline=datetime.fromisoformat(self.config["project"]["experiment_deadline"]);self.reports=self.root/"reports/phase5";self.artifacts=self.root/"artifacts/phase5";self.metrics=self.artifacts/"metrics";self.figures=self.artifacts/"figures";self.checkpoints=self.root/"checkpoints/phase5";self.manifests=self.root/"manifests/phase5";self.logs=self.root/"logs/phase5";self.markers=self.artifacts/"stages"
        for p in [self.reports,self.artifacts,self.metrics,self.figures,self.checkpoints,self.manifests,self.logs,self.markers]:p.mkdir(parents=True,exist_ok=True)
        self.contract=hashlib.sha256((self.root/"configs/phase5/protocol.yaml").read_bytes()+(self.root/"src/phase5_models.py").read_bytes()+(self.root/"src/run_phase5.py").read_bytes()+b"phase5_v2").hexdigest()
    def marker(self,stage):return self.markers/f"{stage}.json"
    def done(self,stage):return self.resume and not self.force and self.marker(stage).exists() and json.loads(self.marker(stage).read_text(encoding="utf-8")).get("contract")==self.contract
    def mark(self,stage,payload):write_json(self.marker(stage),{"stage":stage,"completed_at":now(),"contract":self.contract,"runner_hash":sha(self.root/"src/run_phase5.py"),**payload})
    def split(self,fold):
        frame=pd.read_csv(self.root/self.config["data"]["split_source"]);return {name:frame[(frame.fold==fold)&(frame.split==name)].piece_id.tolist() for name in ["train","validation","test"]}
    def load(self,ids):
        out={}
        for pid in ids:
            with np.load(self.root/self.config["data"]["cache"]/f"{pid}.npz",allow_pickle=False) as z:out[pid]={k:z[k] for k in z.files}
        return out
    @staticmethod
    def specs():return {"score_hard_cap":("score_only","hard","original_capped"),"curves_hard_cap":("curves_only","hard","original_capped"),"combined_hard_cap":("combined","hard","original_capped"),"combined_hard_mass":("combined","hard","mass_balanced"),"combined_soft_cap":("combined","soft","original_capped"),"combined_soft_mass":("combined","soft","mass_balanced"),"combined_hard_moddrop":("combined_moddrop","hard","original_capped")}
    def training_budget(self):
        path=self.manifests/"training_budget.json"
        state=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"maximum_seconds":float(self.config["project"]["maximum_new_training_hours"])*3600,"completed_seconds":0.0,"runs":[],"active":None}
        if state.get("active"):
            started=datetime.fromisoformat(state["active"]["started_at"]);elapsed=max(0.0,(datetime.now().astimezone()-started).total_seconds());state["completed_seconds"]+=elapsed;state["runs"].append({**state["active"],"elapsed_seconds":elapsed,"status":"interrupted_reconciled"});state["active"]=None;write_json(path,state)
        return path,state
    def train_one(self,name,fold,seed,train,val):
        modality,target,weight=self.specs()[name];directory=self.checkpoints/name/f"seed{seed}"/f"fold{fold}";budget_path,budget=self.training_budget();remaining=float(budget["maximum_seconds"])-float(budget["completed_seconds"])
        if remaining<=0:raise ExperimentDeadline("Phase5 cumulative two-hour training budget exhausted")
        started=datetime.now().astimezone();effective_deadline=min(self.deadline,started+timedelta(seconds=remaining));budget["active"]={"name":name,"fold":fold,"seed":seed,"started_at":started.isoformat(timespec="seconds")};write_json(budget_path,budget)
        try:
            model,cn,sn,raw,info=train_matched_bigru(train,val,self.config,directory,effective_deadline,seed,modality,target,weight,self.resume,contract_hash=self.contract)
        except Exception:
            elapsed=(datetime.now().astimezone()-started).total_seconds();budget["completed_seconds"]+=elapsed;budget["runs"].append({**budget["active"],"elapsed_seconds":elapsed,"status":"failed"});budget["active"]=None;write_json(budget_path,budget);raise
        elapsed=(datetime.now().astimezone()-started).total_seconds();budget["completed_seconds"]+=elapsed;budget["runs"].append({**budget["active"],"elapsed_seconds":elapsed,"status":"complete"});budget["active"]=None;write_json(budget_path,budget);info["training_elapsed_seconds"]=elapsed;prefix=f"{name}_seed{seed}_fold{fold}";perf,piece,summary=evaluate_single_performance(raw,val,float(info["threshold"]));perf.to_csv(self.metrics/f"{prefix}_validation_performance.csv",index=False);piece.to_csv(self.metrics/f"{prefix}_validation_piece.csv",index=False);write_json(self.metrics/f"{prefix}_info.json",info);return {"setting":name,"fold":fold,"seed":seed,"modality":modality,"target":target,"weight":weight,"validation_f1_tol0":summary["macro_f1_tol0"],"validation_precision_tol1":summary["macro_precision_tol1"],"validation_recall_tol1":summary["macro_recall_tol1"],"validation_f1_tol1":summary["macro_f1_tol1"],"validation_f1_tol2":summary["macro_f1_tol2"],"validation_pr_auc":summary["macro_pr_auc"],"threshold":info["threshold"],"best_epoch":info["best_epoch"],"last_epoch":info["last_epoch"],"train_minus_validation_f1":info["train_minus_validation_f1"],"probability_variance":info["probability_variance"],"cross_performance_probability_std":info["cross_performance_probability_std"],"prediction_density":info["prediction_density"],"mean_raw_peak_width":info["mean_raw_peak_width"],"curve_gradient":info["input_gradient_curve_mean_abs"],"score_gradient":info["input_gradient_score_mean_abs"],"pos_weight":info["balance"]["pos_weight"],"positive_mass":info["balance"]["positive_mass"],"training_elapsed_seconds":elapsed}
    def init(self):
        if self.done("init"):print("[resume] init");return
        sources=["reports/phase3/final_report.md","reports/phase4/final_report.md","reports/phase4/completion_audit.json","reports/phase4/resume_validation.json","manifests/phase4/frozen_run_contract.json","artifacts/phase2/splits/opus_split_manifest.csv"];write_json(self.manifests/"frozen_prior_evidence.json",{"created_at":now(),"files":[{"path":p,"sha256":sha(self.root/p)} for p in sources],"phase4_correction":"0/5 fold wins means lost on all five folds"});split=pd.read_csv(self.root/self.config["data"]["split_source"]);rows=[]
        for f in range(5):
            parts={s:split[(split.fold==f)&(split.split==s)] for s in ["train","validation","test"]};rows.append({"fold":f,"piece_overlap":sum(len(set(parts[a].piece_id)&set(parts[b].piece_id)) for a,b in [("train","validation"),("train","test"),("validation","test")]),"opus_overlap":sum(len(set(parts[a].opus.astype(str))&set(parts[b].opus.astype(str))) for a,b in [("train","validation"),("train","test"),("validation","test")])})
        pd.DataFrame(rows).to_csv(self.manifests/"split_reaudit.csv",index=False)
        status_path=self.reports/"STATUS.md";decision_path=self.reports/"decision_log.md"
        if not status_path.exists():status_path.write_text(f"# Phase 5 Status\n\nStatus: INIT_COMPLETE\nStarted: {now()}\nTraining deadline: {self.deadline.isoformat()}\n",encoding="utf-8")
        if not decision_path.exists():decision_path.write_text("# Phase 5 Decision Log\n\n## Frozen protocol\n\nFold 0/1 validation and seeds 42/43 are fixed before training. Same-capacity BiGRU controls change only modality or target/weight. Outer tests remain unopened until the gate is frozen.\n",encoding="utf-8")
        self.mark("init",{"frozen_files":len(sources),"zero_overlap":True})
    def develop(self):
        if self.done("develop"):print("[resume] develop");return
        rows=[];names=["score_hard_cap","curves_hard_cap","combined_hard_cap","combined_hard_mass","combined_soft_cap","combined_soft_mass"]
        for fold in self.config["development"]["folds"]:
            split=self.split(fold);train,val=self.load(split["train"]),self.load(split["validation"])
            for seed in self.config["project"]["seeds"]:
                for name in names:rows.append(self.train_one(name,int(fold),int(seed),train,val))
        frame=pd.DataFrame(rows);frame.to_csv(self.metrics/"development_results.csv",index=False);balance=[]
        for fold in self.config["development"]["folds"]:
            train=self.load(self.split(fold)["train"])
            for target in ["hard","soft"]:
                td=target_data(train,target,int(self.config["targets"]["soft_radius_beats"]),float(self.config["targets"]["soft_neighbor_weight"]))
                for weight in ["original_capped","mass_balanced"]:balance.append({"fold":fold,"target":target,"weight":weight,**loss_balance(td,weight,float(self.config["targets"]["original_pos_weight_cap"]))})
        pd.DataFrame(balance).to_csv(self.metrics/"loss_contribution_audit.csv",index=False);self.mark("develop",{"runs":len(rows)})
    def gate_table(self,frame):
        control=frame[frame.setting=="combined_hard_cap"][["fold","seed","validation_f1_tol1"]].rename(columns={"validation_f1_tol1":"control_f1"});merged=frame.merge(control,on=["fold","seed"]);merged["delta"]=merged.validation_f1_tol1-merged.control_f1;rows=[]
        for name,g in merged[merged.setting!="combined_hard_cap"].groupby("setting"):
            fold_delta=g.groupby("fold").delta.mean();rows.append({"setting":name,"mean_validation_f1":g.validation_f1_tol1.mean(),"mean_delta":g.delta.mean(),"fold0_delta":fold_delta.get(0,np.nan),"fold1_delta":fold_delta.get(1,np.nan),"minimum_fold_delta":fold_delta.min(),"eligible":g.delta.mean()>=float(self.config["gate"]["mean_delta_vs_matched_control"]) and fold_delta.min()>=-float(self.config["gate"]["maximum_single_fold_drop"])})
        return pd.DataFrame(rows).sort_values(["eligible","mean_delta"],ascending=[False,False])
    def select(self):
        if self.done("select"):print("[resume] select");return
        frame=pd.read_csv(self.metrics/"development_results.csv");means=frame.groupby("setting").validation_f1_tol1.mean();curve_predictive=means.get("curves_hard_cap",0)>=.20;combined_increment=means.get("combined_hard_cap",0)-means.get("score_hard_cap",0);correction_run=bool(curve_predictive and combined_increment<.005)
        if correction_run:
            rows=[]
            for fold in self.config["development"]["folds"]:
                split=self.split(fold);train,val=self.load(split["train"]),self.load(split["validation"])
                for seed in self.config["project"]["seeds"]:rows.append(self.train_one("combined_hard_moddrop",int(fold),int(seed),train,val))
            frame=pd.concat([frame,pd.DataFrame(rows)],ignore_index=True);frame.to_csv(self.metrics/"development_results.csv",index=False)
        gate=self.gate_table(frame);gate.to_csv(self.metrics/"development_gate.csv",index=False);eligible=gate[gate.eligible].head(int(self.config["gate"]["maximum_outer_candidates"]));payload={"created_at":now(),"outer_test_unopened":True,"criterion":{"mean_delta":self.config["gate"]["mean_delta_vs_matched_control"],"max_single_fold_drop":self.config["gate"]["maximum_single_fold_drop"]},"curve_predictive_threshold":.20,"curve_predictive":curve_predictive,"combined_minus_score":combined_increment,"optional_correction_run":correction_run,"selected":eligible.setting.tolist(),"gate":gate.to_dict("records")};write_json(self.manifests/"frozen_selection.json",payload);self.mark("select",{"selected":payload["selected"],"optional_correction_run":correction_run})
    def outer(self):
        if self.done("outer"):print("[resume] outer");return
        selection=json.loads((self.manifests/"frozen_selection.json").read_text(encoding="utf-8"));selected=selection["selected"]
        if not selected:write_json(self.metrics/"outer_gate_stop.json",{"status":"no_candidate","selected":[]});self.mark("outer",{"status":"gate_stop"});return
        name=selected[0];rows=[]
        for fold in range(5):
            split=self.split(fold);train,val,test=(self.load(split[x]) for x in ["train","validation","test"]);modality,target,weight=self.specs()[name];model,cn,sn,raw_val,info=train_matched_bigru(train,val,self.config,self.checkpoints/name/"seed42"/f"fold{fold}",self.deadline,42,modality,target,weight,self.resume);raw_test=raw_predictions(model,test,cn,sn,next(model.parameters()).device,modality);perf,piece,summary=evaluate_single_performance(raw_test,test,float(info["threshold"]));perf.to_csv(self.metrics/f"{name}_seed42_fold{fold}_test_performance.csv",index=False);piece.to_csv(self.metrics/f"{name}_seed42_fold{fold}_test_piece.csv",index=False);rows.append({"fold":fold,**summary})
        pd.DataFrame(rows).to_csv(self.metrics/"outer_runs.csv",index=False);candidate=pd.concat([pd.read_csv(self.metrics/f"{name}_seed42_fold{f}_test_piece.csv").assign(fold=f) for f in range(5)],ignore_index=True);base=pd.concat([pd.read_csv(self.root/f"artifacts/phase3/metrics/seed42_fold{f}_A2_Small_BiGRU_test_piece.csv").assign(fold=f) for f in range(5)],ignore_index=True);delta=candidate.set_index("piece_id").f1_tol1-base.set_index("piece_id").f1_tol1;lo,hi=paired_bootstrap(delta.to_numpy());fold_delta=candidate.groupby("fold").f1_tol1.mean()-base.groupby("fold").f1_tol1.mean();write_json(self.metrics/"outer_comparison.json",{"setting":name,"macro_precision_tol1":candidate.precision_tol1.mean(),"macro_recall_tol1":candidate.recall_tol1.mean(),"macro_f1_tol1":candidate.f1_tol1.mean(),"macro_f1_tol0":candidate.f1_tol0.mean(),"macro_f1_tol2":candidate.f1_tol2.mean(),"macro_pr_auc":candidate.pr_auc.mean(),"delta_vs_a2":delta.mean(),"ci_low":lo,"ci_high":hi,"fold_wins":int((fold_delta>0).sum()),"works":candidate.piece_id.nunique()});self.mark("outer",{"status":"complete","selected":name})
    def savefig(self,fig,name):
        for ext,dpi in [("png",300),("svg",None),("pdf",None)]:fig.savefig(self.figures/f"{name}.{ext}",dpi=dpi,bbox_inches="tight")
        plt.close(fig)
    def report(self):
        if self.done("report"):print("[resume] report");return
        frame=pd.read_csv(self.metrics/"development_results.csv");gate=pd.read_csv(self.metrics/"development_gate.csv");selection=json.loads((self.manifests/"frozen_selection.json").read_text(encoding="utf-8"));balance=pd.read_csv(self.metrics/"loss_contribution_audit.csv");summary=frame.groupby("setting").agg(f1=("validation_f1_tol1","mean"),f1_std=("validation_f1_tol1","std"),exact=("validation_f1_tol0","mean"),tol2=("validation_f1_tol2","mean"),pr_auc=("validation_pr_auc","mean"),gap=("train_minus_validation_f1","mean"),peak_width=("mean_raw_peak_width","mean"),density=("prediction_density","mean"),best_epoch=("best_epoch","mean")).reset_index();summary.to_csv(self.metrics/"development_summary.csv",index=False);fig,ax=plt.subplots(figsize=(10,4));ax.bar(summary.setting,summary.f1,yerr=summary.f1_std,color="#4C78A8",capsize=3);ax.tick_params(axis="x",rotation=30);ax.set(ylabel="Fold0/1, seed42/43 validation F1@±1",title="Phase 5 matched-capacity diagnostics");self.savefig(fig,"phase5_development")
        fig,ax=plt.subplots(figsize=(8,4));pivot=balance.pivot_table(index=["fold","target"],columns="weight",values="pos_weight").reset_index();x=np.arange(len(pivot));ax.bar(x-.18,pivot.original_capped,.36,label="original capped");ax.bar(x+.18,pivot.mass_balanced,.36,label="mass balanced");ax.set_xticks(x,[f"f{r.fold}-{r.target}" for r in pivot.itertuples()]);ax.set(ylabel="positive weight",title="Target mass changes the uncapped loss balance");ax.legend();self.savefig(fig,"loss_weight_interaction")
        outer=(self.metrics/"outer_comparison.json").read_text(encoding="utf-8") if (self.metrics/"outer_comparison.json").exists() else "Skipped: no candidate met the frozen gate."
        text=f"""# Phase 5 Final Report\n\nGenerated: {now()}. All findings are exploratory.\n\n## Development summary\n\n{markdown_table(summary)}\n\n## Frozen gate\n\n{markdown_table(gate)}\n\nSelected: `{selection['selected']}`. Optional modality-dropout correction run: {selection['optional_correction_run']}. Outer tests were unopened until this manifest was frozen.\n\n## Outer result\n\n{outer}\n\n## Interpretation\n\nThese are from-scratch matched-capacity controls, not test-time mean replacement. `epoch 0` is the first completed training epoch. The original capped target weighting and mass-balanced weighting have numerically distinct positive coefficients. Uni-modal bias papers motivate the diagnostic but do not prove it occurs here.\n\n## Correction\n\nPhase 4's `0/5 fold wins` means the soft model lost on all five folds.\n\n## Reproduction\n\n`powershell -ExecutionPolicy Bypass -File '.\\scripts\\run_phase5.ps1' -Resume -Stage all`\n""";(self.reports/"final_report.md").write_text(text,encoding="utf-8");meeting=self.root/"reports/meeting_report";meeting.mkdir(parents=True,exist_ok=True);cn="# Phase 5：同架构模态与损失诊断\n\n"+text+"\n\n一手依据：[PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.BCEWithLogitsLoss.html)、[Du et al. 2023](https://proceedings.mlr.press/v202/du23e/du23e.pdf)、[Zhang et al. 2024](https://proceedings.mlr.press/v235/zhang24aa.html)。文献只用于提出可证伪假设，不替代本项目实验。\n";(meeting/"PHASE5_REPORT_CN.md").write_text(cn,encoding="utf-8");source=meeting/"figure_sources";source.mkdir(parents=True,exist_ok=True);summary.to_csv(source/"phase5_development_summary.csv",index=False);gate.to_csv(source/"phase5_gate.csv",index=False);balance.to_csv(source/"phase5_loss_contribution.csv",index=False)
        for p in self.figures.glob("*"):
            (meeting/p.name).write_bytes(p.read_bytes())
        self.mark("report",{"selected":selection["selected"],"figures":2})
    def audit(self):
        tests=subprocess.run([str(self.root/".venv/Scripts/python.exe"),"-m","pytest","tests/phase5","-q"],cwd=self.root,capture_output=True,text=True,encoding="utf-8",errors="replace");frozen=json.loads((self.manifests/"frozen_prior_evidence.json").read_text(encoding="utf-8"));mismatch=[]
        for item in frozen["files"]:
            p=self.root/item["path"]
            if not p.exists() or sha(p)!=item["sha256"]:mismatch.append(item["path"])
        selection=json.loads((self.manifests/"frozen_selection.json").read_text(encoding="utf-8"));split=pd.read_csv(self.manifests/"split_reaudit.csv");violations=[]
        for d in self.checkpoints.rglob("fold*"):
            if d.is_dir() and set(p.name for p in d.glob("*.pt"))!={"best.pt","latest.pt"}:violations.append(str(d))
        checks={"prior_hashes":not mismatch,"zero_overlap":bool((split[["piece_overlap","opus_overlap"]]==0).all().all()),"tests":tests.returncode==0,"outer_unopened_during_selection":selection["outer_test_unopened"],"weight_contrasts_distinct":bool((pd.read_csv(self.metrics/"loss_contribution_audit.csv").groupby(["fold","target"]).pos_weight.nunique()==2).all()),"checkpoint_policy":not violations,"reports":(self.reports/"final_report.md").exists() and (self.root/"reports/meeting_report/PHASE5_REPORT_CN.md").exists()}
        if selection["selected"]:checks["outer_43_works"]=json.loads((self.metrics/"outer_comparison.json").read_text(encoding="utf-8"))["works"]==43
        else:checks["gate_stop_recorded"]=(self.metrics/"outer_gate_stop.json").exists()
        payload={"status":"complete" if all(checks.values()) else "incomplete","created_at":now(),"checks":checks,"prior_mismatches":mismatch,"checkpoint_violations":violations,"pytest":tests.stdout+tests.stderr};write_json(self.reports/"completion_audit.json",payload)
        if not all(checks.values()):raise AssertionError(payload)
        self.mark("audit",{"checks":len(checks)});(self.reports/"STATUS.md").write_text((self.reports/"STATUS.md").read_text(encoding="utf-8")+f"\n## Complete {now()}\n\nChecks: {len(checks)}; selected: {selection['selected']}.\n",encoding="utf-8")
    def run(self,stage):getattr(self,stage)()


def main():
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--stage",choices=[*STAGES,"all"],default="all");p.add_argument("--resume",action="store_true");p.add_argument("--force",action="store_true");a=p.parse_args();runner=Phase5Pipeline(a.root,a.resume,a.force)
    for stage in STAGES if a.stage=="all" else [a.stage]:
        started=time.time();print(f"[{now()}] START phase5 {stage}",flush=True)
        try:runner.run(stage)
        except Exception as exc:
            failure={"timestamp":now(),"stage":stage,"error":repr(exc),"traceback":traceback.format_exc()};(runner.logs/"failures.jsonl").parent.mkdir(parents=True,exist_ok=True)
            with (runner.logs/"failures.jsonl").open("a",encoding="utf-8") as h:h.write(json.dumps(failure,ensure_ascii=False)+"\n")
            print(failure["traceback"],flush=True);return 1
        print(f"[{now()}] DONE phase5 {stage} elapsed={time.time()-started:.2f}",flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
