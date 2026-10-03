"""Actual 300->500 BiGRU continuation, preserving optimizer/sampler/RNG."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import current_gru_study as source
from .score_context_study import ROOT,read,write,sha,normalizer,GRID
from .current_gru_models import make_model
from .recurrence_duration_core import restore,update,improves,selected_step
from .window_exposure_audit import exposure
from .three_round_round2 import split_ids,checkpoint_threshold
from .phase7_models import CurvePieceBalancedSampler
from .phase6_models import positive_weight
from .phase2_models import choose_single_threshold
from .local_context_study import predictions,metrics
from .context_inference_audit_v2 import save_raw,error
from .audit_external_stem_transfer import checked_raw
from .interstart_decoder import fit_prior
from .recurrence_message_probe import mix_probabilities
from . import run_halo_decoder_composition as dec

OUT=ROOT/'reports/gru_500_study';ART=ROOT/'artifacts/gru_500_study'
DEVICE=torch.device('cuda');COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']
CELLS=['best_300','terminal_300','best_500','terminal_500']
def load(p):return torch.load(p,map_location=DEVICE,weights_only=False)

def prepare():
    assert read(ROOT/'reports/trajectory_average_study/audit.json')['status']=='complete'
    for p in (OUT,ART/'runs',ART/'checkpoints',ART/'metrics',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(source.OUT/'contract.json')['hashes'])
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'src/recurrence_duration_core.py',ROOT/'tests/test_recurrence_duration_core.py',ROOT/'tests/test_gru_500_restore.py',ROOT/'src/window_exposure_audit.py',ROOT/'src/run_halo_decoder_composition.py'):hashes[str(p)]=sha(p)
    for f in (0,1):
        ids=split_ids(f);assert not set(ids['train'])&set(ids['validation'])
        manifest=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv');part=manifest[manifest.fold==f]
        assert not set(part[part.split=='train'].opus)&set(part[part.split=='validation'].opus)
        for s in (42,43):
            for fn in ('best.pt','latest.pt'):
                p=source.ART/'checkpoints'/f'G_seed{s}_fold{f}'/fn;hashes[str(p)]=sha(p)
            for folder,k in [(source.ART,'G'),(ROOT/'artifacts/recurrence_depth_study','C3')]:
                for suffix in ('.json','_predictions.csv.gz'):
                    p=folder/'metrics'/f'{k}_seed{s}_fold{f}{suffix}';hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items());digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));return digest

def train_one(fold,seed,contract):
    run=f'G_seed{seed}_fold{fold}';done=ART/'runs'/f'{run}.json'
    if done.exists():assert read(done)['contract']==contract;return
    budget=read(ROOT/'reports/research_continuation_20260915/BUDGET.json');assert budget['observed_used_percent']<budget['stop_new_runs_used_percent']
    ids=split_ids(fold);tr=source.dataset(ids['train'],'G');va=source.dataset(ids['validation'],'G');norm=normalizer(tr)
    model=make_model('G',seed).to(DEVICE);assert sum(p.numel() for p in model.parameters())==6005
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);sampler=CurvePieceBalancedSampler(tr,norm,64,32,seed)
    crit=torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(positive_weight(tr,10),device=DEVICE))
    original=load(source.ART/'checkpoints'/run/'latest.pt');assert original['step']==300
    np.testing.assert_array_equal(original['mean'],norm.mean);np.testing.assert_array_equal(original['std'],norm.std)
    exp300=exposure(tr,seed,300);assert original['sampler']==exp300['sampler']
    restore(original,model,opt,sampler);a=update(model,opt,sampler,crit,DEVICE);state=copy.deepcopy(model.state_dict());opta=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone();cuda=[v.clone() for v in torch.cuda.get_rng_state_all()];samp=sampler.state()
    restore(original,model,opt,sampler);b=update(model,opt,sampler,crit,DEVICE)
    assert a==b and samp==sampler.state() and torch.equal(rng,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    for i,vals in opta['state'].items():
        for k,v in vals.items():torch.testing.assert_close(v,opt.state_dict()['state'][i][k],atol=0,rtol=0)
    write(OUT/f'{run}_restore_probe.json',dict(contract=contract,two_restored_gpu_steps_exact=True,optimizer_exact=True,source_step=300))
    dest=ART/'checkpoints'/run;dest.mkdir(exist_ok=True)
    if not (dest/'latest.pt').exists():
        for cell,fn in [('best_300','best.pt'),('terminal_300','latest.pt')]:
            cp=load(source.ART/'checkpoints'/run/fn);cp['origin_contract']=cp['contract'];cp['contract']=contract;cp['origin_seconds']=cp['seconds'];cp['seconds']=0.
            torch.save(cp,dest/f'{cell}.pt')
        torch.save(load(dest/'best_300.pt'),dest/'best.pt');torch.save(load(dest/'terminal_300.pt'),dest/'latest.pt')
    cp=load(dest/'latest.pt');assert cp['contract']==contract and 300<=cp['step']<=500
    restore(cp,model,opt,sampler);step=cp['step'];history=cp['history'];best_score=cp['best_score'];prior=cp['seconds'];start=time.monotonic()
    write(OUT/'STATE.json',dict(status='training',run=run,pid=os.getpid(),step=step))
    def snap():return dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),step=step,history=history,best_score=best_score,seconds=prior+time.monotonic()-start,kind='G',fold=fold,seed=seed,contract=contract,mean=norm.mean,std=norm.std,origin_contract=original['contract'])
    while step<500:
        if prior+time.monotonic()-start>600:torch.save(snap(),dest/'latest.pt');raise TimeoutError(run)
        loss,grad=update(model,opt,sampler,crit,DEVICE);step+=1
        if step%50==0:
            raw=predictions(model,va,norm,DEVICE);threshold,_=choose_single_threshold(raw,va,GRID);score=metrics(raw,va,threshold)[2];history.append(dict(step=step,loss=loss,grad=grad,**score))
            if improves(score['macro_f1_tol1'],best_score):best_score=score['macro_f1_tol1'];torch.save(snap(),dest/'best.pt')
            print(run,step,'F1',score['macro_f1_tol1'],flush=True)
        if step%25==0:torch.save(snap(),dest/'latest.pt')
    last=load(dest/'latest.pt');best=load(dest/'best.pt');assert selected_step(last['history'])==best['step']
    torch.save(last,dest/'terminal_500.pt');torch.save(best,dest/'best_500.pt')
    exp500=exposure(tr,seed,500);assert last['sampler']==exp500['sampler']
    write(OUT/f'{run}_exposure.json',dict(at300=exp300,at500=exp500,source_final_sampler_exact=True))
    write(done,dict(run=run,fold=fold,seed=seed,contract=contract,seconds=prior+time.monotonic()-start,best_step=best['step']))

def audit(contract):
    dec.ART=ART/'decoder';dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);rows=[];checks=[];hashes={}
    write(OUT/'STATE.json',dict(status='auditing',pid=os.getpid()))
    for fold in (0,1):
        ids=split_ids(fold);tr=source.dataset(ids['train'],'G');va=source.dataset(ids['validation'],'G');norm=normalizer(tr);prior=fit_prior(tr)
        for seed in (42,43):
            run=f'G_seed{seed}_fold{fold}';cnn=checked_raw(pd.read_csv(ROOT/'artifacts/recurrence_depth_study/metrics'/f'C3_seed{seed}_fold{fold}_predictions.csv.gz'),va)
            for cell in CELLS:
                cp_path=ART/'checkpoints'/run/f'{cell}.pt';hashes[str(cp_path)]=sha(cp_path);cp=load(cp_path)
                assert cp['contract']==contract;model=make_model('G',seed).to(DEVICE);model.load_state_dict(cp['model']);threshold=checkpoint_threshold(cp)
                raw=predictions(model,va,norm,DEVICE);path=ART/'metrics'/f'{run}_{cell}_predictions.csv.gz'
                if not path.exists():save_raw(raw,path,va)
                cached=checked_raw(pd.read_csv(path),va);err=error(raw,cached);assert err<2e-4
                score=metrics(cached,va,threshold)[2];expected=next(h for h in cp['history'] if h['step']==cp['step'])
                assert max(abs(score[c]-expected[c]) for c in COLS)<1e-10
                if cell=='best_300':
                    old=read(source.ART/'metrics'/f'{run}.json');assert max(abs(score[c]-old[c]) for c in COLS)<1e-10
                train_score=metrics(predictions(model,tr,norm,DEVICE),tr,threshold)[2]
                checks.append(dict(run=run,cell=cell,step=cp['step'],replay_error=err,train_f1=train_score['macro_f1_tol1'],dev_f1=score['macro_f1_tol1']))
                eraw={p:{q:(a+cnn[p][q])*.5 for q,a in pp.items()} for p,pp in cached.items()}
                for kind,probs in [('G',cached),('E',eraw)]:
                    th,_=choose_single_threshold(probs,va,GRID);score=metrics(probs,va,th)[2]
                    rows.append(dict(kind=kind,fold=fold,seed=seed,cell=cell,policy='raw',selected_step=cp['step'],**score));adjusted={}
                    for p,pp in probs.items():
                        with np.load(ROOT/'artifacts/recurrence_mean_control/graphs'/f'{p}.npz') as z:g=z['M']
                        adjusted[p]={q:mix_probabilities(a,g) for q,a in pp.items()}
                    score=dec.evaluate(adjusted,va,th,prior,1.,f'{kind}_seed{seed}_fold{fold}_{cell}_M10',contract)
                    rows.append(dict(kind=kind,fold=fold,seed=seed,cell=cell,policy='M10',selected_step=cp['step'],**score))
                pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'results.csv',index=False);frame.groupby(['kind','cell','policy'])[COLS].mean().to_csv(OUT/'means.csv');comp=[]
    for kind in ('G','E'):
        for policy in ('raw','M10'):
            for select in ('best','terminal'):
                df=frame[(frame.kind==kind)&(frame.policy==policy)];a=df[df.cell==f'{select}_500'].set_index(['fold','seed']);b=df[df.cell==f'{select}_300'].set_index(['fold','seed']);d=a[COLS]-b[COLS]
                comp.append(dict(kind=kind,policy=policy,selection=select,**d.mean().to_dict(),positive=int((d.macro_f1_tol1>0).sum()),new_best=int((a.selected_step>300).sum()),passed=bool(select=='best' and d.macro_f1_tol1.mean()>=.015 and (d.macro_f1_tol1>0).sum()>=3 and d.macro_f1_tol0.mean()>=0 and d.raw_ap.mean()>=0 and (a.selected_step>300).sum()>=3)))
    assert all(sha(p)==h for p,h in hashes.items());frozen=read(OUT/'contract.json')['hashes'];assert all(sha(p)==h for p,h in frozen.items())
    write(OUT/'comparisons.json',comp);write(OUT/'audit.json',dict(status='complete',continued_runs=4,evaluated_states=16,decode_cells=32,source_and_output_hashes_unchanged=True,restored_gpu_probes=4,outer_test_used=False,checkpoint_hashes=hashes))
    write(OUT/'STATE.json',dict(status='complete',pid=None));print(frame.groupby(['kind','cell','policy'])[COLS].mean().to_string())

def main():
    torch.set_num_threads(2);contract=prepare()
    for fold in (0,1):
        for seed in (42,43):train_one(fold,seed,contract)
    audit(contract)

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(time=time.time(),traceback=traceback.format_exc()));write(OUT/'STATE.json',dict(status='failed',pid=None));raise
