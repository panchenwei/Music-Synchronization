"""Replay original training unchanged, then compare local weight averages."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import score_context_study as engine
from . import run_recurrence_depth_study as source
from .recurrence_depth_models import make_model
from .three_round_round2 import split_ids
from .local_context_study import metrics,predictions
from .phase2_models import choose_single_threshold
from .context_inference_audit_v2 import save_raw,error
from .audit_external_stem_transfer import checked_raw
from .recurrence_message_probe import mix_probabilities
from .interstart_decoder import fit_prior
from . import run_halo_decoder_composition as dec

ROOT=engine.ROOT;OUT=ROOT/'reports/trajectory_average_study';ART=ROOT/'artifacts/trajectory_average_study'
COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']

def average_states(states):
    assert len(states)>0 and all(list(s)==list(states[0]) for s in states)
    assert all(torch.is_floating_point(t) for s in states for t in s.values())
    return {k:torch.stack([s[k] for s in states]).mean(0) for k in states[0]}

def prepare():
    for p in (OUT,ART/'metrics',ART/'checkpoints',ART/'snapshots',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(engine.read(source.OUT/'contract.json')['hashes'])
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'tests/test_trajectory_average.py',ROOT/'src/interstart_decoder.py',ROOT/'src/recurrence_message_probe.py',ROOT/'src/run_halo_decoder_composition.py'):hashes[str(p)]=engine.sha(p)
    for f in (0,1):
        for s in (42,43):
            for fn in ('best.pt','latest.pt'):
                p=source.ART/'checkpoints'/f'C3_seed{s}_fold{f}'/fn;hashes[str(p)]=engine.sha(p)
            for k,folder in [('C3',source.ART),('G',ROOT/'artifacts/current_gru_study')]:
                for suffix in ('.json','_predictions.csv.gz'):
                    p=folder/'metrics'/f'{k}_seed{s}_fold{f}{suffix}';hashes[str(p)]=engine.sha(p)
    for p in (ROOT/'artifacts/recurrence_mean_control/graphs').glob('*.npz'):hashes[str(p)]=engine.sha(p)
    assert all(engine.sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert engine.read(OUT/'contract.json')['contract']==digest
    engine.write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    if not (OUT/'STATE.json').exists():engine.write(OUT/'STATE.json',dict(status='ready',seconds=0,completed=[],pid=None))
    return digest

def replay(fold,seed,contract):
    b=engine.read(ROOT/'reports/research_continuation_20260915/BUDGET.json');assert b['observed_used_percent']<b['stop_new_runs_used_percent']
    target=ART/'snapshots'/f'C3_seed{seed}_fold{fold}';target.mkdir(exist_ok=True)
    holder={};original_adamw=torch.optim.AdamW
    def factory(kind,seed):
        holder['model']=make_model('C3',seed);return holder['model']
    def optimizer(*args,**kwargs):
        opt=original_adamw(*args,**kwargs)
        def hook(opt,args,kwargs):
            step=int(opt.state[opt.param_groups[0]['params'][0]]['step'].item())
            if step%25==0:
                state={k:v.detach().cpu().clone() for k,v in holder['model'].state_dict().items()}
                torch.save(state,target/f'step{step}.pt')
        opt.register_step_post_hook(hook);return opt
    engine.OUT=OUT;engine.ART=ART;engine.CAP=1800.;engine.dataset=source.dataset;engine.make_model=factory
    try:
        torch.optim.AdamW=optimizer;engine.train_one('C3',fold,seed,contract)
    finally:torch.optim.AdamW=original_adamw
    assert all((target/f'step{t}.pt').exists() for t in range(25,301,25))
    for fn in ('best.pt','latest.pt'):
        a=torch.load(source.ART/'checkpoints'/f'C3_seed{seed}_fold{fold}'/fn,map_location='cpu',weights_only=False)
        z=torch.load(ART/'checkpoints'/f'C3_seed{seed}_fold{fold}'/fn,map_location='cpu',weights_only=False)
        assert a['step']==z['step'] and a['sampler']==z['sampler'];assert a['optimizer']['param_groups']==z['optimizer']['param_groups']
        for k,v in a['model'].items():torch.testing.assert_close(v,z['model'][k],atol=0,rtol=0)
        for i,values in a['optimizer']['state'].items():
            for k,v in values.items():torch.testing.assert_close(v,z['optimizer']['state'][i][k],atol=0,rtol=0)
        assert torch.equal(a['rng'],z['rng']) and all(torch.equal(u,v) for u,v in zip(a['cuda_rng'],z['cuda_rng']))
    print('EXACT ORIGINAL TRAJECTORY',fold,seed,flush=True)

def candidate(fold,seed,contract):
    name=f'A_seed{seed}_fold{fold}';res=ART/'metrics'/f'{name}.json'
    if res.exists():assert engine.read(res)['contract']==contract;return
    ids=split_ids(fold);tr=source.dataset(ids['train'],'C3');va=source.dataset(ids['validation'],'C3');norm=engine.normalizer(tr)
    model=make_model('C3',seed).cuda();assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    path=ART/'snapshots'/f'C3_seed{seed}_fold{fold}';best=-1;history=[];beststate=None;beststep=None
    for step in range(50,301,50):
        times=[50] if step==50 else [step-50,step-25,step]
        states=[torch.load(path/f'step{t}.pt',map_location='cpu',weights_only=True) for t in times]
        state=average_states(states);model.load_state_dict(state);raw=predictions(model,va,norm,torch.device('cuda'))
        threshold,_=choose_single_threshold(raw,va,engine.GRID);score=metrics(raw,va,threshold)[2]
        history.append(dict(step=step,averaged_steps=times,**score))
        if score['macro_f1_tol1']>best+1e-9:best=score['macro_f1_tol1'];beststate=state;beststep=step
        print(name,step,'F1',score['macro_f1_tol1'],flush=True)
    dest=ART/'checkpoints'/name;dest.mkdir(exist_ok=True)
    torch.save(dict(model=beststate,step=beststep,history=history,contract=contract,mean=norm.mean,std=norm.std),dest/'best.pt')
    model.load_state_dict(beststate);raw=predictions(model,va,norm,torch.device('cuda'));threshold=next(h['threshold'] for h in history if h['step']==beststep)
    perfs,pieces,score=metrics(raw,va,threshold);save_raw(raw,ART/'metrics'/f'{name}_predictions.csv.gz',va)
    perfs.to_csv(ART/'metrics'/f'{name}_performances.csv',index=False);pieces.to_csv(ART/'metrics'/f'{name}_pieces.csv',index=False)
    engine.write(res,dict(kind='A',seed=seed,fold=fold,best_step=beststep,history=history,contract=contract,**score))

def evaluate(contract):
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder';rows=[];checks=[]
    for fold in (0,1):
        ids=split_ids(fold);tr=source.dataset(ids['train'],'C3');va=source.dataset(ids['validation'],'C3');norm=engine.normalizer(tr);prior=fit_prior(tr)
        for seed in (42,43):
            raw={}
            for k,folder in [('A',ART),('C3',source.ART),('G',ROOT/'artifacts/current_gru_study')]:raw[k]=checked_raw(pd.read_csv(folder/'metrics'/f'{k}_seed{seed}_fold{fold}_predictions.csv.gz'),va)
            cp=torch.load(ART/'checkpoints'/f'A_seed{seed}_fold{fold}'/'best.pt',map_location='cpu',weights_only=False)
            model=make_model('C3',seed).cuda();model.load_state_dict(cp['model']);rp=predictions(model,va,norm,torch.device('cuda'));err=error(raw['A'],rp);assert err<2e-4
            meta=engine.read(ART/'metrics'/f'A_seed{seed}_fold{fold}.json');assert max(abs(metrics(rp,va,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
            checks.append(dict(seed=seed,fold=fold,replay_error=err))
            for name,k in [('EA','A'),('EC','C3')]:raw[name]={p:{q:(a+raw['G'][p][q])*.5 for q,a in pp.items()} for p,pp in raw[k].items()}
            for k in ('A','C3','EA','EC'):
                threshold,_=choose_single_threshold(raw[k],va,engine.GRID);score=metrics(raw[k],va,threshold)[2]
                rows.append(dict(kind=k,seed=seed,fold=fold,policy='raw',**score));adjusted={}
                for p,pp in raw[k].items():
                    with np.load(ROOT/'artifacts/recurrence_mean_control/graphs'/f'{p}.npz') as z:g=z['M']
                    adjusted[p]={q:mix_probabilities(a,g) for q,a in pp.items()}
                score=dec.evaluate(adjusted,va,threshold,prior,1.,f'{k}_seed{seed}_fold{fold}_M10',contract)
                rows.append(dict(kind=k,seed=seed,fold=fold,policy='M10',**score))
    df=pd.DataFrame(rows);df.to_csv(OUT/'results.csv',index=False);df.groupby(['kind','policy'])[COLS].mean().to_csv(OUT/'means.csv');comp=[]
    for a,b in [('A','C3'),('EA','EC')]:
        for pol in ('raw','M10'):
            x=df[(df.kind==a)&(df.policy==pol)].set_index(['fold','seed']);y=df[(df.kind==b)&(df.policy==pol)].set_index(['fold','seed']);d=x[COLS]-y[COLS]
            comp.append(dict(candidate=a,control=b,policy=pol,**d.mean().to_dict(),positive=int((d.macro_f1_tol1>0).sum()),passed=bool(d.macro_f1_tol1.mean()>=.015 and (d.macro_f1_tol1>0).sum()>=3 and d.macro_f1_tol0.mean()>=0 and d.raw_ap.mean()>=0)))
    hashes=engine.read(OUT/'contract.json')['hashes'];assert all(engine.sha(p)==h for p,h in hashes.items())
    engine.write(OUT/'comparisons.json',comp);engine.write(OUT/'audit.json',dict(status='complete',original_trajectories_exact=4,averaged_candidates=4,original_checkpoints_preserved=True,outer_test_used=False,replays=checks,hashes_unchanged=True))
    engine.write(OUT/'STATE.json',dict(status='complete',pid=None));print(df.groupby(['kind','policy'])[COLS].mean().to_string())

def main():
    torch.set_num_threads(2);contract=prepare()
    for fold in (0,1):
        for seed in (42,43):replay(fold,seed,contract);candidate(fold,seed,contract)
    evaluate(contract)

if __name__=='__main__':main()
