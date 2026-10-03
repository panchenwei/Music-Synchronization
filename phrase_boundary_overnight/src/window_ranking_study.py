"""Isolated fixed-C3 pair-ranking experiment; never edits frozen studies."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from . import score_context_study as base
from . import run_recurrence_depth_study as source
from .recurrence_depth_models import make_model
from .local_context_study import predictions, metrics
from .phase2_models import choose_single_threshold
from .three_round_round2 import split_ids, checkpoint_threshold
from .context_inference_audit_v2 import save_raw, error
from .audit_external_stem_transfer import checked_raw
from .interstart_decoder import fit_prior
from .recurrence_message_probe import mix_probabilities
from . import run_halo_decoder_composition as dec

ROOT=base.ROOT
OUT=ROOT/'reports/window_ranking_study'
ART=ROOT/'artifacts/window_ranking_study'
COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']

def rank_loss(logits, labels, mask):
    valid=mask.bool()
    pairs=(labels.bool() & valid)[:,:,None] & ((~labels.bool()) & valid)[:,None,:]
    # rows are positives, columns are negatives; no stochastic sampling.
    losses=F.softplus(logits[:,None,:]-logits[:,:,None])
    count=pairs.sum((1,2)); present=count>0
    per=(losses*pairs).sum((1,2))/count.clamp_min(1)
    return (per*present).sum()/present.sum().clamp_min(1)

def combined(bce, logits, labels, mask, weight):
    if weight==0:return bce
    return (bce+weight*rank_loss(logits,labels,mask))/(1+weight)

def guard():
    b=base.read(ROOT/'reports/research_continuation_20260915/BUDGET.json')
    assert b['observed_used_percent']<b['stop_new_runs_used_percent'], 'Budget soft stop'

def prepare():
    for p in (OUT,ART/'metrics',ART/'checkpoints',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(base.read(source.OUT/'contract.json')['hashes'])
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'tests/test_window_ranking.py',ROOT/'src/interstart_decoder.py',ROOT/'src/run_halo_decoder_composition.py',ROOT/'src/recurrence_message_probe.py'):
        hashes[str(p)]=base.sha(p)
    for f in (0,1):
        ids=split_ids(f);assert not set(ids['train'])&set(ids['validation'])
        for s in (42,43):
            for k,folder in [('C3',source.ART),('G',ROOT/'artifacts/current_gru_study')]:
                for suffix in ('.json','_predictions.csv.gz'):
                    p=folder/'metrics'/f'{k}_seed{s}_fold{f}{suffix}';hashes[str(p)]=base.sha(p)
    for p in (ROOT/'artifacts/recurrence_mean_control/graphs').glob('*.npz'):hashes[str(p)]=base.sha(p)
    assert all(base.sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert base.read(OUT/'contract.json')['contract']==digest
    base.write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    return digest

def train(kind,fold,seed,contract):
    name=f'{kind}_seed{seed}_fold{fold}';res=ART/'metrics'/f'{name}.json'
    if res.exists():assert base.read(res)['contract']==contract;return
    guard();ids=split_ids(fold);tr=source.dataset(ids['train'],'C3');va=source.dataset(ids['validation'],'C3');norm=base.normalizer(tr)
    model=make_model('C3',seed).cuda();sampler=base.CurvePieceBalancedSampler(tr,norm,64,32,seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    crit=nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(base.positive_weight(tr,10),device='cuda'))
    dest=ART/'checkpoints'/name;dest.mkdir(exist_ok=True);latest=dest/'latest.pt';best=dest/'best.pt'
    step=0;history=[];trace=[];best_score=-1.;prior=0.;weight=0. if kind=='Z' else .25
    if latest.exists():
        cp=torch.load(latest,map_location='cuda',weights_only=False);assert cp['contract']==contract
        model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);sampler.load_state(cp['sampler'])
        step=cp['step'];history=cp['history'];trace=cp['trace'];best_score=cp['best_score'];prior=cp['seconds']
        torch.set_rng_state(cp['rng'].cpu());torch.cuda.set_rng_state_all([a.cpu() for a in cp['cuda_rng']])
    began=time.monotonic();base.write(OUT/'STATE.json',dict(status='training',run=name,pid=os.getpid()))
    def snap():return dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state(),step=step,history=history,trace=trace,best_score=best_score,seconds=prior+time.monotonic()-began,contract=contract,kind=kind,fold=fold,seed=seed,mean=norm.mean,std=norm.std,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    while step<300:
        if prior+time.monotonic()-began>600:torch.save(snap(),latest);raise TimeoutError(name)
        model.train();x,y,mask,valid=(a.cuda() for a in sampler.batch());opt.zero_grad(set_to_none=True)
        logits=model(x,padding_mask=~valid.bool());bce=(crit(logits,y)*mask).sum()/mask.sum().clamp_min(1)
        loss=combined(bce,logits,y,mask,weight);assert torch.isfinite(loss)
        loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gn);opt.step();step+=1
        trace.append(dict(step=step,lr=opt.param_groups[0]['lr'],bce=float(bce.detach()),loss=float(loss.detach()),grad=float(gn)))
        if step%50==0:
            raw=predictions(model,va,norm,torch.device('cuda'));threshold,_=choose_single_threshold(raw,va,base.GRID);score=metrics(raw,va,threshold)[2]
            history.append(dict(step=step,loss=float(loss.detach()),grad=float(gn),**score))
            if score['macro_f1_tol1']>best_score+1e-9:best_score=score['macro_f1_tol1'];torch.save(snap(),best)
            print(name,step,'F1',score['macro_f1_tol1'],flush=True)
        if step%25==0:torch.save(snap(),latest)
    cp=torch.load(best,map_location='cuda',weights_only=False);model.load_state_dict(cp['model']);threshold=checkpoint_threshold(cp)
    raw=predictions(model,va,norm,torch.device('cuda'));perfs,pieces,score=metrics(raw,va,threshold)
    save_raw(raw,ART/'metrics'/f'{name}_predictions.csv.gz',va);perfs.to_csv(ART/'metrics'/f'{name}_performances.csv',index=False);pieces.to_csv(ART/'metrics'/f'{name}_pieces.csv',index=False)
    pd.DataFrame(trace).to_csv(ART/'metrics'/f'{name}_trace.csv',index=False)
    base.write(res,dict(kind=kind,fold=fold,seed=seed,contract=contract,best_step=cp['step'],params=sum(p.numel() for p in model.parameters()),seconds=prior+time.monotonic()-began,**score))

def reproduce():
    for fn in ('best.pt','latest.pt'):
        a=torch.load(source.ART/'checkpoints/C3_seed42_fold0'/fn,map_location='cpu',weights_only=False)
        b=torch.load(ART/'checkpoints/Z_seed42_fold0'/fn,map_location='cpu',weights_only=False)
        assert a['step']==b['step'] and a['sampler']==b['sampler']
        for k,v in a['model'].items():torch.testing.assert_close(v,b['model'][k],atol=0,rtol=0)
        assert a['optimizer']['param_groups']==b['optimizer']['param_groups']
        for i,values in a['optimizer']['state'].items():
            for k,v in values.items():torch.testing.assert_close(v,b['optimizer']['state'][i][k],atol=0,rtol=0)
        assert torch.equal(a['rng'],b['rng']) and all(torch.equal(x,y) for x,y in zip(a['cuda_rng'],b['cuda_rng']))
    base.write(OUT/'zero_weight_reproduction.json',dict(exact_weights_optimizer_sampler_rng=True))

def evaluate(contract):
    dec.ART=ART/'decoder';dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);rows=[];checks=[]
    for f in (0,1):
        ids=split_ids(f);tr=source.dataset(ids['train'],'C3');va=source.dataset(ids['validation'],'C3');norm=base.normalizer(tr);prior=fit_prior(tr)
        for s in (42,43):
            raw={}
            for k,folder in [('R',ART),('C3',source.ART),('G',ROOT/'artifacts/current_gru_study')]:
                raw[k]=checked_raw(pd.read_csv(folder/'metrics'/f'{k}_seed{s}_fold{f}_predictions.csv.gz'),va)
            cp=torch.load(ART/'checkpoints'/f'R_seed{s}_fold{f}'/'best.pt',map_location='cpu',weights_only=False)
            m=make_model('C3',s).cuda();m.load_state_dict(cp['model']);rp=predictions(m,va,norm,torch.device('cuda'))
            err=error(raw['R'],rp);assert err<2e-4
            meta=base.read(ART/'metrics'/f'R_seed{s}_fold{f}.json');assert max(abs(metrics(rp,va,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
            checks.append(dict(fold=f,seed=s,replay_error=err,checkpoint_hash=base.sha(ART/'checkpoints'/f'R_seed{s}_fold{f}'/'best.pt')))
            for name,k in [('ER','R'),('EC','C3')]:raw[name]={p:{q:(a+raw['G'][p][q])*.5 for q,a in pp.items()} for p,pp in raw[k].items()}
            for k in ('R','C3','ER','EC'):
                threshold,_=choose_single_threshold(raw[k],va,base.GRID)
                score=metrics(raw[k],va,threshold)[2];rows.append(dict(kind=k,fold=f,seed=s,policy='raw',**score))
                adjusted={}
                for p,pp in raw[k].items():
                    with np.load(ROOT/'artifacts/recurrence_mean_control/graphs'/f'{p}.npz') as z:g=z['M']
                    adjusted[p]={q:mix_probabilities(a,g) for q,a in pp.items()}
                score=dec.evaluate(adjusted,va,threshold,prior,1.,f'{k}_seed{s}_fold{f}_M10',contract)
                rows.append(dict(kind=k,fold=f,seed=s,policy='M10',**score))
    df=pd.DataFrame(rows);df.to_csv(OUT/'results.csv',index=False);df.groupby(['kind','policy'])[COLS].mean().to_csv(OUT/'means.csv')
    comparisons=[]
    for a,b in [('R','C3'),('ER','EC')]:
        for pol in ('raw','M10'):
            x=df[(df.kind==a)&(df.policy==pol)].set_index(['fold','seed']);y=df[(df.kind==b)&(df.policy==pol)].set_index(['fold','seed']);d=x[COLS]-y[COLS]
            comparisons.append(dict(candidate=a,control=b,policy=pol,**d.mean().to_dict(),positive=int((d.macro_f1_tol1>0).sum()),passed=bool(d.macro_f1_tol1.mean()>=.015 and (d.macro_f1_tol1>0).sum()>=3 and d.macro_f1_tol0.mean()>=0 and d.raw_ap.mean()>=0)))
    hashes=base.read(OUT/'contract.json')['hashes'];assert all(base.sha(p)==h for p,h in hashes.items())
    base.write(OUT/'comparisons.json',comparisons);base.write(OUT/'audit.json',dict(status='complete',candidate_runs=4,zero_reproduction_runs=1,replays=checks,frozen_hashes_unchanged=True,outer_test_used=False))
    base.write(OUT/'STATE.json',dict(status='complete',pid=None));print(df.groupby(['kind','policy'])[COLS].mean().to_string(),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','evaluate'],default='all');args=p.parse_args();torch.set_num_threads(2)
    contract=prepare()
    if args.stage=='all':
        train('Z',0,42,contract);reproduce()
        for f in (0,1):
            for s in (42,43):train('R',f,s,contract)
    evaluate(contract)

if __name__=='__main__':main()
