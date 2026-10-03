"""Frozen per-beat piano-roll branch study with a matched random-feature control."""
from __future__ import annotations
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from .score_context_study import make_model,normalizer,sha,read,write
from .three_round_round2 import dataset as base_dataset,split_ids,checkpoint_threshold
from .slice_energy_study import DCML
from .slice_energy_features import voiced_events
from .data import discover_dcml_pieces
from .score_piano_roll import piano_roll
from .score_roll_branch import RollBoundary, RollSampler, roll_predictions
from .score_novelty_study import dataset as control_dataset
from .phase7_models import CurvePieceBalancedSampler
from .phase6_models import positive_weight
from .phase2_models import choose_single_threshold
from .local_context_study import metrics
predictions=roll_predictions

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/score_roll_study';ART=ROOT/'artifacts/score_roll_study'
KINDS=('L','R');GRID=np.arange(.1,.91,.05).round(2).tolist();CAP=2400.

def dataset(ids,kind):
    if kind not in KINDS:raise ValueError('Unknown roll setting')
    data=control_dataset(ids,'B')
    for pid,item in data.items():
        item['piano_roll']=np.load(ART/'cache'/f'{pid}.npy',allow_pickle=False)
    return data


def prepare():
    for p in (OUT,ART/'cache',ART/'checkpoints',ART/'metrics'):p.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/score_novelty_study/completion_audit.json')['status']=='complete'
    hashes=dict(read(ROOT/'reports/score_novelty_study/contract.json')['hashes'])
    assert all(sha(p)==h for p,h in hashes.items())
    assert read(ROOT/'reports/focal_boundary_study/completion_audit.json')['status']=='complete'
    pieces=discover_dcml_pieces(DCML);ids=sorted({p for f in (0,1) for k in ('train','validation') for p in split_ids(f)[k]})
    rows=[]
    for pid in ids:
        with np.load(ROOT/'artifacts/slice_energy_study/cache'/f'{pid}.npz') as z:n=len(z['start_labels'])
        dest=ART/'cache'/f'{pid}.npy'
        events=voiced_events(pieces[pid],n)
        if not dest.exists():np.save(dest,piano_roll(events,n))
        roll=np.load(dest,allow_pickle=False);assert roll.shape==(n,2,128,4) and np.isfinite(roll).all() and roll.min()>=0 and roll.max()<=1
        hashes[str(dest)]=sha(dest)
        onsets=[(min(int(np.floor(o*4+1e-9)),n*4-1),p) for o,d,p,*_ in events if d>0 and 0<=o<n]
        rows.append(dict(piece_id=pid,beats=n,events=len(events),positive_duration_onsets=len(onsets),distinct_pitch_ticks=len(set(onsets)),onset_collisions=len(onsets)-len(set(onsets)),bytes=int(roll.nbytes)))
    pd.DataFrame(rows).to_csv(OUT/'feature_audit.csv',index=False)
    for p in (Path(__file__),ROOT/'tests/test_score_piano_roll.py',ROOT/'tests/test_score_roll_branch.py',ROOT/'src/score_piano_roll.py',ROOT/'src/score_roll_branch.py',OUT/'PROTOCOL.md',ROOT/'artifacts/score_novelty_study/summary.csv'):
        hashes[str(p)]=sha(p)
    contract=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    write(OUT/'contract.json',dict(contract=contract,hashes=hashes))
    complete=[read(p) for p in (ART/'metrics').glob('*_fold*.json')];assert all(r['contract']==contract for r in complete)
    write(OUT/'STATE.json',dict(status='ready',completed=[r['run_id'] for r in complete],seconds=sum(r['seconds'] for r in complete),pid=None,contract=contract))
    return contract

def train_one(kind,fold,seed,contract):
    run=f'{kind}_seed{seed}_fold{fold}';res=ART/'metrics'/f'{run}.json'
    if res.exists():assert read(res)['contract']==contract;print('CACHED',run,flush=True);return
    dest=ART/'checkpoints'/run;dest.mkdir(exist_ok=True);best=dest/'best.pt';latest=dest/'latest.pt';ids=split_ids(fold)
    assert not set(ids['train'])&set(ids['validation'])
    train=dataset(ids['train'],kind);val=dataset(ids['validation'],kind);norm=normalizer(train)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=RollBoundary(kind,seed).to(device)
    sampler=RollSampler(train,norm,64,32,seed);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    crit=nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(positive_weight(train,10),device=device))
    step=0;history=[];best_score=-1.;prior=0.
    if latest.exists():
        s=torch.load(latest,map_location=device,weights_only=False);assert (s['contract'],s['kind'],s['fold'],s['seed'])==(contract,kind,fold,seed)
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);sampler.load_state(s['sampler']);step=s['step'];history=s['history'];best_score=s['best_score'];prior=s['seconds'];torch.set_rng_state(s['rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda_rng']])
    state=read(OUT/'STATE.json');start=time.monotonic();state.update(status='running',current_run=run,pid=os.getpid());write(OUT/'STATE.json',state)
    def snapshot():return dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state(),step=step,history=history,best_score=best_score,seconds=prior+time.monotonic()-start,contract=contract,kind=kind,fold=fold,seed=seed,mean=norm.mean,std=norm.std,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    while step<300:
        if state['seconds']+prior+time.monotonic()-start>=CAP:torch.save(snapshot(),latest);raise TimeoutError('2400s batch cap')
        model.train();x,y,mask,valid,roll=(v.to(device) for v in sampler.batch());opt.zero_grad(set_to_none=True)
        loss=(crit(model(x,roll,padding_mask=~valid.bool()),y)*mask).sum()/mask.sum().clamp_min(1)
        assert torch.isfinite(loss);loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gn);opt.step();step+=1
        if step%50==0:
            raw=predictions(model,val,norm,device);threshold,_=choose_single_threshold(raw,val,GRID);_,_,score=metrics(raw,val,threshold);history.append(dict(step=step,loss=float(loss.detach()),grad=float(gn),**score))
            if score['macro_f1_tol1']>best_score+1e-9:best_score=score['macro_f1_tol1'];torch.save(snapshot(),best)
            print(run,step,'F1',round(score['macro_f1_tol1'],4),'exact',round(score['macro_f1_tol0'],4),'AP',round(score['raw_ap'],4),flush=True)
        if step%25==0:torch.save(snapshot(),latest)
    s=torch.load(best,map_location=device,weights_only=False);model.load_state_dict(s['model']);threshold=checkpoint_threshold(s);raw=predictions(model,val,norm,device);perfs,pieces,score=metrics(raw,val,threshold);elapsed=prior+time.monotonic()-start
    result=dict(run_id=run,kind=kind,fold=fold,seed=seed,contract=contract,best_step=s['step'],params=sum(p.numel() for p in model.parameters()),trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),seconds=elapsed,gpu_peak_bytes=int(torch.cuda.max_memory_allocated()) if device.type=='cuda' else 0,history=history,**score)
    perfs.to_csv(ART/'metrics'/f'{run}_performances.csv',index=False);pieces.to_csv(ART/'metrics'/f'{run}_pieces.csv',index=False)
    rows=[(pid,perf,b,float(p),int(val[pid]['labels'][b]),int(val[pid]['label_mask'][b])) for pid,pp in raw.items() for perf,arr in pp.items() for b,p in enumerate(arr)]
    pd.DataFrame(rows,columns=['piece_id','performance_id','beat','probability','label','valid']).to_csv(ART/'metrics'/f'{run}_predictions.csv.gz',index=False)
    write(res,result);state['completed'].append(run);state['seconds']+=elapsed;state.update(status='between_runs');write(OUT/'STATE.json',state)

def report():
    df=pd.DataFrame([{k:v for k,v in read(p).items() if k!='history'} for p in (ART/'metrics').glob('*_fold*.json')]);assert len(df)==8
    df.to_csv(ART/'summary.csv',index=False);cols=['macro_f1_tol0','macro_f1_tol1','raw_ap','macro_precision_tol1','macro_recall_tol1','seconds'];means=df.groupby('kind')[cols].mean();means.to_csv(OUT/'model_means.csv')
    comparisons=[];controls=pd.read_csv(ROOT/'artifacts/score_novelty_study/summary.csv');b=controls[controls.kind=='B'].set_index(['fold','seed'])
    for kind in KINDS:
        a=df[df.kind==kind].set_index(['fold','seed']);d=a[cols[:3]]-b[cols[:3]]
        comparisons.append(dict(candidate=kind,f1_delta=d.macro_f1_tol1.mean(),exact_delta=d.macro_f1_tol0.mean(),ap_delta=d.raw_ap.mean(),f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),promotion=bool(d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>0 and d.raw_ap.mean()>0 and (d.macro_f1_tol1>0).sum()>=3 and (d.raw_ap>0).sum()>=3)))
    pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False)
    old=pd.read_csv(ROOT/'artifacts/score_context_study/summary.csv');old=old[old.kind=='B'].set_index(['fold','seed']);error=float((b[cols[:3]]-old[cols[:3]]).abs().max().max())
    write(OUT/'baseline_reproduction.json',dict(max_error=error,passed=error<1e-6,baseline_reused_not_retrained=True));assert all(sha(p)==h for p,h in read(OUT/'contract.json')['hashes'].items())
    state=read(OUT/'STATE.json');state.update(status='training_complete',pid=None);write(OUT/'STATE.json',state)
    print('MEANS\n'+means.to_string(),flush=True);print('COMPARISONS\n'+pd.DataFrame(comparisons).to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','all','report'],default='all');args=p.parse_args();torch.set_num_threads(2)
    try:
        if args.stage=='report':report();return
        contract=prepare()
        if args.stage=='prepare':print(contract);return
        for fold in (0,1):
            for seed in (42,43):
                for kind in KINDS:train_one(kind,fold,seed,contract)
        report()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        if (OUT/'STATE.json').exists():s=read(OUT/'STATE.json');s.update(status='failed',pid=None);write(OUT/'STATE.json',s)
        raise

if __name__=='__main__':main()
