"""Frozen new-seed replication of the selected NR ensemble."""
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
from .local_context_study import predictions as core_predictions
from .fixed_ensemble_study import aligned_average,raw_from_frame

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/ensemble_seed_replication';ART=ROOT/'artifacts/ensemble_seed_replication'
KINDS=('B','N','R');GRID=np.arange(.1,.91,.05).round(2).tolist();CAP=2400.

def dataset(ids,kind):
    if kind not in KINDS:raise ValueError('Unknown member')
    data=control_dataset(ids,'N' if kind=='N' else 'B')
    if kind=='R':
        for pid,item in data.items():
            item['piano_roll']=np.load(ROOT/'artifacts/score_roll_study/cache'/f'{pid}.npy',allow_pickle=False)
    return data


def build_model(kind,seed):
    return RollBoundary('R',seed) if kind=='R' else make_model('P',seed)


def predictions(model,data,norm,device):
    return (roll_predictions if isinstance(model,RollBoundary) else core_predictions)(model,data,norm,device)


def prepare():
    for p in (OUT,ART/'checkpoints',ART/'metrics',ART/'ensemble'):p.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/fixed_ensemble_study/completion_audit.json')['status']=='complete'
    hashes=dict(read(ROOT/'reports/fixed_ensemble_study/contract.json')['hashes'])
    assert all(sha(p)==h for p,h in hashes.items())
    for p in (Path(__file__),ROOT/'tests/test_ensemble_seed_replication.py',OUT/'PROTOCOL.md'):
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
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=build_model(kind,seed).to(device)
    sampler=(RollSampler if kind=='R' else CurvePieceBalancedSampler)(train,norm,64,32,seed);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
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
        model.train();batch=[v.to(device) for v in sampler.batch()];x,y,mask,valid=batch[:4];opt.zero_grad(set_to_none=True)
        logits=model(x,batch[4],padding_mask=~valid.bool()) if kind=='R' else model(x,padding_mask=~valid.bool())
        loss=(crit(logits,y)*mask).sum()/mask.sum().clamp_min(1)
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
    df=pd.DataFrame([{k:v for k,v in read(p).items() if k!='history'} for p in (ART/'metrics').glob('*_fold*.json')]);assert len(df)==12
    df.to_csv(ART/'summary.csv',index=False)
    cols=['macro_f1_tol0','macro_f1_tol1','raw_ap','macro_precision_tol1','macro_recall_tol1']
    df.groupby('kind')[cols].mean().to_csv(OUT/'member_means.csv')
    rows=[]
    for fold in (0,1):
        data=dataset(split_ids(fold)['validation'],'B')
        for seed in (44,45):
            frames=[pd.read_csv(ART/'metrics'/f'{k}_seed{seed}_fold{fold}_predictions.csv.gz') for k in ('N','R')]
            fused=aligned_average(frames);raw=raw_from_frame(fused,data)
            threshold,_=choose_single_threshold(raw,data,GRID);perfs,pieces,score=metrics(raw,data,threshold)
            run=f'NR_seed{seed}_fold{fold}'
            fused.to_csv(ART/'ensemble'/f'{run}_predictions.csv.gz',index=False)
            perfs.to_csv(ART/'ensemble'/f'{run}_performances.csv',index=False);pieces.to_csv(ART/'ensemble'/f'{run}_pieces.csv',index=False)
            result=dict(run_id=run,kind='NR',fold=fold,seed=seed,**score);write(ART/'ensemble'/f'{run}.json',result);rows.append(result)
    nr=pd.DataFrame(rows);nr.to_csv(ART/'ensemble_summary.csv',index=False);nr[cols].mean().to_csv(OUT/'ensemble_means.csv')
    comparisons=[];a=nr.set_index(['fold','seed'])
    for ref in KINDS:
        b=df[df.kind==ref].set_index(['fold','seed']);delta=a[cols[:3]]-b[cols[:3]]
        comparisons.append(dict(reference=ref,f1_delta=delta.macro_f1_tol1.mean(),exact_delta=delta.macro_f1_tol0.mean(),ap_delta=delta.raw_ap.mean(),f1_positive=int((delta.macro_f1_tol1>0).sum()),ap_positive=int((delta.raw_ap>0).sum()),replication_supported_vs_B=bool(ref=='B' and delta.macro_f1_tol1.mean()>=.015 and delta.macro_f1_tol0.mean()>0 and delta.raw_ap.mean()>0 and (delta.macro_f1_tol1>0).sum()>=3 and (delta.raw_ap>0).sum()>=3)))
    pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False)
    old=pd.read_csv(ROOT/'artifacts/fixed_ensemble_study/summary.csv');old=old[old.kind=='NR']
    pd.concat([old,nr],ignore_index=True).to_csv(OUT/'all_four_seeds_NR.csv',index=False)
    oldb=pd.read_csv(ROOT/'artifacts/score_novelty_study/summary.csv');oldb=oldb[oldb.kind=='B']
    pd.concat([oldb,df[df.kind=='B']],ignore_index=True).to_csv(OUT/'all_four_seeds_B.csv',index=False)
    assert all(sha(p)==h for p,h in read(OUT/'contract.json')['hashes'].items())
    state=read(OUT/'STATE.json');state.update(status='training_complete',pid=None);write(OUT/'STATE.json',state)
    print('NR NEW SEEDS\\n'+nr[cols].mean().to_string(),flush=True)
    print('COMPARISONS\\n'+pd.DataFrame(comparisons).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','all','report'],default='all');args=p.parse_args();torch.set_num_threads(2)
    try:
        if args.stage=='report':report();return
        contract=prepare()
        if args.stage=='prepare':print(contract);return
        for fold in (0,1):
            for seed in (44,45):
                for kind in KINDS:train_one(kind,fold,seed,contract)
        report()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        if (OUT/'STATE.json').exists():s=read(OUT/'STATE.json');s.update(status='failed',pid=None);write(OUT/'STATE.json',s)
        raise

if __name__=='__main__':main()
