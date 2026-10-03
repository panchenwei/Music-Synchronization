"""External score-only pretraining, followed by a matched frozen-stem transfer test."""
import argparse, hashlib, json, os, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from .score_context_study import ROOT, read, write, sha, normalizer
from .models import Normalizer
from .score_roll_branch import RollBoundary, RollSampler, roll_predictions
from .label_repaired_rebaseline import dataset as repaired_dataset
from .three_round_round2 import split_ids
from .phase6_models import positive_weight
from .phase2_models import choose_single_threshold
from .local_context_study import metrics
from .fixed_ensemble_study import aligned_average, raw_from_frame
from . import score_roll_study as target_engine

OUT=ROOT/'reports/external_stem_transfer'; ART=ROOT/'artifacts/external_stem_transfer'
GRID=np.arange(.1,.91,.05).round(2).tolist()
COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap','macro_precision_tol1','macro_recall_tol1']


def external_split():
    df=pd.read_csv(ROOT/'reports/external_score_probe/piece_audit.csv')
    assert len(df)==79 and (df.status=='usable').all()
    df['split']=np.where(df.corpus=='grieg_lyric_pieces','train','validation')
    assert df.groupby('split').size().to_dict()=={'train':66,'validation':13}
    a=df[df.split=='train'];b=df[df.split=='validation']
    assert not set(a.piece_id)&set(b.piece_id) and not set(a.corpus)&set(b.corpus)
    assert not set(zip(a.corpus,a.opus))&set(zip(b.corpus,b.opus))
    assert int(a.valid_positives.sum())==519 and int(b.valid_positives.sum())==84
    return df


def external_dataset(split):
    df=external_split();data={}
    for pid in df.loc[df.split==split,'piece_id']:
        with np.load(ROOT/'artifacts/external_score_probe/cache'/f'{pid}.npz',allow_pickle=False) as z:
            item={k:z[k].copy() for k in ('piano_roll','labels','label_mask')}
        n=len(item['labels']);item.update(curves=np.zeros((1,n,58),np.float32),performance_ids=np.array(['score']),pitch_profiles=np.zeros((n,24),np.float32))
        data[pid]=item
    return data


def dataset(ids,kind):
    assert kind in ('R','T')
    data=repaired_dataset(ids,'B')
    for pid,item in data.items():item['piano_roll']=np.load(ROOT/'artifacts/local_coordinate_roll/cache'/f'{pid}.npy',allow_pickle=False)
    return data


def transplant(model,state):
    model.stem.load_state_dict({k.removeprefix('stem.'):v for k,v in state.items() if k.startswith('stem.')},strict=True)
    model.stem.requires_grad_(False)
    return model


def build_model(kind,seed):
    assert kind in ('R','T')
    model=RollBoundary('R',seed)
    if kind=='T':
        s=torch.load(ART/'external'/f'seed{seed}'/'best.pt',map_location='cpu',weights_only=False)
        assert s['contract']==read(OUT/'contract.json')['contract'] and s['seed']==seed
        transplant(model,s['model'])
    return model


def prepare():
    for p in (OUT,ART/'external',ART/'checkpoints',ART/'metrics',ART/'ensemble'):p.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/label_repaired_rebaseline/completion_audit.json')['status']=='complete'
    hashes=dict(read(ROOT/'reports/label_repaired_rebaseline/contract.json')['hashes'])
    for p in ('reports/external_score_probe/source_hashes.json','reports/local_coordinate_roll/source_hashes.json'):
        hashes.update(read(ROOT/p));hashes[str(ROOT/p)]=sha(ROOT/p)
    files=[Path(__file__),ROOT/'src/audit_external_stem_transfer.py',ROOT/'tests/test_external_stem_transfer.py',OUT/'PROTOCOL.md',ROOT/'reports/external_score_probe/piece_audit.csv']
    files+=list((ROOT/'artifacts/label_repaired_rebaseline/metrics').glob('N_*_predictions.csv.gz'))
    files+=[ROOT/'artifacts/label_repaired_rebaseline/summary.csv',ROOT/'artifacts/label_repaired_rebaseline/ensemble_summary.csv']
    for p in files:hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    contract=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    write(OUT/'contract.json',dict(contract=contract,hashes=hashes));external_split().to_csv(OUT/'external_split.csv',index=False)
    if not (OUT/'STATE.json').exists():write(OUT/'STATE.json',dict(status='ready',contract=contract,seconds=0.,pid=None,completed=[]))
    return contract


def save_predictions(raw,data,dest,threshold):
    perfs,pieces,score=metrics(raw,data,threshold)
    perfs.to_csv(str(dest)+'_performances.csv',index=False);pieces.to_csv(str(dest)+'_pieces.csv',index=False)
    rows=[(pid,perf,b,float(p),int(data[pid]['labels'][b]),int(data[pid]['label_mask'][b])) for pid,pp in raw.items() for perf,arr in pp.items() for b,p in enumerate(arr)]
    pd.DataFrame(rows,columns=['piece_id','performance_id','beat','probability','label','valid']).to_csv(str(dest)+'_predictions.csv.gz',index=False)
    return score


def pretrain(seed,contract):
    dest=ART/'external'/f'seed{seed}';dest.mkdir(exist_ok=True);resultpath=dest/'result.json'
    if resultpath.exists():assert read(resultpath)['contract']==contract;print('CACHED external',seed,flush=True);return
    train=external_dataset('train');val=external_dataset('validation');norm=Normalizer(np.zeros(58,np.float32),np.ones(58,np.float32))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=RollBoundary('L',seed).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);sampler=RollSampler(train,norm,64,32,seed)
    crit=nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(positive_weight(train,10),device=device))
    step=0;history=[];best=-1.;prior=0.;latest=dest/'latest.pt'
    if latest.exists():
        s=torch.load(latest,map_location=device,weights_only=False);assert s['contract']==contract and s['seed']==seed
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);sampler.load_state(s['sampler']);step=s['step'];history=s['history'];best=s['best_ap'];prior=s['seconds'];torch.set_rng_state(s['rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda_rng']])
    state=read(OUT/'STATE.json');start=time.monotonic();state.update(status='running',current_run=f'external_seed{seed}',pid=os.getpid());write(OUT/'STATE.json',state)
    def snapshot():return dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state(),step=step,history=history,best_ap=best,seconds=prior+time.monotonic()-start,contract=contract,seed=seed,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    while step<600:
        if state['seconds']+prior+time.monotonic()-start>=3600:torch.save(snapshot(),latest);raise TimeoutError('3600s training cap')
        model.train();x,y,mask,valid,roll=(v.to(device) for v in sampler.batch());opt.zero_grad(set_to_none=True)
        loss=(crit(model(x,roll,padding_mask=~valid.bool()),y)*mask).sum()/mask.sum().clamp_min(1)
        assert torch.isfinite(loss);loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gn);opt.step();step+=1
        if step%100==0:
            raw=roll_predictions(model,val,norm,device);threshold,_=choose_single_threshold(raw,val,GRID);_,_,score=metrics(raw,val,threshold)
            history.append(dict(step=step,loss=float(loss.detach()),grad=float(gn),**score))
            if score['raw_ap']>best+1e-9:best=score['raw_ap'];torch.save(snapshot(),dest/'best.pt')
            print('external',seed,step,'F1',round(score['macro_f1_tol1'],4),'AP',round(score['raw_ap'],4),flush=True)
        if step%25==0:torch.save(snapshot(),latest)
    s=torch.load(dest/'best.pt',map_location=device,weights_only=False);model.load_state_dict(s['model']);threshold=s['history'][-1]['threshold']
    score=save_predictions(roll_predictions(model,val,norm,device),val,dest/'validation',threshold)
    elapsed=prior+time.monotonic()-start;result=dict(contract=contract,seed=seed,best_step=s['step'],selection_metric='raw_ap',params=sum(p.numel() for p in model.parameters()),seconds=elapsed,history=history,gpu_peak_bytes=int(torch.cuda.max_memory_allocated()) if device.type=='cuda' else 0,**score)
    write(resultpath,result);state['seconds']+=elapsed;state['completed'].append(f'external_seed{seed}');state.update(status='between_runs');write(OUT/'STATE.json',state)


def report():
    df=pd.DataFrame([{k:v for k,v in read(p).items() if k!='history'} for p in (ART/'metrics').glob('*_fold*.json')]);assert len(df)==8
    df.to_csv(ART/'summary.csv',index=False);df.groupby('kind')[COLS].mean().to_csv(OUT/'model_means.csv');rows=[]
    for fold in (0,1):
        data=dataset(split_ids(fold)['validation'],'R')
        for seed in (42,43):
            n=pd.read_csv(ROOT/'artifacts/label_repaired_rebaseline/metrics'/f'N_seed{seed}_fold{fold}_predictions.csv.gz')
            for kind in ('R','T'):
                member=pd.read_csv(ART/'metrics'/f'{kind}_seed{seed}_fold{fold}_predictions.csv.gz');fused=aligned_average([n,member]);raw=raw_from_frame(fused,data);threshold,_=choose_single_threshold(raw,data,GRID)
                run=f'N{kind}_seed{seed}_fold{fold}';score=save_predictions(raw,data,ART/'ensemble'/run,threshold);r=dict(run_id=run,kind='N'+kind,fold=fold,seed=seed,**score);write(ART/'ensemble'/f'{run}.json',r);rows.append(r)
    en=pd.DataFrame(rows);en.to_csv(ART/'ensemble_summary.csv',index=False);en.groupby('kind')[COLS].mean().to_csv(OUT/'ensemble_means.csv')
    comparisons=[]
    for data,cand,ref in ((df,'T','R'),(en,'NT','NR')):
        a=data[data.kind==cand].set_index(['fold','seed']);b=data[data.kind==ref].set_index(['fold','seed']);d=a[COLS]-b[COLS]
        comparisons.append(dict(candidate=cand,reference=ref,f1_delta=d.macro_f1_tol1.mean(),exact_delta=d.macro_f1_tol0.mean(),ap_delta=d.raw_ap.mean(),f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),promotion=bool(d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>0 and d.raw_ap.mean()>0 and (d.macro_f1_tol1>0).sum()>=3 and (d.raw_ap>0).sum()>=3)))
    pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False)
    repair=[]
    for kind,new,path in (('R',df,ROOT/'artifacts/label_repaired_rebaseline/summary.csv'),('NR',en,ROOT/'artifacts/label_repaired_rebaseline/ensemble_summary.csv')):
        old=pd.read_csv(path);a=new[new.kind==kind].set_index(['fold','seed']);b=old[old.kind==kind].set_index(['fold','seed']);d=a[COLS]-b[COLS]
        repair.append(dict(kind=kind,**{c+'_delta':float(d[c].mean()) for c in COLS}))
    pd.DataFrame(repair).to_csv(OUT/'input_repair_effect.csv',index=False)
    state=read(OUT/'STATE.json');state.update(status='training_complete',pid=None);write(OUT/'STATE.json',state)
    print('TARGET\n'+df.groupby('kind')[COLS].mean().to_string(),flush=True);print('ENSEMBLES\n'+en.groupby('kind')[COLS].mean().to_string(),flush=True);print('COMPARISONS\n'+pd.DataFrame(comparisons).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','all','report','audit'],default='all');a=p.parse_args();torch.set_num_threads(2)
    try:
        if a.stage=='audit':
            from .audit_external_stem_transfer import main as audit
            audit();return
        if a.stage=='report':report();return
        contract=prepare()
        if a.stage=='prepare':print(contract);return
        for seed in (42,43):pretrain(seed,contract)
        # Bind only this worker process; historical frozen source is untouched.
        target_engine.OUT=OUT;target_engine.ART=ART;target_engine.dataset=dataset;target_engine.RollBoundary=build_model;target_engine.CAP=3600.
        for fold in (0,1):
            for seed in (42,43):
                for kind in ('R','T'):target_engine.train_one(kind,fold,seed,contract)
        report()
        from .audit_external_stem_transfer import main as audit
        audit()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        if (OUT/'STATE.json').exists():s=read(OUT/'STATE.json');s.update(status='failed',pid=None);write(OUT/'STATE.json',s)
        raise


if __name__=='__main__':main()
