"""Isolated normalization / pitch-transposition study with checkpointed RNGs."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from .score_context_study import dataset as score_dataset, normalizer as score_normalizer, make_model, sha, read, write
from .three_round_round2 import split_ids, checkpoint_threshold
from .models import Normalizer
from .phase7_models import CurvePieceBalancedSampler
from .phase6_models import positive_weight
from .phase2_models import choose_single_threshold
from .local_context_study import predictions, metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/pitch_invariance_study'; ART=ROOT/'artifacts/pitch_invariance_study'
KINDS=('P','N','A'); GRID=np.arange(.1,.91,.05).round(2).tolist(); CAP=1800.

def normalization(data,kind):
    n=score_normalizer(data)
    if kind=='P':return n
    x=np.concatenate([v['pitch_profiles'] for v in data.values()]).reshape(-1,2,12)
    mean=x.mean(axis=(0,2));std=x.std(axis=(0,2));std[std<1e-6]=1.
    return Normalizer(np.r_[n.mean[:34],np.repeat(mean,12)],np.r_[n.std[:34],np.repeat(std,12)])

def rotate_windows(x, shifts):
    """Same shift for every beat and both chroma blocks; never shift time."""
    out=x.clone()
    for i,s in enumerate(shifts):
        for a,b in ((34,46),(46,58)):out[i,:,a:b]=torch.roll(x[i,:,a:b],int(s),dims=-1)
    return out

class Sampler(CurvePieceBalancedSampler):
    def __init__(self,data,norm,seed,augment):
        super().__init__(data,norm,64,32,seed)
        self.augment=augment;self.augmentation_rng=np.random.default_rng(seed+100000)
        if augment:
            for a,b in ((34,46),(46,58)):
                assert np.ptp(norm.mean[a:b])==0 and np.ptp(norm.std[a:b])==0
    def state(self):return {**super().state(),'augmentation_rng':copy.deepcopy(self.augmentation_rng.bit_generator.state)}
    def load_state(self,state):
        super().load_state(state);self.augmentation_rng.bit_generator.state=state['augmentation_rng']
    def batch(self):
        x,y,m,v=super().batch()
        if self.augment:x=rotate_windows(x,self.augmentation_rng.integers(0,12,len(x)))
        return x,y,m,v

def prepare():
    for p in (OUT,ART/'metrics',ART/'checkpoints'):p.mkdir(parents=True,exist_ok=True)
    parent=read(ROOT/'reports/score_context_study/contract.json')
    assert read(ROOT/'reports/score_context_study/completion_audit.json')['status']=='complete'
    assert all(sha(p)==h for p,h in parent['hashes'].items())
    hashes=dict(parent['hashes'])
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'tests/test_pitch_invariance.py',ROOT/'artifacts/score_context_study/summary.csv'):
        hashes[str(p)]=sha(p)
    contract=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    write(OUT/'contract.json',dict(contract=contract,hashes=hashes))
    complete=[read(p) for p in (ART/'metrics').glob('*_fold*.json')]
    assert all(r['contract']==contract for r in complete)
    write(OUT/'STATE.json',dict(status='ready',completed=[r['run_id'] for r in complete],seconds=sum(r['seconds'] for r in complete),pid=None,contract=contract))
    rows=[]
    for fold in (0,1):
        ids=split_ids(fold);assert not set(ids['train'])&set(ids['validation'])
        data=score_dataset(ids['train'],'P');p=normalization(data,'P');n=normalization(data,'N')
        np.testing.assert_array_equal(p.mean[:34],n.mean[:34]);np.testing.assert_array_equal(p.std[:34],n.std[:34])
        np.savez(OUT/f'normalizers_fold{fold}.npz',P_mean=p.mean,P_std=p.std,N_mean=n.mean,N_std=n.std)
        rows.append(dict(fold=fold,train_works=len(ids['train']),development_works=len(ids['validation']),work_overlap=0,outer_test_used=False))
    pd.DataFrame(rows).to_csv(OUT/'split_audit.csv',index=False)
    return contract

def train_one(kind,fold,seed,contract):
    run=f'{kind}_seed{seed}_fold{fold}';result_path=ART/'metrics'/f'{run}.json'
    if result_path.exists():assert read(result_path)['contract']==contract;print('CACHED',run,flush=True);return
    state=read(OUT/'STATE.json');dest=ART/'checkpoints'/run;dest.mkdir(exist_ok=True)
    best=dest/'best.pt';latest=dest/'latest.pt';ids=split_ids(fold)
    train=score_dataset(ids['train'],'P');val=score_dataset(ids['validation'],'P');norm=normalization(train,kind)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=make_model('P',seed).to(device)
    sampler=Sampler(train,norm,seed,kind=='A');opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    crit=nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(positive_weight(train,10),device=device))
    step=0;history=[];best_score=-1.;prior=0.
    if latest.exists():
        s=torch.load(latest,map_location=device,weights_only=False);assert (s['contract'],s['kind'],s['fold'],s['seed'])==(contract,kind,fold,seed)
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);sampler.load_state(s['sampler'])
        step=s['step'];history=s['history'];best_score=s['best_score'];prior=s['seconds'];torch.set_rng_state(s['rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda_rng']])
    start=time.monotonic();state.update(status='running',current_run=run,pid=os.getpid());write(OUT/'STATE.json',state)
    def snapshot():return dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state(),step=step,history=history,best_score=best_score,seconds=prior+time.monotonic()-start,contract=contract,kind=kind,fold=fold,seed=seed,mean=norm.mean,std=norm.std,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    while step<300:
        if state['seconds']+prior+time.monotonic()-start>=CAP:torch.save(snapshot(),latest);raise TimeoutError('Frozen 1800s batch cap reached')
        model.train();x,y,mask,valid=(v.to(device) for v in sampler.batch());opt.zero_grad(set_to_none=True)
        loss=(crit(model(x,padding_mask=~valid.bool()),y)*mask).sum()/mask.sum().clamp_min(1)
        assert torch.isfinite(loss);loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gn);opt.step();step+=1
        if step%50==0:
            raw=predictions(model,val,norm,device);threshold,_=choose_single_threshold(raw,val,GRID);_,_,score=metrics(raw,val,threshold)
            history.append(dict(step=step,loss=float(loss.detach()),grad=float(gn),**score))
            if score['macro_f1_tol1']>best_score+1e-9:best_score=score['macro_f1_tol1'];torch.save(snapshot(),best)
            print(run,step,'F1',round(score['macro_f1_tol1'],4),'exact',round(score['macro_f1_tol0'],4),'AP',round(score['raw_ap'],4),flush=True)
        if step%25==0:torch.save(snapshot(),latest)
    s=torch.load(best,map_location=device,weights_only=False);model.load_state_dict(s['model']);threshold=checkpoint_threshold(s);raw=predictions(model,val,norm,device)
    perfs,pieces,score=metrics(raw,val,threshold);elapsed=prior+time.monotonic()-start
    result=dict(run_id=run,kind=kind,fold=fold,seed=seed,contract=contract,best_step=s['step'],params=sum(p.numel() for p in model.parameters()),seconds=elapsed,gpu_peak_bytes=int(torch.cuda.max_memory_allocated()) if device.type=='cuda' else 0,history=history,**score)
    perfs.to_csv(ART/'metrics'/f'{run}_performances.csv',index=False);pieces.to_csv(ART/'metrics'/f'{run}_pieces.csv',index=False)
    rows=[(pid,perf,b,float(p),int(val[pid]['labels'][b]),int(val[pid]['label_mask'][b])) for pid,pp in raw.items() for perf,arr in pp.items() for b,p in enumerate(arr)]
    pd.DataFrame(rows,columns=['piece_id','performance_id','beat','probability','label','valid']).to_csv(ART/'metrics'/f'{run}_predictions.csv.gz',index=False)
    write(result_path,result);state['completed'].append(run);state['seconds']+=elapsed;state.update(status='between_runs');write(OUT/'STATE.json',state)

def report():
    frame=pd.DataFrame([{k:v for k,v in read(p).items() if k!='history'} for p in (ART/'metrics').glob('*_fold*.json')]);assert len(frame)==12
    assert frame.groupby('kind').size().to_dict()=={'P':4,'N':4,'A':4}
    frame.to_csv(ART/'summary.csv',index=False);cols=['macro_f1_tol0','macro_f1_tol1','raw_ap','macro_precision_tol1','macro_recall_tol1','seconds']
    means=frame.groupby('kind')[cols].mean();means.to_csv(OUT/'model_means.csv')
    old=pd.read_csv(ROOT/'artifacts/score_context_study/summary.csv');bridge=frame[frame.kind=='P'].merge(old[old.kind=='P'],on=['fold','seed'],suffixes=('_new','_old'))
    bridge_errors={c:float((bridge[c+'_new']-bridge[c+'_old']).abs().max()) for c in cols[:3]}
    write(OUT/'bridge_reproduction.json',dict(cells=len(bridge),errors=bridge_errors,passed=len(bridge)==4 and max(bridge_errors.values())<1e-6))
    comparisons=[]
    for a,b in (('N','P'),('A','P'),('A','N'),('N','B'),('A','B')):
        candidate=frame[frame.kind==a].set_index(['fold','seed']);control=(old if b=='B' else frame);control=control[control.kind==b].set_index(['fold','seed'])
        d=candidate[cols[:3]]-control[cols[:3]]
        comparisons.append(dict(candidate=a,control=b,f1_delta=d.macro_f1_tol1.mean(),exact_delta=d.macro_f1_tol0.mean(),ap_delta=d.raw_ap.mean(),f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),promotion=bool(b=='B' and max(bridge_errors.values())<1e-6 and d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>0 and d.raw_ap.mean()>0 and (d.macro_f1_tol1>0).sum()>=3 and (d.raw_ap>0).sum()>=3)))
    pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False)
    assert all(sha(p)==h for p,h in read(OUT/'contract.json')['hashes'].items())
    state=read(OUT/'STATE.json');state.update(status='training_complete',pid=None);write(OUT/'STATE.json',state)
    print('MEANS\n'+means.to_string(),flush=True);print('COMPARISONS\n'+pd.DataFrame(comparisons).to_string(index=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','all','report'],default='all');args=parser.parse_args();torch.set_num_threads(2)
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
