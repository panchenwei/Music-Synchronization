"""Matched start-only versus start + independent structural-end supervision."""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from .score_context_study import ROOT,read,write,sha,make_model,normalizer
from .label_repaired_rebaseline import dataset as original_dataset
from .three_round_round2 import split_ids,checkpoint_threshold
from .slice_energy_study import DCML
from .data import discover_dcml_pieces
from .phrase_end_targets import from_source
from .phase7_models import CurvePieceBalancedSampler
from .phase6_models import positive_weight
from .phase2_models import choose_single_threshold
from .local_context_study import metrics,predictions
from .models import Normalizer

OUT=ROOT/'reports/phrase_end_auxiliary';ART=ROOT/'artifacts/phrase_end_auxiliary'
GRID=np.arange(.1,.91,.05).round(2).tolist();KINDS=('C','E');COLS=('macro_f1_tol1','macro_f1_tol0','raw_ap')


class DualBoundary(nn.Module):
    def __init__(self,seed):
        super().__init__();self.core=make_model('P',seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+5000);self.end_head=nn.Linear(32,1)
    def forward(self,x,padding_mask=None,both=False):
        start,hidden=self.core(x,padding_mask=padding_mask,return_hidden=True)
        end=self.end_head(hidden).squeeze(-1)
        if padding_mask is not None:end=end.masked_fill(padding_mask,0)
        return (start,end) if both else start


def end_data(data):
    return {p:{**v,'labels':v['end_labels'],'label_mask':v['end_mask']} for p,v in data.items()}


class EndView(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):return self.model(x,both=True)[1]


class DualSampler:
    def __init__(self,data,norm,seed):
        self.start=CurvePieceBalancedSampler(data,norm,64,32,seed)
        self.end=CurvePieceBalancedSampler(end_data(data),norm,64,32,seed)
    def state(self):return dict(start=self.start.state(),end=self.end.state())
    def load_state(self,s):self.start.load_state(s['start']);self.end.load_state(s['end'])
    def batch(self):
        a=self.start.batch();b=self.end.batch()
        assert torch.equal(a[0],b[0]) and torch.equal(a[3],b[3])
        return (*a,b[1],b[2])


def dataset(ids):
    data=original_dataset(ids,'B')
    for pid,v in data.items():
        with np.load(ART/'cache'/f'{pid}.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(v['labels'],z['start_labels']);np.testing.assert_array_equal(v['label_mask'],z['start_mask'])
            v['end_labels']=z['end_labels'].copy();v['end_mask']=z['end_mask'].copy()
    return data


def prepare():
    for p in (OUT,ART/'cache',ART/'checkpoints',ART/'metrics'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(ROOT/'reports/label_repaired_rebaseline/contract.json')['hashes']);assert all(sha(p)==h for p,h in hashes.items())
    files=[Path(__file__),OUT/'PROTOCOL.md',ROOT/'src/phrase_end_targets.py',ROOT/'tests/test_phrase_end_auxiliary.py',ROOT/'src/audit_phrase_end_auxiliary.py']
    pieces=discover_dcml_pieces(DCML);counts=[]
    ids=sorted(set(p for f in (0,1) for g in ('train','validation') for p in split_ids(f)[g]))
    for pid in ids:
        piece=pieces[pid];cp=ROOT/'artifacts/coordinate_repair_preview'/f'{pid}.npz'
        with np.load(cp,allow_pickle=False) as z:sy=z['labels'].copy();sm=z['label_mask'].copy()
        y,m,mapping=from_source(pd.read_csv(piece.harmony_path,sep='\t'),pd.read_csv(piece.measures_path,sep='\t'),len(sy))
        path=ART/'cache'/f'{pid}.npz';arrays=dict(start_labels=sy,start_mask=sm,end_labels=y,end_mask=m)
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                for k,v in arrays.items():np.testing.assert_array_equal(v,z[k])
        else:np.savez_compressed(path,**arrays)
        counts.append(dict(piece_id=pid,starts=int((sy*sm).sum()),ends=int((y*m).sum()),end_valid=int(m.sum()),end_ambiguous=sum(r['status']=='ambiguous_tie' for r in mapping)))
        files.extend([piece.harmony_path,piece.measures_path,cp,path])
    pd.DataFrame(counts).to_csv(OUT/'target_counts.csv',index=False)
    for p in files:hashes[str(p)]=sha(p)
    contract=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    write(OUT/'contract.json',dict(contract=contract,hashes=hashes))
    done=[read(p) for p in (ART/'metrics').glob('*_fold*.json')];assert all(r['contract']==contract for r in done)
    write(OUT/'STATE.json',dict(status='ready',contract=contract,pid=None,completed=[r['run_id'] for r in done],seconds=sum(r['seconds'] for r in done)))
    return contract


def train_one(kind,fold,seed,contract):
    run=f'{kind}_seed{seed}_fold{fold}';res=ART/'metrics'/f'{run}.json'
    if res.exists():assert read(res)['contract']==contract;print('CACHED',run,flush=True);return
    dest=ART/'checkpoints'/run;dest.mkdir(exist_ok=True);latest=dest/'latest.pt';best=dest/'best.pt'
    ids=split_ids(fold);assert not set(ids['train'])&set(ids['validation'])
    train=dataset(ids['train']);val=dataset(ids['validation']);norm=normalizer(train);device=torch.device('cuda')
    model=DualBoundary(seed).to(device);sampler=DualSampler(train,norm,seed);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    def criterion(d):return nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(positive_weight(d,10),device=device))
    cs,ce=criterion(train),criterion(end_data(train));weight=.25 if kind=='E' else 0.
    step=0;history=[];best_score=-1.;prior=0.
    if latest.exists():
        s=torch.load(latest,map_location=device,weights_only=False);assert (s['contract'],s['kind'],s['fold'],s['seed'])==(contract,kind,fold,seed)
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);sampler.load_state(s['sampler']);step=s['step'];history=s['history'];best_score=s['best_score'];prior=s['seconds'];torch.set_rng_state(s['rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda_rng']])
    state=read(OUT/'STATE.json');began=time.monotonic();state.update(status='running',pid=os.getpid(),current_run=run);write(OUT/'STATE.json',state)
    def snapshot():return dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state(),step=step,history=history,best_score=best_score,seconds=prior+time.monotonic()-began,contract=contract,kind=kind,fold=fold,seed=seed,mean=norm.mean,std=norm.std,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),aux_weight=weight)
    torch.cuda.reset_peak_memory_stats()
    while step<300:
        if state['seconds']+prior+time.monotonic()-began>=1800:torch.save(snapshot(),latest);raise TimeoutError('1800s round cap')
        model.train();x,y,m,v,ey,em=[a.to(device) for a in sampler.batch()];opt.zero_grad(set_to_none=True)
        a,b=model(x,padding_mask=~v.bool(),both=True);ls=(cs(a,y)*m).sum()/m.sum().clamp_min(1);le=(ce(b,ey)*em).sum()/em.sum().clamp_min(1);loss=ls+weight*le
        assert torch.isfinite(loss);loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gn);opt.step();step+=1
        if step%50==0:
            raw=predictions(model,val,norm,device);threshold,_=choose_single_threshold(raw,val,GRID);_,_,score=metrics(raw,val,threshold)
            history.append(dict(step=step,loss=float(loss.detach()),start_loss=float(ls.detach()),end_loss=float(le.detach()),grad=float(gn),**score))
            if score['macro_f1_tol1']>best_score+1e-9:best_score=score['macro_f1_tol1'];torch.save(snapshot(),best)
            print(run,step,'F1',round(score['macro_f1_tol1'],4),'AP',round(score['raw_ap'],4),flush=True)
        if step%25==0:torch.save(snapshot(),latest)
    s=torch.load(best,map_location=device,weights_only=False);model.load_state_dict(s['model']);threshold=checkpoint_threshold(s)
    raw=predictions(model,val,norm,device);perfs,pieces,score=metrics(raw,val,threshold)
    ev=end_data(val);eraw=predictions(EndView(model),ev,norm,device);et,_=choose_single_threshold(eraw,ev,GRID);_,_,escore=metrics(eraw,ev,et)
    rows=[(p,k,b,float(prob),int(val[p]['labels'][b]),int(val[p]['label_mask'][b]),float(eraw[p][k][b]),int(val[p]['end_labels'][b]),int(val[p]['end_mask'][b])) for p,pp in raw.items() for k,arr in pp.items() for b,prob in enumerate(arr)]
    pd.DataFrame(rows,columns=['piece_id','performance_id','beat','probability','label','valid','end_probability','end_label','end_valid']).to_csv(ART/'metrics'/f'{run}_predictions.csv.gz',index=False)
    perfs.to_csv(ART/'metrics'/f'{run}_performances.csv',index=False);pieces.to_csv(ART/'metrics'/f'{run}_pieces.csv',index=False)
    elapsed=prior+time.monotonic()-began;result=dict(run_id=run,kind=kind,fold=fold,seed=seed,contract=contract,best_step=s['step'],params=sum(p.numel() for p in model.parameters()),seconds=elapsed,gpu_peak_bytes=int(torch.cuda.max_memory_allocated()),history=history,**score,**{'end_'+k:v for k,v in escore.items()})
    write(res,result);state['completed'].append(run);state['seconds']+=elapsed;state.update(status='between_runs',pid=os.getpid());write(OUT/'STATE.json',state)


def report():
    df=pd.DataFrame([{k:v for k,v in read(p).items() if k!='history'} for p in (ART/'metrics').glob('*_fold*.json')]);assert len(df)==8
    df.to_csv(ART/'summary.csv',index=False);cols=list(COLS)+['end_'+c for c in COLS];means=df.groupby('kind')[cols].mean();means.to_csv(OUT/'means.csv')
    c=df[df.kind=='C'].set_index(['fold','seed']);e=df[df.kind=='E'].set_index(['fold','seed']);delta=e[list(COLS)]-c[list(COLS)];delta.to_csv(OUT/'paired_deltas.csv')
    write(OUT/'comparison.json',dict(mean_delta=delta.mean().to_dict(),f1_positive=int((delta.macro_f1_tol1>0).sum()),ap_positive=int((delta.raw_ap>0).sum()),promotion=bool(delta.macro_f1_tol1.mean()>=.015 and delta.macro_f1_tol0.mean()>0 and delta.raw_ap.mean()>0 and (delta.macro_f1_tol1>0).sum()>=3 and (delta.raw_ap>0).sum()>=3)))
    state=read(OUT/'STATE.json');state.update(status='training_complete',pid=None);write(OUT/'STATE.json',state);print(means.to_string(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','prepare','report','audit'],default='all');a=p.parse_args();torch.set_num_threads(2)
    try:
        if a.stage=='audit':
            from .audit_phrase_end_auxiliary import main as audit
            audit();return
        if a.stage=='report':report();return
        contract=prepare()
        if a.stage=='prepare':print(contract);return
        for fold in (0,1):
            for seed in (42,43):
                for kind in KINDS:train_one(kind,fold,seed,contract)
        report()
        from .audit_phrase_end_auxiliary import main as audit
        audit()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        if (OUT/'STATE.json').exists():s=read(OUT/'STATE.json');s.update(status='failed',pid=None);write(OUT/'STATE.json',s)
        raise


if __name__=='__main__':main()
