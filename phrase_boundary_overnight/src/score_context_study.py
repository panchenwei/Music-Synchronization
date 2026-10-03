"""Frozen score representation x context experiment; isolated outputs, resumable."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from .models import Normalizer
from .three_round_round2 import dataset as old_dataset,normalizer as old_normalizer,make_model34,split_ids,checkpoint_threshold
from .slice_energy_study import DCML
from .slice_energy_features import voiced_events
from .data import discover_dcml_pieces
from .phase3_models import fit_train_normalizer
from .phase7_models import CurvePieceBalancedSampler
from .phase6_models import positive_weight
from .phase2_models import choose_single_threshold
from .local_context_study import metrics,predictions

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/score_context_study';ART=ROOT/'artifacts/score_context_study'
KINDS=('B','P','C','PC');GRID=np.arange(.1,.91,.05).round(2).tolist();CAP=2400.

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def pitch_profiles(events,n):
    """Time-weighted sounding pitch-class and instantaneous-lowest-note profiles."""
    result=np.zeros((n,24),np.float32)
    valid=[(float(o),float(d),int(p)) for o,d,p,*_ in events if d>0]
    for b in range(n):
        active=[(o,d,p) for o,d,p in valid if o<b+1 and o+d>b]
        for o,d,p in active:result[b,p%12]+=max(0.,min(b+1,o+d)-max(b,o))
        den=float(result[b,:12].sum())
        if den:result[b,:12]/=den
        edges=sorted(set([float(b),float(b+1)]+[v for o,d,p in active for v in (max(float(b),o),min(float(b+1),o+d))]))
        for a,z in zip(edges[:-1],edges[1:]):
            pitches=[p for o,d,p in active if o<(a+z)/2<o+d]
            if pitches:result[b,12+min(pitches)%12]+=z-a
        den=float(result[b,12:].sum())
        if den:result[b,12:]/=den
    assert np.isfinite(result).all() and (result>=0).all()
    return result

class ContextFrontend(nn.Module):
    def __init__(self,old,wide):
        super().__init__();self.norm=old.norm;self.convs=old.convs;self.pointwise=old.pointwise;self.dropout=old.dropout;self.wide=wide
    def forward(self,h,padding_mask=None):
        mask=torch.zeros(h.shape[:2],dtype=torch.bool,device=h.device) if padding_mask is None else padding_mask
        h=h.masked_fill(mask[...,None],0);z=self.norm(h).masked_fill(mask[...,None],0).transpose(1,2)
        conv=self.convs[0];out=conv(z)
        if self.wide:out=(out+F.conv1d(z,conv.weight,conv.bias,padding=8,dilation=4,groups=conv.groups))*.5
        out=self.pointwise(F.gelu(out)).transpose(1,2)
        return (h+self.dropout(out)).masked_fill(mask[...,None],0)

def make_model(kind,seed):
    m=make_model34(seed);old=m.input_projection
    with torch.random.fork_rng(devices=[]):linear=nn.Linear(58,32)
    with torch.no_grad():linear.weight.zero_();linear.weight[:,:34].copy_(old.weight);linear.bias.copy_(old.bias)
    m.input_projection=linear;m.frontend=ContextFrontend(m.frontend,kind in ('C','PC'))
    return m

def dataset(ids,kind):
    data=old_dataset(ids,'B')
    for pid,item in data.items():
        feat=np.load(ART/'cache'/f'{pid}.npy',allow_pickle=False)
        if kind in ('B','C'):feat=np.zeros_like(feat)
        item['pitch_profiles']=feat;item['curves']=np.concatenate([item['curves'],np.broadcast_to(feat,(*item['curves'].shape[:2],24))],axis=-1)
    return data

def normalizer(data):
    base=old_normalizer(data);extra=fit_train_normalizer([v['pitch_profiles'] for v in data.values()])
    return Normalizer(np.r_[base.mean,extra.mean],np.r_[base.std,extra.std])

def prepare():
    for p in (OUT,ART/'cache',ART/'checkpoints',ART/'metrics'):p.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),OUT/'PROTOCOL.md',ROOT/'artifacts/phase2/splits/opus_split_manifest.csv']
    for s in ('models.py','phase2_models.py','phase3_models.py','phase6_models.py','phase7_models.py','local_context_study.py','slice_energy_features.py','slice_energy_study.py','three_round_round2.py','data.py','evaluation.py'):files.append(ROOT/'src'/s)
    pieces=discover_dcml_pieces(DCML);ids=sorted(set(p for f in (0,1) for s in ('train','validation') for p in split_ids(f)[s]));rows=[]
    for pid in ids:
        cache=ROOT/'artifacts/slice_energy_study/cache'/f'{pid}.npz';files.extend([cache,pieces[pid].notes_path,pieces[pid].measures_path])
        with np.load(cache) as a:n=len(a['start_labels'])
        out=ART/'cache'/f'{pid}.npy'
        if not out.exists():np.save(out,pitch_profiles(voiced_events(pieces[pid],n),n))
        files.append(out);x=np.load(out);assert x.shape==(n,24)
        rows.append(dict(piece_id=pid,beats=n,sounding_beats=int((x[:,:12].sum(1)>0).sum())))
    hashes={str(p):sha(p) for p in files};contract=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract,'Source contract changed; do not overwrite completed study'
    write(OUT/'contract.json',dict(contract=contract,hashes=hashes));pd.DataFrame(rows).to_csv(OUT/'feature_audit.csv',index=False)
    splitrows=[]
    for f in (0,1):
        d=split_ids(f);assert not(set(d['train'])&set(d['validation']))
        frame=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv');part=frame[frame.fold==f]
        assert not set(part[part.split=='train'].opus)&set(part[part.split=='validation'].opus)
        splitrows.append(dict(fold=f,train=len(d['train']),validation=len(d['validation']),test_used=False))
    pd.DataFrame(splitrows).to_csv(OUT/'split_audit.csv',index=False)
    if not (OUT/'STATE.json').exists():write(OUT/'STATE.json',dict(status='ready',completed=[],seconds=0.,pid=None,contract=contract))
    return contract

def train_one(kind,fold,seed,contract):
    run=f'{kind}_seed{seed}_fold{fold}';respath=ART/'metrics'/f'{run}.json'
    if respath.exists():assert read(respath)['contract']==contract;print('CACHED',run,flush=True);return
    state=read(OUT/'STATE.json');dest=ART/'checkpoints'/run;dest.mkdir(exist_ok=True);best=dest/'best.pt';latest=dest/'latest.pt'
    ids=split_ids(fold);train=dataset(ids['train'],kind);val=dataset(ids['validation'],kind)
    norm=normalizer(train);model=make_model(kind,seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
    sampler=CurvePieceBalancedSampler(train,norm,64,32,seed);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    crit=nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(positive_weight(train,10),device=device))
    step=0;history=[];best_score=-1.;prior=0.
    if latest.exists():
        s=torch.load(latest,map_location=device,weights_only=False);assert s['contract']==contract
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);sampler.load_state(s['sampler'])
        step=s['step'];history=s['history'];best_score=s['best_score'];prior=s['seconds'];torch.set_rng_state(s['rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda_rng']])
    start=time.monotonic();state.update(status='running',current_run=run,pid=os.getpid());write(OUT/'STATE.json',state)
    def snapshot():return dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state(),step=step,history=history,best_score=best_score,seconds=prior+time.monotonic()-start,contract=contract,kind=kind,fold=fold,seed=seed,mean=norm.mean,std=norm.std,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    while step<300:
        if state['seconds']+prior+time.monotonic()-start>=CAP:
            torch.save(snapshot(),latest);raise TimeoutError('Frozen per-batch 2400s cap reached')
        model.train();x,y,mask,valid=(v.to(device) for v in sampler.batch());opt.zero_grad(set_to_none=True);logit=model(x,padding_mask=~valid.bool());loss=(crit(logit,y)*mask).sum()/mask.sum().clamp_min(1)
        assert torch.isfinite(loss);loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gn);opt.step();step+=1
        if step%50==0:
            raw=predictions(model,val,norm,device);threshold,_=choose_single_threshold(raw,val,GRID);_,_,score=metrics(raw,val,threshold)
            history.append(dict(step=step,loss=float(loss.detach()),grad=float(gn),**score))
            if score['macro_f1_tol1']>best_score+1e-9:best_score=score['macro_f1_tol1'];torch.save(snapshot(),best)
            print(run,step,'F1',round(score['macro_f1_tol1'],4),'exact',round(score['macro_f1_tol0'],4),'AP',round(score['raw_ap'],4),flush=True)
        if step%25==0:torch.save(snapshot(),latest)
    s=torch.load(best,map_location=device,weights_only=False);model.load_state_dict(s['model']);threshold=checkpoint_threshold(s);raw=predictions(model,val,norm,device)
    perfs,pieces,score=metrics(raw,val,threshold);elapsed=prior+time.monotonic()-start
    result=dict(run_id=run,kind=kind,fold=fold,seed=seed,contract=contract,best_step=int(s['step']),params=sum(p.numel() for p in model.parameters()),seconds=elapsed,gpu_peak_bytes=int(torch.cuda.max_memory_allocated()) if device.type=='cuda' else 0,**score)
    perfs.to_csv(ART/'metrics'/f'{run}_performances.csv',index=False);pieces.to_csv(ART/'metrics'/f'{run}_pieces.csv',index=False)
    prediction_rows=[(pid,perf,b,float(p),int(val[pid]['labels'][b]),int(val[pid]['label_mask'][b])) for pid,pp in raw.items() for perf,arr in pp.items() for b,p in enumerate(arr)]
    pd.DataFrame(prediction_rows,columns=['piece_id','performance_id','beat','probability','label','valid']).to_csv(ART/'metrics'/f'{run}_predictions.csv.gz',index=False)
    write(respath,result);state['completed'].append(run);state['seconds']+=elapsed;state.update(status='between_runs',pid=os.getpid());write(OUT/'STATE.json',state)

def report():
    rows=[read(p) for p in sorted((ART/'metrics').glob('*_fold*.json'))];frame=pd.DataFrame(rows)
    if frame.empty:return
    frame.to_csv(ART/'summary.csv',index=False)
    cols=['macro_f1_tol0','macro_f1_tol1','macro_f1_tol2','raw_ap','macro_precision_tol1','macro_recall_tol1','seconds','params']
    means=frame.groupby('kind')[cols].mean();means.to_csv(OUT/'model_means.csv')
    comparisons=[]
    for candidate,control in [('P','B'),('C','B'),('PC','B'),('PC','P'),('PC','C')]:
        a=frame[frame.kind==candidate].set_index(['fold','seed']);b=frame[frame.kind==control].set_index(['fold','seed']);common=a.index.intersection(b.index)
        if len(common)==0:continue
        d=a.loc[common,cols]-b.loc[common,cols]
        comparisons.append(dict(candidate=candidate,control=control,cells=len(common),f1_delta=d.macro_f1_tol1.mean(),exact_delta=d.macro_f1_tol0.mean(),ap_delta=d.raw_ap.mean(),f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),passed=bool(len(common)==4 and d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>0 and d.raw_ap.mean()>0 and (d.macro_f1_tol1>0).sum()>=3 and (d.raw_ap>0).sum()>=3)))
    pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False)
    sources=read(OUT/'contract.json')['hashes'];unchanged=all(sha(p)==h for p,h in sources.items());assert unchanged
    state=read(OUT/'STATE.json');state.update(status='complete' if len(frame)==16 else 'partial',pid=None);write(OUT/'STATE.json',state)
    write(OUT/'audit.json',dict(completed_runs=len(frame),full_matrix=len(frame)==16,source_hashes_unchanged=unchanged,outer_test_evaluated=False,parameter_counts=sorted(frame.params.unique().tolist()),training_validation_seconds=state['seconds'],under_cap=state['seconds']<=CAP))
    print('MEANS\n'+means.to_string(),flush=True);print('COMPARISONS\n'+pd.DataFrame(comparisons).to_string(index=False),flush=True)
    old=pd.read_csv(ROOT/'artifacts/three_round_study/round2/summary.csv');base=frame[frame.kind=='B'].merge(old[old.kind=='B'],on=['fold','seed'],suffixes=('_new','_old'))
    if len(base):base[['fold','seed','macro_f1_tol1_new','macro_f1_tol1_old']].to_csv(OUT/'baseline_reproduction.csv',index=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','prepare','report'],default='all');args=parser.parse_args()
    torch.set_num_threads(2)
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
        if (OUT/'STATE.json').exists():
            s=read(OUT/'STATE.json');s.update(status='failed',pid=None);write(OUT/'STATE.json',s)
        raise

if __name__=='__main__':main()
