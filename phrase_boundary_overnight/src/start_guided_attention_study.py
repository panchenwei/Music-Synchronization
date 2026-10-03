"""Frozen single-factor start-guided attention training and audit."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from . import score_context_study as engine
from . import run_recurrence_depth_study as base
from . import run_halo_decoder_composition as dec
from .start_guided_attention import make_model
from .score_context_study import ROOT,read,write,sha,normalizer,GRID,CurvePieceBalancedSampler,positive_weight,choose_single_threshold,checkpoint_threshold
from .three_round_round2 import split_ids
from .local_context_study import metrics,predictions
from .audit_external_stem_transfer import checked_raw
from .context_inference_audit_v2 import error
from .interstart_decoder import fit_prior
from .recurrence_message_probe import mix_probabilities

OUT=ROOT/'reports/start_guided_attention';ART=ROOT/'artifacts/start_guided_attention'
COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap'];CAP=4800.

def dataset(ids,kind):
    assert kind in ('G','D','Z')
    return base.dataset(ids,'C3')

def prepare():
    for p in (OUT,ART/'metrics',ART/'checkpoints',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(base.OUT/'contract.json')['hashes'])
    files=[Path(__file__),ROOT/'src/start_guided_attention.py',ROOT/'src/recurrence_local_attention.py',ROOT/'tests/test_start_guided_attention.py',OUT/'PROTOCOL.md',
        ROOT/'src/interstart_decoder.py',ROOT/'src/run_halo_decoder_composition.py',ROOT/'src/recurrence_message_probe.py']
    files+=list((ROOT/'artifacts/recurrence_mean_control/graphs').glob('*.npz'))
    for f in (0,1):
        for s in (42,43):
            for suffix in ('.json','_predictions.csv.gz'):files.append(base.ART/'metrics'/f'C3_seed{s}_fold{f}{suffix}')
    for s in (42,43):
        a=make_model('G',s).state_dict()
        for k in ('D','Z'):
            for key,v in make_model(k,s).state_dict().items():torch.testing.assert_close(v,a[key],atol=0,rtol=0)
    for p in files:hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    write(OUT/'initialization_audit.json',dict(same_initial_weights=True,params=sum(p.numel() for p in make_model('G',42).parameters()),same_dataset_and_sampler=True))
    if not (OUT/'STATE.json').exists():write(OUT/'STATE.json',dict(status='ready',completed=[],seconds=0.,pid=None))
    return digest

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
        main_loss=loss;aux=model.guidance(y,mask,kind);loss=loss+(0. if kind=='Z' else .1)*aux
        assert torch.isfinite(loss);loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gn);opt.step();step+=1
        if step%50==0:
            raw=predictions(model,val,norm,device);threshold,_=choose_single_threshold(raw,val,GRID);_,_,score=metrics(raw,val,threshold)
            history.append(dict(step=step,loss=float(loss.detach()),main_bce=float(main_loss.detach()),auxiliary_kl=float(aux.detach()),grad=float(gn),**score))
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

def audit():
    began=time.monotonic();contract=read(OUT/'contract.json');checks=[];rows=[];histories=[]
    hashes={str(p):sha(p) for p in (ART/'checkpoints').glob('*/*.pt')};assert len(hashes)==24
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder'
    for f in (0,1):
        ids=split_ids(f);train=dataset(ids['train'],'G');val=dataset(ids['validation'],'G');norm=normalizer(train);prior=fit_prior(train)
        for s in (42,43):
            for kind,source in [('C3',base.ART),('G',ART),('D',ART),('Z',ART)]:
                assert time.monotonic()-began<1800
                name=f'{kind}_seed{s}_fold{f}';meta=read(source/'metrics'/f'{name}.json')
                raw=checked_raw(pd.read_csv(source/'metrics'/f'{name}_predictions.csv.gz'),val)
                assert max(abs(metrics(raw,val,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
                if kind!='C3':
                    best=torch.load(source/'checkpoints'/name/'best.pt',map_location='cpu',weights_only=False)
                    last=torch.load(source/'checkpoints'/name/'latest.pt',map_location='cpu',weights_only=False)
                    assert best['contract']==last['contract']==meta['contract']==contract['contract'] and last['step']==300
                    assert (best['kind'],best['fold'],best['seed'])==(kind,f,s)
                    assert best['step']==meta['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
                    np.testing.assert_array_equal(norm.mean,best['mean']);np.testing.assert_array_equal(norm.std,best['std'])
                    m=make_model(kind,s).cuda();m.load_state_dict(best['model']);replay=predictions(m,val,norm,torch.device('cuda'))
                    err=error(raw,replay);assert err<2e-4
                    assert max(abs(metrics(replay,val,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
                    checks.append(dict(run=name,replay_error=err,params=meta['params']))
                    histories.extend([dict(run=name,kind=kind,**h) for h in last['history']])
                    pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False)
                    pd.DataFrame(histories).to_csv(OUT/'training_history.csv',index=False)
                adjusted={}
                for pid,pp in raw.items():
                    with np.load(ROOT/'artifacts/recurrence_mean_control/graphs'/f'{pid}.npz',allow_pickle=False) as z:g=z['M']
                    adjusted[pid]={q:mix_probabilities(a,g) for q,a in pp.items()}
                for policy,x,strength in [('raw',raw,0.),('M10',adjusted,1.)]:
                    r=dec.evaluate(x,val,meta['threshold'],prior,strength,f'{name}_{policy}',contract['contract'])
                    if policy=='raw':assert max(abs(r[c]-meta[c]) for c in COLS)<1e-10
                    rows.append(dict(kind=kind,seed=s,fold=f,policy=policy,**r))
                pd.DataFrame(rows).to_csv(ART/'summary.csv',index=False);print('AUDITED',name,flush=True)
    df=pd.DataFrame(rows);assert len(df)==32 and len(checks)==12
    means=df.groupby(['kind','policy'])[COLS].mean();means.to_csv(OUT/'means.csv');comp=[]
    for ref in ('C3','Z','D'):
        for policy in ('raw','M10'):
            a=df[(df.kind=='G')&(df.policy==policy)].set_index(['fold','seed']);b=df[(df.kind==ref)&(df.policy==policy)].set_index(['fold','seed']);d=a[COLS]-b[COLS]
            comp.append(dict(reference=ref,policy=policy,**d.mean().to_dict(),positive=int((d.macro_f1_tol1>0).sum()),passed=bool(d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>=0 and d.raw_ap.mean()>=0 and (d.macro_f1_tol1>0).sum()>=3)))
    write(OUT/'comparisons.json',comp)
    assert all(sha(p)==h for p,h in hashes.items()) and all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'checkpoint_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=12,gpu_replays=12,checkpoints=24,decoded_cells=32,
        old_metrics_recomputed=4,hashes_unchanged=True,test_used=False,training_seconds=sum(read(p)['seconds'] for p in (ART/'metrics').glob('*.json')),audit_seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None));print(means.to_string(),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if args.stage=='audit':audit();return
    digest=prepare()
    if args.stage=='prepare':print(digest);return
    for f in (0,1):
        for s in (42,43):
            for kind in ('G','D','Z'):
                guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent']
                train_one(kind,f,s,digest)
    audit()


if __name__=='__main__':
    try:
        with threadpool_limits(2):main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',{**(read(OUT/'STATE.json') if (OUT/'STATE.json').exists() else {}),'status':'failed','pid':None});raise

