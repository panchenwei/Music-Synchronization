"""Isolated frozen-loop copy: only AdamW learning-rate expression changes."""
import os,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from .score_context_study import ROOT,read,write,normalizer,GRID
from .three_round_round2 import split_ids,checkpoint_threshold
from .phase7_models import CurvePieceBalancedSampler
from .phase6_models import positive_weight
from .phase2_models import choose_single_threshold
from .local_context_study import metrics,predictions
from . import run_recurrence_depth_study as prior

OUT=ROOT/'reports/recurrence_lr_study';ART=ROOT/'artifacts/recurrence_lr_study';CAP=1500.
LEARNING_RATES={'C3':.001,'L3':.0003}


def dataset(ids,kind):
    assert kind in LEARNING_RATES
    return prior.dataset(ids,'C3')


def make_model(kind,seed):
    assert kind in LEARNING_RATES
    return prior.make_model('C3',seed)


def train_one(kind,fold,seed,contract):
    run=f'{kind}_seed{seed}_fold{fold}';respath=ART/'metrics'/f'{run}.json'
    if respath.exists():assert read(respath)['contract']==contract;print('CACHED',run,flush=True);return
    state=read(OUT/'STATE.json');dest=ART/'checkpoints'/run;dest.mkdir(exist_ok=True);best=dest/'best.pt';latest=dest/'latest.pt'
    ids=split_ids(fold);train=dataset(ids['train'],kind);val=dataset(ids['validation'],kind)
    norm=normalizer(train);model=make_model(kind,seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
    sampler=CurvePieceBalancedSampler(train,norm,64,32,seed);opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATES[kind],weight_decay=.0001)
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
