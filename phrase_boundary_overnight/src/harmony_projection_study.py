"""Fixed main-priority gradient projection with matched content controls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from . import harmony_auxiliary_study as old
from . import start_guided_attention_study as auditing
from .auxiliary_gradient_projection import merge
from .harmony_auxiliary import make_model,HarmonySampler
from .score_context_study import ROOT,read,write,sha,normalizer,GRID,positive_weight,choose_single_threshold,checkpoint_threshold
from .three_round_round2 import split_ids
from .local_context_study import metrics,predictions

OUT=ROOT/'reports/harmony_projection';ART=ROOT/'artifacts/harmony_projection';CAP=4800.
dataset=old.dataset


def prepare():
    assert read(ROOT/'reports/harmony_gradient_probe/completion_audit.json')['projected_gradient_followup_gate']
    for p in (OUT,ART/'metrics',ART/'checkpoints',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(old.OUT/'contract.json')['hashes'])
    for p in (Path(__file__),ROOT/'src/auxiliary_gradient_projection.py',ROOT/'tests/test_auxiliary_gradient_projection.py',OUT/'PROTOCOL.md'):
        hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    if not (OUT/'STATE.json').exists():write(OUT/'STATE.json',dict(status='ready',completed=[],seconds=0.,pid=None))
    return digest


def train_one(kind,fold,seed,contract):
    run=f'{kind}_seed{seed}_fold{fold}';respath=ART/'metrics'/f'{run}.json'
    if respath.exists():assert read(respath)['contract']==contract;print('CACHED',run,flush=True);return
    state=read(OUT/'STATE.json');dest=ART/'checkpoints'/run;dest.mkdir(exist_ok=True);best=dest/'best.pt';latest=dest/'latest.pt'
    ids=split_ids(fold);train=dataset(ids['train'],kind);val=dataset(ids['validation'],kind);norm=normalizer(train)
    model=make_model(kind,seed).cuda();params=list(model.parameters());sampler=HarmonySampler(train,norm,seed,'D' if kind=='D' else 'G')
    opt=torch.optim.AdamW(params,lr=.001,weight_decay=.0001);device=torch.device('cuda')
    crit=nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(positive_weight(train,10),device=device))
    step=0;history=[];gradients=[];best_score=-1.;prior=0.
    if latest.exists():
        s=torch.load(latest,map_location=device,weights_only=False);assert s['contract']==contract
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);sampler.load_state(s['sampler'])
        step=s['step'];history=s['history'];gradients=s['gradients'];best_score=s['best_score'];prior=s['seconds']
        torch.set_rng_state(s['rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda_rng']])
    start=time.monotonic();state.update(status='running',current_run=run,pid=os.getpid());write(OUT/'STATE.json',state)
    def snapshot():return dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state(),step=step,history=history,gradients=gradients,
        best_score=best_score,seconds=prior+time.monotonic()-start,contract=contract,kind=kind,fold=fold,seed=seed,mean=norm.mean,std=norm.std,
        rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    torch.cuda.reset_peak_memory_stats()
    while step<300:
        if state['seconds']+prior+time.monotonic()-start>=CAP:torch.save(snapshot(),latest);raise TimeoutError('4800s training cap')
        model.train();x,y,mask,valid,hy,hm=(v.to(device) for v in sampler.batch());opt.zero_grad(set_to_none=True)
        logit,hlogit=model(x,padding_mask=~valid.bool(),both=True)
        main=(crit(logit,y)*mask).sum()/mask.sum().clamp_min(1)
        aux=.25*(nn.functional.cross_entropy(hlogit.transpose(1,2),hy,reduction='none')*hm).sum()/hm.sum().clamp_min(1)/np.log(7)
        assert torch.isfinite(main+aux)
        gm=torch.autograd.grad(main,params,retain_graph=True,allow_unused=True);ga=torch.autograd.grad(aux,params,allow_unused=True)
        merged,stats=merge(gm,ga,kind!='Z')
        for p,g in zip(params,merged):p.grad=g
        gn=nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(gn);opt.step();step+=1;gradients.append(dict(step=step,**stats))
        if step%50==0:
            raw=predictions(model,val,norm,device);threshold,_=choose_single_threshold(raw,val,GRID);_,_,score=metrics(raw,val,threshold)
            history.append(dict(step=step,loss=float((main+aux).detach()),main_bce=float(main.detach()),weighted_aux=float(aux.detach()),grad=float(gn),**score))
            if score['macro_f1_tol1']>best_score+1e-9:best_score=score['macro_f1_tol1'];torch.save(snapshot(),best)
            print(run,step,'F1',round(score['macro_f1_tol1'],4),'AP',round(score['raw_ap'],4),'projected',round(np.mean([r['applied'] for r in gradients[-50:]]),3),flush=True)
        if step%25==0:torch.save(snapshot(),latest)
    saved=torch.load(best,map_location=device,weights_only=False);model.load_state_dict(saved['model']);threshold=checkpoint_threshold(saved)
    raw=predictions(model,val,norm,device);perfs,pieces,score=metrics(raw,val,threshold);elapsed=prior+time.monotonic()-start
    result=dict(run_id=run,kind=kind,fold=fold,seed=seed,contract=contract,best_step=int(saved['step']),params=6152,seconds=elapsed,
        gpu_peak_bytes=int(torch.cuda.max_memory_allocated()),**score)
    perfs.to_csv(ART/'metrics'/f'{run}_performances.csv',index=False);pieces.to_csv(ART/'metrics'/f'{run}_pieces.csv',index=False)
    pd.DataFrame(gradients).to_csv(ART/'metrics'/f'{run}_gradients.csv',index=False)
    pd.DataFrame([(pid,perf,b,float(p),int(val[pid]['labels'][b]),int(val[pid]['label_mask'][b])) for pid,pp in raw.items() for perf,arr in pp.items() for b,p in enumerate(arr)],
        columns=['piece_id','performance_id','beat','probability','label','valid']).to_csv(ART/'metrics'/f'{run}_predictions.csv.gz',index=False)
    write(respath,result);state['completed'].append(run);state['seconds']+=elapsed;state.update(status='between_runs',pid=os.getpid());write(OUT/'STATE.json',state)


def audit():
    auditing.OUT=OUT;auditing.ART=ART;auditing.dataset=dataset;auditing.make_model=make_model;auditing.audit()
    rows=[]
    for p in sorted((ART/'metrics').glob('*_gradients.csv')):
        frame=pd.read_csv(p);assert len(frame)==300
        if p.name.startswith('Z_'):assert frame.applied.sum()==0
        else:assert frame.projected_dot.min()>-1e-4
        rows.append(dict(run=p.stem,**frame[['cosine','conflicting','applied','aux_main_ratio']].mean().to_dict()))
    assert len(rows)==12;pd.DataFrame(rows).to_csv(OUT/'gradient_audit.csv',index=False)
    write(OUT/'projection_audit.json',dict(status='complete',runs=12,steps_audited=3600,main_priority=True,no_new_forward=True))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if args.stage=='audit':audit();return
    digest=prepare()
    if args.stage=='prepare':print(digest,flush=True);return
    for f in (0,1):
        for s in (42,43):
            for k in ('G','D','Z'):
                guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent']
                train_one(k,f,s,digest)
    audit()


if __name__=='__main__':
    try:
        with threadpool_limits(2):main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',{**(read(OUT/'STATE.json') if (OUT/'STATE.json').exists() else {}),'status':'failed','pid':None});raise
