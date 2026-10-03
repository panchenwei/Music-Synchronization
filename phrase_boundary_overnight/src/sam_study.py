"""Fixed-rho SAM experiment; isolated artifacts, frozen evaluation reuse."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib,json,time,traceback
from pathlib import Path
import pandas as pd
import torch
from . import window_ranking_study as shared
from .sam_update import step as sam_step

base=shared.base;ROOT=base.ROOT;OUT=ROOT/'reports/sam_study';ART=ROOT/'artifacts/sam_study'


def prepare():
    assert base.read(ROOT/'reports/gru_500_study/audit.json')['status']=='complete'
    for p in (OUT,ART/'metrics',ART/'checkpoints',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(base.read(ROOT/'reports/window_ranking_study/contract.json')['hashes'])
    for p in (Path(__file__),ROOT/'src/sam_update.py',ROOT/'tests/test_sam_update.py',OUT/'PROTOCOL.md'):hashes[str(p)]=base.sha(p)
    for f in (0,1):
        for s in (42,43):
            for folder in ('recurrence_depth_study','recurrence_duration_study'):
                for fn in ('best.pt','latest.pt'):
                    p=ROOT/'artifacts'/folder/'checkpoints'/f'C3_seed{s}_fold{f}'/fn;hashes[str(p)]=base.sha(p)
    assert all(base.sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert base.read(OUT/'contract.json')['contract']==digest
    base.write(OUT/'contract.json',dict(contract=digest,hashes=hashes));return digest


def train(kind,fold,seed,contract):
    name=f'{kind}_seed{seed}_fold{fold}';res=ART/'metrics'/f'{name}.json'
    if res.exists():assert base.read(res)['contract']==contract;return
    shared.guard();ids=shared.split_ids(fold);tr=shared.source.dataset(ids['train'],'C3');va=shared.source.dataset(ids['validation'],'C3');norm=base.normalizer(tr)
    model=shared.make_model('C3',seed).cuda();assert sum(p.numel() for p in model.parameters())==5921
    sampler=base.CurvePieceBalancedSampler(tr,norm,64,32,seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    crit=torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(base.positive_weight(tr,10),device='cuda'))
    dest=ART/'checkpoints'/name;dest.mkdir(exist_ok=True);latest=dest/'latest.pt';best=dest/'best.pt'
    step=0;history=[];trace=[];best_score=-1.;prior=0.;rho=0. if kind=='Z' else .05
    if latest.exists():
        cp=torch.load(latest,map_location='cuda',weights_only=False);assert cp['contract']==contract
        model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);sampler.load_state(cp['sampler'])
        step=cp['step'];history=cp['history'];trace=cp['trace'];best_score=cp['best_score'];prior=cp['seconds']
        torch.set_rng_state(cp['rng'].cpu());torch.cuda.set_rng_state_all([a.cpu() for a in cp['cuda_rng']])
    began=time.monotonic();base.write(OUT/'STATE.json',dict(status='training',run=name,pid=os.getpid()))
    def snap():return dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state(),step=step,history=history,trace=trace,best_score=best_score,seconds=prior+time.monotonic()-began,contract=contract,kind=kind,fold=fold,seed=seed,mean=norm.mean,std=norm.std,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    while step<300:
        if prior+time.monotonic()-began>600:torch.save(snap(),latest);raise TimeoutError(name)
        batch=tuple(a.cuda() for a in sampler.batch());values=sam_step(model,opt,batch,crit,rho);step+=1
        trace.append(dict(step=step,lr=opt.param_groups[0]['lr'],rho=rho,**values))
        if step%50==0:
            raw=shared.predictions(model,va,norm,torch.device('cuda'));threshold,_=shared.choose_single_threshold(raw,va,base.GRID);score=shared.metrics(raw,va,threshold)[2]
            history.append(dict(step=step,loss=values['loss'],grad=values['grad'],**score))
            if score['macro_f1_tol1']>best_score+1e-9:best_score=score['macro_f1_tol1'];torch.save(snap(),best)
            print(name,step,'F1',score['macro_f1_tol1'],flush=True)
        if step%25==0:torch.save(snap(),latest)
    cp=torch.load(best,map_location='cuda',weights_only=False);model.load_state_dict(cp['model']);threshold=shared.checkpoint_threshold(cp)
    raw=shared.predictions(model,va,norm,torch.device('cuda'));perfs,pieces,score=shared.metrics(raw,va,threshold)
    shared.save_raw(raw,ART/'metrics'/f'{name}_predictions.csv.gz',va);perfs.to_csv(ART/'metrics'/f'{name}_performances.csv',index=False);pieces.to_csv(ART/'metrics'/f'{name}_pieces.csv',index=False)
    pd.DataFrame(trace).to_csv(ART/'metrics'/f'{name}_trace.csv',index=False)
    base.write(res,dict(kind=kind,method='SAM' if rho else 'zero',rho=rho,fold=fold,seed=seed,contract=contract,best_step=cp['step'],params=5921,seconds=prior+time.monotonic()-began,**score))


def main():
    torch.set_num_threads(2);shared.OUT=OUT;shared.ART=ART
    contract=prepare();train('Z',0,42,contract);shared.reproduce()
    for f in (0,1):
        for s in (42,43):train('R',f,s,contract)
    base.write(OUT/'STATE.json',dict(status='auditing',pid=os.getpid()))
    shared.evaluate(contract)


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);base.write(OUT/'failure.json',dict(time=time.time(),traceback=traceback.format_exc()));base.write(OUT/'STATE.json',dict(status='failed',pid=None));raise
