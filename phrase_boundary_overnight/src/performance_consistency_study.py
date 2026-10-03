"""Two-sided curve trends against a same-size zero-feature control."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np,pandas as pd,torch
from . import score_context_study as engine
from . import run_recurrence_depth_study as base
from .recurrence_depth_models import make_model as c3_model
from torch import nn
from . import run_halo_decoder_composition as dec

from .models import Normalizer
from .local_context_study import predictions
from .score_context_study import ROOT,read,write,sha,normalizer as base_normalizer
from .three_round_round2 import split_ids
from .local_context_study import metrics
from .audit_external_stem_transfer import checked_raw
from .interstart_decoder import fit_prior

OUT=ROOT/'reports/performance_consistency_study';ART=ROOT/'artifacts/performance_consistency_study';KINDS=('P','Z');COLS=base.COLS;PARAMS={'P':5921,'Z':5921}


from .performance_pair_sampler import PairSampler,consistency_forward
from .score_context_study import positive_weight,choose_single_threshold,checkpoint_threshold,GRID
CAP=1800.

class PairBoundary(nn.Module):
    def __init__(self,kind,seed):
        super().__init__();assert kind in KINDS;self.core=c3_model('C3',seed)
    def forward(self,x,padding_mask=None):return self.core(x,padding_mask=padding_mask)

def dataset(ids,kind):
    assert kind in KINDS
    return base.dataset(ids,'C3')

normalizer=base_normalizer
make_model=PairBoundary


def prepare():
    for p in (OUT,ART/'metrics',ART/'checkpoints',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/curve_arc_ap_selection/completion_audit.json')['status']=='complete'
    assert read(ROOT/'reports/c3_performance_agreement/completion_audit.json')['status']=='complete'
    hashes=dict(read(base.OUT/'contract.json')['hashes']);hashes.update(read(ROOT/'reports/c3_performance_agreement/source_hashes.json'))
    for p in (Path(__file__),ROOT/'tests/test_performance_pair_sampler.py',OUT/'PROTOCOL.md',ROOT/'src/performance_pair_sampler.py',ROOT/'artifacts/interstart_decoder_study/summary.csv'):hashes[str(p)]=sha(p)
    checks=[]
    for f in (0,1):
        ids=split_ids(f);old=dataset(ids['train'],'Z');a=normalizer(old)
        for kind in KINDS:
            new=dataset(ids['train'],kind);b=normalizer(new)
            np.testing.assert_array_equal(a.mean,b.mean);np.testing.assert_array_equal(a.std,b.std)
            for pid,item in new.items():
                np.testing.assert_array_equal(item['curves'][...,:58],old[pid]['curves'][...,:58])
                for key in ('labels','label_mask','performance_ids'):np.testing.assert_array_equal(item[key],old[pid][key])
            for seed in (42,43):
                sa=engine.CurvePieceBalancedSampler(old,a,64,32,seed);sb=engine.CurvePieceBalancedSampler(new,b,64,32,seed)
                for _ in range(3):
                    ba=sa.batch();bb=sb.batch();torch.testing.assert_close(ba[0][...,:58],bb[0][...,:58],atol=0,rtol=0)
                    for x,y in zip(ba[1:],bb[1:]):torch.testing.assert_close(x,y,atol=0,rtol=0)
            checks.append(dict(fold=f,kind=kind,norm_equal=True,old58_labels_sampling_equal=True))
        for seed in (42,43):
            run=f'C3_seed{seed}_fold{f}'
            for suffix in ('.json','_predictions.csv.gz'):
                p=base.ART/'metrics'/f'{run}{suffix}';hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));pd.DataFrame(checks).to_csv(OUT/'input_audit.csv',index=False)
    done=[read(p) for p in (ART/'metrics').glob('*_fold*.json')];assert all(r['contract']==digest for r in done)
    write(OUT/'STATE.json',dict(status='ready',contract=digest,completed=[r['run_id'] for r in done],seconds=sum(r['seconds'] for r in done),pid=None))
    return digest


def audit():
    began=time.monotonic();contract=read(OUT/'contract.json');rows=[];checks=[];decoded=[]
    cp_hash={str(p):sha(p) for p in (ART/'checkpoints').glob('*/*.pt')};assert len(cp_hash)==16
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder'
    for f in (0,1):
        ids=split_ids(f);train=dataset(ids['train'],'P');val=dataset(ids['validation'],'P');norm=normalizer(train);prior=fit_prior(train)
        manifest=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv');part=manifest[manifest.fold==f]
        assert not set(part[part.split=='train'].opus)&set(part[part.split=='validation'].opus)
        write(OUT/f'prior_fold{f}.json',dict(prior=prior,train_ids=ids['train']))
        for s in (42,43):
            for oldkind,source in (('C3',base.ART),):
                run=f'{oldkind}_seed{s}_fold{f}';r=read(source/'metrics'/f'{run}.json')
                raw=checked_raw(pd.read_csv(source/'metrics'/f'{run}_predictions.csv.gz'),val)
                assert max(abs(metrics(raw,val,r['threshold'])[2][c]-r[c]) for c in COLS)<1e-10
            for k in KINDS:
                train=dataset(ids['train'],k);val=dataset(ids['validation'],k)
                run=f'{k}_seed{s}_fold{f}';r=read(ART/'metrics'/f'{run}.json');cp=ART/'checkpoints'/run
                best=torch.load(cp/'best.pt',map_location='cpu',weights_only=False);last=torch.load(cp/'latest.pt',map_location='cpu',weights_only=False)
                assert (best['contract'],best['kind'],best['fold'],best['seed'])==(contract['contract'],k,f,s)
                assert last['contract']==r['contract']==contract['contract'] and last['step']==300
                assert best['step']==r['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
                np.testing.assert_array_equal(norm.mean,best['mean']);np.testing.assert_array_equal(norm.std,best['std'])
                raw=checked_raw(pd.read_csv(ART/'metrics'/f'{run}_predictions.csv.gz'),val)
                assert max(abs(metrics(raw,val,r['threshold'])[2][c]-r[c]) for c in COLS)<1e-10
                m=PairBoundary(k,s);m.load_state_dict(best['model'])
                assert r['params']==PARAMS[k]
                m.cuda().eval()
                replay=predictions(m,val,norm,torch.device('cuda'))
                err=max(float(abs(replay[p][q]-raw[p][q]).max()) for p in raw for q in raw[p]);assert err<2e-4
                assert max(abs(metrics(replay,val,r['threshold'])[2][c]-r[c]) for c in COLS)<1e-10
                ts=metrics(predictions(m,train,norm,torch.device('cuda')),train,r['threshold'])[2]
                checks.append(dict(run_id=run,replay_error=err,train_f1=ts['macro_f1_tol1'],gap=ts['macro_f1_tol1']-r['macro_f1_tol1']))
                rows.append({key:value for key,value in r.items() if key!='history'})
                for strength in (0.,.5):
                    row=dec.evaluate(raw,val,r['threshold'],prior,strength,f'{run}_lambda{strength:g}',contract['contract'])
                    if strength==0:assert max(abs(row[c]-r[c]) for c in COLS)<1e-10
                    decoded.append(dict(kind=k,fold=f,seed=s,**row))
                pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',run,flush=True)
    neural=pd.DataFrame(rows);neural.to_csv(ART/'summary.csv',index=False)
    d=pd.DataFrame(decoded);assert len(d)==16;d.to_csv(ART/'decoder/summary.csv',index=False)
    ref=pd.read_csv(ROOT/'artifacts/interstart_decoder_study/summary.csv');ref=ref[ref.strength.isin((0.,.5))].copy();ref['kind']='C3'
    allrows=pd.concat([d,ref],ignore_index=True);comparisons=[]
    allrows.groupby(['kind','strength'])[COLS].mean().to_csv(OUT/'means.csv')
    for candidate,control in (('P','C3'),('P','Z'),('Z','C3')):
        for strength in (0.,.5):
            a=allrows[(allrows.kind==candidate)&(allrows.strength==strength)].set_index(['fold','seed'])
            b=allrows[(allrows.kind==control)&(allrows.strength==strength)].set_index(['fold','seed']);delta=a[COLS]-b[COLS]
            delta.to_csv(OUT/f'{candidate}_minus_{control}_lambda{strength:g}.csv')
            comparisons.append(dict(candidate=candidate,control=control,strength=strength,**delta.mean().to_dict(),f1_positive=int((delta.macro_f1_tol1>0).sum()),passed=bool(delta.macro_f1_tol1.mean()>=.015 and delta.macro_f1_tol0.mean()>=0 and (delta.macro_f1_tol1>0).sum()>=3)))
    write(OUT/'comparisons.json',comparisons)
    assert all(sha(p)==h for p,h in cp_hash.items()) and all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'checkpoint_hashes.json',cp_hash)
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=8,full_gpu_replays=8,checkpoints=16,decoder_cells=16,old_metrics_recomputed=4,hashes_unchanged=True,norms_rebuilt=True,test_used=False,training_seconds=float(neural.seconds.sum()),audit_seconds=time.monotonic()-began))
    write(OUT/'decoder_state/STATE.json',dict(status='complete',pid=None));write(OUT/'STATE.json',dict(status='complete',pid=None,contract=contract['contract']))
    print('COMPLETE\n'+allrows.groupby(['kind','strength'])[COLS].mean().to_string(),flush=True)


def train_one(kind,fold,seed,contract):
    run=f'{kind}_seed{seed}_fold{fold}';respath=ART/'metrics'/f'{run}.json'
    if respath.exists():assert read(respath)['contract']==contract;print('CACHED',run,flush=True);return
    state=read(OUT/'STATE.json');dest=ART/'checkpoints'/run;dest.mkdir(exist_ok=True);best=dest/'best.pt';latest=dest/'latest.pt'
    ids=split_ids(fold);train=dataset(ids['train'],kind);val=dataset(ids['validation'],kind)
    norm=normalizer(train);model=make_model(kind,seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
    sampler=PairSampler(train,norm,64,32,seed);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
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
        paired=sampler.pair_batch.to(device);penalty=consistency_forward(model,logit,paired,valid,mask);loss=loss+(.05 if kind=='P' else 0.)*penalty
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

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=parser.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    engine.OUT=OUT;engine.ART=ART;engine.CAP=1800.;engine.dataset=dataset;engine.make_model=PairBoundary;engine.predictions=predictions;engine.normalizer=normalizer
    if args.stage=='audit':audit();return
    digest=prepare()
    if args.stage=='prepare':print(digest);return
    for f in (0,1):
        for s in (42,43):
            for k in KINDS:
                guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent'],'Budget soft stop'
                train_one(k,f,s,digest)
    write(OUT/'STATE.json',{**read(OUT/'STATE.json'),'status':'auditing','pid':os.getpid()});audit()


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',{**(read(OUT/'STATE.json') if (OUT/'STATE.json').exists() else {}),'status':'failed','pid':None});raise
