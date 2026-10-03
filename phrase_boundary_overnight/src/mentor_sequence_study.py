"""Mentor-guided controlled context study; isolated, budgeted and resumable."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, time, traceback, copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .mentor_sequence_models import make_model, ContextSampler
from .run_recurrence_depth_study import dataset
from .score_context_study import ROOT, read, write, sha, normalizer, GRID
from .local_context_study import predictions, metrics
from .phase2_models import choose_single_threshold
from .phase6_models import positive_weight
from .context_inference_audit_v2 import save_raw, error
from .audit_external_stem_transfer import checked_raw

OUT=ROOT/'reports/mentor_continuation_20260916'
ART=ROOT/'artifacts/mentor_sequence_20260916'
KINDS=('C_full','G64','Gfull','T64','Tfull')
COLS=['macro_f1_tol1','macro_f1_tol0','macro_precision_tol1','macro_recall_tol1','raw_ap']


def splits():
    f=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv',dtype={'opus':str})
    excluded=set(f[f.fold.isin([0,1])&(f.split=='validation')].piece_id)
    pool=f[['piece_id','opus']].drop_duplicates();pool=pool[~pool.piece_id.isin(excluded)]
    assert len(excluded)==19 and len(pool)==24
    rows=[]
    for fold,opuses in enumerate((['06','30'],['07','67'])):
        for r in pool.itertuples():rows.append(dict(fold=fold,piece_id=r.piece_id,opus=r.opus,split='validation' if r.opus in opuses else 'train'))
    frame=pd.DataFrame(rows).sort_values(['fold','split','piece_id'])
    for fold in (0,1):
        q=frame[frame.fold==fold];a=q[q.split=='train'];b=q[q.split=='validation']
        assert len(a)==17 and len(b)==7 and not(set(a.opus)&set(b.opus))
    return frame,sorted(excluded)


def digest_array(a):
    a=np.asarray(a)
    if a.dtype.kind in 'OUS':content=json.dumps(a.tolist(),ensure_ascii=False).encode()
    else:content=np.ascontiguousarray(a).tobytes()
    return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+content).hexdigest()


def prepare():
    for p in (OUT,ART,ART/'checkpoints',ART/'metrics'):p.mkdir(parents=True,exist_ok=True)
    frame,excluded=splits();frame.to_csv(OUT/'sequence_split_manifest.csv',index=False)
    data=dataset(sorted(frame.piece_id.unique()),'C3')
    inputs={p:{key:digest_array(v[key]) for key in ('curves','labels','label_mask','performance_ids','pitch_profiles')} for p,v in data.items()}
    paths=list((ROOT/'src').glob('*.py'))+[OUT/'SEQUENCE_PROTOCOL.md',OUT/'sequence_split_manifest.csv',ROOT/'tests/test_mentor_sequence.py']
    sources={str(p):sha(p) for p in paths}
    value=dict(sources=sources,inputs=inputs,excluded_recent_dev=excluded,independent_test=False)
    contract=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    path=OUT/'sequence_contract.json'
    if path.exists():assert read(path)['contract']==contract,'Frozen study contract changed'
    else:write(path,dict(contract=contract,**value))
    pd.DataFrame([dict(piece_id=p,beats=len(v['labels']),performances=len(v['curves']),positive=int((v['labels']*v['label_mask']).sum())) for p,v in data.items()]).to_csv(OUT/'sequence_data_inventory.csv',index=False)
    return frame,data,contract


def guard(new=False):
    b=read(OUT/'BUDGET.json');stop=b['stop_new_runs_used_percent'] if new else b['absolute_stop_used_percent']
    if b['observed_used_percent']>=stop:raise TimeoutError('Account budget stop')
    completed=sum(read(p)['seconds'] for p in (ART/'metrics').glob('*_fold*.json'))
    return completed


def gpu_probe(data):
    path=OUT/'sequence_gpu_preflight.json'
    if path.exists():return
    norm=normalizer(data);sampler=ContextSampler(data,norm,42,True)
    x,y,mask,valid=[v.cuda() for v in sampler.batch()]
    rows=[]
    for kind in KINDS:
        m=make_model(kind,42).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
        def update():
            opt.zero_grad(set_to_none=True);loss=(torch.nn.functional.binary_cross_entropy_with_logits(m(x,padding_mask=~valid.bool()),y,reduction='none')*mask).sum()/mask.sum();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();return float(loss.detach())
        first=update();weights=copy.deepcopy(m.state_dict());state=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state();crng=torch.cuda.get_rng_state_all()
        expected=update();after=copy.deepcopy(m.state_dict());m.load_state_dict(weights);opt.load_state_dict(state);torch.set_rng_state(rng);torch.cuda.set_rng_state_all(crng)
        actual=update();assert expected==actual and all(torch.equal(v,m.state_dict()[k]) for k,v in after.items())
        rows.append(dict(kind=kind,params=sum(p.numel() for p in m.parameters()),shape=list(x.shape),first_loss=first,exact_gpu_update_replay=True))
        del m,opt
    write(path,dict(status='passed',rows=rows,probe_only_not_training_results=True))


def train_one(kind,fold,seed,train,val,contract):
    run=f'{kind}_seed{seed}_fold{fold}';resultpath=ART/'metrics'/f'{run}.json'
    if resultpath.exists():assert read(resultpath)['contract']==contract;print('CACHED',run,flush=True);return
    completed=guard(True);dest=ART/'checkpoints'/run;dest.mkdir(exist_ok=True);latest=dest/'latest.pt'
    norm=normalizer(train);model=make_model(kind,seed).cuda();sampler=ContextSampler(train,norm,seed,kind not in ('G64','T64'))
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);pw=positive_weight(train,10)
    criterion=torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(pw,device='cuda'))
    step=0;history=[];best=-1.;prior=0.;exposure=0
    if latest.exists():
        s=torch.load(latest,map_location='cuda',weights_only=False);assert s['contract']==contract
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);sampler.load_state(s['sampler']);step=s['step'];history=s['history'];best=s['best_score'];prior=s['seconds'];exposure=s['supervised_positions']
        torch.set_rng_state(s['rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in s['cuda_rng']])
    began=time.monotonic();torch.cuda.reset_peak_memory_stats()
    def snapshot():return dict(kind=kind,fold=fold,seed=seed,model=model.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state(),step=step,history=history,best_score=best,seconds=prior+time.monotonic()-began,supervised_positions=exposure,contract=contract,mean=norm.mean,std=norm.std,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    write(OUT/'SEQUENCE_STATE.json',dict(status='running',run=run,pid=os.getpid(),contract=contract))
    try:
        while step<600:
            if step%25==0:
                guard();elapsed=prior+time.monotonic()-began
                if elapsed>=1500 or completed+elapsed>=18000:raise TimeoutError('Fixed training time cap reached')
            model.train();x,y,mask,valid=[v.cuda() for v in sampler.batch()];opt.zero_grad(set_to_none=True)
            loss=(criterion(model(x,padding_mask=~valid.bool()),y)*mask).sum()/mask.sum().clamp_min(1)
            assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(gn);opt.step();step+=1;exposure+=int(mask.sum())
            if step%100==0:
                raw=predictions(model,val,norm,torch.device('cuda'));assert model.training
                threshold,_=choose_single_threshold(raw,val,GRID);_,_,score=metrics(raw,val,threshold)
                history.append(dict(step=step,threshold=float(threshold),loss=float(loss.detach()),gradient=float(gn),lr=float(opt.param_groups[0]['lr']),**score))
                if score['macro_f1_tol1']>best+1e-9:best=score['macro_f1_tol1'];torch.save(snapshot(),dest/'best.pt')
                print(run,'step',step,'F1',round(score['macro_f1_tol1'],5),'AP',round(score['raw_ap'],5),flush=True)
            if step%25==0:torch.save(snapshot(),latest)
    except BaseException:
        torch.save(snapshot(),latest);raise
    s=torch.load(dest/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(s['model']);threshold=s['history'][-1]['threshold']
    np.testing.assert_array_equal(norm.mean,s['mean']);np.testing.assert_array_equal(norm.std,s['std'])
    raw=predictions(model,val,norm,torch.device('cuda'));perfs,pieces,score=metrics(raw,val,threshold)
    path=ART/'metrics'/f'{run}_predictions.csv.gz';save_raw(raw,path,val);stored=checked_raw(pd.read_csv(path),val)
    _,_,recomputed=metrics(stored,val,threshold)
    replay=make_model(kind,seed).cuda();replay.load_state_dict(torch.load(dest/'best.pt',map_location='cuda',weights_only=False)['model'])
    delta=error(raw,predictions(replay,val,norm,torch.device('cuda')));assert delta<1e-7
    assert all(abs(score[k]-recomputed[k])<1e-10 for k in COLS)
    train_score=metrics(predictions(model,train,norm,torch.device('cuda')),train,threshold)[2]
    result=dict(run_id=run,kind=kind,fold=fold,seed=seed,contract=contract,best_step=s['step'],last_step=step,threshold=threshold,params=sum(p.numel() for p in model.parameters()),seconds=prior+time.monotonic()-began,supervised_positions=exposure,pos_weight=pw,peak_gpu_bytes=torch.cuda.max_memory_allocated(),checkpoint_replay_error=delta,stored_probability_metric_replay=True,train_at_dev_threshold=train_score,**score)
    perfs.to_csv(ART/'metrics'/f'{run}_performances.csv',index=False);pieces.to_csv(ART/'metrics'/f'{run}_pieces.csv',index=False);write(resultpath,result)
    print('AUDITED',run,'best',s['step'],'F1',round(score['macro_f1_tol1'],5),'seconds',round(result['seconds']),flush=True)


def report():
    rows=[read(p) for p in (ART/'metrics').glob('*_fold*.json')]
    if not rows:return
    f=pd.DataFrame(rows);f.to_csv(OUT/'sequence_runs.csv',index=False);f.groupby('kind')[COLS+['seconds','params']].mean().to_csv(OUT/'sequence_means.csv')
    comparisons=[]
    for a,b in [('Gfull','G64'),('Tfull','T64'),('Gfull','C_full'),('Tfull','C_full')]:
        x=f[f.kind==a].set_index(['fold','seed']);y=f[f.kind==b].set_index(['fold','seed']);ix=x.index.intersection(y.index)
        if not len(ix):continue
        d=x.loc[ix,COLS]-y.loc[ix,COLS];pos=int((d.macro_f1_tol1>0).sum());v=d.mean()
        comparisons.append(dict(candidate=a,reference=b,cells=len(ix),**{k+'_delta':float(v[k]) for k in COLS},positive_cells=pos,passed=bool(len(ix)==4 and v.macro_f1_tol1>=.015 and pos>=3 and v.macro_f1_tol0>=0 and v.raw_ap>=0)))
    pd.DataFrame(comparisons).to_csv(OUT/'sequence_comparisons.csv',index=False)
    sources=read(OUT/'sequence_contract.json')['sources'];assert all(sha(p)==h for p,h in sources.items())
    write(OUT/'SEQUENCE_STATE.json',dict(status='complete' if len(rows)==20 else 'partial',completed=len(rows),planned=20,source_hashes_unchanged=True,independent_test=False,pid=None))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-runs',type=int,default=20);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    frame,data,contract=prepare();gpu_probe(data)
    if args.prepare_only:return
    count=0
    try:
        for seed in (42,43):
            for fold in (0,1):
                part=frame[frame.fold==fold];train={p:data[p] for p in part[part.split=='train'].piece_id};val={p:data[p] for p in part[part.split=='validation'].piece_id}
                for kind in KINDS:
                    if count>=args.max_runs:return
                    train_one(kind,fold,seed,train,val,contract);count+=1;report()
    except BaseException as e:
        write(OUT/'sequence_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),pid=os.getpid()));raise
    finally:report()


if __name__=='__main__':main()
