"""Preregistered score/basic-audio/CQT ablation on new sonata-held-out groups."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from .score_context_study import ROOT,read,write,sha,GRID
from .models import Normalizer
from .recurrence_depth_models import make_model as cnn
from .phase2_models import nms_probabilities
from .evaluation import evaluate_piece
from .phase6_models import window_starts

OUT=ROOT/'reports/external_audio_trial';ART=ROOT/'artifacts/external_audio_trial';FEATURES=ROOT/'reports/external_audio_features'
COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap','macro_precision_tol1','macro_recall_tol1']


def make_model(seed):
    model=cnn('C3',seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+414);model.input_projection=nn.Linear(121,32)
    assert sum(p.numel() for p in model.parameters())==7937
    return model


def mask_modalities(x,kind):
    x=x.copy()
    if kind=='S':x[...,28:120]=0
    elif kind=='E':x[...,32:120]=0
    else:assert kind=='F'
    return x


def splits(groups):
    ordered=np.random.default_rng(20260914).permutation(sorted(set(groups))).tolist();parts=np.array_split(ordered,3);result={}
    for f,part in enumerate(parts):
        test=list(part);other=[g for g in ordered if g not in test];val=other[:2];train=other[2:]
        assert not set(train)&set(val) and not set(test)&set(train+val)
        result[str(f)]=dict(train=train,validation=val,test=test)
    return result


def load_data():
    frame=pd.read_csv(FEATURES/'feature_manifest.csv');data={}
    for r in frame.itertuples():
        assert sha(r.path)==r.sha256
        with np.load(r.path,allow_pickle=False) as z:
            token=Path(r.path).stem
            data[token]=dict(features=z['features'].copy(),labels=z['labels'].copy(),label_mask=z['label_mask'].copy(),
                piece_id=r.piece_id,performance_id=r.performance,group=r.group,audio_hash=r.audio_sha256)
    return data


def subset(data,groups):return {k:v for k,v in data.items() if v['group'] in groups}


def normalizer(data):
    # All permitted input frames, not only the ground-truth known mask.
    x=np.concatenate([v['features'] for v in data.values()]);return Normalizer(x.mean(0),np.maximum(x.std(0),1e-4))


class Sampler:
    def __init__(self,data,norm,kind,seed):
        self.data=data;self.norm=norm;self.kind=kind;self.rng=np.random.default_rng(seed)
        self.pieces=sorted({v['piece_id'] for v in data.values()});self.records={p:sorted(k for k,v in data.items() if v['piece_id']==p) for p in self.pieces}
    def state(self):return copy.deepcopy(self.rng.bit_generator.state)
    def load_state(self,state):self.rng.bit_generator.state=state
    def batch(self):
        rows=[]
        for _ in range(32):
            p=str(self.rng.choice(self.pieces));key=str(self.rng.choice(self.records[p]));v=self.data[key];n=len(v['labels'])
            start=int(self.rng.choice(window_starts(n,64,32)));valid=min(64,n-start)
            x=mask_modalities(self.norm.apply(v['features'][start:start+64]),self.kind).astype(np.float32)
            y=v['labels'][start:start+64];m=v['label_mask'][start:start+64];a=np.ones(valid,np.float32)
            rows.append((np.pad(x,((0,64-valid),(0,0))),np.pad(y,(0,64-valid)),np.pad(m,(0,64-valid)),np.pad(a,(0,64-valid))))
        return tuple(torch.from_numpy(np.stack([r[i] for r in rows])) for i in range(4))


def predict(model,data,norm,kind):
    model.eval();result={}
    with torch.no_grad():
        for key,v in sorted(data.items()):
            x=torch.from_numpy(mask_modalities(norm.apply(v['features']),kind).astype(np.float32)[None]).cuda()
            result[key]=model(x).sigmoid().cpu().numpy()[0]
    return result


def metrics(raw,data,threshold):
    rows=[]
    for key,p in sorted(raw.items()):
        v=data[key];known=v['label_mask']>.5;y=v['labels'][known]
        row=evaluate_piece(key,nms_probabilities(p),v['labels'],v['label_mask'],threshold)
        row.update(record=key,piece_id=v['piece_id'],performance_id=v['performance_id'],group=v['group'],
            raw_ap=float(average_precision_score(y,p[known])))
        rows.append(row)
    frame=pd.DataFrame(rows);cols=['f1_tol1','f1_tol0','precision_tol1','recall_tol1','raw_ap']
    works=frame.groupby(['piece_id','group'],as_index=False)[cols].mean()
    result={('raw_ap' if c=='raw_ap' else 'macro_'+c):float(works[c].mean()) for c in cols};result['threshold']=float(threshold)
    return frame,works,result


def choose(raw,data):
    rows=[metrics(raw,data,t)[2] for t in GRID]
    return sorted(rows,key=lambda r:(r['macro_f1_tol1'],r['macro_precision_tol1'],r['threshold']),reverse=True)[0]


def prepare():
    assert read(FEATURES/'completion_audit.json')['status']=='complete'
    for p in (OUT,ART/'checkpoints',ART/'metrics'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(FEATURES/'source_hashes.json'));data=load_data();sp=splits([v['group'] for v in data.values()]);checks=[]
    if (OUT/'splits.json').exists():assert read(OUT/'splits.json')==sp
    else:write(OUT/'splits.json',sp)
    for f,parts in sp.items():
        sets={s:subset(data,g) for s,g in parts.items()}
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            assert not {v['audio_hash'] for v in sets[a].values()}&{v['audio_hash'] for v in sets[b].values()}
            assert not {v['piece_id'] for v in sets[a].values()}&{v['piece_id'] for v in sets[b].values()}
        for s,d in sets.items():checks.append(dict(fold=f,split=s,groups=len(parts[s]),pieces=len({v['piece_id'] for v in d.values()}),records=len(d),known=sum(int(v['label_mask'].sum()) for v in d.values()),positive=sum(int(v['labels'].sum()) for v in d.values())))
    pd.DataFrame(checks).to_csv(OUT/'split_audit.csv',index=False)
    for p in [Path(__file__),OUT/'PROTOCOL.md',OUT/'splits.json',FEATURES/'feature_manifest.csv',ROOT/'tests/test_external_audio_trial.py',
              ROOT/'src/evaluation.py',ROOT/'src/phase2_models.py',ROOT/'src/recurrence_depth_models.py',ROOT/'src/models.py']+list((ROOT/'artifacts/external_audio_features').glob('*.npz')):hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    return data,sp,digest


def train_one(data,sp,kind,fold,seed,contract):
    name=f'{kind}_seed{seed}_fold{fold}';dest=ART/'checkpoints'/name;dest.mkdir(exist_ok=True);resultpath=ART/'metrics'/f'{name}.json'
    if resultpath.exists():assert read(resultpath)['contract']==contract;print('CACHED',name,flush=True);return
    train=subset(data,sp[str(fold)]['train']);val=subset(data,sp[str(fold)]['validation']);norm=normalizer(train);model=make_model(seed).cuda()
    sampler=Sampler(train,norm,kind,seed);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    pos=sum(v['labels'].sum() for v in train.values());known=sum(v['label_mask'].sum() for v in train.values())
    crit=nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(min((known-pos)/max(pos,1),10),device='cuda'))
    step=0;best_score=-1.;history=[];prior=0.;latest=dest/'latest.pt';best=dest/'best.pt'
    if latest.exists():
        cp=torch.load(latest,map_location='cuda',weights_only=False);assert cp['contract']==contract
        model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);sampler.load_state(cp['sampler']);step=cp['step'];best_score=cp['best_score'];history=cp['history'];prior=cp['seconds']
        torch.set_rng_state(cp['rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in cp['cuda_rng']])
    began=time.monotonic();write(OUT/'STATE.json',dict(status='training',run=name,pid=os.getpid()))
    def state():return dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state(),step=step,best_score=best_score,history=history,
        seconds=prior+time.monotonic()-began,kind=kind,fold=fold,seed=seed,contract=contract,mean=norm.mean,std=norm.std,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    while step<300:
        if prior+time.monotonic()-began>600:torch.save(state(),latest);raise TimeoutError('600s per run cap')
        model.train();x,y,m,v=(z.cuda() for z in sampler.batch());opt.zero_grad(set_to_none=True);logit=model(x,padding_mask=~v.bool())
        loss=(crit(logit,y)*m).sum()/m.sum().clamp_min(1);assert torch.isfinite(loss);loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gn);opt.step();step+=1
        if step%50==0:
            score=choose(predict(model,val,norm,kind),val);history.append(dict(step=step,loss=float(loss.detach()),grad=float(gn),**score))
            if score['macro_f1_tol1']>best_score+1e-9:best_score=score['macro_f1_tol1'];torch.save(state(),best)
            print(name,step,'validation F1',round(score['macro_f1_tol1'],4),flush=True)
        if step%25==0:torch.save(state(),latest)
    cp=torch.load(best,map_location='cuda',weights_only=False);model.load_state_dict(cp['model']);selection=max(cp['history'],key=lambda r:r['macro_f1_tol1'])
    raw=predict(model,val,norm,kind);frame,works,score=metrics(raw,val,selection['threshold'])
    assert max(abs(score[c]-selection[c]) for c in COLS)<1e-10
    np.savez_compressed(ART/'metrics'/f'{name}_validation.npz',**raw)
    write(resultpath,dict(run=name,kind=kind,fold=fold,seed=seed,contract=contract,best_step=cp['step'],params=7937,seconds=prior+time.monotonic()-began,**score))


def audit(data,sp,contract):
    began=time.monotonic();assert len(list((ART/'metrics').glob('*.json')))==18
    hashes=read(OUT/'contract.json')['hashes'];checkpoints={str(p):sha(p) for p in (ART/'checkpoints').glob('*/*.pt')};assert len(checkpoints)==36
    rows=[];checks=[];workrows=[]
    for f in range(3):
        train=subset(data,sp[str(f)]['train']);val=subset(data,sp[str(f)]['validation']);test=subset(data,sp[str(f)]['test']);norm=normalizer(train)
        for seed in (42,43):
            for kind in ('S','E','F'):
                assert time.monotonic()-began<1200
                name=f'{kind}_seed{seed}_fold{f}';meta=read(ART/'metrics'/f'{name}.json');dest=ART/'checkpoints'/name
                best=torch.load(dest/'best.pt',map_location='cpu',weights_only=False);last=torch.load(dest/'latest.pt',map_location='cpu',weights_only=False)
                assert best['contract']==last['contract']==meta['contract']==contract and last['step']==300
                assert best['step']==meta['best_step']==max(last['history'],key=lambda r:r['macro_f1_tol1'])['step']
                np.testing.assert_array_equal(norm.mean,best['mean']);np.testing.assert_array_equal(norm.std,best['std'])
                model=make_model(seed).cuda();model.load_state_dict(best['model']);valraw=predict(model,val,norm,kind)
                with np.load(ART/'metrics'/f'{name}_validation.npz',allow_pickle=False) as z:
                    err=max(float(abs(valraw[k]-z[k]).max()) for k in valraw);assert err<2e-4
                assert max(abs(metrics(valraw,val,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
                # Test outcomes are only read after every preregistered model trained.
                testpath=ART/'metrics'/f'{name}_test.npz'
                testpred=predict(model,test,norm,kind)
                if testpath.exists():
                    with np.load(testpath,allow_pickle=False) as z:
                        for k in testpred:np.testing.assert_allclose(testpred[k],z[k],atol=2e-4,rtol=0)
                else:np.savez_compressed(testpath,**testpred)
                with np.load(testpath,allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
                frame,works,score=metrics(saved,test,meta['threshold']);frame.to_csv(ART/'metrics'/f'{name}_test_performances.csv',index=False)
                assert max(abs(metrics(testpred,test,meta['threshold'])[2][c]-score[c]) for c in COLS)<1e-10
                workrows.extend([dict(kind=kind,seed=seed,fold=f,**r) for r in works.to_dict('records')]);rows.append(dict(kind=kind,seed=seed,fold=f,**score))
                checks.append(dict(run=name,validation_replay_error=err,test_metrics_recomputed=True));print('AUDITED',name,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'fold_metrics.csv',index=False);work=pd.DataFrame(workrows);work.to_csv(OUT/'test_work_metrics.csv',index=False);pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False)
    # Each held-out movement contributes equally, then average fixed seeds.
    means=work.groupby('kind')[['f1_tol1','f1_tol0','precision_tol1','recall_tol1','raw_ap']].mean();means.to_csv(OUT/'means.csv')
    comparisons=[];rng=np.random.default_rng(20260914)
    for kind,ref in [('E','S'),('F','S'),('F','E')]:
        keys=['fold','seed','piece_id','group'];a=work[work.kind==kind].set_index(keys);b=work[work.kind==ref].set_index(keys);d=a[['f1_tol1','f1_tol0','raw_ap']]-b[['f1_tol1','f1_tol0','raw_ap']]
        groups=sorted(work.group.unique());delta=d.reset_index().groupby(['group','piece_id']).mean(numeric_only=True).reset_index();samples=[]
        for _ in range(2000):
            selected=rng.choice(groups,len(groups),replace=True);values=np.concatenate([delta.loc[delta.group==g,'f1_tol1'].to_numpy() for g in selected]);samples.append(values.mean())
        fold=d.groupby(['fold','seed']).mean();lo,hi=np.quantile(samples,[.025,.975])
        comparisons.append(dict(candidate=kind,reference=ref,**d.mean().to_dict(),positive_cells=int((fold.f1_tol1>0).sum()),conditional_group_ci=[float(lo),float(hi)],
            passed=bool(d.f1_tol1.mean()>=.015 and d.f1_tol0.mean()>=0 and d.raw_ap.mean()>=0 and (fold.f1_tol1>0).sum()>=4)))
    write(OUT/'comparisons.json',comparisons)
    assert all(sha(p)==h for p,h in hashes.items()) and all(sha(p)==h for p,h in checkpoints.items());write(OUT/'checkpoint_hashes.json',checkpoints)
    write(OUT/'completion_audit.json',dict(status='complete',training_runs=18,checkpoints=36,gpu_validation_replays=18,test_inferences=18,
        source_hashes_unchanged=True,training_seconds=sum(read(p)['seconds'] for p in (ART/'metrics').glob('*.json')),audit_seconds=time.monotonic()-began,
        protocol='First fixed-matrix external sonata-held-out reference-alignment trial; not comparable to Chopin development64.14%'))
    write(OUT/'STATE.json',dict(status='complete',pid=None));print(means.to_string(),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);data,sp,digest=prepare()
    if args.stage=='prepare':print(digest,flush=True);return
    if args.stage!='audit':
        started=time.monotonic()
        for f in range(3):
            for seed in (42,43):
                for kind in ('S','E','F'):
                    guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent']
                    assert time.monotonic()-started<5400
                    train_one(data,sp,kind,f,seed,digest)
    audit(data,sp,digest)


if __name__=='__main__':
    try:
        with threadpool_limits(2):main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed',pid=None));raise
