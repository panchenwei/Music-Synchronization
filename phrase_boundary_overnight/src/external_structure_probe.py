"""Label-blind fixed checkerboard-change features, external development only."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from . import external_audio_trial as base
from .score_context_study import ROOT,read,write,sha

OUT=ROOT/'reports/external_structure_probe';ART=ROOT/'artifacts/external_structure_probe'
COLS=['f1_tol1','f1_tol0','precision_tol1','recall_tol1','raw_ap']


def shuffled(x,available,token):
    rng=np.random.default_rng(20260914+int(hashlib.sha256(token.encode()).hexdigest()[:8],16))
    x=x.copy();idx=np.flatnonzero(available);x[idx]=x[rng.permutation(idx)];return x


def novelty(x,available,width=4):
    x=np.asarray(x,np.float64);available=np.asarray(available,bool)
    assert x.ndim==2 and available.shape==(len(x),) and width==4 and np.isfinite(x).all()
    unit=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-10);unit[~available]=0
    sums=np.vstack([np.zeros((1,x.shape[1])),unit.cumsum(0)]);counts=np.r_[0,available.cumsum()];out=np.zeros(len(x))
    for b in range(len(x)):
        lo=max(0,b-width);hi=min(len(x),b+width);nl=counts[b]-counts[lo];nr=counts[hi]-counts[b]
        if nl and nr:
            d=(sums[b]-sums[lo])/nl-(sums[hi]-sums[b])/nr;out[b]=max(float(d@d)*.25,0.)
    return out


def main():
    began=time.monotonic()
    for p in (OUT,ART):p.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/external_mert_features/completion_audit.json')['status']=='complete'
    hashes=dict(read(ROOT/'reports/external_mert_features/source_hashes.json'))
    paths=[Path(__file__),OUT/'PROTOCOL.md',ROOT/'tests/test_external_structure_probe.py',base.OUT/'splits.json',ROOT/'src/external_audio_trial.py',ROOT/'src/evaluation.py',ROOT/'src/phase2_models.py']
    for p in paths:hashes[str(p)]=sha(p)
    data=base.load_data();features={};available={}
    for k,v in data.items():
        mp=ROOT/'artifacts/external_mert_features'/f'{k}.npz';hashes[str(mp)]=sha(mp)
        with np.load(mp,allow_pickle=False) as z:
            features[k]={'M':z['embeddings'].copy(),'C':v['features'][:,32:120].copy()}
            available[k]=z['availability'].astype(bool)&(v['features'][:,-1]>.5)
        assert int((available[k]*v['label_mask']).sum())==int(v['label_mask'].sum())
    assert all(sha(p)==h for p,h in hashes.items());digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));sp=read(base.OUT/'splits.json');rows=[];workrows=[];checks=[]
    for f in range(3):
        train=base.subset(data,sp[str(f)]['train']);val=base.subset(data,sp[str(f)]['validation']);held=base.subset(data,sp[str(f)]['test'])
        for rep in ('M','C'):
            scaler=StandardScaler().fit(np.concatenate([features[k][rep][available[k]] for k in sorted(train)]))
            np.savez_compressed(ART/f'{rep}_fold{f}_scaler.npz',mean=scaler.mean_,scale=scaler.scale_)
            for permute in (False,True):
                kind=rep+('D' if permute else '');raw={}
                for k in data:
                    x=features[k][rep];x=shuffled(x,available[k],k) if permute else x
                    raw[k]=novelty(scaler.transform(x),available[k])
                scale=max(float(np.quantile(np.concatenate([raw[k][available[k]] for k in sorted(train)]),.95)),1e-10)
                probs={k:a/(a+scale) for k,a in raw.items()}
                vs=base.choose({k:probs[k] for k in val},val);threshold=vs['threshold']
                name=f'{kind}_fold{f}';dest=ART/f'{name}_predictions.npz'
                if dest.exists():
                    with np.load(dest,allow_pickle=False) as z:
                        for k in probs:np.testing.assert_array_equal(probs[k],z[k])
                else:np.savez_compressed(dest,**probs)
                with np.load(dest,allow_pickle=False) as z:saved={k:z[k].copy() for k in held}
                pf,work,score=base.metrics(saved,held,threshold);pf.to_csv(ART/f'{name}_performances.csv',index=False)
                check=base.metrics({k:probs[k] for k in held},held,threshold)[2]
                assert max(abs(score[c]-check[c]) for c in base.COLS)<1e-12
                rows.append(dict(fold=f,kind=kind,threshold=threshold,train_scale=scale,validation_f1=vs['macro_f1_tol1'],**{c:score[c] for c in base.COLS}))
                workrows.extend([dict(fold=f,kind=kind,**r) for r in work.to_dict('records')])
                checks.append(dict(fold=f,kind=kind,train_groups=sp[str(f)]['train'],fit_frames=int(scaler.n_samples_seen_),known=sum(int(data[k]['label_mask'].sum()) for k in data),threshold_validation_only=True))
                print('PROBED',name,score,flush=True);assert time.monotonic()-began<900
    df=pd.DataFrame(rows);df.to_csv(OUT/'fold_metrics.csv',index=False);work=pd.DataFrame(workrows);work.to_csv(OUT/'held_development_work_metrics.csv',index=False)
    means=work.groupby('kind')[COLS].mean();means.to_csv(OUT/'means.csv');comparisons=[];rng=np.random.default_rng(20260914)
    for kind,ref in [('M','MD'),('M','C'),('C','CD')]:
        keys=['fold','piece_id','group'];d=work[work.kind==kind].set_index(keys)[COLS]-work[work.kind==ref].set_index(keys)[COLS]
        groups=sorted(work.group.unique());delta=d.reset_index();boot=[]
        for _ in range(2000):boot.append(np.concatenate([delta.loc[delta.group==g,'f1_tol1'].to_numpy() for g in rng.choice(groups,len(groups),replace=True)]).mean())
        cells=d.groupby('fold').mean();comparisons.append(dict(candidate=kind,reference=ref,**d[['f1_tol1','f1_tol0','raw_ap']].mean().to_dict(),positive_folds=int((cells.f1_tol1>0).sum()),conditional_group_ci=np.quantile(boot,[.025,.975]).tolist(),passed=bool(d.f1_tol1.mean()>=.015 and d.f1_tol0.mean()>=0 and d.raw_ap.mean()>=0 and (cells.f1_tol1>0).sum()>=2)))
    write(OUT/'comparisons.json',comparisons);write(OUT/'input_audit.json',checks)
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'artifact_hashes.json',{str(p):sha(p) for p in ART.glob('*')})
    write(OUT/'completion_audit.json',dict(status='complete',training_runs=0,cells=12,new_blind_test=False,hashes_unchanged=True,prediction_metrics_recomputed=True,pretraining_overlap_unknown=True,seconds=time.monotonic()-began))
    print(means.to_string(),flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
