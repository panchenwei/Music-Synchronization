"""Low-capacity score/novelty probe, identical frozen inputs to the CNN trial."""
import hashlib,json,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from . import external_novelty_trial as source
from .score_context_study import ROOT,read,write,sha

OUT=ROOT/'reports/external_linear_novelty';ART=ROOT/'artifacts/external_linear_novelty'
COLS=['f1_tol1','f1_tol0','precision_tol1','recall_tol1','raw_ap'];KEEP=list(range(29))+[120]


def arrays(data,norm,kind):
    result={}
    for k,v in sorted(data.items()):
        x=norm.apply(v['features'])[:,KEEP].astype(np.float64)
        if kind=='S':x[:,28]=0
        result[k]=x
    return result


def training_arrays(data,x):
    counts={}
    for k,v in data.items():counts[v['piece_id']]=counts.get(v['piece_id'],0)+int(v['label_mask'].sum())
    xx=[];yy=[];ww=[];ids=[]
    for k,v in sorted(data.items()):
        use=v['label_mask']>.5;y=v['labels'][use]
        xx.append(x[k][use]);yy.append(y);ww.append(np.full(len(y),1/counts[v['piece_id']]));ids.extend([v['piece_id']]*len(y))
    x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww);base_weights=w.copy()
    positive=min(float(w[y==0].sum()/w[y==1].sum()),10.);w[y>.5]*=positive;w/=w.mean()
    assert np.isfinite(x).all() and set(np.unique(y))=={0,1}
    return x,y,w,base_weights,ids,positive


def main():
    began=time.monotonic()
    for p in (OUT,ART):p.mkdir(parents=True,exist_ok=True)
    assert read(source.OUT/'completion_audit.json')['status']=='complete'
    hashes=dict(read(source.OUT/'contract.json')['hashes'])
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'tests/test_external_linear_novelty.py'):hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items());digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));data=source.base.load_data();splits=read(source.template.OLD_OUT/'splits.json');workrows=[];audits=[];allrows=[]
    for fold in range(3):
        for kind in ('S','E','F'):
            guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent']
            condition=source.load_fold(data,fold,kind);parts={s:source.base.subset(condition,g) for s,g in splits[str(fold)].items()};norm=source.base.normalizer(parts['train'])
            transformed={s:arrays(d,norm,kind) for s,d in parts.items()};x,y,w,bw,ids,pos=training_arrays(parts['train'],transformed['train'])
            totals=pd.DataFrame(dict(piece_id=ids,weight=bw)).groupby('piece_id').weight.sum().to_numpy();np.testing.assert_allclose(totals,1,rtol=1e-12)
            cp=ART/f'{kind}_fold{fold}.joblib'
            if cp.exists():
                state=joblib.load(cp);assert state['contract']==digest;model=state['model'];np.testing.assert_array_equal(state['mean'],norm.mean);np.testing.assert_array_equal(state['std'],norm.std)
            else:
                model=LogisticRegression(C=.1,solver='lbfgs',max_iter=1000,tol=1e-8)
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning);model.fit(x,y,sample_weight=w)
                assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
                joblib.dump(dict(model=model,mean=norm.mean,std=norm.std,contract=digest),cp)
            replay=joblib.load(cp)['model'];raw={s:{k:model.predict_proba(a)[:,1] for k,a in xx.items()} for s,xx in transformed.items()}
            threshold=source.base.choose(raw['validation'],parts['validation'])['threshold'];maxerror=0.
            for split in ('train','validation','test'):
                pp=ART/f'{kind}_fold{fold}_{split}.npz'
                if pp.exists():
                    with np.load(pp,allow_pickle=False) as z:
                        for k,v in raw[split].items():np.testing.assert_array_equal(z[k],v)
                else:np.savez_compressed(pp,**raw[split])
                with np.load(pp,allow_pickle=False) as z:
                    reloaded={k:z[k].copy() for k in raw[split]}
                for k,a in transformed[split].items():maxerror=max(maxerror,float(np.max(np.abs(replay.predict_proba(a)[:,1]-reloaded[k]))))
                perfs,works,score=source.base.metrics(reloaded,parts[split],threshold);allrows.append(dict(kind=kind,fold=fold,split=split,**score))
                perfs.to_csv(OUT/f'{kind}_fold{fold}_{split}_performances.csv',index=False)
                if split=='test':workrows.extend([dict(kind=kind,fold=fold,**r) for r in works.to_dict('records')])
            assert maxerror<1e-12
            audits.append(dict(kind=kind,fold=fold,params=int(model.coef_.size+model.intercept_.size),iterations=int(model.n_iter_[0]),threshold=threshold,known=len(y),positives=int(y.sum()),positive_weight=pos,work_weights_equal=True,replay_error=maxerror))
            print('LINEAR',kind,fold,allrows[-1],flush=True);assert time.monotonic()-began<1200
    work=pd.DataFrame(workrows);work.to_csv(OUT/'held_development_work_metrics.csv',index=False);means=work.groupby('kind')[COLS].mean();means.to_csv(OUT/'means.csv');pd.DataFrame(allrows).to_csv(OUT/'all_split_metrics.csv',index=False);pd.DataFrame(audits).to_csv(OUT/'run_audit.csv',index=False)
    keys=['fold','piece_id','group'];a=work[work.kind=='E'].set_index(keys);comparisons=[]
    for ref in ('S','F'):
        d=a[COLS]-work[work.kind==ref].set_index(keys)[COLS];cells=d.groupby('fold').mean()
        comparisons.append(dict(reference=ref,**d.mean().to_dict(),positive_folds=int((cells.f1_tol1>0).sum()),passed=bool(d.f1_tol1.mean()>=.015 and d.f1_tol0.mean()>=0 and d.raw_ap.mean()>=0 and (cells.f1_tol1>0).sum()>=2)))
    write(OUT/'comparisons.json',comparisons);assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'artifact_hashes.json',{str(p):sha(p) for p in ART.glob('*')});write(OUT/'completion_audit.json',dict(status='complete',training_runs=9,model_replays=9,saved_models=9,probability_files=27,new_blind_test=False,source_hashes_unchanged=True,seconds=time.monotonic()-began))
    print(means.to_string(),flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
