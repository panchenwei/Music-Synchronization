"""CBM-inspired 7-band global segmentation, explicit length-prior control."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from . import external_audio_trial as base
from .external_structure_probe import shuffled
from .evaluation import evaluate_piece
from .score_context_study import ROOT,read,write,sha

OUT=ROOT/'reports/external_cbm_probe';ART=ROOT/'artifacts/external_cbm_probe'
COLS=['f1_tol1','f1_tol0','precision_tol1','recall_tol1']


def band_scores(x,band=7,max_length=32):
    x=np.asarray(x,np.float64);n=len(x);unit=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-10)
    # Score [start,end) = 2 * sum positive-lag similarities / length.
    prefixes={lag:np.r_[0.,np.einsum('ij,ij->i',unit[:-lag],unit[lag:]).cumsum()] for lag in range(1,min(band,n-1)+1)}
    scores=np.full((n+1,max_length+1),-np.inf)
    for end in range(1,n+1):
        lengths=np.arange(1,min(max_length,end)+1);starts=end-lengths;total=np.zeros(len(lengths))
        for lag,prefix in prefixes.items():
            use=lengths>lag
            if use.any():total[use]+=2*(prefix[end-lag]-prefix[starts[use]])
        scores[end,lengths]=total/lengths
    return scores


def segment(x,band=7,max_length=32):
    scores=band_scores(x,band,max_length);n=len(x);dp=np.full(n+1,-np.inf);dp[0]=0.;prev=np.full(n+1,-1,int)
    for end in range(1,n+1):
        # Ascending antecedents, fixed first maximum; no ground-truth tie breaking.
        starts=np.arange(max(0,end-max_length),end);values=dp[starts]+scores[end,end-starts];idx=int(values.argmax())
        dp[end]=values[idx];prev[end]=starts[idx]
    cuts=[];end=n
    while end>0:
        start=int(prev[end]);assert 0<=start<end
        if start>0:cuts.append(start)
        end=start
    cuts=np.array(sorted(cuts),int);return cuts,float(dp[n])


def main():
    began=time.monotonic()
    for p in (OUT,ART):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(ROOT/'reports/external_structure_probe/contract.json')['hashes'])
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'tests/test_external_cbm_probe.py',ROOT/'reports/external_structure_probe/held_development_work_metrics.csv'):hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items());digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));data=base.load_data();sp=read(base.OUT/'splits.json');emb={};avail={}
    for k,v in data.items():
        with np.load(ROOT/'artifacts/external_mert_features'/f'{k}.npz',allow_pickle=False) as z:
            emb[k]=z['embeddings'].copy();avail[k]=z['availability'].astype(bool)&(v['features'][:,-1]>.5)
        assert int((avail[k]*v['label_mask']).sum())==int(v['label_mask'].sum())
    rows=[];workrows=[];selection=[];lengths=[]
    for f in range(3):
        train=base.subset(data,sp[str(f)]['train']);held=base.subset(data,sp[str(f)]['test'])
        scaler=StandardScaler().fit(np.concatenate([emb[k][avail[k]] for k in sorted(train)]))
        with np.load(ROOT/'artifacts/external_structure_probe'/f'M_fold{f}_scaler.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(scaler.mean_,z['mean']);np.testing.assert_array_equal(scaler.scale_,z['scale'])
        for kind in ('M','D','U'):
            perfs=[]
            for k,v in sorted(held.items()):
                raw=emb[k] if kind!='D' else shuffled(emb[k],avail[k],k)
                x=scaler.transform(raw) if kind!='U' else np.ones((len(raw),1));x[~avail[k]]=0
                cuts,objective=segment(x);n=len(x);assert ((cuts>0)&(cuts<n)).all()
                dest=ART/f'{kind}_fold{f}_{k}.npz'
                if dest.exists():
                    with np.load(dest,allow_pickle=False) as z:np.testing.assert_array_equal(cuts,z['cuts']);assert abs(float(z['objective'])-objective)<1e-10
                else:np.savez_compressed(dest,cuts=cuts,objective=objective)
                with np.load(dest,allow_pickle=False) as z:stored=z['cuts'].copy()
                binary=np.zeros(n);binary[stored]=1;score=evaluate_piece(k,binary,v['labels'],v['label_mask'],.5)
                row=dict(record=k,piece_id=v['piece_id'],group=v['group'],performance=v['performance_id'],**{c:score[c] for c in COLS});perfs.append(row)
                check=np.zeros(n);check[cuts]=1;again=evaluate_piece(k,check,v['labels'],v['label_mask'],.5)
                assert all(score[f'{a}_tol{t}']==again[f'{a}_tol{t}'] for a in ('tp','fp','fn') for t in (0,1))
                lengths.extend(dict(kind=kind,fold=f,record=k,length=int(d)) for d in np.diff(np.r_[0,cuts,n]));selection.append(dict(kind=kind,fold=f,record=k,objective=objective,cuts=len(cuts)))
                assert time.monotonic()-began<900
            frame=pd.DataFrame(perfs);frame.to_csv(OUT/f'{kind}_fold{f}_performances.csv',index=False);work=frame.groupby(['piece_id','group'],as_index=False)[COLS].mean()
            workrows.extend([dict(kind=kind,fold=f,**r) for r in work.to_dict('records')]);rows.append(dict(kind=kind,fold=f,**work[COLS].mean().to_dict()));print('CBM',kind,f,rows[-1],flush=True)
    work=pd.DataFrame(workrows);work.to_csv(OUT/'held_development_work_metrics.csv',index=False);means=work.groupby('kind')[COLS].mean();means.to_csv(OUT/'means.csv')
    pd.DataFrame(rows).to_csv(OUT/'fold_metrics.csv',index=False);pd.DataFrame(lengths).to_csv(OUT/'segment_lengths.csv',index=False);write(OUT/'path_audit.json',selection)
    old=pd.read_csv(ROOT/'reports/external_structure_probe/held_development_work_metrics.csv');old=old[old.kind=='M'];keys=['fold','piece_id','group'];a=work[work.kind=='M'].set_index(keys);comparisons=[];rng=np.random.default_rng(20260914)
    for label,b in [('D',work[work.kind=='D']),('U',work[work.kind=='U']),('local_novelty',old)]:
        d=a[COLS]-b.set_index(keys)[COLS];delta=d.reset_index();groups=sorted(delta.group.unique());samples=[]
        for _ in range(2000):samples.append(np.concatenate([delta.loc[delta.group==g,'f1_tol1'].to_numpy() for g in rng.choice(groups,len(groups),replace=True)]).mean())
        cells=d.groupby('fold').mean();comparisons.append(dict(reference=label,**d.mean().to_dict(),positive_folds=int((cells.f1_tol1>0).sum()),conditional_group_ci=np.quantile(samples,[.025,.975]).tolist(),passed=bool(d.f1_tol1.mean()>=.015 and d.f1_tol0.mean()>=0 and (cells.f1_tol1>0).sum()>=2)))
    write(OUT/'comparisons.json',comparisons);assert all(sha(p)==h for p,h in hashes.items());write(OUT/'artifact_hashes.json',{str(p):sha(p) for p in ART.glob('*')})
    write(OUT/'completion_audit.json',dict(status='complete',training_runs=0,cells=9,path_replays=len(selection),source_hashes_unchanged=True,scalers_reproduced=True,threshold_fits=0,new_blind_test=False,seconds=time.monotonic()-began))
    print(means.to_string(),flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
