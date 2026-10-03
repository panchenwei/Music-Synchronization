"""Fixed FMP/MSAF-style adaptive peak selection on cached novelty scores."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d,median_filter
from threadpoolctl import threadpool_limits
from . import external_audio_trial as base
from .evaluation import evaluate_piece
from .score_context_study import ROOT,read,write,sha

OUT=ROOT/'reports/external_adaptive_peak_probe';ART=ROOT/'artifacts/external_adaptive_peak_probe'
SOURCE=ROOT/'artifacts/external_structure_probe';COLS=['f1_tol1','f1_tol0','precision_tol1','recall_tol1']


def pick(x,global_threshold=None):
    x=np.asarray(x,dtype=float);assert x.ndim==1 and len(x)>0 and np.isfinite(x).all()
    smooth=gaussian_filter1d(x,4.,mode='reflect')
    threshold=median_filter(smooth,size=16,mode='reflect')+.05*x.mean() if global_threshold is None else np.full(len(x),global_threshold)
    peak=np.zeros(len(x),bool)
    peak[1:-1]=(smooth[1:-1]>smooth[:-2])&(smooth[1:-1]>smooth[2:])&(smooth[1:-1]>threshold[1:-1])
    return np.flatnonzero(peak),smooth,threshold


def main():
    began=time.monotonic()
    for p in (OUT,ART):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(ROOT/'reports/external_structure_probe/contract.json')['hashes']);hashes.update(read(ROOT/'reports/external_structure_probe/artifact_hashes.json'))
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'tests/test_external_adaptive_peak_probe.py',ROOT/'reports/external_structure_probe/fold_metrics.csv',ROOT/'reports/external_structure_probe/held_development_work_metrics.csv'):hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items());digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));data=base.load_data();splits=read(base.OUT/'splits.json');old=pd.read_csv(ROOT/'reports/external_structure_probe/fold_metrics.csv');workrows=[];positions=[]
    for fold in range(3):
        held=base.subset(data,splits[str(fold)]['test']);threshold=float(old[(old.fold==fold)&(old.kind=='M')].iloc[0].threshold)
        for kind,source,global_threshold in [('A','M',None),('S','M',threshold),('D','MD',None),('C','C',None)]:
            with np.load(SOURCE/f'{source}_fold{fold}_predictions.npz',allow_pickle=False) as z:raw={k:z[k].copy() for k in held}
            rows=[]
            for key,item in held.items():
                cuts,smooth,local=pick(raw[key],global_threshold);binary=np.zeros(len(raw[key]));binary[cuts]=1
                metrics=evaluate_piece(key,binary,item['labels'],item['label_mask'],.5)
                rows.append(dict(record=key,piece_id=item['piece_id'],group=item['group'],**{c:metrics[c] for c in COLS}))
                p=ART/f'{kind}_fold{fold}_{key}.npz'
                if p.exists():
                    with np.load(p,allow_pickle=False) as z:np.testing.assert_array_equal(z['cuts'],cuts)
                else:np.savez_compressed(p,cuts=cuts,smooth=smooth,local_threshold=local)
                with np.load(p,allow_pickle=False) as z:stored=z['cuts'].copy()
                replay=np.zeros(len(binary));replay[stored]=1;check=evaluate_piece(key,replay,item['labels'],item['label_mask'],.5)
                assert all(check[f'{c}_tol{t}']==metrics[f'{c}_tol{t}'] for c in ('tp','fp','fn') for t in (0,1))
                positions.append(dict(kind=kind,fold=fold,record=key,cuts=len(cuts)));assert time.monotonic()-began<600
            frame=pd.DataFrame(rows);frame.to_csv(OUT/f'{kind}_fold{fold}_performances.csv',index=False)
            work=frame.groupby(['piece_id','group'],as_index=False)[COLS].mean();workrows.extend([dict(kind=kind,fold=fold,**r) for r in work.to_dict('records')]);print(kind,fold,work[COLS].mean().to_dict(),flush=True)
    work=pd.DataFrame(workrows);work.to_csv(OUT/'held_development_work_metrics.csv',index=False);means=work.groupby('kind')[COLS].mean();means.to_csv(OUT/'means.csv')
    previous=pd.read_csv(ROOT/'reports/external_structure_probe/held_development_work_metrics.csv');keys=['fold','piece_id','group'];a=work[work.kind=='A'].set_index(keys);comparisons=[]
    for ref,b in [('global_M',previous[previous.kind=='M']),('smooth_global_S',work[work.kind=='S']),('shuffled_D',work[work.kind=='D']),('CQT_C',work[work.kind=='C'])]:
        d=a[COLS]-b.set_index(keys)[COLS];cells=d.groupby('fold').mean()
        comparisons.append(dict(reference=ref,**d.mean().to_dict(),positive_folds=int((cells.f1_tol1>0).sum()),passed=bool(d.f1_tol1.mean()>=.015 and d.f1_tol0.mean()>=0 and (cells.f1_tol1>0).sum()>=2)))
    write(OUT/'comparisons.json',comparisons);write(OUT/'position_audit.json',positions)
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'artifact_hashes.json',{str(p):sha(p) for p in ART.glob('*')})
    write(OUT/'completion_audit.json',dict(status='complete',training_runs=0,decode_cells=12,position_replays=len(positions),threshold_fits=0,source_hashes_unchanged=True,new_blind_test=False,seconds=time.monotonic()-began))
    print(means.to_string(),flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
