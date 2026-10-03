"""Consecutive-start interval evaluation, not DCML phrase-end evaluation."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent;sys.path[:0]=[str(ROOT),str(BASE)]
import numpy as np
import pandas as pd
from src.score_context_study import read,write,sha
from score_only_export_training import data_u,guard
OUT=BASE/'interval_quality'

def intervals(points,mask):
    p=sorted(set(int(x) for x in points));return [(a,b) for a,b in zip(p[:-1],p[1:]) if 0<=a<b<len(mask) and mask[a:b+1].all()]

def match(a,b,tol):
    d=np.zeros((len(a)+1,len(b)+1),int)
    for i,x in enumerate(a):
        for j,y in enumerate(b):d[i+1,j+1]=max(d[i,j+1],d[i+1,j],d[i,j]+int(abs(x[0]-y[0])<=tol and abs(x[1]-y[1])<=tol))
    return int(d[-1,-1])

def score(pred,truth,tol):
    tp=match(pred,truth,tol);fp=len(pred)-tp;fn=len(truth)-tp;p=tp/max(tp+fp,1);r=tp/max(tp+fn,1);f=2*tp/max(2*tp+fp+fn,1);return dict(tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=f)

def main():
    guard(True);OUT.mkdir(exist_ok=True);mask=np.ones(40,bool);assert match([(0,10),(10,20)],[(0,10),(10,20)],0)==2;assert match([(0,20)],[(0,10),(10,20)],1)==0;assert match([(0,5),(5,10)],[(0,10)],1)==0;assert match([(1,11)],[(0,10)],1)==1;assert match([(0,10),(0,10)],[(0,10)],0)==1;assert match([],[(0,1)],1)==0;mask[7]=False;assert intervals([0,10],mask)==[]
    frame,data=data_u();sources={str(Path(__file__)):sha(Path(__file__)),str(OUT/'PROTOCOL.md'):sha(OUT/'PROTOCOL.md')};rows=[];inventory=[];pairs=[]
    for fold in (0,1):
        val={p:data[p] for p in frame[(frame.fold==fold)&(frame.split=='validation')].piece_id}
        for policy in ('raw','B10'):
            path=ROOT/'artifacts/mentor_corrected_seed_ensemble_20260916/decoder'/f'ensemble_fold{fold}_{policy}_positions.csv.gz';sources[str(path)]=sha(path);pf=pd.read_csv(path);groups={(p,str(k)):g.beat.to_numpy(int) for (p,k),g in pf.groupby(['piece_id','performance_id'])}
            for pid,v in val.items():
                mask=v['label_mask'].astype(bool);truth=intervals(np.flatnonzero((v['labels']>.5)&mask),mask);inventory.append(dict(fold=fold,policy=policy,piece_id=pid,valid_starts=int(((v['labels']>.5)&mask).sum()),valid_intervals=len(truth),mask_beats=int(mask.sum())))
                for perf in v['performance_ids']:
                    pred=intervals(groups.get((pid,str(perf)),[]),mask);pairs.append(dict(fold=fold,policy=policy,piece_id=pid,performance_id=str(perf),pred=pred,truth=truth))
                    for tol in (0,1):rows.append(dict(fold=fold,policy=policy,piece_id=pid,performance_id=str(perf),tolerance=tol,**score(pred,truth,tol)))
    f=pd.DataFrame(rows);f.to_csv(OUT/'performance_metrics.csv',index=False);work=f.groupby(['piece_id','policy','tolerance']).mean(numeric_only=True).reset_index();work.to_csv(OUT/'work_metrics.csv',index=False);means=work.groupby(['policy','tolerance'])[['precision','recall','f1']].mean();means.to_csv(OUT/'means.csv');pd.DataFrame(inventory).to_csv(OUT/'coverage.csv',index=False);write(OUT/'intervals.json',pairs)
    stored=read(OUT/'intervals.json');lookup=f.set_index(['fold','policy','piece_id','performance_id','tolerance'])
    for r in stored:
        for tol in (0,1):
            a=score(r['pred'],r['truth'],tol);b=lookup.loc[(r['fold'],r['policy'],r['piece_id'],r['performance_id'],tol)];assert all(a[k]==b[k] for k in ('tp','fp','fn'))
    assert all(sha(p)==h for p,h in sources.items());write(OUT/'source_hashes.json',sources);write(OUT/'completion_audit.json',dict(status='complete',works=14,tolerances=[0,1],intervals_are_consecutive_starts_not_full_phrases=True,unknown_spans_excluded=True,preflight_passed=True,saved_event_counts_replayed=True,source_hashes_unchanged=True,new_training=False));print(means.to_string());print('GT_INTERVALS',pd.DataFrame(inventory).query("policy=='B10'").valid_intervals.sum())

if __name__=='__main__':main()
