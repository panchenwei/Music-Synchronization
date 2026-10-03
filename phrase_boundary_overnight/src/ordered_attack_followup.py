"""Descriptive frozen-prediction follow-up on C3's preidentified hard sites."""
from pathlib import Path
import numpy as np
import pandas as pd
from .score_context_study import ROOT,read,write,sha
from .error_review_30 import matches
OUT=ROOT/'reports/ordered_attack_followup';ART=ROOT/'artifacts/ordered_attack_study'

def main():
    OUT.mkdir(exist_ok=True);assert read(ROOT/'reports/ordered_attack_study/completion_audit.json')['status']=='complete'
    hashes={str(__file__):sha(__file__)};old=ROOT/'reports/current_boundary_diagnosis/joint_sites.csv';hashes[str(old)]=sha(old)
    baseline=pd.read_csv(old);hard=baseline[(baseline.strength==.5)&(baseline['max']<=.2)]
    hardset=set(zip(hard.piece_id,hard.beat));assert len(hardset)==34
    cases=pd.read_csv(ROOT/'reports/error_review_30/cases.csv');sites=[];follow=[];perf_rows=[]
    for f in (0,1):
        for s in (42,43):
            for k in ('R','U','Z'):
                run=f'{k}_seed{s}_fold{f}';p=ART/'metrics'/f'{run}_predictions.csv.gz';pos=ART/'decoder'/f'{run}_lambda0.5_positions.csv.gz'
                hashes.update({str(v):sha(v) for v in (p,pos)})
                frame=pd.read_csv(p);pg={(a,b):g.beat.to_numpy(int) for (a,b),g in pd.read_csv(pos).groupby(['piece_id','performance_id'])}
                for pid,piece in frame.groupby('piece_id'):
                    hits={};nperf=0
                    for perf,g in piece.groupby('performance_id'):
                        g=g.sort_values('beat');mask=g.valid.to_numpy(bool);truth=g.loc[(g.label>.5)&g.valid.astype(bool),'beat'].to_numpy(int)
                        pred=pg.get((pid,perf),np.array([],int));pred=pred[mask[pred]];pairs,fp,fn=matches(pred,truth);nperf+=1
                        for t in truth:hits[t]=hits.get(t,0)+int(any(tt==t for _,tt in pairs))
                        far=sum(min(abs(truth-b))>3 for b in fp)
                        perf_rows.append(dict(kind=k,fold=f,seed=s,piece_id=pid,performance_id=perf,tp=len(pairs),fp=len(fp),fn=len(fn),far_fp=far))
                        for r in cases[(cases.piece_id==pid)&(cases.performance_id==perf)].itertuples():
                            target=int(r.matched_truth) if r.category=='TP' else int(r.beat)
                            follow.append(dict(kind=k,fold=f,seed=s,case_id=r.case_id,category=r.category,probability=float(g.iloc[int(r.beat)].probability),near_prediction=bool(any(abs(pred-int(r.beat))<=1)),truth_hit=bool(any(t==target for _,t in pairs))))
                    sites.extend(dict(kind=k,fold=f,seed=s,piece_id=pid,beat=int(t),hit_fraction=v/nperf,was_c3_hard=(pid,t) in hardset) for t,v in hits.items())
    st=pd.DataFrame(sites);summary=[]
    for k,g in st.groupby('kind'):
        joint=g.groupby(['piece_id','beat']).hit_fraction.max()
        hh=g[g.was_c3_hard]
        summary.append(dict(kind=k,sites=len(joint),hard_sites=int((joint<=.2).sum()),old_hard_mean_hit=float(hh.hit_fraction.mean()),old_hard_still_hard=int((hh.groupby(['piece_id','beat']).hit_fraction.max()<=.2).sum())))
    st.to_csv(OUT/'sites.csv',index=False);pd.DataFrame(follow).to_csv(OUT/'old_30_followup.csv',index=False);pd.DataFrame(perf_rows).to_csv(OUT/'performance_counts.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False);assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'source_hashes.json',hashes);write(OUT/'completion_audit.json',dict(status='complete',training=0,test_used=False,c3_hard_sites=34,prediction_hashes_unchanged=True))
    print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__':main()
