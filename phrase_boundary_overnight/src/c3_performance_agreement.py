"""Frozen within-work cross-performance averaging diagnostic, not deployment F1."""
import time
import numpy as np
import pandas as pd
from . import run_recurrence_depth_study as base
from .score_context_study import ROOT,read,write,sha
from .three_round_round2 import split_ids
from .fixed_ensemble_study import raw_from_frame,matched_truth
from .local_context_study import metrics
OUT=ROOT/'reports/c3_performance_agreement'

def main():
    started=time.monotonic();OUT.mkdir(exist_ok=True);rows=[];sites=[];hashes={str(__file__):sha(__file__)}
    write(OUT/'STATE.json',dict(status='running'))
    for fold in (0,1):
        val=base.dataset(split_ids(fold)['validation'],'C3')
        for seed in (42,43):
            run=f'C3_seed{seed}_fold{fold}';path=base.ART/'metrics'/f'{run}_predictions.csv.gz';hashes[str(path)]=sha(path)
            rp=base.ART/'metrics'/f'{run}.json';hashes[str(rp)]=sha(rp);r=read(rp);df=pd.read_csv(path)
            raw=raw_from_frame(df,val);sc=metrics(raw,val,r['threshold'])[2]
            assert max(abs(sc[c]-r[c]) for c in base.COLS)<1e-10
            averaged={}
            for pid,perfs in raw.items():
                mat=np.stack(list(perfs.values()));mean=mat.mean(0);averaged[pid]={p:mean for p in perfs}
                mask=val[pid]['label_mask'].astype(bool);labels=val[pid]['labels'];hits={}
                for p in mat:
                    hit,truth=matched_truth(p,labels,mask,r['threshold'])
                    for t in truth:hits[t]=hits.get(t,0)+int(t in hit)
                for t,hit in hits.items():sites.append(dict(fold=fold,seed=seed,piece_id=pid,beat=t,hit_fraction=hit/len(mat),probability_std=float(mat[:,t].std())))
            pooled=metrics(averaged,val,r['threshold'])[2]
            rows.append(dict(fold=fold,seed=seed,single_f1=sc['macro_f1_tol1'],multi_recording_f1=pooled['macro_f1_tol1'],delta=pooled['macro_f1_tol1']-sc['macro_f1_tol1'],single_ap=sc['raw_ap'],multi_recording_ap=pooled['raw_ap']))
            assert time.monotonic()-started<300
    df=pd.DataFrame(rows);st=pd.DataFrame(sites);df.to_csv(OUT/'summary.csv',index=False);st.to_csv(OUT/'sites.csv',index=False)
    write(OUT/'source_hashes.json',hashes);assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'completion_audit.json',dict(status='complete',runs=4,training_runs=0,thresholds_unchanged=True,hashes_unchanged=True,test_used=False,multiple_recordings_required=True,seconds=time.monotonic()-started))
    write(OUT/'STATE.json',dict(status='complete'));print(df.to_string(index=False));print('SITE FRACTIONS',float((st.hit_fraction<=.2).mean()),float((st.hit_fraction>=.8).mean()))

if __name__=='__main__':main()
