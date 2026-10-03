"""Frozen current C3 errors; all development works, no threshold fitting."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from . import run_recurrence_depth_study as base
from .score_context_study import ROOT,read,write,sha
from .three_round_round2 import split_ids
from .audit_external_stem_transfer import checked_raw
from .local_context_study import metrics
from .error_review_30 import matches,decode as raw_decode
from .evaluation import one_to_one_counts

OUT=ROOT/'reports/current_boundary_diagnosis'

def main():
    started=time.monotonic();OUT.mkdir(exist_ok=True)
    hashes={str(Path(__file__)):sha(__file__),str(OUT/'PROTOCOL.md'):sha(OUT/'PROTOCOL.md')}
    rows=[];sites=[];follow=[];counts=[];data={}
    old=pd.read_csv(ROOT/'reports/error_review_30/cases.csv')
    for fold in (0,1):
        val=base.dataset(split_ids(fold)['validation'],'C3');data.update(val)
        for seed in (42,43):
            run=f'C3_seed{seed}_fold{fold}';mp=base.ART/'metrics'/f'{run}.json';pp=base.ART/'metrics'/f'{run}_predictions.csv.gz'
            hashes.update({str(p):sha(p) for p in (mp,pp)})
            meta=read(mp);raw=checked_raw(pd.read_csv(pp),val)
            score=metrics(raw,val,meta['threshold'])[2]
            assert max(abs(score[c]-meta[c]) for c in base.COLS)<1e-10
            for strength in (0.,.5):
                stem=ROOT/'artifacts/interstart_decoder_study'/f'{run}_lambda{strength:g}'
                pos=Path(str(stem)+'_positions.csv.gz');result=Path(str(stem)+'.json')
                hashes.update({str(p):sha(p) for p in (pos,result)})
                groups={(p,q):g.beat.to_numpy(int) for (p,q),g in pd.read_csv(pos).groupby(['piece_id','performance_id'])}
                local=[]
                for pid,perfs in raw.items():
                    item=val[pid];mask=item['label_mask'].astype(bool);truth=np.flatnonzero((item['labels']>.5)&mask);hitcount={int(t):0 for t in truth}
                    for perf,p in perfs.items():
                        pred=groups.get((pid,perf),np.array([],int));pred=pred[mask[pred]]
                        pairs,fp,fn=matches(pred,truth);assert len(pairs)==one_to_one_counts(pred,truth,1).tp
                        rd=raw_decode(p,item['labels'],mask,meta['threshold'])
                        if strength==0:np.testing.assert_array_equal(pred,rd['pred'])
                        rec=dict(fold=fold,seed=seed,strength=strength,piece_id=pid,performance_id=perf,tp=len(pairs),fp=len(fp),fn=len(fn),f1=2*len(pairs)/max(1,len(pred)+len(truth)))
                        counts.append(rec);local.append(rec)
                        for _,t in pairs:hitcount[t]+=1
                        for category,beats in (('FP',fp),('FN',fn)):
                            for b in beats:
                                dist=int(min(abs(truth-b)))
                                if category=='FP':sub='competition' if dist<=1 else 'near_2_3' if dist<=3 else 'far_over_3'
                                else:
                                    near=np.arange(max(0,b-1),min(len(p),b+2));near=near[mask[near]]
                                    sub='below_threshold' if max(p[near])<meta['threshold'] else 'nms_removed' if max(rd['nms'][near])<meta['threshold'] else 'decoder_or_matching'
                                rows.append(dict(fold=fold,seed=seed,strength=strength,piece_id=pid,performance_id=perf,category=category,beat=int(b),subtype=sub,probability=float(p[b])))
                        for r in old[(old.piece_id==pid)&(old.performance_id==perf)].itertuples():
                            b=int(r.beat);target=int(r.matched_truth) if r.category=='TP' else b
                            follow.append(dict(case_id=r.case_id,old_category=r.category,piece_id=pid,beat=b,seed=seed,strength=strength,probability=float(p[b]),threshold=meta['threshold'],is_current_truth=bool(target in truth),near_prediction=bool(any(abs(pred-b)<=1)),truth_matched=bool(any(t==target for _,t in pairs))))
                    sites.extend(dict(fold=fold,seed=seed,strength=strength,piece_id=pid,beat=t,hit_fraction=c/len(perfs)) for t,c in hitcount.items())
                observed=pd.DataFrame(local).groupby('piece_id').f1.mean().mean()
                assert abs(observed-read(result)['macro_f1_tol1'])<1e-10
    ev=pd.DataFrame(rows);st=pd.DataFrame(sites);pf=pd.DataFrame(counts)
    ev.to_csv(OUT/'events.csv.gz',index=False);st.to_csv(OUT/'sites.csv',index=False);pf.to_csv(OUT/'performance_counts.csv',index=False);pd.DataFrame(follow).to_csv(OUT/'old_30_followup.csv',index=False)
    agg=ev.groupby(['seed','strength','piece_id','category','subtype']).size().rename('count').reset_index()
    keys=['seed','strength','piece_id','category'];agg['fraction']=agg['count']/agg.groupby(keys)['count'].transform('sum')
    # Explicit zeros: average all works having any error of this category.
    means=[]
    for (seed,strength,cat),g in agg.groupby(['seed','strength','category']):
        tab=g.pivot(index='piece_id',columns='subtype',values='fraction').fillna(0)
        means.extend(dict(seed=int(seed),strength=strength,category=cat,subtype=c,work_macro_fraction=float(tab[c].mean())) for c in tab)
    pd.DataFrame(means).to_csv(OUT/'work_macro_subtypes.csv',index=False)
    joint=st.groupby(['strength','piece_id','beat']).hit_fraction.agg(['mean','max']).reset_index()
    joint.to_csv(OUT/'joint_sites.csv',index=False)
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes)
    summary=dict(status='complete',new_training=0,works=len(data),runs=4,decoder_cells=8,hashes_unchanged=True,all_f1_recomputed=True,test_used=False,seconds=time.monotonic()-started)
    write(OUT/'completion_audit.json',summary)
    print(pd.DataFrame(means).groupby(['strength','category','subtype']).work_macro_fraction.mean().to_string())
    print('JOINT HARD',joint.groupby('strength')['max'].apply(lambda x:float((x<=.2).mean())).to_dict());print(summary)

if __name__=='__main__':main()
