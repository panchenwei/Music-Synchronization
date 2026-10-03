"""Re-evaluate the corrected external paper module; no neural training or test access."""
import json
import time
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .pure_energy_eval import load_module,paper,sha,max_match_count,ROOT

OUT=ROOT/'reports/corrected_energy_20260910'
SOURCE=Path(r'C:\Users\pa1018\Desktop\learn\柴柴\复现')

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    mod=load_module('external_corrected_energy',SOURCE/'paper_energy.py')
    manifest=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv')
    val=manifest[(manifest.fold.isin([0,1]))&manifest.split.isin(['validation','val'])]
    assert val.piece_id.nunique()==19 and len(val)==19
    hashes={str(p):sha(p) for p in [SOURCE/'paper_energy.py',SOURCE/'CPOPY.py',SOURCE/'复现/data_analysis.py',
                                   SOURCE/'复现/main.py',Path(__file__),ROOT/'src/pure_energy_eval.py',
                                   ROOT/'artifacts/phase2/splits/opus_split_manifest.csv']}
    rows=[];n=0;delta=0.;time_geometry=True
    for e in val.itertuples(index=False):
        path=ROOT/f'artifacts/slice_energy_study/cache/{e.piece_id}.npz';hashes[str(path)]=sha(path)
        with np.load(path,allow_pickle=False) as z:
            for j,perf in enumerate(z['performance_ids']):
                if time.perf_counter()-start>300:raise TimeoutError('300s CPU cap')
                t=np.exp(z['curves'][j,:,0].astype(float));r=mod.analyze_tempo(t)
                independent=paper(t,(1,3,6,12,24))
                delta=max(delta,float(np.max(np.abs(r['relative_score']-independent[0]))))
                for l in r['levels']:
                    time_geometry &= np.array_equal(np.concatenate([np.arange(a,b) for a,b in zip(l['start'],l['end_exclusive'])]),np.arange(len(t)))
                for method,threshold in [('corrected_L4_fixed010',.1),('corrected_L4_all_positive',0.)]:
                    points=mod.decode_boundaries(r['relative_score'],threshold)
                    for target,label,mask in [('phrase_start','start_labels','start_mask'),('structural_end','labels','label_mask')]:
                        valid=z[mask].astype(bool);y=z[label]>.5;truth=np.flatnonzero(y&valid);pred=points[valid[points]]
                        row={'fold':e.fold,'piece_id':e.piece_id,'performance_id':str(perf),'method':method,'target':target,
                             'true_count':len(truth),'pred_count':len(pred),'valid_beats':int(valid.sum())}
                        for tol in [0,1,2]:
                            tp=max_match_count(pred,truth,tol)
                            row.update({f'tp{tol}':tp,f'fp{tol}':len(pred)-tp,f'fn{tol}':len(truth)-tp,
                                        f'p{tol}':tp/max(len(pred),1),f'r{tol}':tp/max(len(truth),1),
                                        f'f1_{tol}':2*tp/max(len(pred)+len(truth),1)})
                        row['raw_ap']=average_precision_score(y[valid],r['relative_score'][valid]);rows.append(row)
                n+=1
        print(e.piece_id,n,flush=True)
    perfs=pd.DataFrame(rows);works=perfs.groupby(['target','method','fold','piece_id']).mean(numeric_only=True).reset_index()
    folds=works.groupby(['target','method','fold']).mean(numeric_only=True).reset_index()
    means=folds.groupby(['target','method']).mean(numeric_only=True).reset_index().drop(columns='fold')
    perfs.to_csv(OUT/'per_performance.csv',index=False);works.to_csv(OUT/'per_work.csv',index=False)
    folds.to_csv(OUT/'per_fold.csv',index=False);means.to_csv(OUT/'summary.csv',index=False)
    old=pd.read_csv(ROOT/'reports/pure_energy_20260910/summary_equal_fold.csv');comparisons=[]
    for r in means.itertuples(index=False):
        name='paper_L4_fixed010' if r.method.endswith('fixed010') else 'paper_L4_all_positive_peaks'
        before=old[(old.target==r.target)&(old.method==name)].iloc[0]
        comparisons.append({'method':r.method,'target':r.target,'prior_formula_F1':before.f1_1,
                            'corrected_external_F1':r.f1_1,'delta':r.f1_1-before.f1_1})
    pd.DataFrame(comparisons).to_csv(OUT/'formula_reproduction.csv',index=False)
    checks={'906_performances':n==906,'independent_formula_agreement':delta<1e-10,'beat_geometry_preserved':bool(time_geometry),
            'sources_unchanged_during_execution':all(sha(p)==h for p,h in hashes.items()),
            'F1_reproduction':all(abs(c['delta'])<1e-9 for c in comparisons)}
    audit={'status':'complete' if all(checks.values()) else 'needs_review','checks':checks,
           'pieces':len(val),'performances':n,'runtime_seconds':time.perf_counter()-start,
           'max_score_difference':delta,'trained_models':0,'outer_test_evaluated':False,'hashes':hashes}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print(means[['target','method','p1','r1','f1_0','f1_1','f1_2','raw_ap']].to_string(index=False))
    print(json.dumps(checks));assert all(checks.values())

if __name__=='__main__':main()
