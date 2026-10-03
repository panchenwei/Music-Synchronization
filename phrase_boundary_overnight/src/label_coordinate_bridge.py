"""Fixed predictions and thresholds: isolate measurement effect of source label repair."""
import copy,hashlib,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
from .score_context_study import ROOT,read,write,sha
from .score_novelty_study import dataset,split_ids
from .fixed_ensemble_study import raw_from_frame
from .local_context_study import metrics

OUT=ROOT/'reports/label_coordinate_bridge'


def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/tied_roll_study/completion_audit.json')['status']=='complete'
    preview=read(ROOT/'reports/score_coordinate_audit/label_repair_preview.json');assert preview['totals']==dict(old_valid_positives=491,new_valid_positives=495,changed_labels=4,changed_masks=0)
    hashes=dict(read(ROOT/'reports/fixed_ensemble_study/contract.json')['hashes']);assert all(sha(p)==h for p,h in hashes.items())
    paths=[Path(__file__),OUT/'PROTOCOL.md',ROOT/'src/score_local_coordinates.py',ROOT/'src/preview_local_label_repair.py',ROOT/'tests/test_score_local_coordinates.py']
    paths+=list((ROOT/'artifacts/coordinate_repair_preview').glob('*.npz'))
    sources={}
    for kind,study in (('B','score_novelty_study'),('NR','fixed_ensemble_study')):
        for fold in (0,1):
            for seed in (42,43):
                run=f'{kind}_seed{seed}_fold{fold}';root=ROOT/'artifacts'/study/'metrics'
                sources[(kind,fold,seed)]=(root/f'{run}_predictions.csv.gz',root/f'{run}.json');paths.extend(sources[(kind,fold,seed)])
    hashes.update({str(p):sha(p) for p in paths});contract=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    write(OUT/'contract.json',dict(contract=contract,hashes=hashes));write(OUT/'STATE.json',dict(status='running',pid=os.getpid(),contract=contract))
    rows=[]
    for fold in (0,1):
        old=dataset(split_ids(fold)['validation'],'B');new={pid:dict(item) for pid,item in old.items()}
        for pid in new:
            with np.load(ROOT/'artifacts/coordinate_repair_preview'/f'{pid}.npz',allow_pickle=False) as z:
                new[pid]['labels']=z['labels'].copy();new[pid]['label_mask']=z['label_mask'].copy()
            np.testing.assert_array_equal(new[pid]['label_mask'],old[pid]['label_mask'])
        for seed in (42,43):
            for kind in ('B','NR'):
                assert time.monotonic()-started<600
                path,mpath=sources[(kind,fold,seed)];frame=pd.read_csv(path);assert not frame.duplicated(['piece_id','performance_id','beat']).any();assert np.isfinite(frame.probability).all() and frame.probability.between(0,1).all()
                raw=raw_from_frame(frame,old);saved=read(mpath);threshold=saved['threshold'];_,_,a=metrics(raw,old,threshold);_,_,b=metrics(raw,new,threshold)
                err=max(abs(a[c]-saved[c]) for c in ('macro_f1_tol0','macro_f1_tol1','raw_ap'));assert err<1e-10
                r=dict(kind=kind,fold=fold,seed=seed,threshold=threshold,old_replay_error=err)
                for c in ('macro_f1_tol0','macro_f1_tol1','raw_ap','macro_precision_tol1','macro_recall_tol1'):r[c+'_old']=a[c];r[c+'_repaired']=b[c];r[c+'_measurement_delta']=b[c]-a[c]
                rows.append(r);print(kind,fold,seed,'old',a['macro_f1_tol1'],'repaired',b['macro_f1_tol1'],flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'paired_results.csv',index=False);means=df.groupby('kind')[[c for c in df if c.startswith(('macro_','raw_'))]].mean();means.to_csv(OUT/'model_means.csv')
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'completion_audit.json',dict(status='complete',paired_evaluations=8,old_metric_replays=8,source_predictions_and_labels_unchanged=True,new_threshold_search=False,training_runs=0,test_predictions_accessed=False,seconds=time.monotonic()-started,contract=contract));write(OUT/'STATE.json',dict(status='complete',pid=None,contract=contract));print(means.to_string())


if __name__=='__main__':main()
