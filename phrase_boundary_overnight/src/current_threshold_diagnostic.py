"""Isolate threshold portability from representation/model ranking gains."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from . import current_tabular_study as study
from .score_context_study import ROOT,read,write,sha
from .phase2_models import nms_probabilities
from .evaluation import one_to_one_counts

OUT=ROOT/'reports/current_threshold_diagnostic';ART=ROOT/'artifacts/current_threshold_diagnostic'
GRID=np.arange(.1,.91,.05).round(2)
COLS=['f1_tol1','f1_tol0','precision_tol1','recall_tol1']


def select_threshold(frame,exclude=None):
    fit=frame if exclude is None else frame[frame.opus!=exclude]
    assert len(fit) and (exclude is None or exclude not in set(fit.opus))
    means=fit.groupby('threshold',as_index=False)[COLS].mean()
    return float(means.sort_values(['f1_tol1','precision_tol1','threshold'],ascending=False).iloc[0].threshold)


def main():
    began=time.monotonic()
    for p in (OUT,ART):p.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/current_tabular_fusion/completion_audit.json')['status']=='complete'
    hashes=dict(read(ROOT/'reports/current_tabular_fusion/contract.json')['hashes'])
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'tests/test_current_threshold_diagnostic.py'):hashes[str(p)]=sha(p)
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest();assert all(sha(p)==h for p,h in hashes.items())
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));results=[];workrows=[];choices=[]
    manifest=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv',dtype={'opus':str})
    prior=pd.read_csv(ROOT/'artifacts/current_tabular_fusion/summary.csv').set_index(['kind','policy','fold','seed'])
    for fold in (0,1):
        val=study.dataset(study.split_ids(fold)['validation'],'B');opus=manifest[(manifest.fold==fold)&(manifest.split=='validation')].set_index('piece_id').opus.to_dict()
        assert set(opus)==set(val)
        for seed in (42,43):
            raw={}
            for kind,source in [('C3',study.base.ART),('H0',study.ART),('H2',study.ART)]:
                raw[kind]=study.checked_raw(pd.read_csv(source/'metrics'/f'{kind}_seed{seed}_fold{fold}_predictions.csv.gz'),val)
            fixed=read(study.base.ART/'metrics'/f'C3_seed{seed}_fold{fold}.json')['threshold']
            for kind,other in [('C3',None),('T0','H0'),('T2','H2')]:
                path=ART/f'{kind}_fold{fold}_seed{seed}_grid.csv';rows=[]
                if path.exists():frame=pd.read_csv(path,dtype={'opus':str})
                else:
                    for pid,pp in raw['C3'].items():
                        mask=val[pid]['label_mask']>.5;truth=np.flatnonzero((val[pid]['labels']>.5)&mask);cells=[]
                        for perf,p in pp.items():
                            prob=p if other is None else (p+raw[other][pid][perf])*.5;peaks=nms_probabilities(prob)
                            for threshold in GRID:
                                predicted=np.flatnonzero((peaks>=threshold)&mask);c=one_to_one_counts(predicted,truth,1);e=one_to_one_counts(predicted,truth,0)
                                cells.append(dict(threshold=threshold,f1_tol1=c.f1,f1_tol0=e.f1,precision_tol1=c.precision,recall_tol1=c.recall))
                        mean=pd.DataFrame(cells).groupby('threshold',as_index=False)[COLS].mean();rows.extend([dict(piece_id=pid,opus=opus[pid],**r) for r in mean.to_dict('records')])
                    frame=pd.DataFrame(rows);frame.to_csv(path,index=False)
                selected=frame[np.isclose(frame.threshold,fixed)];assert abs(selected.f1_tol1.mean()-prior.loc[(kind,'raw',fold,seed),'macro_f1_tol1'])<1e-10
                for policy in ('fixed','crossfit','oracle_validation'):
                    cells=[]
                    for group in sorted(set(opus.values())):
                        threshold=fixed if policy=='fixed' else select_threshold(frame,group if policy=='crossfit' else None)
                        part=frame[(frame.opus==group)&np.isclose(frame.threshold,threshold)];assert len(part)==sum(o==group for o in opus.values());cells.append(part)
                        fit_ids=sorted(set(frame[frame.opus!=group].piece_id)) if policy=='crossfit' else ([] if policy=='fixed' else sorted(val))
                        choices.append(dict(kind=kind,fold=fold,seed=seed,policy=policy,evaluation_opus=group,threshold=threshold,fit_piece_ids=fit_ids))
                    combined=pd.concat(cells);results.append(dict(kind=kind,fold=fold,seed=seed,policy=policy,**combined[COLS].mean().to_dict()))
                    workrows.extend([dict(kind=kind,fold=fold,seed=seed,policy=policy,**r) for r in combined.to_dict('records')])
                print('DIAGNOSED',kind,fold,seed,flush=True);assert time.monotonic()-began<900
    df=pd.DataFrame(results);df.to_csv(OUT/'fold_metrics.csv',index=False);pd.DataFrame(workrows).to_csv(OUT/'work_metrics.csv',index=False);write(OUT/'threshold_choices.json',choices)
    means=df.groupby(['kind','policy'])[COLS].mean();means.to_csv(OUT/'means.csv');comparisons=[]
    for kind in ('T0','T2'):
        for policy in ('fixed','crossfit','oracle_validation'):
            d=df[(df.kind==kind)&(df.policy==policy)].set_index(['fold','seed'])[COLS]-df[(df.kind=='C3')&(df.policy==policy)].set_index(['fold','seed'])[COLS]
            comparisons.append(dict(candidate=kind,policy=policy,**d.mean().to_dict(),positive=int((d.f1_tol1>0).sum()),passed=bool(d.f1_tol1.mean()>=.015 and d.f1_tol0.mean()>=0 and (d.f1_tol1>0).sum()>=3)))
    write(OUT/'comparisons.json',comparisons);assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'artifact_hashes.json',{str(p):sha(p) for p in ART.glob('*')})
    write(OUT/'completion_audit.json',dict(status='complete',training_runs=0,prediction_sets=12,thresholds=17,fixed_results_reproduced=True,threshold_fit_opus_disjoint=True,checkpoint_selection_not_nested=True,test_used=False,hashes_unchanged=True,seconds=time.monotonic()-began))
    print(means.to_string(),flush=True)


if __name__=='__main__':main()
