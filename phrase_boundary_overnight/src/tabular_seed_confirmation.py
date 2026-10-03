"""Additional seeds for the exploratory point-tree/CNN complementarity signal."""
import hashlib,json,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import current_tabular_study as study
from .current_threshold_diagnostic import GRID,COLS,select_threshold,nms_probabilities,one_to_one_counts
from .score_context_study import ROOT,read,write,sha,normalizer

OUT=ROOT/'reports/tabular_seed_confirmation';ART=ROOT/'artifacts/tabular_seed_confirmation';C3=ROOT/'artifacts/c3_seed_replication'


def prepare():
    for p in (OUT,ART/'models',ART/'metrics',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/c3_seed_replication/completion_audit.json')['status']=='complete'
    hashes=dict(read(study.OUT/'contract.json')['hashes']);hashes.update(read(ROOT/'reports/c3_seed_replication/contract.json')['hashes'])
    files=[Path(__file__),OUT/'PROTOCOL.md',ROOT/'src/current_threshold_diagnostic.py']
    for f in (0,1):
        for s in (44,45):
            for suffix in ('.json','_predictions.csv.gz'):files.append(C3/'metrics'/f'C3_seed{s}_fold{f}{suffix}')
    for p in files:hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items());digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    if not (OUT/'STATE.json').exists():write(OUT/'STATE.json',dict(status='ready',completed=[],seconds=0.,pid=None))
    return digest


def audit(digest):
    began=time.monotonic();hashes=read(OUT/'contract.json')['hashes'];models={str(p):sha(p) for p in (ART/'models').glob('*/*.joblib')};assert len(models)==8
    manifest=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv',dtype={'opus':str});checks=[];decoded=[];cross=[];choices=[]
    study.dec.OUT=OUT/'decoder_state';study.dec.OUT.mkdir(exist_ok=True);study.dec.ART=ART/'decoder'
    for f in (0,1):
        ids=study.split_ids(f);train=study.dataset(ids['train'],'B');val=study.dataset(ids['validation'],'B');norm=normalizer(train);prior=study.fit_prior(train)
        opus=manifest[(manifest.fold==f)&(manifest.split=='validation')].set_index('piece_id').opus.to_dict();assert set(opus)==set(val)
        for seed in (44,45):
            ht=f'H0_seed{seed}_fold{f}';hm=read(ART/'metrics'/f'{ht}.json');best=joblib.load(ART/'models'/ht/'best.joblib');last=joblib.load(ART/'models'/ht/'latest.joblib')
            assert best['contract']==last['contract']==digest and last['step']==150 and best['step']==hm['best_step']
            np.testing.assert_array_equal(norm.mean,best['mean']);np.testing.assert_array_equal(norm.std,best['std'])
            raw=study.checked_raw(pd.read_csv(ART/'metrics'/f'{ht}_predictions.csv.gz'),val);replay=study.tree.raw_predictions(best['model'],val,norm,0);err=study.error(raw,replay);assert err<1e-12
            assert max(abs(study.metrics(raw,val,hm['threshold'])[2][c]-hm[c]) for c in study.COLS)<1e-10
            checks.append(dict(run=ht,replay_error=err));meta=read(C3/'metrics'/f'C3_seed{seed}_fold{f}.json');base=study.checked_raw(pd.read_csv(C3/'metrics'/f'C3_seed{seed}_fold{f}_predictions.csv.gz'),val)
            for kind,prob in [('C3',base),('T0',{p:{q:(a+raw[p][q])*.5 for q,a in pp.items()} for p,pp in base.items()})]:
                adjusted={}
                for pid,pp in prob.items():
                    with np.load(ROOT/'artifacts/recurrence_mean_control/graphs'/f'{pid}.npz',allow_pickle=False) as z:g=z['M']
                    adjusted[pid]={q:study.mix_probabilities(a,g) for q,a in pp.items()}
                for policy,x,strength in [('raw',prob,0.),('M10',adjusted,1.)]:
                    r=study.dec.evaluate(x,val,meta['threshold'],prior,strength,f'{kind}_seed{seed}_fold{f}_{policy}',digest)
                    if kind=='C3' and policy=='raw':assert max(abs(r[c]-meta[c]) for c in study.COLS)<1e-10
                    decoded.append(dict(kind=kind,fold=f,seed=seed,policy=policy,**r))
                path=ART/f'{kind}_fold{f}_seed{seed}_grid.csv'
                if path.exists():frame=pd.read_csv(path,dtype={'opus':str})
                else:
                    rows=[]
                    for pid,pp in prob.items():
                        mask=val[pid]['label_mask']>.5;truth=np.flatnonzero((val[pid]['labels']>.5)&mask);cells=[]
                        for p in pp.values():
                            peaks=nms_probabilities(p)
                            for threshold in GRID:
                                pred=np.flatnonzero((peaks>=threshold)&mask);c=one_to_one_counts(pred,truth,1);e=one_to_one_counts(pred,truth,0)
                                cells.append(dict(threshold=threshold,f1_tol1=c.f1,f1_tol0=e.f1,precision_tol1=c.precision,recall_tol1=c.recall))
                        means=pd.DataFrame(cells).groupby('threshold',as_index=False)[COLS].mean();rows.extend([dict(piece_id=pid,opus=opus[pid],**r) for r in means.to_dict('records')])
                    frame=pd.DataFrame(rows);frame.to_csv(path,index=False)
                selections=[]
                for group in sorted(set(opus.values())):
                    threshold=select_threshold(frame,group);part=frame[(frame.opus==group)&np.isclose(frame.threshold,threshold)];selections.append(part)
                    choices.append(dict(kind=kind,fold=f,seed=seed,excluded_opus=group,threshold=threshold,fit_piece_ids=sorted(set(frame[frame.opus!=group].piece_id))))
                cross.append(dict(kind=kind,fold=f,seed=seed,policy='crossfit',**pd.concat(selections)[COLS].mean().to_dict()))
            print('AUDITED',ht,flush=True);assert time.monotonic()-began<1200
    pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False);pd.DataFrame(decoded).to_csv(OUT/'fixed_metrics.csv',index=False);pd.DataFrame(decoded).groupby(['kind','policy'])[study.COLS].mean().to_csv(OUT/'fixed_means.csv')
    new=pd.DataFrame(cross);new.to_csv(OUT/'crossfit_metrics.csv',index=False);old=pd.read_csv(ROOT/'reports/current_threshold_diagnostic/fold_metrics.csv');old=old[(old.policy=='crossfit')&old.kind.isin(['C3','T0'])]
    allseeds=pd.concat([old,new],ignore_index=True);allseeds.to_csv(OUT/'four_seed_crossfit_metrics.csv',index=False);allseeds.groupby('kind')[COLS].mean().to_csv(OUT/'four_seed_means.csv');new.groupby('kind')[COLS].mean().to_csv(OUT/'new_seed_means.csv')
    comparisons=[]
    for label,df in [('new_seeds',new),('four_seeds',allseeds)]:
        d=df[df.kind=='T0'].set_index(['fold','seed'])[COLS]-df[df.kind=='C3'].set_index(['fold','seed'])[COLS]
        comparisons.append(dict(scope=label,**d.mean().to_dict(),positive=int((d.f1_tol1>0).sum()),passed=bool(d.f1_tol1.mean()>=.015 and d.f1_tol0.mean()>=0 and (d.f1_tol1>0).sum()>=int(.75*len(d)))))
    write(OUT/'comparisons.json',comparisons);write(OUT/'threshold_choices.json',choices);write(OUT/'model_hashes.json',models)
    assert all(sha(p)==h for p,h in hashes.items()) and all(sha(p)==h for p,h in models.items())
    write(OUT/'completion_audit.json',dict(status='complete',training_runs=4,model_replays=4,saved_models=8,decode_cells=16,threshold_grids=8,crossfit_threshold_only=True,checkpoint_selection_not_nested=True,test_used=False,hashes_unchanged=True,training_seconds=sum(read(p)['seconds'] for p in (ART/'metrics').glob('*.json')),audit_seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None));print(comparisons,flush=True)


def main():
    digest=prepare();study.tree.OUT=OUT;study.tree.ART=ART;study.tree.dataset=study.dataset;study.tree.normalizer=normalizer
    for f in (0,1):
        for seed in (44,45):
            guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent'];study.tree.run_one('H0',f,seed,digest)
    audit(digest)


if __name__=='__main__':
    with threadpool_limits(2):main()
