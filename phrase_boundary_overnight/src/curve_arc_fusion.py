"""Frozen equal-weight prediction fusion, without neural retraining."""
import hashlib,json,time,traceback,os
from pathlib import Path
import numpy as np
import pandas as pd
from . import curve_arc_study as arc
from . import run_recurrence_depth_study as base
from . import run_halo_decoder_composition as dec
from .score_context_study import ROOT,read,write,sha,GRID
from .three_round_round2 import split_ids
from .fixed_ensemble_study import aligned_average,raw_from_frame
from .phase2_models import choose_single_threshold
from .local_context_study import metrics
from .interstart_decoder import fit_prior
OUT=ROOT/'reports/curve_arc_fusion';ART=ROOT/'artifacts/curve_arc_fusion';COLS=base.COLS

def main():
    start=time.monotonic()
    for p in (OUT,ART,ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    assert read(arc.OUT/'completion_audit.json')['status']=='complete'
    hashes=dict(read(arc.OUT/'contract.json')['hashes']);hashes.update(read(arc.OUT/'checkpoint_hashes.json'))
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'src/fixed_ensemble_study.py',ROOT/'src/run_halo_decoder_composition.py'):
        hashes[str(p)]=sha(p)
    for source,kinds in ((base.ART,('C3',)),(arc.ART,('R','Z'))):
        for kind in kinds:
            for f in (0,1):
                for s in (42,43):
                    for suffix in ('.json','_predictions.csv.gz'):
                        p=source/'metrics'/f'{kind}_seed{s}_fold{f}{suffix}';hashes[str(p)]=sha(p)
    assert all(sha(p)==v for p,v in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));write(OUT/'STATE.json',dict(status='running',pid=os.getpid()))
    rows=[];checks=[];thresholds=[];dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder'
    for fold in (0,1):
        ids=split_ids(fold);assert not set(ids['train'])&set(ids['validation'])
        train=base.dataset(ids['train'],'C3');val=base.dataset(ids['validation'],'C3');prior=fit_prior(train)
        write(OUT/f'prior_fold{fold}.json',dict(prior=prior,train_ids=ids['train']))
        for seed in (42,43):
            frames={};metas={}
            for k,source in (('C3',base.ART),('R',arc.ART),('Z',arc.ART)):
                run=f'{k}_seed{seed}_fold{fold}';frames[k]=pd.read_csv(source/'metrics'/f'{run}_predictions.csv.gz');metas[k]=read(source/'metrics'/f'{run}.json')
                score=metrics(raw_from_frame(frames[k],val),val,metas[k]['threshold'])[2]
                error=max(abs(score[c]-metas[k][c]) for c in COLS);assert error<1e-10
                checks.append(dict(run_id=run,error=error))
            for member in ('R','Z'):
                assert time.monotonic()-start<600
                kind='E'+member;run=f'{kind}_seed{seed}_fold{fold}'
                fused=aligned_average([frames['C3'],frames[member]])
                path=ART/f'{run}_predictions.csv.gz';fused.to_csv(path,index=False)
                loaded=pd.read_csv(path);np.testing.assert_allclose(loaded.probability,fused.probability,rtol=0,atol=1e-15)
                raw=raw_from_frame(loaded,val);threshold,_=choose_single_threshold(raw,val,GRID)
                fixed=metrics(raw,val,metas['C3']['threshold'])[2];score=metrics(raw,val,threshold)[2]
                thresholds.append(dict(kind=kind,fold=fold,seed=seed,threshold=threshold,c3_threshold=metas['C3']['threshold'],fixed_raw_f1=fixed['macro_f1_tol1'],selected_raw_f1=score['macro_f1_tol1']))
                for strength in (0.,.5):
                    row=dec.evaluate(raw,val,threshold,prior,strength,f'{run}_lambda{strength:g}',digest)
                    if strength==0:assert max(abs(row[c]-score[c]) for c in COLS)<1e-10
                    rows.append(dict(kind=kind,fold=fold,seed=seed,**row))
    d=pd.DataFrame(rows);assert len(d)==16
    d.to_csv(ART/'summary.csv',index=False);pd.DataFrame(checks).to_csv(OUT/'member_recomputation.csv',index=False);pd.DataFrame(thresholds).to_csv(OUT/'thresholds.csv',index=False)
    old=pd.read_csv(arc.ART/'decoder/summary.csv');old=old[old.kind=='R']
    ref=pd.read_csv(ROOT/'artifacts/interstart_decoder_study/summary.csv');ref=ref[ref.strength.isin((0.,.5))].copy();ref['kind']='C3'
    allrows=pd.concat([d,old,ref],ignore_index=True);means=allrows.groupby(['kind','strength'])[COLS].mean();means.to_csv(OUT/'means.csv');comparisons=[]
    for control in ('C3','EZ'):
        a=d[(d.kind=='ER')&(d.strength==.5)].set_index(['fold','seed'])[COLS]
        b=allrows[(allrows.kind==control)&(allrows.strength==.5)].set_index(['fold','seed'])[COLS];delta=a-b
        delta.to_csv(OUT/f'ER_minus_{control}.csv')
        comparisons.append(dict(control=control,**delta.mean().to_dict(),f1_positive=int((delta.macro_f1_tol1>0).sum()),passed=bool(delta.macro_f1_tol1.mean()>=.015 and delta.macro_f1_tol0.mean()>=0 and (delta.macro_f1_tol1>0).sum()>=3)))
    write(OUT/'comparisons.json',comparisons);assert all(sha(p)==v for p,v in hashes.items())
    write(OUT/'completion_audit.json',dict(status='complete',training_runs=0,member_recomputations=len(checks),fusion_files=8,decoder_cells=16,hashes_unchanged=True,test_used=False,seconds=time.monotonic()-start))
    write(OUT/'STATE.json',dict(status='complete',pid=None));write(dec.OUT/'STATE.json',dict(status='complete',pid=None));print(means.to_string(),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        write(OUT/'STATE.json',dict(status='failed',traceback=traceback.format_exc()));raise
