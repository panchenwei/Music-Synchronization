"""One fixed composition of two already tested post-processing components."""
import hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import four_seed_decision_audit as prev
from . import run_recurrence_depth_study as base
from . import run_c3_seed_replication as extra
from . import run_halo_decoder_composition as dec
from .score_context_study import ROOT,read,write,sha
from .three_round_round2 import split_ids
from .interstart_decoder import fit_prior
from .audit_external_stem_transfer import checked_raw

OUT=ROOT/'reports/coverage_decoder_composition';ART=ROOT/'artifacts/coverage_decoder_composition';COLS=prev.COLS

def main():
    began=time.monotonic()
    for p in (OUT,ART,ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    assert read(prev.OUT/'completion_audit.json')['status']=='complete'
    hashes=dict(read(prev.OUT/'contract.json')['hashes'])
    for entry in read(prev.OUT/'probability_hashes.json'):
        if Path(entry['path']).name.startswith('M_'):hashes[entry['path']]=entry['sha256']
    for p in (Path(__file__),OUT/'PROTOCOL.md',prev.ART/'summary.csv'):hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    if (OUT/'completion_audit.json').exists():assert read(OUT/'completion_audit.json')['status']=='complete';print('ALREADY COMPLETE');return
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder'
    old=pd.read_csv(prev.ART/'summary.csv');rows=[]
    for f in (0,1):
        ids=split_ids(f);train=base.dataset(ids['train'],'C3');val=base.dataset(ids['validation'],'C3');prior=fit_prior(train)
        for s in (42,43,44,45):
            source=base.ART if s<44 else extra.ART;meta=read(source/'metrics'/f'C3_seed{s}_fold{f}.json')
            raw=checked_raw(pd.read_csv(prev.ART/f'M_seed{s}_fold{f}_predictions.csv.gz'),val)
            reference=old[(old.condition=='M')&(old.fold==f)&(old.seed==s)].iloc[0]
            for strength in (.5,1.):
                assert time.monotonic()-began<600
                g=read(ROOT/'reports/research_resource_guard.json');assert g['observed_used_percent']<g['post_reset_stop_used_percent']
                r=dec.evaluate(raw,val,meta['threshold'],prior,strength,f'M_seed{s}_fold{f}_lambda{strength:g}',digest)
                assert abs(r['raw_ap']-reference.raw_ap)<1e-10
                if strength==.5:assert max(abs(r[c]-reference[c]) for c in COLS)<1e-10
                rows.append(dict(condition='M05' if strength==.5 else 'M10',fold=f,seed=s,**r))
                pd.DataFrame(rows).to_csv(ART/'summary.csv',index=False)
    frame=pd.DataFrame(rows);assert len(frame)==16
    combined=pd.concat([frame,old[old.condition.isin(['B05','B10'])]],ignore_index=True)
    combined.to_csv(OUT/'all_runs.csv',index=False);means=[]
    for name,seeds in [('all_four',[42,43,44,45]),('original_two',[42,43]),('additional_two',[44,45])]:
        m=combined[combined.seed.isin(seeds)].groupby('condition')[COLS+['macro_precision_tol1','macro_recall_tol1']].mean().reset_index();m['subset']=name;means.append(m)
    means=pd.concat(means);means.to_csv(OUT/'means.csv',index=False);comparisons=[]
    for control in ('M05','B10','B05'):
        a=combined[combined.condition=='M10'].set_index(['fold','seed']);b=combined[combined.condition==control].set_index(['fold','seed']);d=a[COLS]-b[COLS]
        d.to_csv(OUT/f'M10_minus_{control}.csv')
        passed=d.macro_f1_tol1.mean()>=.015 and (d.macro_f1_tol1>0).sum()>=6 and d.macro_f1_tol0.mean()>=0
        comparisons.append(dict(control=control,**d.mean().to_dict(),positive=int((d.macro_f1_tol1>0).sum()),passed=bool(passed)))
    write(OUT/'comparisons.json',comparisons);assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=0,old_m_metrics_recomputed=8,decoder_cells=16,ap_unchanged_for_m=True,source_hashes_unchanged=True,test_used=False,seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None));write(dec.OUT/'STATE.json',dict(status='complete',pid=None));print('COMPLETE\n'+means.to_string(index=False),flush=True)

if __name__=='__main__':
    try:
        with threadpool_limits(2):main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed',pid=None));raise
