"""Development-only threshold adaptation, no training or probability changes."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from . import context_inference_audit_v2 as previous
from . import pure_transformer_reference as ref
from . import run_halo_decoder_composition as dec
from .score_context_study import ROOT,read,write,sha,GRID
from .local_context_study import metrics
from .phase2_models import choose_single_threshold
from .three_round_round2 import split_ids
from .interstart_decoder import fit_prior
from .audit_external_stem_transfer import checked_raw

OUT=ROOT/'reports/context_threshold_audit';ART=ROOT/'artifacts/context_threshold_audit'
COLS=previous.COLS

def main():
    start=time.monotonic();OUT.mkdir(exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    assert read(previous.OUT/'completion_audit.json')['status']=='complete'
    files=[Path(__file__),OUT/'PROTOCOL.md',previous.ART/'summary.csv']
    files+=list(previous.ART.glob('T*_predictions.csv.gz'))
    hashes={str(p):sha(p) for p in files}
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART
    old=pd.read_csv(previous.ART/'summary.csv');results=[];thresholds=[]
    for f in (0,1):
        ids=split_ids(f);train=ref.dataset(ids['train'],'T');val=ref.dataset(ids['validation'],'T');prior=fit_prior(train)
        for s in (42,43):
            meta=read(ref.ART/'metrics'/f'T_seed{s}_fold{f}.json')
            for mode in ('FULL','W64_LOCAL','W64_GLOBAL'):
                assert time.monotonic()-start<1200
                run=f'T_seed{s}_fold{f}_{mode}'
                raw=checked_raw(pd.read_csv(previous.ART/f'{run}_predictions.csv.gz'),val)
                orig=old[(old.kind=='T')&(old.fold==f)&(old.seed==s)&(old['mode']==mode)&(old.strength==0)].iloc[0]
                assert max(abs(metrics(raw,val,meta['threshold'])[2][c]-orig[c]) for c in COLS)<1e-10
                threshold,_=choose_single_threshold(raw,val,GRID)
                if mode=='FULL':assert threshold==meta['threshold']
                thresholds.append(dict(fold=f,seed=s,mode=mode,old_threshold=meta['threshold'],new_threshold=threshold))
                for strength in (0.,.5):
                    r=dec.evaluate(raw,val,threshold,prior,strength,f'{run}_lambda{strength:g}',digest)
                    results.append(dict(kind='T',mode=mode,fold=f,seed=s,**r))
                pd.DataFrame(results).to_csv(ART/'summary.csv',index=False)
                pd.DataFrame(thresholds).to_csv(OUT/'thresholds.csv',index=False)
    frame=pd.DataFrame(results);assert len(frame)==24
    means=frame.groupby(['mode','strength'])[COLS].mean();means.to_csv(OUT/'means.csv')
    deltas=frame.set_index(['mode','fold','seed','strength'])[COLS]-old[old.kind=='T'].set_index(['mode','fold','seed','strength'])[COLS]
    deltas.to_csv(OUT/'adapted_minus_locked.csv')
    assert abs(deltas.raw_ap).max()<1e-10
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'completion_audit.json',dict(status='complete',training_runs=0,threshold_cells=12,decode_cells=24,old_metrics_recomputed=12,full_thresholds_equal=True,ap_unchanged=True,hashes_unchanged=True,test_used=False,seconds=time.monotonic()-start))
    write(dec.OUT/'STATE.json',dict(status='complete',pid=None));print(means.to_string(),flush=True)

if __name__=='__main__':main()
