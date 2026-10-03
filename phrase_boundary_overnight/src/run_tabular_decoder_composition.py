"""Fixed lambda .5 on both audited recurrence trees; no model/threshold search."""
import hashlib,json,time
from pathlib import Path
import pandas as pd
from . import run_recurrence_tabular_study as tree
from . import run_halo_decoder_composition as dec
from .score_context_study import ROOT,read,write,sha
from .three_round_round2 import split_ids
from .audit_external_stem_transfer import checked_raw
from .interstart_decoder import fit_prior

OUT=ROOT/'reports/tabular_decoder_composition';ART=ROOT/'artifacts/tabular_decoder_composition';COLS=tree.COLS


def main():
    began=time.monotonic();OUT.mkdir(exist_ok=True);ART.mkdir(exist_ok=True)
    assert read(tree.OUT/'completion_audit.json')['status']=='complete'
    hashes=dict(read(tree.OUT/'contract.json')['hashes'])
    for p in (Path(__file__),Path(dec.__file__),ROOT/'src/interstart_decoder.py',OUT/'PROTOCOL.md',ROOT/'artifacts/interstart_decoder_study/summary.csv'):hashes[str(p)]=sha(p)
    for k in tree.KINDS:
        for f in (0,1):
            for s in (42,43):
                for suffix in ('.json','_predictions.csv.gz'):
                    p=tree.ART/'metrics'/f'{k}_seed{s}_fold{f}{suffix}';hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items());digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));dec.OUT=OUT;dec.ART=ART;rows=[]
    for f in (0,1):
        ids=split_ids(f);prior=fit_prior(tree.dataset(ids['train'],'B'));val=tree.dataset(ids['validation'],'B')
        write(OUT/f'prior_fold{f}.json',dict(prior=prior,train_ids=ids['train']))
        for s in (42,43):
            for k in tree.KINDS:
                run=f'{k}_seed{s}_fold{f}';meta=read(tree.ART/'metrics'/f'{run}.json');raw=checked_raw(pd.read_csv(tree.ART/'metrics'/f'{run}_predictions.csv.gz'),val)
                for strength in (0.,.5):
                    assert time.monotonic()-began<600
                    r=dec.evaluate(raw,val,meta['threshold'],prior,strength,f'{run}_lambda{strength:g}',digest)
                    if strength==0:assert max(abs(r[c]-meta[c]) for c in COLS)<1e-10
                    rows.append(dict(kind=k,fold=f,seed=s,**r))
    d=pd.DataFrame(rows);assert len(d)==16;d.to_csv(ART/'summary.csv',index=False);d.groupby(['kind','strength'])[COLS].mean().to_csv(OUT/'means.csv')
    old=pd.read_csv(ROOT/'artifacts/interstart_decoder_study/summary.csv');b=old[old.strength==.5].set_index(['fold','seed']);comparisons=[]
    for k in tree.KINDS:
        a=d[(d.kind==k)&(d.strength==.5)].set_index(['fold','seed']);delta=a[COLS]-b[COLS]
        comparisons.append(dict(kind=k,reference='C3_soft',**delta.mean().to_dict(),f1_positive=int((delta.macro_f1_tol1>0).sum()),passed=bool(delta.macro_f1_tol1.mean()>=.015 and delta.macro_f1_tol0.mean()>=0 and (delta.macro_f1_tol1>0).sum()>=3)))
    write(OUT/'comparisons.json',comparisons);assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'completion_audit.json',dict(status='complete',decoder_cells=16,new_training_runs=0,zero_strength_reproduced=True,saved_positions_recomputed=True,source_hashes_unchanged=True,test_used=False,elapsed_seconds=time.monotonic()-began));write(OUT/'STATE.json',dict(status='complete',pid=None,contract=digest))


if __name__=='__main__':main()
