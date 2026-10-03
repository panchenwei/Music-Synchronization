"""Post-hoc fixed ORIGINAL threshold diagnosis; never a promotion test."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from . import sam_study as sam
from . import window_ranking_study as shared

ROOT=sam.ROOT;OUT=ROOT/'reports/sam_threshold_probe';ART=ROOT/'artifacts/sam_threshold_probe'


def main():
    assert shared.base.read(sam.OUT/'audit.json')['status']=='complete'
    OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    hashes=dict(shared.base.read(sam.OUT/'contract.json')['hashes'])
    for p in [Path(__file__),OUT/'PROTOCOL.md',*list((sam.ART/'metrics').glob('R_*predictions.csv.gz'))]:hashes[str(p)]=shared.base.sha(p)
    assert all(shared.base.sha(p)==h for p,h in hashes.items())
    contract=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    shared.dec.ART=ART;shared.dec.OUT=OUT
    rows=[];pieces=[]
    for f in (0,1):
        ids=shared.split_ids(f);tr=shared.source.dataset(ids['train'],'C3');va=shared.source.dataset(ids['validation'],'C3');prior=shared.fit_prior(tr)
        for s in (42,43):
            label=f'seed{s}_fold{f}';a=shared.base.read(sam.ART/'decoder'/f'ER_{label}_M10.json');b=shared.base.read(sam.ART/'decoder'/f'EC_{label}_M10.json')
            if a['threshold']==b['threshold']:
                fixed=a;piecefile=sam.ART/'decoder'/f'ER_{label}_M10_pieces.csv'
            else:
                raw=shared.checked_raw(pd.read_csv(sam.ART/'metrics'/f'R_{label}_predictions.csv.gz'),va)
                gru=shared.checked_raw(pd.read_csv(ROOT/'artifacts/current_gru_study/metrics'/f'G_{label}_predictions.csv.gz'),va)
                adjusted={}
                for p,pp in raw.items():
                    with np.load(ROOT/'artifacts/recurrence_mean_control/graphs'/f'{p}.npz') as z:g=z['M']
                    adjusted[p]={q:shared.mix_probabilities((v+gru[p][q])*.5,g) for q,v in pp.items()}
                name=f'SAM_original_threshold_{label}'
                fixed=shared.dec.evaluate(adjusted,va,b['threshold'],prior,1.,name,contract);piecefile=ART/f'{name}_pieces.csv'
            rows.append(dict(fold=f,seed=s,original_threshold=b['threshold'],sam_threshold=a['threshold'],original_f1=b['macro_f1_tol1'],sam_f1=a['macro_f1_tol1'],fixed_threshold_f1=fixed['macro_f1_tol1'],original_exact=b['macro_f1_tol0'],sam_exact=a['macro_f1_tol0'],fixed_exact=fixed['macro_f1_tol0']))
            base=pd.read_csv(sam.ART/'decoder'/f'EC_{label}_M10_pieces.csv').set_index('piece_id');cand=pd.read_csv(sam.ART/'decoder'/f'ER_{label}_M10_pieces.csv').set_index('piece_id');fixedp=pd.read_csv(piecefile).set_index('piece_id')
            for pid in base.index:
                r=dict(piece_id=pid,fold=f,seed=s)
                for key in ('f1_tol1','precision_tol1','recall_tol1','predicted_boundaries','tp_tol1','fp_tol1','fn_tol1','raw_ap'):
                    r['original_'+key]=base.loc[pid,key];r['sam_'+key]=cand.loc[pid,key];r['fixed_'+key]=fixedp.loc[pid,key]
                r['sam_f1_contribution']=(r['sam_f1_tol1']-r['original_f1_tol1'])/(4*len(base));pieces.append(r)
    df=pd.DataFrame(rows);df.to_csv(OUT/'results.csv',index=False);pd.DataFrame(pieces).sort_values('sam_f1_contribution').to_csv(OUT/'piece_differences.csv',index=False)
    assert all(shared.base.sha(p)==h for p,h in hashes.items())
    shared.base.write(OUT/'audit.json',dict(status='complete',posthoc_diagnostic=True,promotion_allowed=False,new_training_runs=0,new_decode_cells=sum(df.original_threshold!=df.sam_threshold),frozen_hashes_unchanged=True,hashes=hashes))
    print(df.to_string(index=False));print(df[['original_f1','sam_f1','fixed_threshold_f1']].mean().to_string())


if __name__=='__main__':main()
