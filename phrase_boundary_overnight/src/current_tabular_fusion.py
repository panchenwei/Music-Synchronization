"""Frozen equal-weight tree/CNN probability complementarity check."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from . import current_tabular_study as study
from .score_context_study import ROOT,read,write,sha

OUT=ROOT/'reports/current_tabular_fusion';ART=ROOT/'artifacts/current_tabular_fusion'


def main():
    began=time.monotonic()
    for p in (OUT,ART):p.mkdir(parents=True,exist_ok=True)
    assert read(study.OUT/'completion_audit.json')['status']=='complete'
    hashes=dict(read(study.OUT/'contract.json')['hashes'])
    for p in [Path(__file__),OUT/'PROTOCOL.md']+list((study.ART/'metrics').glob('*_predictions.csv.gz'))+list((study.ART/'metrics').glob('*.json')):
        hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items());digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));rows=[]
    study.dec.OUT=OUT;study.dec.ART=ART
    old=pd.read_csv(study.ART/'summary.csv').set_index(['kind','policy','fold','seed'])
    for fold in (0,1):
        ids=study.split_ids(fold);val=study.dataset(ids['validation'],'B');prior=study.fit_prior(study.dataset(ids['train'],'B'))
        for seed in (42,43):
            raw={}
            for kind,source in [('C3',study.base.ART),('H0',study.ART),('H2',study.ART)]:
                name=f'{kind}_seed{seed}_fold{fold}';raw[kind]=study.checked_raw(pd.read_csv(source/'metrics'/f'{name}_predictions.csv.gz'),val)
            threshold=read(study.base.ART/'metrics'/f'C3_seed{seed}_fold{fold}.json')['threshold']
            for kind,tree in [('C3',None),('T0','H0'),('T2','H2')]:
                p=raw['C3'] if tree is None else {w:{q:(a+raw[tree][w][q])*.5 for q,a in pp.items()} for w,pp in raw['C3'].items()}
                adjusted={}
                for w,pp in p.items():
                    with np.load(ROOT/'artifacts/recurrence_mean_control/graphs'/f'{w}.npz',allow_pickle=False) as z:g=z['M']
                    adjusted[w]={q:study.mix_probabilities(a,g) for q,a in pp.items()}
                for policy,x,strength in [('raw',p,0.),('M10',adjusted,1.)]:
                    assert time.monotonic()-began<900
                    row=study.dec.evaluate(x,val,threshold,prior,strength,f'{kind}_seed{seed}_fold{fold}_{policy}',digest)
                    if kind=='C3':assert max(abs(row[c]-old.loc[(kind,policy,fold,seed),c]) for c in study.COLS)<1e-10
                    rows.append(dict(kind=kind,policy=policy,fold=fold,seed=seed,**row))
    df=pd.DataFrame(rows);df.to_csv(ART/'summary.csv',index=False);means=df.groupby(['kind','policy'])[study.COLS].mean();means.to_csv(OUT/'means.csv');comparisons=[]
    for kind,ref in [('T2','C3'),('T2','T0'),('T0','C3')]:
        for policy in ('raw','M10'):
            a=df[(df.kind==kind)&(df.policy==policy)].set_index(['fold','seed']);b=df[(df.kind==ref)&(df.policy==policy)].set_index(['fold','seed']);d=a[study.COLS]-b[study.COLS]
            comparisons.append(dict(candidate=kind,reference=ref,policy=policy,**d.mean().to_dict(),positive=int((d.macro_f1_tol1>0).sum()),passed=bool(d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>=0 and d.raw_ap.mean()>=0 and (d.macro_f1_tol1>0).sum()>=3)))
    write(OUT/'comparisons.json',comparisons);assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'completion_audit.json',dict(status='complete',training_runs=0,decode_cells=24,saved_positions_recomputed=True,c3_reproduced=True,hashes_unchanged=True,test_used=False,seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None));print(means.to_string(),flush=True)


if __name__=='__main__':main()
