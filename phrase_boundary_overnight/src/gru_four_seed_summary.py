"""Combine frozen original and additional GRU seeds; conditional opus bootstrap."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .score_context_study import ROOT,read,write,sha

OUT=ROOT/'reports/gru_four_seed_summary';COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']


def main():
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);frames=[];pieces=[];hashes={}
    for study in ('current_gru_study','current_gru_replication'):
        report=ROOT/'reports'/study;art=ROOT/'artifacts'/study
        assert read(report/'completion_audit.json')['status']=='complete'
        source=read(report/'contract.json')['hashes'];assert all(sha(p)==h for p,h in source.items());hashes.update(source)
        frames.append(pd.read_csv(art/'summary.csv'));hashes[str(art/'summary.csv')]=sha(art/'summary.csv')
        for path in (art/'decoder').glob('*_M10_pieces.csv'):
            kind,seed,fold,_=path.stem.split('_',3)
            v=pd.read_csv(path);v['kind']=kind;v['seed']=int(seed[4:]);v['fold']=int(fold[4:]);pieces.append(v);hashes[str(path)]=sha(path)
    frame=pd.concat(frames,ignore_index=True);assert len(frame)==72
    assert len(frame.drop_duplicates(['kind','policy','seed','fold']))==72
    frame.to_csv(OUT/'all_runs.csv',index=False);means=[]
    for name,seeds in [('original_two',[42,43]),('additional_two',[44,45]),('all_four',[42,43,44,45])]:
        m=frame[frame.seed.isin(seeds)].groupby(['kind','policy'])[COLS+['macro_precision_tol1','macro_recall_tol1']].mean().reset_index();m['subset']=name;means.append(m)
    means=pd.concat(means);means.to_csv(OUT/'means.csv',index=False);comp=[]
    for k in ('G','E'):
        a=frame[(frame.kind==k)&(frame.policy=='M10')].set_index(['fold','seed']);b=frame[(frame.kind=='C3')&(frame.policy=='M10')].set_index(['fold','seed']);d=a[COLS]-b[COLS]
        extra=float(d.reset_index().query('seed>=44').macro_f1_tol1.mean())
        comp.append(dict(kind=k,**d.mean().to_dict(),positive=int((d.macro_f1_tol1>0).sum()),additional_seed_delta=extra,
            passed=bool(d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>=0 and d.raw_ap.mean()>=0 and (d.macro_f1_tol1>0).sum()>=6 and extra>0)))
    write(OUT/'comparisons.json',comp)
    p=pd.concat(pieces,ignore_index=True);p.to_csv(OUT/'piece_metrics.csv',index=False)
    manifest=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv',dtype={'opus':str})
    groups=manifest[manifest.split=='validation'][['fold','piece_id','opus']].drop_duplicates()
    p=p.groupby(['kind','fold','piece_id']).f1_tol1.mean().reset_index().merge(groups,on=['fold','piece_id'],validate='many_to_one')
    stats=[]
    for kind in ('G','E'):
        a=p[p.kind==kind].set_index(['fold','piece_id','opus']).f1_tol1;b=p[p.kind=='C3'].set_index(['fold','piece_id','opus']).f1_tol1;d=(a-b).reset_index(name='delta')
        rng=np.random.default_rng(20260913);samples=[]
        arrays={f:[g.delta.to_numpy() for _,g in d[d.fold==f].groupby('opus')] for f in (0,1)}
        for _ in range(5000):
            samples.append(np.mean([np.concatenate([arrays[f][i] for i in rng.integers(0,len(arrays[f]),len(arrays[f]))]).mean() for f in (0,1)]))
        lo,hi=np.quantile(samples,[.025,.975]);stats.append(dict(kind=kind,estimate=d.groupby('fold').delta.mean().mean(),ci95_low=lo,ci95_high=hi,works=d.piece_id.nunique(),opuses=d.opus.nunique(),conditional_on_selected_models=True))
    pd.DataFrame(stats).to_csv(OUT/'bootstrap.csv',index=False)
    hashes[str(Path(__file__))]=sha(Path(__file__));write(OUT/'source_hashes.json',hashes)
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'completion_audit.json',dict(status='complete',source_hashes_unchanged=True,run_cells=72,seed_count=4,bootstrap_repeats=5000,
        bootstrap_description='Within-fold opus cluster bootstrap after seed averaging; repeated development bias not corrected.',new_training_runs=0,seconds=time.monotonic()-began))
    print(means[means.policy=='M10'].to_string(index=False),flush=True);print(comp,flush=True)


if __name__=='__main__':main()
