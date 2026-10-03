"""Descriptive work/opus uncertainty; does not change the preregistered gate."""
import numpy as np
import pandas as pd
from .ensemble_seed_replication import ROOT,ART,OUT,read,write


def main():
    assert read(OUT/'completion_audit.json')['status']=='complete'
    parts=[]
    for seed in (42,43,44,45):
        for fold in (0,1):
            nrroot=ROOT/'artifacts/fixed_ensemble_study/metrics' if seed<44 else ART/'ensemble'
            broot=ROOT/'artifacts/score_novelty_study/metrics' if seed<44 else ART/'metrics'
            a=pd.read_csv(nrroot/f'NR_seed{seed}_fold{fold}_pieces.csv')
            b=pd.read_csv(broot/f'B_seed{seed}_fold{fold}_pieces.csv')
            merged=a.merge(b,on='piece_id',suffixes=('_NR','_B'),validate='one_to_one')
            assert len(merged)==len(a)==len(b)
            merged['fold']=fold;merged['seed']=seed
            for name in ('f1_tol0','f1_tol1'):
                merged[name+'_delta']=merged[name+'_NR']-merged[name+'_B']
            parts.append(merged[['fold','seed','piece_id','f1_tol0_delta','f1_tol1_delta']])
    frame=pd.concat(parts,ignore_index=True)
    manifest=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv',dtype={'opus':str})
    manifest=manifest[(manifest.fold.isin([0,1]))&(manifest.split=='validation')][['fold','piece_id','opus']]
    frame=frame.merge(manifest,on=['fold','piece_id'],validate='many_to_one')
    frame.to_csv(OUT/'paired_work_deltas.csv',index=False)
    columns=['f1_tol0_delta','f1_tol1_delta'];rows=[];rng=np.random.default_rng(20260910);repetitions=10000
    for label,seeds in (('original_42_43',[42,43]),('replication_44_45',[44,45]),('all_four_seeds',[42,43,44,45])):
        subset=frame[frame.seed.isin(seeds)].groupby(['fold','piece_id','opus'])[columns].mean().reset_index()
        point=subset.groupby('fold')[columns].mean().mean().to_numpy()
        for method in ('work','opus_cluster'):
            samples=np.zeros((repetitions,2))
            for fold,g in subset.groupby('fold'):
                g=g.reset_index(drop=True);values=g[columns].to_numpy()
                if method=='work':
                    indices=rng.integers(0,len(g),size=(repetitions,len(g)))
                    samples+=values[indices].mean(axis=1)/2
                else:
                    groups=[np.flatnonzero(g.opus.to_numpy()==o) for o in sorted(g.opus.unique())]
                    picks=rng.integers(0,len(groups),size=(repetitions,len(groups)))
                    for i,choice in enumerate(picks):
                        indices=np.concatenate([groups[j] for j in choice])
                        samples[i]+=values[indices].mean(axis=0)/2
            for j,metric in enumerate(columns):
                low,high=np.quantile(samples[:,j],[.025,.975])
                rows.append(dict(subset=label,resampling=method,metric=metric,mean_delta=point[j],lower95=float(low),upper95=float(high),works=len(subset),opus=int(subset.opus.nunique()),seeds=len(seeds),replicates=repetitions))
    result=pd.DataFrame(rows);result.to_csv(OUT/'descriptive_uncertainty.csv',index=False)
    write(OUT/'uncertainty_notes.json',dict(status='complete',post_hoc=True,changes_selection_gate=False,
        estimand='Average paired difference: mean seeds within each work, mean works within each fold, equal mean of two folds.',
        note='Conditional on already-trained/selected models and reused development works. Intervals do not correct model-selection bias, and additional seeds are not additional works. Opus clusters are few; no independent-test or definitive significance claim.'))
    print(result.to_string(index=False))


if __name__=='__main__':main()
