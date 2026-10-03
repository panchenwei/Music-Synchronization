"""Post-hoc conditional work/opus intervals for the frozen soft decoder."""
import numpy as np,pandas as pd
from .score_context_study import ROOT,read,write,sha


def main():
    out=ROOT/'reports/interstart_decoder_uncertainty';out.mkdir(exist_ok=True)
    art=ROOT/'artifacts/interstart_decoder_study';parts=[];hashes={}
    assert read(ROOT/'reports/interstart_decoder_study/completion_audit.json')['status']=='complete'
    for fold in (0,1):
        for seed in (42,43):
            paths=[art/f'C3_seed{seed}_fold{fold}_lambda{x}_pieces.csv' for x in ('0','0.5')]
            a,b=[pd.read_csv(p) for p in paths];hashes.update({str(p):sha(p) for p in paths})
            z=a.merge(b,on='piece_id',suffixes=('_base','_soft'),validate='one_to_one');assert len(z)==len(a)==len(b)
            z['fold']=fold;z['seed']=seed
            for c in ('f1_tol1','f1_tol0'):z[c+'_delta']=z[c+'_soft']-z[c+'_base']
            parts.append(z)
    frame=pd.concat(parts,ignore_index=True);manifest=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv',dtype={'opus':str})
    manifest=manifest[(manifest.fold.isin([0,1]))&(manifest.split=='validation')][['fold','piece_id','opus']]
    cols=['f1_tol1_soft','f1_tol0_soft','f1_tol1_delta','f1_tol0_delta']
    work=frame.merge(manifest,on=['fold','piece_id'],validate='many_to_one').groupby(['fold','piece_id','opus'])[cols].mean().reset_index()
    assert len(work)==19;work.to_csv(out/'paired_works.csv',index=False)
    point=work.groupby('fold')[cols].mean().mean().to_numpy();rows=[];rng=np.random.default_rng(20260911)
    for method in ('work','opus_cluster'):
        samples=np.zeros((10000,len(cols)))
        for fold,g in work.groupby('fold'):
            g=g.reset_index(drop=True);v=g[cols].to_numpy()
            if method=='work':samples+=v[rng.integers(0,len(g),(10000,len(g)))].mean(1)/2
            else:
                groups=[np.flatnonzero(g.opus.to_numpy()==o) for o in sorted(g.opus.unique())]
                for i,choice in enumerate(rng.integers(0,len(groups),(10000,len(groups)))):samples[i]+=v[np.concatenate([groups[j] for j in choice])].mean(0)/2
        for j,c in enumerate(cols):
            lo,hi=np.quantile(samples[:,j],[.025,.975]);rows.append(dict(method=method,metric=c,point=point[j],lower95=lo,upper95=hi))
    assert abs(point[0]-.599067508091749)<1e-10 and all(sha(p)==h for p,h in hashes.items())
    result=pd.DataFrame(rows);result.to_csv(out/'intervals.csv',index=False)
    write(out/'audit.json',dict(status='complete',works=19,opus=int(work.opus.nunique()),seeds_averaged_within_work=2,replicates=10000,post_hoc=True,source_hashes=hashes,test_used=False,
        caveat='Conditional on trained/selected models and repeatedly reused development works; no correction for model selection or proof of blind generalization. Few opus clusters; seeds are not independent works.'))
    print(result.to_string(index=False))


if __name__=='__main__':main()
