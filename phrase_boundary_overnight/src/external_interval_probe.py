"""Local audio novelty with a training-only work-balanced duration potential."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.metrics import average_precision_score
from . import external_audio_trial as base
from .interstart_decoder import decode
from .evaluation import evaluate_piece
from .score_context_study import ROOT,read,write,sha

OUT=ROOT/'reports/external_interval_probe';ART=ROOT/'artifacts/external_interval_probe';SOURCE=ROOT/'artifacts/external_structure_probe'
COLS=['f1_tol1','f1_tol0','precision_tol1','recall_tol1','raw_ap']


def fit_prior(data):
    bywork={};seen=set();used=[]
    for key,item in sorted(data.items()):
        identity=(item['piece_id'],item['audio_hash'])
        if identity in seen:continue
        seen.add(identity);mask=item['label_mask']>.5;starts=np.flatnonzero((item['labels']>.5)&mask)
        gaps=[int(b-a) for a,b in zip(starts[:-1],starts[1:]) if mask[a:b+1].all()]
        if not gaps:continue
        assert max(gaps)<=256
        bywork.setdefault(item['piece_id'],[]).append(np.bincount(np.asarray(gaps)-1,minlength=256)/len(gaps));used.append(dict(record=key,piece_id=item['piece_id'],audio_hash=item['audio_hash'],interval_appearances=len(gaps)))
    assert bywork,'No complete training intervals; do not invent prior'
    hist=np.mean([np.mean(v,axis=0) for v in bywork.values()],axis=0);mean=float(hist@np.arange(1,257))
    kernel=np.exp(-.5*np.arange(-3,4,dtype=float)**2);kernel/=kernel.sum();smooth=np.convolve(hist,kernel,'same');smooth/=smooth.sum()
    return dict(histogram=smooth.tolist(),mean=mean,works=len(bywork),intervals=sum(r['interval_appearances'] for r in used),support=256,geometric_mixture=.5,smoothing_sigma=1.,records=used)


def main():
    began=time.monotonic()
    for p in (OUT,ART):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(ROOT/'reports/external_structure_probe/contract.json')['hashes']);hashes.update(read(ROOT/'reports/external_structure_probe/artifact_hashes.json'))
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'src/interstart_decoder.py',ROOT/'tests/test_external_interval_probe.py',ROOT/'reports/external_structure_probe/fold_metrics.csv'):hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items());digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));data=base.load_data();sp=read(base.OUT/'splits.json');old=pd.read_csv(ROOT/'reports/external_structure_probe/fold_metrics.csv');rows=[];workrows=[];positions=[]
    for f in range(3):
        train=base.subset(data,sp[str(f)]['train']);held=base.subset(data,sp[str(f)]['test']);prior=fit_prior(train)
        write(OUT/f'prior_fold{f}.json',dict(prior=prior,train_groups=sp[str(f)]['train']));reference=old[(old.fold==f)&(old.kind=='M')].iloc[0];threshold=float(reference.threshold)
        with np.load(SOURCE/f'M_fold{f}_predictions.npz',allow_pickle=False) as z:m={k:z[k].copy() for k in held}
        with np.load(SOURCE/f'MD_fold{f}_predictions.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in held}
        for kind,raw,strength in [('M0',m,0.),('MP',m,1.),('DP',d,1.),('UP',{k:np.full(len(v['labels']),.5) for k,v in held.items()},1.)]:
            perfs=[]
            for k,v in held.items():
                cut=decode(raw[k],threshold,prior,strength);binary=np.zeros(len(raw[k]));binary[cut]=1
                score=evaluate_piece(k,binary,v['labels'],v['label_mask'],.5);mask=v['label_mask']>.5
                perfs.append(dict(record=k,piece_id=v['piece_id'],group=v['group'],raw_ap=float(average_precision_score(v['labels'][mask],raw[k][mask])),**{c:score[c] for c in COLS if c!='raw_ap'}))
                path=ART/f'{kind}_fold{f}_{k}.npz'
                if path.exists():
                    with np.load(path,allow_pickle=False) as z:np.testing.assert_array_equal(z['cuts'],cut)
                else:np.savez_compressed(path,cuts=cut)
                with np.load(path,allow_pickle=False) as z:saved=z['cuts'].copy()
                replay=np.zeros(len(binary));replay[saved]=1;sc=evaluate_piece(k,replay,v['labels'],v['label_mask'],.5)
                assert all(sc[f'{c}_tol{t}']==score[f'{c}_tol{t}'] for c in ('tp','fp','fn') for t in (0,1));positions.append(dict(kind=kind,fold=f,record=k,cuts=len(cut)))
                assert time.monotonic()-began<900
            frame=pd.DataFrame(perfs);frame.to_csv(OUT/f'{kind}_fold{f}_performances.csv',index=False);work=frame.groupby(['piece_id','group'],as_index=False)[COLS].mean();score=work[COLS].mean().to_dict()
            if kind=='M0':assert max(abs(score[c]-reference['raw_ap' if c=='raw_ap' else 'macro_'+c]) for c in COLS)<1e-10
            workrows.extend([dict(kind=kind,fold=f,**r) for r in work.to_dict('records')]);rows.append(dict(kind=kind,fold=f,**score));print('INTERVAL',kind,f,score,flush=True)
    work=pd.DataFrame(workrows);work.to_csv(OUT/'held_development_work_metrics.csv',index=False);means=work.groupby('kind')[COLS].mean();means.to_csv(OUT/'means.csv');pd.DataFrame(rows).to_csv(OUT/'fold_metrics.csv',index=False)
    keys=['fold','piece_id','group'];a=work[work.kind=='MP'].set_index(keys);b=work[work.kind=='M0'].set_index(keys);np.testing.assert_array_equal(a.raw_ap,b.raw_ap)
    rng=np.random.default_rng(20260914);comparisons=[]
    for ref in ('M0','DP','UP'):
        d=a[COLS]-work[work.kind==ref].set_index(keys)[COLS];delta=d.reset_index();groups=sorted(delta.group.unique());samples=[]
        for _ in range(2000):samples.append(np.concatenate([delta.loc[delta.group==g,'f1_tol1'].to_numpy() for g in rng.choice(groups,len(groups),replace=True)]).mean())
        cells=d.groupby('fold').mean();comparisons.append(dict(reference=ref,**d.mean().to_dict(),positive_folds=int((cells.f1_tol1>0).sum()),conditional_group_ci=np.quantile(samples,[.025,.975]).tolist(),passed=bool(d.f1_tol1.mean()>=.015 and d.f1_tol0.mean()>=0 and (cells.f1_tol1>0).sum()>=2)))
    write(OUT/'comparisons.json',comparisons);write(OUT/'positions_audit.json',positions);assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'artifact_hashes.json',{str(p):sha(p) for p in ART.glob('*')})
    write(OUT/'completion_audit.json',dict(status='complete',training_runs=0,decode_cells=12,position_replays=len(positions),priors_train_work_balanced=True,original_novelty_reproduced=True,AP_unchanged_by_decoder=True,hashes_unchanged=True,new_blind_test=False,seconds=time.monotonic()-began))
    print(means.to_string(),flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
