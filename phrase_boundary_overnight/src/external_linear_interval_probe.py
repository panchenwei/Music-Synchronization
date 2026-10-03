"""Fixed duration decoder on frozen low-capacity supervised probabilities."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from . import external_linear_novelty as trial
from . import external_audio_trial as base
from .interstart_decoder import decode
from .evaluation import evaluate_piece
from .score_context_study import ROOT,read,write,sha

OUT=ROOT/'reports/external_linear_interval_probe';ART=ROOT/'artifacts/external_linear_interval_probe';COLS=trial.COLS


def main():
    began=time.monotonic()
    for p in (OUT,ART):p.mkdir(parents=True,exist_ok=True)
    assert read(trial.OUT/'completion_audit.json')['status']=='complete' and all(v['passed'] for v in read(trial.OUT/'comparisons.json'))
    hashes=dict(read(trial.OUT/'contract.json')['hashes']);hashes.update(read(trial.OUT/'artifact_hashes.json'))
    files=[Path(__file__),OUT/'PROTOCOL.md',ROOT/'src/interstart_decoder.py',trial.OUT/'run_audit.csv',trial.OUT/'held_development_work_metrics.csv']
    files+=list((ROOT/'reports/external_interval_probe').glob('prior_fold*.json'))
    for p in files:hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items());digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));data=base.load_data();splits=read(base.OUT/'splits.json');choices=pd.read_csv(trial.OUT/'run_audit.csv');old=pd.read_csv(trial.OUT/'held_development_work_metrics.csv');workrows=[];paths=[]
    for fold in range(3):
        held=base.subset(data,splits[str(fold)]['test']);priorfile=read(ROOT/f'reports/external_interval_probe/prior_fold{fold}.json');assert priorfile['train_groups']==splits[str(fold)]['train'];prior=priorfile['prior']
        for kind in ('E','S','F'):
            threshold=float(choices[(choices.kind==kind)&(choices.fold==fold)].iloc[0].threshold)
            with np.load(trial.ART/f'{kind}_fold{fold}_test.npz',allow_pickle=False) as z:raw={k:z[k].copy() for k in held}
            for policy,strength in [('raw',0.),('interval',1.)]:
                rows=[]
                for key,item in held.items():
                    cuts=decode(raw[key],threshold,prior,strength);binary=np.zeros(len(raw[key]));binary[cuts]=1
                    score=evaluate_piece(key,binary,item['labels'],item['label_mask'],.5);known=item['label_mask']>.5
                    rows.append(dict(record=key,piece_id=item['piece_id'],group=item['group'],raw_ap=float(average_precision_score(item['labels'][known],raw[key][known])),**{c:score[c] for c in COLS if c!='raw_ap'}))
                    path=ART/f'{kind}_{policy}_fold{fold}_{key}.npz'
                    if path.exists():
                        with np.load(path,allow_pickle=False) as z:np.testing.assert_array_equal(z['cuts'],cuts)
                    else:np.savez_compressed(path,cuts=cuts)
                    with np.load(path,allow_pickle=False) as z:stored=z['cuts'].copy()
                    replay=np.zeros(len(binary));replay[stored]=1;check=evaluate_piece(key,replay,item['labels'],item['label_mask'],.5)
                    assert all(score[f'{c}_tol{t}']==check[f'{c}_tol{t}'] for c in ('tp','fp','fn') for t in (0,1));paths.append(dict(kind=kind,policy=policy,fold=fold,record=key,cuts=len(cuts)))
                frame=pd.DataFrame(rows);frame.to_csv(OUT/f'{kind}_{policy}_fold{fold}_performances.csv',index=False);work=frame.groupby(['piece_id','group'],as_index=False)[COLS].mean()
                if policy=='raw':np.testing.assert_allclose(work.set_index(['piece_id','group'])[COLS],old[(old.kind==kind)&(old.fold==fold)].set_index(['piece_id','group'])[COLS],rtol=0,atol=1e-10)
                workrows.extend([dict(kind=kind,policy=policy,fold=fold,**r) for r in work.to_dict('records')]);print(kind,policy,fold,work[COLS].mean().to_dict(),flush=True);assert time.monotonic()-began<900
    work=pd.DataFrame(workrows);work.to_csv(OUT/'held_development_work_metrics.csv',index=False);means=work.groupby(['kind','policy'])[COLS].mean();means.to_csv(OUT/'means.csv');keys=['fold','piece_id','group'];a=work[(work.kind=='E')&(work.policy=='interval')].set_index(keys);comparisons=[]
    for kind,policy in [('E','raw'),('S','interval'),('F','interval')]:
        b=work[(work.kind==kind)&(work.policy==policy)].set_index(keys);d=a[COLS]-b[COLS]
        if kind=='E':np.testing.assert_array_equal(a.raw_ap,b.raw_ap)
        cells=d.groupby('fold').mean();comparisons.append(dict(reference=kind+'_'+policy,**d.mean().to_dict(),positive_folds=int((cells.f1_tol1>0).sum()),passed=bool(d.f1_tol1.mean()>=.015 and d.f1_tol0.mean()>=0 and (cells.f1_tol1>0).sum()>=2)))
    write(OUT/'comparisons.json',comparisons);write(OUT/'positions_audit.json',paths);assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'artifact_hashes.json',{str(p):sha(p) for p in ART.glob('*')});write(OUT/'completion_audit.json',dict(status='complete',training_runs=0,decode_cells=18,position_replays=len(paths),original_linear_metrics_reproduced=True,AP_unchanged_by_decoder=True,source_hashes_unchanged=True,new_blind_test=False,seconds=time.monotonic()-began));print(means.to_string(),flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
