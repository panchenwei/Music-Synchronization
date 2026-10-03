"""Matched frozen-MERT/temporal-shuffle/score controls, external development only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from . import external_audio_trial as base
from .score_context_study import ROOT,read,write,sha
from .prepare_external_mert_trial import OUT,ART

OLD_OUT=base.OUT;OLD_ART=base.ART;COLS=base.COLS


def mask_modalities(x,kind):
    assert kind in ('S','E','F');x=x.copy()
    if kind=='S':x[...,28:120]=0
    return x


def load_fold(data,fold,kind):
    result={}
    for key,v in data.items():
        with np.load(ART/'cache'/f'fold{fold}'/f'{key}.npz',allow_pickle=False) as z:feat=z['F' if kind=='F' else 'E']
        x=v['features'].copy();x[:,28:120]=0;x[:,28:60]=feat
        result[key]={**v,'features':x}
    return result


def prepare():
    assert read(OUT/'preparation_audit.json')['status']=='prepared'
    for p in (ART/'checkpoints',ART/'metrics'):p.mkdir(exist_ok=True)
    data=base.load_data();sp=read(OLD_OUT/'splits.json');hashes=dict(read(OUT/'source_hashes.json'));checks=[]
    for f in range(3):
        old=base.subset(data,sp[str(f)]['train']);onorm=base.normalizer(old)
        for kind in ('S','E','F'):
            new=base.subset(load_fold(data,f,kind),sp[str(f)]['train']);norm=base.normalizer(new)
            for col in (list(range(28))+[120]):
                assert norm.mean[col]==onorm.mean[col] and norm.std[col]==onorm.std[col]
            for key in old:
                for field in ('labels','label_mask'):np.testing.assert_array_equal(new[key][field],old[key][field])
                np.testing.assert_array_equal(new[key]['features'][:,:28],old[key]['features'][:,:28])
                if kind=='S':np.testing.assert_array_equal(mask_modalities(norm.apply(new[key]['features']),'S'),base.mask_modalities(onorm.apply(old[key]['features']),'S'))
            checks.append(dict(fold=f,kind=kind,original_score_labels_coverage_unchanged=True,original_score_norm_equal=True))
    for p in (Path(__file__),ROOT/'src/external_audio_trial.py',ROOT/'tests/test_external_mert_trial.py',OUT/'PROTOCOL.md'):
        hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items());digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));pd.DataFrame(checks).to_csv(OUT/'input_audit.csv',index=False)
    return data,sp,digest


def audit(data,sp,contract):
    began=time.monotonic();assert len(list((ART/'metrics').glob('*.json')))==18
    hashes=read(OUT/'contract.json')['hashes'];checkpoints={str(p):sha(p) for p in (ART/'checkpoints').glob('*/*.pt')};assert len(checkpoints)==36
    rows=[];checks=[];workrows=[]
    for f in range(3):
        for seed in (42,43):
            for kind in ('S','E','F'):
                assert time.monotonic()-began<1200
                condition=load_fold(data,f,kind);train=base.subset(condition,sp[str(f)]['train']);val=base.subset(condition,sp[str(f)]['validation'])
                held=base.subset(condition,sp[str(f)]['test']);norm=base.normalizer(train)
                name=f'{kind}_seed{seed}_fold{f}';meta=read(ART/'metrics'/f'{name}.json');folder=ART/'checkpoints'/name
                best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);last=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
                assert best['contract']==last['contract']==meta['contract']==contract and last['step']==300 and meta['params']==7937
                assert best['step']==meta['best_step']==max(last['history'],key=lambda r:r['macro_f1_tol1'])['step']
                np.testing.assert_array_equal(norm.mean,best['mean']);np.testing.assert_array_equal(norm.std,best['std'])
                model=base.make_model(seed).cuda();model.load_state_dict(best['model']);raw=base.predict(model,val,norm,kind)
                with np.load(ART/'metrics'/f'{name}_validation.npz',allow_pickle=False) as z:
                    err=max(float(abs(raw[k]-z[k]).max()) for k in raw);assert err<2e-4
                assert max(abs(base.metrics(raw,val,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
                hp=base.predict(model,held,norm,kind);path=ART/'metrics'/f'{name}_held_development.npz'
                if path.exists():
                    with np.load(path,allow_pickle=False) as z:
                        for k in hp:np.testing.assert_allclose(hp[k],z[k],atol=2e-4,rtol=0)
                else:np.savez_compressed(path,**hp)
                with np.load(path,allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
                frame,works,score=base.metrics(saved,held,meta['threshold']);frame.to_csv(ART/'metrics'/f'{name}_held_performances.csv',index=False)
                assert max(abs(base.metrics(hp,held,meta['threshold'])[2][c]-score[c]) for c in COLS)<1e-10
                old_error=None
                if kind=='S':
                    with np.load(OLD_ART/'metrics'/f'{name}_test.npz',allow_pickle=False) as z:old_error=max(float(abs(hp[k]-z[k]).max()) for k in hp)
                checks.append(dict(run=name,validation_replay_error=err,old_S_prediction_error=old_error))
                rows.append(dict(kind=kind,fold=f,seed=seed,**score));workrows.extend([dict(kind=kind,fold=f,seed=seed,**r) for r in works.to_dict('records')])
                pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',name,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'fold_metrics.csv',index=False);work=pd.DataFrame(workrows);work.to_csv(OUT/'held_development_work_metrics.csv',index=False)
    cols=['f1_tol1','f1_tol0','precision_tol1','recall_tol1','raw_ap'];means=work.groupby('kind')[cols].mean();means.to_csv(OUT/'means.csv');comparisons=[]
    rng=np.random.default_rng(20260914)
    for kind,ref in [('E','S'),('E','F'),('F','S')]:
        keys=['fold','seed','piece_id','group'];a=work[work.kind==kind].set_index(keys);b=work[work.kind==ref].set_index(keys);d=a[cols]-b[cols]
        delta=d.reset_index().groupby(['group','piece_id']).mean(numeric_only=True).reset_index();groups=sorted(work.group.unique());samples=[]
        for _ in range(2000):
            selected=rng.choice(groups,len(groups),replace=True);samples.append(np.concatenate([delta.loc[delta.group==g,'f1_tol1'].to_numpy() for g in selected]).mean())
        cells=d.groupby(['fold','seed']).mean();lo,hi=np.quantile(samples,[.025,.975])
        comparisons.append(dict(candidate=kind,reference=ref,**d[['f1_tol1','f1_tol0','raw_ap']].mean().to_dict(),positive_cells=int((cells.f1_tol1>0).sum()),
            conditional_group_ci=[float(lo),float(hi)],passed=bool(d.f1_tol1.mean()>=.015 and d.f1_tol0.mean()>=0 and d.raw_ap.mean()>=0 and (cells.f1_tol1>0).sum()>=4)))
    write(OUT/'comparisons.json',comparisons)
    assert all(sha(p)==h for p,h in hashes.items()) and all(sha(p)==h for p,h in checkpoints.items());write(OUT/'checkpoint_hashes.json',checkpoints)
    write(OUT/'completion_audit.json',dict(status='complete',training_runs=18,checkpoints=36,gpu_validation_replays=18,held_development_inferences=18,
        source_hashes_unchanged=True,new_blind_test=False,pretraining_overlap_unknown=True,encoder_frozen=True,
        training_seconds=sum(read(p)['seconds'] for p in (ART/'metrics').glob('*.json')),audit_seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None));print(means.to_string(),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);data,sp,digest=prepare()
    if args.stage=='prepare':print(digest,flush=True);return
    base.OUT=OUT;base.ART=ART;base.mask_modalities=mask_modalities
    if args.stage!='audit':
        began=time.monotonic()
        for f in range(3):
            for seed in (42,43):
                for kind in ('S','E','F'):
                    guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent']
                    assert time.monotonic()-began<5400
                    base.train_one(load_fold(data,f,kind),sp,kind,f,seed,digest)
    audit(data,sp,digest)


if __name__=='__main__':
    try:
        with threadpool_limits(2):main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed',pid=None));raise
