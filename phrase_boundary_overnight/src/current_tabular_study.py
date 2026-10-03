"""Recheck the frozen shallow-tree recipe on current, not obsolete, inputs."""
import argparse,hashlib,json,time,traceback,os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import tabular_context_study as tree
from . import run_recurrence_depth_study as base
from . import run_halo_decoder_composition as dec
from .score_context_study import ROOT,read,write,sha,normalizer
from .three_round_round2 import split_ids
from .local_context_study import metrics
from .audit_external_stem_transfer import checked_raw
from .context_inference_audit_v2 import error
from .interstart_decoder import fit_prior
from .recurrence_message_probe import mix_probabilities

OUT=ROOT/'reports/current_tabular';ART=ROOT/'artifacts/current_tabular';COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']


def dataset(ids,kind):
    assert kind=='B'
    return base.dataset(ids,'C3')


def prepare():
    for p in (OUT,ART/'models',ART/'metrics',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(base.OUT/'contract.json')['hashes'])
    files=[Path(__file__),ROOT/'src/tabular_context_study.py',ROOT/'tests/test_tabular_context.py',OUT/'PROTOCOL.md',
        ROOT/'src/run_recurrence_depth_study.py',ROOT/'src/run_halo_decoder_composition.py',ROOT/'src/interstart_decoder.py',ROOT/'src/recurrence_message_probe.py']
    files+=list((ROOT/'artifacts/recurrence_mean_control/graphs').glob('*.npz'))
    for f in (0,1):
        data=dataset(split_ids(f)['train'],'B');norm=normalizer(data);assert len(norm.mean)==58
        for v in data.values():assert tree.context(norm.apply(v['curves'][0]),2).shape==(len(v['labels']),290)
        for s in (42,43):
            for suffix in ('.json','_predictions.csv.gz'):files.append(base.ART/'metrics'/f'C3_seed{s}_fold{f}{suffix}')
    for p in files:hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes,sklearn_version=tree.sklearn.__version__))
    if not (OUT/'STATE.json').exists():write(OUT/'STATE.json',dict(status='ready',completed=[],seconds=0.,pid=None))
    return digest


def audit():
    began=time.monotonic();contract=read(OUT/'contract.json');rows=[];checks=[]
    hashes={str(p):sha(p) for p in (ART/'models').glob('*/*.joblib')};assert len(hashes)==16
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder'
    for f in (0,1):
        ids=split_ids(f);train=dataset(ids['train'],'B');val=dataset(ids['validation'],'B');norm=normalizer(train);prior=fit_prior(train)
        for seed in (42,43):
            for kind,source in [('C3',base.ART),('H0',ART),('H2',ART)]:
                assert time.monotonic()-began<1200
                name=f'{kind}_seed{seed}_fold{f}';meta=read(source/'metrics'/f'{name}.json');raw=checked_raw(pd.read_csv(source/'metrics'/f'{name}_predictions.csv.gz'),val)
                assert max(abs(metrics(raw,val,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
                if kind!='C3':
                    folder=ART/'models'/name;best=joblib.load(folder/'best.joblib');last=joblib.load(folder/'latest.joblib')
                    assert best['contract']==last['contract']==meta['contract']==contract['contract'] and last['step']==150
                    assert best['step']==meta['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
                    np.testing.assert_array_equal(norm.mean,best['mean']);np.testing.assert_array_equal(norm.std,best['std'])
                    assert not best['model'].do_early_stopping_ and best['model'].n_features_in_==(58 if kind=='H0' else 290)
                    replay=tree.raw_predictions(best['model'],val,norm,tree.KINDS[kind]);err=error(raw,replay);assert err<1e-12
                    score=metrics(replay,val,meta['threshold'])[2];assert max(abs(score[c]-meta[c]) for c in COLS)<1e-10
                    ts=metrics(tree.raw_predictions(best['model'],train,norm,tree.KINDS[kind]),train,meta['threshold'])[2]
                    assert abs(ts['macro_f1_tol1']-meta['train_f1'])<1e-10
                    checks.append(dict(run=name,replay_error=err,features=meta['features'],train_f1=ts['macro_f1_tol1'],gap=meta['gap']))
                    pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False)
                adjusted={}
                for pid,pp in raw.items():
                    with np.load(ROOT/'artifacts/recurrence_mean_control/graphs'/f'{pid}.npz',allow_pickle=False) as z:g=z['M']
                    adjusted[pid]={q:mix_probabilities(a,g) for q,a in pp.items()}
                for policy,x,strength in [('raw',raw,0.),('M10',adjusted,1.)]:
                    result=dec.evaluate(x,val,meta['threshold'],prior,strength,f'{name}_{policy}',contract['contract'])
                    if policy=='raw':assert max(abs(result[c]-meta[c]) for c in COLS)<1e-10
                    rows.append(dict(kind=kind,seed=seed,fold=f,policy=policy,**result))
                pd.DataFrame(rows).to_csv(ART/'summary.csv',index=False);print('AUDITED',name,flush=True)
    assert len(rows)==24 and len(checks)==8
    df=pd.DataFrame(rows);means=df.groupby(['kind','policy'])[COLS].mean();means.to_csv(OUT/'means.csv');comparisons=[]
    for kind,ref in [('H0','C3'),('H2','C3'),('H2','H0')]:
        for policy in ('raw','M10'):
            a=df[(df.kind==kind)&(df.policy==policy)].set_index(['fold','seed']);b=df[(df.kind==ref)&(df.policy==policy)].set_index(['fold','seed']);d=a[COLS]-b[COLS]
            comparisons.append(dict(candidate=kind,reference=ref,policy=policy,**d.mean().to_dict(),positive=int((d.macro_f1_tol1>0).sum()),
                passed=bool(d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>=0 and d.raw_ap.mean()>=0 and (d.macro_f1_tol1>0).sum()>=3)))
    write(OUT/'comparisons.json',comparisons)
    assert all(sha(p)==h for p,h in hashes.items()) and all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'model_hashes.json',hashes);write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=8,model_replays=8,
        saved_models=16,decoded_cells=24,old_c3_recomputed=4,hashes_unchanged=True,test_used=False,
        training_seconds=sum(read(p)['seconds'] for p in (ART/'metrics').glob('*.json')),audit_seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None));print(means.to_string(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=p.parse_args()
    if args.stage=='audit':audit();return
    digest=prepare()
    if args.stage=='prepare':print(digest,flush=True);return
    tree.OUT=OUT;tree.ART=ART;tree.dataset=dataset;tree.normalizer=normalizer
    for f in (0,1):
        for s in (42,43):
            for k in ('H0','H2'):
                guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent']
                tree.run_one(k,f,s,digest)
    audit()


if __name__=='__main__':
    try:
        with threadpool_limits(2):main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',{**(read(OUT/'STATE.json') if (OUT/'STATE.json').exists() else {}),'status':'failed','pid':None});raise
