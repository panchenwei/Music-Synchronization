"""Frozen, fold-specific predictive information x boundary training experiment."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from . import score_context_study as engine
from . import run_recurrence_depth_study as base
from . import run_halo_decoder_composition as dec
from .prepare_predictive_information import ART as FEATURES
from .predictive_information_model import make_model
from .score_context_study import ROOT,read,write,sha,normalizer as base_normalizer
from .phase3_models import fit_train_normalizer
from .models import Normalizer
from .three_round_round2 import split_ids
from .local_context_study import metrics,predictions
from .audit_external_stem_transfer import checked_raw
from .context_inference_audit_v2 import error
from .interstart_decoder import fit_prior
from .recurrence_message_probe import mix_probabilities

OUT=ROOT/'reports/predictive_information_boundary';ART=ROOT/'artifacts/predictive_information_boundary'
COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']


def dataset(ids,kind):
    assert kind in ('G','D','Z')
    matches=[f for f in (0,1) if any(set(ids)==set(split_ids(f)[s]) for s in ('train','validation'))]
    assert len(matches)==1,'Ambiguous fold for feature fitting'
    data=base.dataset(ids,'C3')
    for pid,v in data.items():
        with np.load(FEATURES/f'fold{matches[0]}'/f'{pid}.npz',allow_pickle=False) as z:feat=z[kind].copy()
        assert feat.shape==(len(v['labels']),14) and np.isfinite(feat).all()
        v['information']=feat;v['curves']=np.concatenate([v['curves'],np.broadcast_to(feat,(*v['curves'].shape[:2],14))],-1)
    return data


def normalizer(data):
    original=base_normalizer(data);extra=fit_train_normalizer([v['information'] for v in data.values()])
    assert len(original.mean)==58
    return Normalizer(np.r_[original.mean,extra.mean],np.r_[original.std,extra.std])


def prepare():
    assert read(ROOT/'reports/predictive_information/completion_audit.json')['predictor_gate_passed']
    for p in (OUT,ART/'metrics',ART/'checkpoints',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(base.OUT/'contract.json')['hashes']);hashes.update(read(ROOT/'reports/predictive_information/source_hashes.json'))
    files=[Path(__file__),ROOT/'src/predictive_information_model.py',ROOT/'tests/test_predictive_information_model.py',ROOT/'reports/predictive_information/BOUNDARY_PROTOCOL.md',
        ROOT/'src/interstart_decoder.py',ROOT/'src/run_halo_decoder_composition.py',ROOT/'src/recurrence_message_probe.py',ROOT/'src/score_context_study.py',
        ROOT/'src/run_recurrence_depth_study.py',ROOT/'src/recurrence_depth_models.py']
    files+=list((ROOT/'artifacts/recurrence_mean_control/graphs').glob('*.npz'));checks=[]
    for f in (0,1):
        ids=split_ids(f);old=base.dataset(ids['train'],'C3');on=base_normalizer(old)
        manifest=pd.read_csv(ROOT/'artifacts/phase2/splits/opus_split_manifest.csv');part=manifest[manifest.fold==f]
        assert not set(part[part.split=='train'].opus)&set(part[part.split=='validation'].opus)
        for kind in ('G','D','Z'):
            new=dataset(ids['train'],kind);norm=normalizer(new)
            np.testing.assert_array_equal(norm.mean[:58],on.mean);np.testing.assert_array_equal(norm.std[:58],on.std)
            for pid in old:
                np.testing.assert_array_equal(new[pid]['curves'][...,:58],old[pid]['curves'])
                for key in ('labels','label_mask','performance_ids'):np.testing.assert_array_equal(new[pid][key],old[pid][key])
            for s in (42,43):
                a=engine.CurvePieceBalancedSampler(new,norm,64,32,s).batch();b=engine.CurvePieceBalancedSampler(old,on,64,32,s).batch()
                torch.testing.assert_close(a[0][...,:58],b[0],atol=0,rtol=0)
                for x,y in zip(a[1:],b[1:]):torch.testing.assert_close(x,y,atol=0,rtol=0)
            checks.append(dict(fold=f,kind=kind,original58_norm_labels_sampler_equal=True,opus_disjoint=True))
        for s in (42,43):
            for suffix in ('.json','_predictions.csv.gz'):files.append(base.ART/'metrics'/f'C3_seed{s}_fold{f}{suffix}')
    for p in files:hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));pd.DataFrame(checks).to_csv(OUT/'input_audit.csv',index=False)
    if not (OUT/'STATE.json').exists():write(OUT/'STATE.json',dict(status='ready',completed=[],seconds=0.,pid=None))
    return digest


def audit():
    began=time.monotonic();contract=read(OUT/'contract.json');checks=[];rows=[];histories=[]
    hashes={str(p):sha(p) for p in (ART/'checkpoints').glob('*/*.pt')};assert len(hashes)==24
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder'
    for f in (0,1):
        ids=split_ids(f)
        for s in (42,43):
            for kind,source in [('C3',base.ART),('G',ART),('D',ART),('Z',ART)]:
                if kind=='C3':
                    train=base.dataset(ids['train'],'C3');val=base.dataset(ids['validation'],'C3');norm=base_normalizer(train)
                else:
                    train=dataset(ids['train'],kind);val=dataset(ids['validation'],kind);norm=normalizer(train)
                prior=fit_prior(train);assert time.monotonic()-began<1800
                name=f'{kind}_seed{s}_fold{f}';meta=read(source/'metrics'/f'{name}.json')
                raw=checked_raw(pd.read_csv(source/'metrics'/f'{name}_predictions.csv.gz'),val)
                assert max(abs(metrics(raw,val,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
                if kind!='C3':
                    best=torch.load(source/'checkpoints'/name/'best.pt',map_location='cpu',weights_only=False)
                    last=torch.load(source/'checkpoints'/name/'latest.pt',map_location='cpu',weights_only=False)
                    assert best['contract']==last['contract']==meta['contract']==contract['contract'] and last['step']==300
                    assert (best['kind'],best['fold'],best['seed'])==(kind,f,s)
                    assert best['step']==meta['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
                    assert meta['params']==6369
                    np.testing.assert_array_equal(norm.mean,best['mean']);np.testing.assert_array_equal(norm.std,best['std'])
                    m=make_model(kind,s).cuda();m.load_state_dict(best['model']);replay=predictions(m,val,norm,torch.device('cuda'))
                    err=error(raw,replay);assert err<2e-4
                    assert max(abs(metrics(replay,val,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
                    checks.append(dict(run=name,replay_error=err,params=meta['params']));histories.extend([dict(run=name,kind=kind,**h) for h in last['history']])
                    pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False);pd.DataFrame(histories).to_csv(OUT/'training_history.csv',index=False)
                adjusted={}
                for pid,pp in raw.items():
                    with np.load(ROOT/'artifacts/recurrence_mean_control/graphs'/f'{pid}.npz',allow_pickle=False) as z:g=z['M']
                    adjusted[pid]={q:mix_probabilities(a,g) for q,a in pp.items()}
                for policy,x,strength in [('raw',raw,0.),('M10',adjusted,1.)]:
                    r=dec.evaluate(x,val,meta['threshold'],prior,strength,f'{name}_{policy}',contract['contract'])
                    if policy=='raw':assert max(abs(r[c]-meta[c]) for c in COLS)<1e-10
                    rows.append(dict(kind=kind,seed=s,fold=f,policy=policy,**r))
                pd.DataFrame(rows).to_csv(ART/'summary.csv',index=False);print('AUDITED',name,flush=True)
    df=pd.DataFrame(rows);assert len(df)==32 and len(checks)==12
    means=df.groupby(['kind','policy'])[COLS].mean();means.to_csv(OUT/'means.csv');comp=[]
    for ref in ('C3','Z','D'):
        for policy in ('raw','M10'):
            a=df[(df.kind=='G')&(df.policy==policy)].set_index(['fold','seed']);b=df[(df.kind==ref)&(df.policy==policy)].set_index(['fold','seed']);d=a[COLS]-b[COLS]
            comp.append(dict(reference=ref,policy=policy,**d.mean().to_dict(),positive=int((d.macro_f1_tol1>0).sum()),passed=bool(d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>=0 and d.raw_ap.mean()>=0 and (d.macro_f1_tol1>0).sum()>=3)))
    write(OUT/'comparisons.json',comp)
    assert all(sha(p)==h for p,h in hashes.items()) and all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'checkpoint_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=12,gpu_replays=12,checkpoints=24,decoded_cells=32,
        old_metrics_recomputed=4,condition_specific_norms=True,hashes_unchanged=True,test_used=False,
        training_seconds=sum(read(p)['seconds'] for p in (ART/'metrics').glob('*.json')),audit_seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None));print(means.to_string(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if args.stage=='audit':audit();return
    digest=prepare()
    if args.stage=='prepare':print(digest,flush=True);return
    engine.OUT=OUT;engine.ART=ART;engine.CAP=4800.;engine.dataset=dataset;engine.normalizer=normalizer;engine.make_model=make_model
    for f in (0,1):
        for s in (42,43):
            for k in ('G','D','Z'):
                guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent']
                engine.train_one(k,f,s,digest)
    audit()


if __name__=='__main__':
    try:
        with threadpool_limits(2):main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',{**(read(OUT/'STATE.json') if (OUT/'STATE.json').exists() else {}),'status':'failed','pid':None});raise
