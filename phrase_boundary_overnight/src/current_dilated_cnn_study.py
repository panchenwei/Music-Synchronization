"""Current C3 temporal-dilation single-change study."""
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
from .current_dilated_cnn import make_model
from .score_context_study import ROOT,read,write,sha,normalizer
from .models import Normalizer
from .three_round_round2 import split_ids
from .local_context_study import metrics,predictions
from .audit_external_stem_transfer import checked_raw
from .context_inference_audit_v2 import error
from .interstart_decoder import fit_prior
from .recurrence_message_probe import mix_probabilities

OUT=ROOT/'reports/current_dilated_cnn';ART=ROOT/'artifacts/current_dilated_cnn'
COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']


def dataset(ids,kind):
    assert kind=='D'
    return base.dataset(ids,'C3')


def prepare():
    for p in (OUT,ART/'metrics',ART/'checkpoints',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(base.OUT/'contract.json')['hashes'])
    files=[Path(__file__),ROOT/'src/current_dilated_cnn.py',ROOT/'tests/test_current_dilated_cnn.py',OUT/'PROTOCOL.md',
        ROOT/'src/interstart_decoder.py',ROOT/'src/run_halo_decoder_composition.py',ROOT/'src/recurrence_message_probe.py']
    files+=list((ROOT/'artifacts/recurrence_mean_control/graphs').glob('*.npz'))
    for f in (0,1):
        for seed in (42,43):
            for suffix in ('.json','_predictions.csv.gz'):files.append(base.ART/'metrics'/f'C3_seed{seed}_fold{f}{suffix}')
    for p in files:hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    if not (OUT/'STATE.json').exists():write(OUT/'STATE.json',dict(status='ready',completed=[],seconds=0.,pid=None))
    return digest

def audit():
    began=time.monotonic();contract=read(OUT/'contract.json');checks=[];rows=[]
    hashes={str(p):sha(p) for p in (ART/'checkpoints').glob('*/*.pt')};assert len(hashes)==8
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder'
    for f in (0,1):
        ids=split_ids(f);train=dataset(ids['train'],'D');val=dataset(ids['validation'],'D');norm=normalizer(train);prior=fit_prior(train)
        for s in (42,43):
            for kind,source in [('C3',base.ART),('D',ART)]:
                assert time.monotonic()-began<1200
                name=f'{kind}_seed{s}_fold{f}';meta=read(source/'metrics'/f'{name}.json')
                raw=checked_raw(pd.read_csv(source/'metrics'/f'{name}_predictions.csv.gz'),val)
                assert max(abs(metrics(raw,val,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
                if kind=='D':
                    best=torch.load(source/'checkpoints'/name/'best.pt',map_location='cpu',weights_only=False)
                    last=torch.load(source/'checkpoints'/name/'latest.pt',map_location='cpu',weights_only=False)
                    assert best['contract']==last['contract']==meta['contract']==contract['contract'] and last['step']==300
                    assert (best['kind'],best['fold'],best['seed'])==(kind,f,s)
                    assert best['step']==meta['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
                    np.testing.assert_array_equal(norm.mean,best['mean']);np.testing.assert_array_equal(norm.std,best['std'])
                    m=make_model('D',s).cuda();m.load_state_dict(best['model']);replay=predictions(m,val,norm,torch.device('cuda'))
                    err=error(raw,replay);assert err<2e-4
                    assert max(abs(metrics(replay,val,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
                    checks.append(dict(run=name,replay_error=err));pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False)
                adjusted={}
                for pid,pp in raw.items():
                    with np.load(ROOT/'artifacts/recurrence_mean_control/graphs'/f'{pid}.npz',allow_pickle=False) as z:g=z['M']
                    adjusted[pid]={q:mix_probabilities(a,g) for q,a in pp.items()}
                for policy,x,strength in [('raw',raw,0.),('M10',adjusted,1.)]:
                    r=dec.evaluate(x,val,meta['threshold'],prior,strength,f'{name}_{policy}',contract['contract'])
                    if policy=='raw':assert max(abs(r[c]-meta[c]) for c in COLS)<1e-10
                    rows.append(dict(kind=kind,seed=s,fold=f,policy=policy,**r))
                pd.DataFrame(rows).to_csv(ART/'summary.csv',index=False);print('AUDITED',name,flush=True)
    df=pd.DataFrame(rows);assert len(df)==16 and len(checks)==4
    means=df.groupby(['kind','policy'])[COLS].mean();means.to_csv(OUT/'means.csv');comp=[]
    for ref in ('C3',):
        for policy in ('raw','M10'):
            a=df[(df.kind=='D')&(df.policy==policy)].set_index(['fold','seed']);b=df[(df.kind==ref)&(df.policy==policy)].set_index(['fold','seed']);d=a[COLS]-b[COLS]
            comp.append(dict(reference=ref,policy=policy,**d.mean().to_dict(),positive=int((d.macro_f1_tol1>0).sum()),passed=bool(d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>=0 and d.raw_ap.mean()>=0 and (d.macro_f1_tol1>0).sum()>=3)))
    write(OUT/'comparisons.json',comp)
    assert all(sha(p)==h for p,h in hashes.items()) and all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'checkpoint_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=4,gpu_replays=4,checkpoints=8,decoded_cells=16,
        old_metrics_recomputed=4,hashes_unchanged=True,test_used=False,training_seconds=sum(read(p)['seconds'] for p in (ART/'metrics').glob('*.json')),audit_seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None));print(means.to_string(),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if args.stage=='audit':audit();return
    digest=prepare()
    if args.stage=='prepare':print(digest);return
    engine.OUT=OUT;engine.ART=ART;engine.CAP=2400.;engine.dataset=dataset;engine.normalizer=normalizer;engine.split_ids=split_ids;engine.make_model=make_model
    for f in (0,1):
        for s in (42,43):
            for kind in ('D',):
                guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent']
                engine.train_one(kind,f,s,digest)
    audit()


if __name__=='__main__':
    try:
        with threadpool_limits(2):main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',{**(read(OUT/'STATE.json') if (OUT/'STATE.json').exists() else {}),'status':'failed','pid':None});raise
