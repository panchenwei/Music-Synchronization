"""Single-change head-reset control after failed external transfer."""
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
from . import external_recurrence_transfer as old
from . import run_halo_decoder_composition as dec
from .recurrence_depth_models import make_model as cnn
from .transfer_head_reset import transfer_without_head
from .score_context_study import ROOT,read,write,sha,normalizer
from .models import Normalizer
from .three_round_round2 import split_ids
from .local_context_study import metrics,predictions
from .audit_external_stem_transfer import checked_raw
from .context_inference_audit_v2 import error
from .interstart_decoder import fit_prior
from .recurrence_message_probe import mix_probabilities

OUT=ROOT/'reports/external_head_reset';ART=ROOT/'artifacts/external_head_reset'
COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap'];TARGET_FOLD=None


def dataset(ids,kind):
    assert kind in ('H','Q')
    return base.dataset(ids,'C3')


def make_model(kind,seed):
    assert kind in ('H','Q') and TARGET_FOLD in (0,1)
    m=cnn('C3',seed);source='E' if kind=='H' else 'S'
    cp=torch.load(old.ART/'external/checkpoints'/f'{source}_seed{seed}_fold2/best.pt',map_location='cpu',weights_only=False)
    en=Normalizer(cp['mean'],cp['std']);tn=normalizer(dataset(split_ids(TARGET_FOLD)['train'],kind))
    return transfer_without_head(m,cp['model'],en,tn)


def prepare():
    global TARGET_FOLD
    assert read(old.OUT/'completion_audit.json')['status']=='complete'
    for p in (OUT,ART/'metrics',ART/'checkpoints',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(old.OUT/'contract.json')['hashes'])
    files=[Path(__file__),ROOT/'src/transfer_head_reset.py',ROOT/'tests/test_transfer_head_reset.py',OUT/'PROTOCOL.md',old.OUT/'completion_audit.json']
    files+=list((old.ART/'external/checkpoints').glob('*/best.pt'))
    for f in (0,1):
        TARGET_FOLD=f;old.TARGET_FOLD=f
        for s in (42,43):
            for kind,ref in [('H','T'),('Q','S')]:
                a=old.make_target(ref,s);b=make_model(kind,s)
                initial=cnn('C3',s).output.state_dict()
                for k,v in a.state_dict().items():
                    if not k.startswith('output.'):torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
                for k,v in initial.items():torch.testing.assert_close(v,b.output.state_dict()[k],atol=0,rtol=0)
            for kind,source in [('C3',base.ART),('T',old.ART),('S',old.ART)]:
                for suffix in ('.json','_predictions.csv.gz'):files.append(source/'metrics'/f'{kind}_seed{s}_fold{f}{suffix}')
    for p in files:hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes))
    write(OUT/'intervention_audit.json',dict(only_output_changed=True,comparisons=8))
    if not (OUT/'STATE.json').exists():write(OUT/'STATE.json',dict(status='ready',completed=[],seconds=0.,pid=None))
    return digest


def audit():
    began=time.monotonic();contract=read(OUT/'contract.json');checks=[];rows=[]
    hashes={str(p):sha(p) for p in (ART/'checkpoints').glob('*/*.pt')};assert len(hashes)==16
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder'
    for f in (0,1):
        ids=split_ids(f);train=dataset(ids['train'],'H');val=dataset(ids['validation'],'H');norm=normalizer(train);prior=fit_prior(train)
        for s in (42,43):
            for kind,source in [('C3',base.ART),('T',old.ART),('S',old.ART),('H',ART),('Q',ART)]:
                assert time.monotonic()-began<1500
                name=f'{kind}_seed{s}_fold{f}';meta=read(source/'metrics'/f'{name}.json')
                raw=checked_raw(pd.read_csv(source/'metrics'/f'{name}_predictions.csv.gz'),val)
                assert max(abs(metrics(raw,val,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
                if kind in ('H','Q'):
                    best=torch.load(source/'checkpoints'/name/'best.pt',map_location='cpu',weights_only=False)
                    last=torch.load(source/'checkpoints'/name/'latest.pt',map_location='cpu',weights_only=False)
                    assert best['contract']==last['contract']==meta['contract']==contract['contract'] and last['step']==300
                    assert (best['kind'],best['fold'],best['seed'])==(kind,f,s)
                    assert best['step']==meta['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
                    np.testing.assert_array_equal(norm.mean,best['mean']);np.testing.assert_array_equal(norm.std,best['std'])
                    m=cnn('C3',s).cuda();m.load_state_dict(best['model']);replay=predictions(m,val,norm,torch.device('cuda'))
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
    df=pd.DataFrame(rows);assert len(df)==40 and len(checks)==8
    means=df.groupby(['kind','policy'])[COLS].mean();means.to_csv(OUT/'means.csv');comp=[]
    for ref in ('C3','T','Q'):
        for policy in ('raw','M10'):
            a=df[(df.kind=='H')&(df.policy==policy)].set_index(['fold','seed']);b=df[(df.kind==ref)&(df.policy==policy)].set_index(['fold','seed']);d=a[COLS]-b[COLS]
            comp.append(dict(reference=ref,policy=policy,**d.mean().to_dict(),positive=int((d.macro_f1_tol1>0).sum()),passed=bool(d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>=0 and d.raw_ap.mean()>=0 and (d.macro_f1_tol1>0).sum()>=3)))
    write(OUT/'comparisons.json',comp)
    assert all(sha(p)==h for p,h in hashes.items()) and all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'checkpoint_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='complete',new_training_runs=8,gpu_replays=8,checkpoints=16,decoded_cells=40,
        old_metrics_recomputed=12,hashes_unchanged=True,test_used=False,training_seconds=sum(read(p)['seconds'] for p in (ART/'metrics').glob('*.json')),audit_seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None));print(means.to_string(),flush=True)


def main():
    global TARGET_FOLD
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if args.stage=='audit':audit();return
    digest=prepare()
    if args.stage=='prepare':print(digest);return
    engine.OUT=OUT;engine.ART=ART;engine.CAP=3600.;engine.dataset=dataset;engine.normalizer=normalizer;engine.split_ids=split_ids;engine.make_model=make_model
    for f in (0,1):
        TARGET_FOLD=f
        for s in (42,43):
            for kind in ('H','Q'):
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
