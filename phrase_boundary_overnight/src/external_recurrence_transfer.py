"""Isolated real-label/shuffled-label external pretraining and target finetuning."""
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
from .recurrence_depth_models import make_model as cnn
from .recurrence_transfer_core import transfer,shuffled_labels
from .score_context_study import ROOT,read,write,sha,normalizer
from .models import Normalizer
from .phase3_models import fit_train_normalizer
from .three_round_round2 import split_ids
from .local_context_study import metrics,predictions
from .audit_external_stem_transfer import checked_raw
from .context_inference_audit_v2 import error
from .interstart_decoder import fit_prior
from .recurrence_message_probe import mix_probabilities

OUT=ROOT/'reports/external_recurrence_transfer';ART=ROOT/'artifacts/external_recurrence_transfer'
DATA=ROOT/'artifacts/external_recurrence_data/cache';DREPORT=ROOT/'reports/external_recurrence_data'
COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap'];TARGET_FOLD=None


def external_ids():
    d=pd.read_csv(DREPORT/'piece_audit.csv');d=d[d.status=='usable']
    return {split:sorted(d[d.split==split].piece_id) for split in ('train','validation')}


def external_data(ids,kind):
    result={}
    train=set(external_ids()['train'])
    for pid in ids:
        with np.load(DATA/f'{pid}.npz',allow_pickle=False) as z:
            feat=z['recurrence'].copy();labels=z['labels'].copy();mask=z['label_mask'].copy()
        if kind=='S' and pid in train:labels=shuffled_labels(labels,mask,pid)
        x=np.zeros((1,len(labels),58),np.float32);x[0,:,34:]=feat
        result[pid]=dict(curves=x,labels=labels,label_mask=mask,performance_ids=np.array(['symbolic_score']),pitch_profiles=feat)
    return result


def external_norm(data):
    n=fit_train_normalizer([v['pitch_profiles'] for v in data.values()])
    return Normalizer(np.r_[np.zeros(34,np.float32),n.mean],np.r_[np.ones(34,np.float32),n.std])


def target_data(ids,kind):return base.dataset(ids,'C3')


def make_target(kind,seed):
    assert kind in ('T','S') and TARGET_FOLD in (0,1)
    m=cnn('C3',seed);source='E' if kind=='T' else 'S'
    cp=torch.load(ART/'external/checkpoints'/f'{source}_seed{seed}_fold2'/'best.pt',map_location='cpu',weights_only=False)
    en=Normalizer(cp['mean'],cp['std']);tn=normalizer(target_data(split_ids(TARGET_FOLD)['train'],kind))
    return transfer(m,cp['model'],en,tn)


def prepare():
    assert read(DREPORT/'STATE.json')['status']=='prepared'
    for p in (OUT,OUT/'external',ART/'external/metrics',ART/'external/checkpoints',ART/'metrics',ART/'checkpoints',ART/'decoder'):p.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(base.OUT/'contract.json')['hashes']);hashes.update(read(DREPORT/'source_hashes.json'))
    files=[Path(__file__),ROOT/'src/recurrence_transfer_core.py',ROOT/'tests/test_recurrence_transfer.py',OUT/'PROTOCOL.md',DREPORT/'piece_audit.csv',DREPORT/'STATE.json',ROOT/'src/interstart_decoder.py',ROOT/'src/run_halo_decoder_composition.py',ROOT/'src/recurrence_message_probe.py']
    files+=list((ROOT/'artifacts/recurrence_mean_control/graphs').glob('*.npz'))
    ids=external_ids();assert not set(ids['train'])&set(ids['validation'])
    a=external_data(ids['train'],'E');b=external_data(ids['train'],'S')
    en=external_norm(a);sn=external_norm(b);np.testing.assert_array_equal(en.mean,sn.mean);np.testing.assert_array_equal(en.std,sn.std)
    controls=[]
    for pid in a:
        np.testing.assert_array_equal(a[pid]['curves'],b[pid]['curves']);np.testing.assert_array_equal(a[pid]['label_mask'],b[pid]['label_mask'])
        mask=a[pid]['label_mask']>.5
        assert a[pid]['labels'][mask].sum()==b[pid]['labels'][mask].sum()
        controls.append(dict(piece_id=pid,changed_positions=int((a[pid]['labels']!=b[pid]['labels']).sum())))
    pd.DataFrame(controls).to_csv(OUT/'shuffled_label_audit.csv',index=False)
    for f in (0,1):
        for s in (42,43):
            for suffix in ('.json','_predictions.csv.gz'):files.append(base.ART/'metrics'/f'C3_seed{s}_fold{f}{suffix}')
    for p in files:hashes[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));write(OUT/'external_split.json',ids)
    for p in (OUT,OUT/'external'):
        if not (p/'STATE.json').exists():write(p/'STATE.json',dict(status='ready',completed=[],seconds=0.,pid=None))
    return digest


def audit():
    global TARGET_FOLD
    began=time.monotonic();contract=read(OUT/'contract.json');checks=[];rows=[]
    cps=list((ART/'checkpoints').glob('*/*.pt'))+list((ART/'external/checkpoints').glob('*/*.pt'))
    assert len(cps)==24;hashes={str(p):sha(p) for p in cps}
    for stage,source,kinds,folds in [('external',ART/'external',('E','S'),(2,)),('target',ART,('T','S'),(0,1))]:
        for f in folds:
            TARGET_FOLD=f
            for s in (42,43):
                for kind in kinds:
                    assert time.monotonic()-began<1500
                    ids=external_ids() if stage=='external' else split_ids(f)
                    datafn=external_data if stage=='external' else target_data
                    train=datafn(ids['train'],kind);val=datafn(ids['validation'],kind)
                    norm=external_norm(train) if stage=='external' else normalizer(train)
                    name=f'{kind}_seed{s}_fold{f}';meta=read(source/'metrics'/f'{name}.json')
                    best=torch.load(source/'checkpoints'/name/'best.pt',map_location='cpu',weights_only=False)
                    last=torch.load(source/'checkpoints'/name/'latest.pt',map_location='cpu',weights_only=False)
                    assert best['contract']==last['contract']==meta['contract']==contract['contract'] and last['step']==300
                    assert (best['kind'],best['fold'],best['seed'])==(kind,f,s)
                    assert best['step']==meta['best_step']==max(last['history'],key=lambda h:h['macro_f1_tol1'])['step']
                    np.testing.assert_array_equal(norm.mean,best['mean']);np.testing.assert_array_equal(norm.std,best['std'])
                    raw=checked_raw(pd.read_csv(source/'metrics'/f'{name}_predictions.csv.gz'),val)
                    assert max(abs(metrics(raw,val,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
                    model=cnn('C3',s).cuda();model.load_state_dict(best['model']);replay=predictions(model,val,norm,torch.device('cuda'))
                    err=error(raw,replay);assert err<2e-4
                    assert max(abs(metrics(replay,val,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
                    checks.append(dict(stage=stage,run=name,replay_error=err,**{c:meta[c] for c in COLS}))
                    pd.DataFrame(checks).to_csv(OUT/'run_audit.csv',index=False);print('AUDITED',stage,name,flush=True)
    dec.OUT=OUT/'decoder_state';dec.OUT.mkdir(exist_ok=True);dec.ART=ART/'decoder'
    for f in (0,1):
        ids=split_ids(f);train=target_data(ids['train'],'T');val=target_data(ids['validation'],'T');prior=fit_prior(train)
        for s in (42,43):
            for kind,source in [('C3',base.ART),('T',ART),('S',ART)]:
                name=f'{kind}_seed{s}_fold{f}';meta=read(source/'metrics'/f'{name}.json')
                raw=checked_raw(pd.read_csv(source/'metrics'/f'{name}_predictions.csv.gz'),val)
                assert max(abs(metrics(raw,val,meta['threshold'])[2][c]-meta[c]) for c in COLS)<1e-10
                adjusted={}
                for pid,pp in raw.items():
                    with np.load(ROOT/'artifacts/recurrence_mean_control/graphs'/f'{pid}.npz',allow_pickle=False) as z:g=z['M']
                    adjusted[pid]={q:mix_probabilities(a,g) for q,a in pp.items()}
                for policy,x,strength in [('raw',raw,0.),('M10',adjusted,1.)]:
                    r=dec.evaluate(x,val,meta['threshold'],prior,strength,f'{name}_{policy}',contract['contract'])
                    if policy=='raw':assert max(abs(r[c]-meta[c]) for c in COLS)<1e-10
                    rows.append(dict(kind=kind,seed=s,fold=f,policy=policy,**r))
            pd.DataFrame(rows).to_csv(ART/'summary.csv',index=False)
    frame=pd.DataFrame(rows);assert len(frame)==24
    means=frame.groupby(['kind','policy'])[COLS].mean();means.to_csv(OUT/'means.csv');comp=[]
    for ref in ('C3','S'):
        for policy in ('raw','M10'):
            a=frame[(frame.kind=='T')&(frame.policy==policy)].set_index(['fold','seed'])
            b=frame[(frame.kind==ref)&(frame.policy==policy)].set_index(['fold','seed']);d=a[COLS]-b[COLS]
            comp.append(dict(reference=ref,policy=policy,**d.mean().to_dict(),positive=int((d.macro_f1_tol1>0).sum()),passed=bool(d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>=0 and d.raw_ap.mean()>=0 and (d.macro_f1_tol1>0).sum()>=3)))
    write(OUT/'comparisons.json',comp)
    assert all(sha(p)==h for p,h in hashes.items()) and all(sha(p)==h for p,h in contract['hashes'].items())
    write(OUT/'checkpoint_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='complete',external_training_runs=4,target_training_runs=8,gpu_replays=12,
        checkpoints=24,decoded_cells=24,hashes_unchanged=True,normalizers_rebuilt=True,test_used=False,
        training_seconds=sum(read(p)['seconds'] for source in (ART,ART/'external') for p in (source/'metrics').glob('*.json')),
        audit_seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None));print(means.to_string(),flush=True)


def main():
    global TARGET_FOLD
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('all','prepare','audit'),default='all');args=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if args.stage=='audit':audit();return
    digest=prepare()
    if args.stage=='prepare':print(digest);return
    engine.OUT=OUT/'external';engine.ART=ART/'external';engine.CAP=3600.;engine.dataset=external_data;engine.normalizer=external_norm
    engine.split_ids=lambda f:external_ids();engine.make_model=lambda k,s:cnn('C3',s)
    for s in (42,43):
        for kind in ('E','S'):engine.train_one(kind,2,s,digest)
    engine.OUT=OUT;engine.ART=ART;engine.dataset=target_data;engine.normalizer=normalizer;engine.split_ids=split_ids;engine.make_model=make_target
    for f in (0,1):
        TARGET_FOLD=f
        for s in (42,43):
            for kind in ('T','S'):
                guard=read(ROOT/'reports/research_resource_guard.json');assert guard['observed_used_percent']<guard['post_reset_stop_used_percent']
                total=read(OUT/'STATE.json')['seconds']+read(OUT/'external/STATE.json')['seconds'];assert total<3600
                engine.train_one(kind,f,s,digest)
    audit()


if __name__=='__main__':
    try:
        with threadpool_limits(2):main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',{**(read(OUT/'STATE.json') if (OUT/'STATE.json').exists() else {}),'status':'failed','pid':None});raise
