"""Isolated frozen tonal feature round, reusing immutable training implementation."""
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import score_novelty_study as engine
from .score_context_study import make_model,normalizer,sha,read,write
from .three_round_round2 import dataset as base_dataset,split_ids
from .data import discover_dcml_pieces
from .slice_energy_study import DCML
from .slice_energy_features import voiced_events
from .tonal_context_features import duration_histogram,key_correlations,tonic_weights,relative_profiles

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/tonal_context_study';ART=ROOT/'artifacts/tonal_context_study'
KINDS=('K','S')


def dataset(ids,kind):
    assert kind in KINDS
    data=base_dataset(ids,'B')
    for pid,item in data.items():
        x=np.load(ART/'cache'/f'{pid}_{kind}.npy',allow_pickle=False)
        item['pitch_profiles']=x
        item['curves']=np.concatenate([item['curves'],np.broadcast_to(x,(*item['curves'].shape[:2],24))],axis=-1)
    return data


def bind_engine():
    # Only process-local function globals; frozen historical files remain unchanged.
    engine.OUT=OUT;engine.ART=ART;engine.dataset=dataset;engine.CAP=1800.


def prepare():
    for p in (OUT,ART/'cache',ART/'checkpoints',ART/'metrics'):p.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/ensemble_seed_replication/completion_audit.json')['status']=='complete'
    hashes=dict(read(ROOT/'reports/score_novelty_study/contract.json')['hashes'])
    assert all(sha(p)==h for p,h in hashes.items())
    pieces=discover_dcml_pieces(DCML)
    ids=sorted({p for f in (0,1) for s in ('train','validation') for p in split_ids(f)[s]})
    rows=[]
    for pid in ids:
        x=np.load(ROOT/'artifacts/score_context_study/cache'/f'{pid}.npy',allow_pickle=False)
        events=voiced_events(pieces[pid],len(x));h=duration_histogram(events,len(x));scores=key_correlations(h)
        for kind in KINDS:
            dest=ART/'cache'/f'{pid}_{kind}.npy';expected=relative_profiles(x,h,kind)
            if not dest.exists():np.save(dest,expected)
            np.testing.assert_array_equal(np.load(dest,allow_pickle=False),expected)
            hashes[str(dest)]=sha(dest);w=tonic_weights(h,kind)
            rows.append(dict(piece_id=pid,kind=kind,beats=len(x),tonic=int(np.argmax(w)),max_correlation=float(scores.max()),tonic_max_weight=float(w.max()),tonic_entropy=float(-(w[w>0]*np.log(w[w>0])).sum())))
    for p in (Path(__file__),ROOT/'src/tonal_context_features.py',ROOT/'src/audit_tonal_context.py',ROOT/'tests/test_tonal_context.py',OUT/'PROTOCOL.md',ROOT/'artifacts/score_novelty_study/summary.csv'):
        hashes[str(p)]=sha(p)
    # Input and mask equality checked on real training data before fitting anything.
    base=base_dataset(split_ids(0)['train'],'B')
    for kind in KINDS:
        data=dataset(base.keys(),kind)
        for pid in base:
            for field in ('labels','label_mask'):np.testing.assert_array_equal(data[pid][field],base[pid][field])
            np.testing.assert_array_equal(data[pid]['curves'][...,:34],base[pid]['curves'])
    contract=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    write(OUT/'contract.json',dict(contract=contract,hashes=hashes))
    pd.DataFrame(rows).to_csv(OUT/'feature_audit.csv',index=False)
    done=[read(p) for p in (ART/'metrics').glob('*_fold*.json')]
    assert all(r['contract']==contract for r in done)
    write(OUT/'STATE.json',dict(status='ready',completed=[r['run_id'] for r in done],seconds=sum(r['seconds'] for r in done),pid=None,contract=contract))
    return contract


def report():
    df=pd.DataFrame([{k:v for k,v in read(p).items() if k!='history'} for p in (ART/'metrics').glob('*_fold*.json')])
    assert len(df)==8
    df.to_csv(ART/'summary.csv',index=False)
    cols=['macro_f1_tol0','macro_f1_tol1','raw_ap','macro_precision_tol1','macro_recall_tol1','seconds']
    means=df.groupby('kind')[cols].mean();means.to_csv(OUT/'model_means.csv')
    old=pd.read_csv(ROOT/'artifacts/score_context_study/summary.csv');rows=[]
    for control in ('B','P'):
        b=old[old.kind==control].set_index(['fold','seed'])
        for kind in KINDS:
            a=df[df.kind==kind].set_index(['fold','seed']);d=a[cols[:3]]-b[cols[:3]]
            rows.append(dict(candidate=kind,control=control,f1_delta=d.macro_f1_tol1.mean(),exact_delta=d.macro_f1_tol0.mean(),ap_delta=d.raw_ap.mean(),f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),promotion=bool(control=='B' and d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>0 and d.raw_ap.mean()>0 and (d.macro_f1_tol1>0).sum()>=3 and (d.raw_ap>0).sum()>=3)))
    pd.DataFrame(rows).to_csv(OUT/'comparisons.csv',index=False)
    assert all(sha(p)==h for p,h in read(OUT/'contract.json')['hashes'].items())
    state=read(OUT/'STATE.json');state.update(status='training_complete',pid=None);write(OUT/'STATE.json',state)
    print('MEANS\n'+means.to_string(),flush=True);print('COMPARISONS\n'+pd.DataFrame(rows).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','prepare','report'],default='all');a=p.parse_args();torch.set_num_threads(2)
    try:
        if a.stage=='report':report();return
        contract=prepare()
        if a.stage=='prepare':print(contract);return
        bind_engine()
        for fold in (0,1):
            for seed in (42,43):
                for kind in KINDS:engine.train_one(kind,fold,seed,contract)
        report()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        if (OUT/'STATE.json').exists():s=read(OUT/'STATE.json');s.update(status='failed',pid=None);write(OUT/'STATE.json',s)
        raise


if __name__=='__main__':main()
