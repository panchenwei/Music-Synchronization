"""Label-only re-baseline; all inputs and model families retained exactly."""
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import ensemble_seed_replication as engine
from .score_context_study import ROOT,read,write,sha,make_model,normalizer
from .score_novelty_study import dataset as old_core_dataset
from .three_round_round2 import split_ids
from .fixed_ensemble_study import aligned_average,raw_from_frame
from .phase2_models import choose_single_threshold
from .local_context_study import metrics

OUT=ROOT/'reports/label_repaired_rebaseline';ART=ROOT/'artifacts/label_repaired_rebaseline'
KINDS=('B','N','R');GRID=engine.GRID;build_model=engine.build_model;predictions=engine.predictions


def dataset(ids,kind):
    assert kind in KINDS
    data=old_core_dataset(ids,'N' if kind=='N' else 'B')
    for pid,item in data.items():
        with np.load(ROOT/'artifacts/coordinate_repair_preview'/f'{pid}.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(item['label_mask'],z['label_mask'])
            item['labels']=z['labels'].copy();item['label_mask']=z['label_mask'].copy()
        if kind=='R':item['piano_roll']=np.load(ROOT/'artifacts/score_roll_study/cache'/f'{pid}.npy',allow_pickle=False)
    return data


def prepare():
    for p in (OUT,ART/'metrics',ART/'checkpoints',ART/'ensemble'):p.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/label_coordinate_bridge/completion_audit.json')['status']=='complete'
    hashes=dict(read(ROOT/'reports/ensemble_seed_replication/contract.json')['hashes']);assert all(sha(p)==h for p,h in hashes.items())
    files=[Path(__file__),OUT/'PROTOCOL.md',ROOT/'src/audit_label_rebaseline.py',ROOT/'src/score_local_coordinates.py',ROOT/'tests/test_score_local_coordinates.py']
    files+=list((ROOT/'artifacts/coordinate_repair_preview').glob('*.npz'))
    for p in files:hashes[str(p)]=sha(p)
    # Raw inputs and normalizers must be identical; only labels change.
    for kind in KINDS:
        ids=split_ids(0)['train'];old=old_core_dataset(ids,'N' if kind=='N' else 'B');new=dataset(ids,kind)
        for pid in old:
            for field in ('curves','pitch_profiles','label_mask'):np.testing.assert_array_equal(old[pid][field],new[pid][field])
        a=normalizer(old);b=normalizer(new);np.testing.assert_array_equal(a.mean,b.mean);np.testing.assert_array_equal(a.std,b.std)
    contract=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    write(OUT/'contract.json',dict(contract=contract,hashes=hashes))
    done=[read(p) for p in (ART/'metrics').glob('*_fold*.json')];assert all(r['contract']==contract for r in done)
    write(OUT/'STATE.json',dict(status='ready',pid=None,completed=[r['run_id'] for r in done],seconds=sum(r['seconds'] for r in done),contract=contract,label_version='measure_local_495'))
    return contract


def report():
    df=pd.DataFrame([{k:v for k,v in read(p).items() if k!='history'} for p in (ART/'metrics').glob('*_fold*.json')]);assert len(df)==12
    df.to_csv(ART/'summary.csv',index=False);cols=['macro_f1_tol0','macro_f1_tol1','raw_ap','macro_precision_tol1','macro_recall_tol1'];df.groupby('kind')[cols].mean().to_csv(OUT/'member_means.csv');rows=[]
    for fold in (0,1):
        data=dataset(split_ids(fold)['validation'],'B')
        for seed in (42,43):
            frames=[pd.read_csv(ART/'metrics'/f'{k}_seed{seed}_fold{fold}_predictions.csv.gz') for k in ('N','R')]
            fused=aligned_average(frames);raw=raw_from_frame(fused,data);threshold,_=choose_single_threshold(raw,data,GRID);perfs,pieces,score=metrics(raw,data,threshold);run=f'NR_seed{seed}_fold{fold}'
            fused.to_csv(ART/'ensemble'/f'{run}_predictions.csv.gz',index=False);perfs.to_csv(ART/'ensemble'/f'{run}_performances.csv',index=False);pieces.to_csv(ART/'ensemble'/f'{run}_pieces.csv',index=False)
            result=dict(run_id=run,kind='NR',fold=fold,seed=seed,label_version='measure_local_495',**score);write(ART/'ensemble'/f'{run}.json',result);rows.append(result)
    nr=pd.DataFrame(rows);nr.to_csv(ART/'ensemble_summary.csv',index=False);nr[cols].mean().to_csv(OUT/'ensemble_means.csv');comparisons=[];a=nr.set_index(['fold','seed'])
    for ref in KINDS:
        b=df[df.kind==ref].set_index(['fold','seed']);d=a[cols[:3]]-b[cols[:3]]
        comparisons.append(dict(reference=ref,f1_delta=d.macro_f1_tol1.mean(),exact_delta=d.macro_f1_tol0.mean(),ap_delta=d.raw_ap.mean(),f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum()),promotion_vs_B=bool(ref=='B' and d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>0 and d.raw_ap.mean()>0 and (d.macro_f1_tol1>0).sum()>=3 and (d.raw_ap>0).sum()>=3)))
    pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False)
    bridge=pd.read_csv(ROOT/'reports/label_coordinate_bridge/paired_results.csv');delta_rows=[]
    for kind,new in (('B',df[df.kind=='B']),('NR',nr)):
        old=bridge[bridge.kind==kind].set_index(['fold','seed']);new=new.set_index(['fold','seed']);result=dict(kind=kind)
        for c in cols[:3]:result[c+'_delta_same_repaired_labels']=float((new[c]-old[c+'_repaired']).mean())
        delta_rows.append(result)
    pd.DataFrame(delta_rows).to_csv(OUT/'retraining_vs_fixed_old_predictions.csv',index=False)
    assert all(sha(p)==h for p,h in read(OUT/'contract.json')['hashes'].items())
    state=read(OUT/'STATE.json');state.update(status='training_complete',pid=None);write(OUT/'STATE.json',state)
    print('MEMBERS\n'+df.groupby('kind')[cols].mean().to_string(),flush=True);print('NR\n'+nr[cols].mean().to_string(),flush=True);print('COMPARISONS\n'+pd.DataFrame(comparisons).to_string(index=False),flush=True);print('RETRAINING EFFECT\n'+pd.DataFrame(delta_rows).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','prepare','report','audit'],default='all');a=p.parse_args();torch.set_num_threads(2)
    from .audit_label_rebaseline import main as audit
    try:
        if a.stage=='audit':audit();return
        if a.stage=='report':report();return
        contract=prepare()
        if a.stage=='prepare':print(contract);return
        engine.OUT=OUT;engine.ART=ART;engine.dataset=dataset
        for fold in (0,1):
            for seed in (42,43):
                for kind in KINDS:engine.train_one(kind,fold,seed,contract)
        report();audit()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        if (OUT/'STATE.json').exists():s=read(OUT/'STATE.json');s.update(status='failed',pid=None);write(OUT/'STATE.json',s)
        raise


if __name__=='__main__':main()
