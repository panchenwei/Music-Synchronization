"""Frozen neural models x fixed decoder, plus explicit score-agnostic prior control."""
import hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from . import run_context_halo_study as halo
from .score_context_study import ROOT,read,write,sha
from .three_round_round2 import split_ids
from .audit_external_stem_transfer import checked_raw
from .evaluation import evaluate_piece
from .interstart_decoder import fit_prior,decode

OUT=ROOT/'reports/halo_decoder_composition';ART=ROOT/'artifacts/halo_decoder_composition'
KINDS=('C3','A2','HC3','HA2');COLS=['macro_f1_tol1','macro_f1_tol0','raw_ap']


def source(kind):return halo.prior.ART if kind=='C3' else (halo.att.ART if kind=='A2' else halo.ART)


def evaluate(raw,val,threshold,prior,strength,name,contract):
    target=ART/f'{name}.json'
    if target.exists():
        row=read(target);assert row['contract']==contract and sha(ART/f'{name}_positions.csv.gz')==row['positions_hash'];return row
    began=time.monotonic();rows=[];positions=[]
    write(OUT/'STATE.json',dict(status='running',pid=os.getpid(),cell=name,contract=contract))
    for pid,perfs in raw.items():
        mask=val[pid]['label_mask'].astype(bool);labels=val[pid]['labels']
        for perf,probs in perfs.items():
            indices=decode(probs,threshold,prior,strength)
            binary=np.zeros(len(probs));binary[indices]=1
            score=evaluate_piece(pid,binary,labels,mask,.5)
            ap=average_precision_score(labels[mask],probs[mask]) if labels[mask].sum() else np.nan
            rows.append(dict(**score,performance_id=perf,raw_ap=float(ap)))
            positions.extend((pid,perf,int(b)) for b in indices)
    ppath=ART/f'{name}_positions.csv.gz';pd.DataFrame(positions,columns=['piece_id','performance_id','beat']).to_csv(ppath,index=False)
    pf=pd.DataFrame(rows);pf.to_csv(ART/f'{name}_performances.csv',index=False)
    pieces=pf.groupby('piece_id').mean(numeric_only=True);pieces.to_csv(ART/f'{name}_pieces.csv')
    cached=pd.read_csv(ppath);groups={(p,q):v.beat.to_numpy(int) for (p,q),v in cached.groupby(['piece_id','performance_id'])}
    for row in rows:
        pid=row['piece_id'];binary=np.zeros(len(val[pid]['labels']));binary[groups.get((pid,row['performance_id']),np.array([],int))]=1
        score=evaluate_piece(pid,binary,val[pid]['labels'],val[pid]['label_mask'],.5)
        assert all(score[k]==row[k] for k in ('tp_tol0','fp_tol0','fn_tol0','tp_tol1','fp_tol1','fn_tol1'))
    summary={f'macro_{key}':float(pieces[key].mean()) for key in ('f1_tol1','f1_tol0','precision_tol1','recall_tol1')}
    row=dict(contract=contract,threshold=threshold,strength=strength,seconds=time.monotonic()-began,raw_ap=float(pieces.raw_ap.mean()),positions_hash=sha(ppath),**summary)
    write(target,row);print(name,summary,flush=True);return row


def main():
    began=time.monotonic()
    for path in (OUT,ART):path.mkdir(parents=True,exist_ok=True)
    assert read(halo.OUT/'completion_audit.json')['status']=='complete'
    hashes=dict(read(halo.OUT/'contract.json')['hashes'])
    reference=ROOT/'artifacts/interstart_decoder_study/summary.csv';hashes[str(reference)]=sha(reference)
    for kind in KINDS:
        for fold in (0,1):
            for seed in (42,43):
                for suffix in ('.json','_predictions.csv.gz'):
                    path=source(kind)/'metrics'/f'{kind}_seed{seed}_fold{fold}{suffix}';hashes[str(path)]=sha(path)
    for path in (Path(__file__),ROOT/'src/interstart_decoder.py',OUT/'PROTOCOL.md'):hashes[str(path)]=sha(path)
    assert all(sha(p)==h for p,h in hashes.items())
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==digest
    write(OUT/'contract.json',dict(contract=digest,hashes=hashes));results=[]
    for fold in (0,1):
        ids=split_ids(fold);prior=fit_prior(halo.dataset(ids['train'],'C3'));val=halo.dataset(ids['validation'],'C3')
        write(OUT/f'prior_fold{fold}.json',dict(prior=prior,train_ids=ids['train']))
        for kind in KINDS:
            for seed in (42,43):
                run=f'{kind}_seed{seed}_fold{fold}';meta=read(source(kind)/'metrics'/f'{run}.json')
                raw=checked_raw(pd.read_csv(source(kind)/'metrics'/f'{run}_predictions.csv.gz'),val)
                for strength in (0.,.5):
                    assert time.monotonic()-began<1800
                    row=evaluate(raw,val,meta['threshold'],prior,strength,f'{run}_lambda{strength:g}',digest)
                    if strength==0:assert max(abs(row[c]-meta[c]) for c in COLS)<1e-10
                    results.append(dict(kind=kind,fold=fold,seed=seed,**row))
        # One deterministic control per fold, NOT two independent random seeds.
        raw={pid:{'prior_only':np.full(len(v['labels']),.5)} for pid,v in val.items()}
        row=evaluate(raw,val,.5,prior,.5,f'PRIOR_ONLY_fold{fold}',digest)
        results.append(dict(kind='PRIOR_ONLY',fold=fold,seed=-1,**row))
    frame=pd.DataFrame(results);assert len(frame)==34
    frame.to_csv(ART/'summary.csv',index=False);frame.groupby(['kind','strength'])[COLS].mean().to_csv(OUT/'means.csv')
    old=pd.read_csv(reference);a=frame[(frame.kind=='C3')&(frame.strength==.5)].set_index(['fold','seed']);b=old[old.strength==.5].set_index(['fold','seed'])
    assert abs(a[COLS]-b[COLS]).to_numpy().max()<1e-10
    comparisons=[]
    for candidate,control in (('HC3','C3'),('HA2','A2'),('A2','C3'),('HA2','C3')):
        a=frame[(frame.kind==candidate)&(frame.strength==.5)].set_index(['fold','seed']);b=frame[(frame.kind==control)&(frame.strength==.5)].set_index(['fold','seed']);d=a[COLS]-b[COLS]
        d.to_csv(OUT/f'{candidate}_minus_{control}.csv')
        passed=d.macro_f1_tol1.mean()>=.015 and d.macro_f1_tol0.mean()>=0 and (d.macro_f1_tol1>0).sum()>=3
        comparisons.append(dict(candidate=candidate,control=control,**d.mean().to_dict(),f1_positive=int((d.macro_f1_tol1>0).sum()),passed=bool(passed)))
    write(OUT/'comparisons.json',comparisons)
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'completion_audit.json',dict(status='complete',neural_decode_cells=32,prior_only_cells=2,
        new_training_runs=0,original_neural_metrics_reproduced=True,c3_soft_decoder_reproduced=True,
        source_hashes_unchanged=True,saved_event_counts_recomputed=True,priors_train_only=True,test_used=False,elapsed_seconds=time.monotonic()-began))
    write(OUT/'STATE.json',dict(status='complete',pid=None,contract=digest))


if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        write(OUT/'STATE.json',dict(status='failed',pid=None));raise
