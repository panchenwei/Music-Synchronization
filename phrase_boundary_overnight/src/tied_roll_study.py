"""Isolated tie-aware input correction with historical training and audit replay."""
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import score_roll_study as engine
from . import audit_score_roll as verifier
from .score_context_study import read,write,sha,normalizer
from .score_novelty_study import dataset as control_dataset
from .three_round_round2 import split_ids
from .data import discover_dcml_pieces
from .slice_energy_study import DCML
from .score_tied_events import tied_rows,tied_piano_roll

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/tied_roll_study';ART=ROOT/'artifacts/tied_roll_study';KINDS=('L','R')


def dataset(ids,kind):
    assert kind in KINDS
    data=control_dataset(ids,'B')
    for pid,item in data.items():item['piano_roll']=np.load(ART/'cache'/f'{pid}.npy',allow_pickle=False)
    return data


def bind():
    engine.OUT=OUT;engine.ART=ART;engine.dataset=dataset
    verifier.OUT=OUT;verifier.ART=ART;verifier.dataset=dataset


def prepare():
    for p in (OUT,ART/'cache',ART/'checkpoints',ART/'metrics'):p.mkdir(parents=True,exist_ok=True)
    assert read(ROOT/'reports/tonal_context_study/completion_audit.json')['status']=='complete'
    hashes=dict(read(ROOT/'reports/score_roll_study/contract.json')['hashes']);assert all(sha(p)==h for p,h in hashes.items())
    pieces=discover_dcml_pieces(DCML);rows=[]
    for pid in sorted({p for f in (0,1) for s in ('train','validation') for p in split_ids(f)[s]}):
        old=np.load(ROOT/'artifacts/score_roll_study/cache'/f'{pid}.npy',allow_pickle=False)
        events,ties=tied_rows(pieces[pid],len(old));fixed,counts=tied_piano_roll(events,ties,len(old))
        np.testing.assert_array_equal(fixed[:,0],old[:,0]);assert (fixed[:,1]<=old[:,1]).all()
        dest=ART/'cache'/f'{pid}.npy'
        if not dest.exists():np.save(dest,fixed)
        np.testing.assert_array_equal(np.load(dest,allow_pickle=False),fixed);hashes[str(dest)]=sha(dest)
        rows.append(dict(piece_id=pid,beats=len(old),**counts,old_onset_cells=int(old[:,1].sum()),new_onset_cells=int(fixed[:,1].sum()),removed_onset_cells=int((old[:,1]-fixed[:,1]).sum()),changed_beats=int((old[:,1]!=fixed[:,1]).any(axis=(1,2)).sum())))
    for p in (Path(__file__),ROOT/'src/score_tied_events.py',ROOT/'src/audit_score_roll.py',ROOT/'tests/test_score_tied_events.py',OUT/'PROTOCOL.md',ROOT/'artifacts/score_roll_study/summary.csv'):
        hashes[str(p)]=sha(p)
    # The non-roll fields remain literally identical to the matched control.
    base=control_dataset(split_ids(0)['train'],'B');changed=dataset(base.keys(),'R')
    for pid in base:
        for field in ('curves','labels','label_mask','pitch_profiles'):np.testing.assert_array_equal(base[pid][field],changed[pid][field])
    contract=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')['contract']==contract
    write(OUT/'contract.json',dict(contract=contract,hashes=hashes));pd.DataFrame(rows).to_csv(OUT/'feature_audit.csv',index=False)
    done=[read(p) for p in (ART/'metrics').glob('*_fold*.json')];assert all(r['contract']==contract for r in done)
    write(OUT/'STATE.json',dict(status='ready',completed=[r['run_id'] for r in done],seconds=sum(r['seconds'] for r in done),pid=None,contract=contract))
    return contract


def report():
    engine.report()
    new=pd.read_csv(ART/'summary.csv');old=pd.read_csv(ROOT/'artifacts/score_roll_study/summary.csv');cols=['macro_f1_tol0','macro_f1_tol1','raw_ap'];rows=[]
    for kind in KINDS:
        a=new[new.kind==kind].set_index(['fold','seed']);b=old[old.kind==kind].set_index(['fold','seed']);d=a[cols]-b[cols]
        d.to_csv(OUT/f'{kind}_minus_notehead_roll_cells.csv')
        rows.append(dict(kind=kind,f1_delta=d.macro_f1_tol1.mean(),exact_delta=d.macro_f1_tol0.mean(),ap_delta=d.raw_ap.mean(),f1_positive=int((d.macro_f1_tol1>0).sum()),ap_positive=int((d.raw_ap>0).sum())))
    pd.DataFrame(rows).to_csv(OUT/'input_correction_comparisons.csv',index=False)
    print('INPUT CORRECTION\n'+pd.DataFrame(rows).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','prepare','report','audit'],default='all');a=p.parse_args();torch.set_num_threads(2);bind()
    try:
        if a.stage=='audit':verifier.main();return
        if a.stage=='report':report();return
        contract=prepare()
        if a.stage=='prepare':print(contract);return
        for fold in (0,1):
            for seed in (42,43):
                for kind in KINDS:engine.train_one(kind,fold,seed,contract)
        report();verifier.main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'failures.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()))+'\n')
        if (OUT/'STATE.json').exists():s=read(OUT/'STATE.json');s.update(status='failed',pid=None);write(OUT/'STATE.json',s)
        raise


if __name__=='__main__':main()
