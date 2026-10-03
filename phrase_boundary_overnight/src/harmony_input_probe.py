"""Fixed linear diagnostic: do the current inputs expose harmonic degree?"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from threadpoolctl import threadpool_limits
from .score_context_study import ROOT,read,write,sha,pitch_profiles
from .harmony_auxiliary_study import dataset,OUT as HOUT
from .data import discover_dcml_pieces
from .slice_energy_study import DCML
from .three_round_round2 import split_ids
from .score_local_coordinates import local_events

OUT=ROOT/'reports/harmony_input_probe';ART=ROOT/'artifacts/harmony_input_probe'


def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    hashes=dict(read(HOUT/'contract.json')['hashes']);sources=discover_dcml_pieces(DCML)
    for p in (Path(__file__),OUT/'PROTOCOL.md',ROOT/'src/score_local_coordinates.py'):hashes[str(p)]=sha(p)
    data=dataset(sorted({p for f in (0,1) for split in ('train','validation') for p in split_ids(f)[split]}),'G');profiles={}
    for pid,v in data.items():
        source=sources[pid];events,_=local_events(pd.read_csv(source.notes_path,sep='\t'),pd.read_csv(source.measures_path,sep='\t'),len(v['labels']))
        profiles[pid]=pitch_profiles(events,len(v['labels']));path=ART/f'{pid}.npy'
        if path.exists():np.testing.assert_array_equal(np.load(path,allow_pickle=False),profiles[pid])
        else:np.save(path,profiles[pid])
        for p in (source.notes_path,source.measures_path,path):hashes[str(p)]=sha(p)
    rows=[];pred=[]
    for f in (0,1):
        ids=split_ids(f)
        for kind in ('B','P','S'):
            sets={}
            for split in ('train','validation'):
                xs=[];ys=[];pids=[]
                for pid in ids[split]:
                    v=data[pid];mask=v['harmony_mask']>.5;x=v['curves'].mean(0);extra=profiles[pid].copy()
                    if kind=='S':extra=extra[np.random.default_rng(int(sha(ART/f'{pid}.npy')[:8],16)).permutation(len(extra))]
                    if kind!='B':x=np.c_[x,extra]
                    xs.append(x[mask]);ys.append(v['harmony_labels'][mask]);pids.extend([pid]*int(mask.sum()))
                sets[split]=(np.concatenate(xs),np.concatenate(ys),pids)
            x,y,_=sets['train'];model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=42))
            model.fit(x,y);majority=int(np.bincount(y,minlength=7).argmax())
            for split,(x,y,pids) in sets.items():
                p=model.predict(x)
                for pid in sorted(set(pids)):
                    m=np.asarray(pids)==pid;a=y[m];b=p[m]
                    rows.append(dict(fold=f,kind=kind,split=split,piece_id=pid,n=int(m.sum()),accuracy=accuracy_score(a,b),
                        macro_f1=f1_score(a,b,average='macro',zero_division=0),majority_accuracy=float((a==majority).mean())))
                pred.extend(dict(fold=f,kind=kind,split=split,piece_id=pid,true=int(a),predicted=int(b)) for pid,a,b in zip(pids,y,p))
            assert time.monotonic()-start<300
    df=pd.DataFrame(rows);df.to_csv(OUT/'piece_metrics.csv',index=False);pd.DataFrame(pred).to_csv(OUT/'predictions.csv.gz',index=False)
    df.groupby(['kind','split'])[['accuracy','macro_f1','majority_accuracy']].mean().to_csv(OUT/'means.csv')
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='complete',linear_fits=6,boundary_training_runs=0,hashes_unchanged=True,seconds=time.monotonic()-start,
        caution='Two development folds; linear diagnostic only. Does not prove all nonlinear models can or cannot recover harmony. No new boundary F1.'))
    print(df.groupby(['kind','split'])[['accuracy','macro_f1','majority_accuracy']].mean().to_string(),flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
