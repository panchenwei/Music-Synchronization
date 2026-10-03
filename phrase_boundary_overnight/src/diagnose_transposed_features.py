"""Training-fold-only descriptive feature saturation; no label-based refitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .phrase_end_auxiliary import ROOT, split_ids, read, write, sha


def main():
    out=ROOT/'reports/transposed_recurrence_study/feature_diagnosis';out.mkdir(parents=True,exist_ok=True)
    hashes=read(ROOT/'reports/transposed_recurrence_study/contract.json')['hashes']
    assert all(sha(p)==h for p,h in hashes.items())
    arrays={k:[] for k in ('O','T')};labels=[]
    ids=split_ids(0)['train']
    for pid in ids:
        with np.load(ROOT/'artifacts/coordinate_repair_preview'/f'{pid}.npz',allow_pickle=False) as z:
            y=z['labels'];valid=z['label_mask'].astype(bool)
        with np.load(ROOT/'artifacts/motif_recurrence_study/cache'/f'{pid}.npz',allow_pickle=False) as z:o=z['O']
        t=np.load(ROOT/'artifacts/transposed_recurrence_study/cache'/f'{pid}.npy',allow_pickle=False)
        arrays['O'].append(o[valid]);arrays['T'].append(t[valid]);labels.append(y[valid])
    y=np.concatenate(labels).astype(bool);rows=[]
    for kind,chunks in arrays.items():
        x=np.concatenate(chunks)
        for cue,name in enumerate(('forward','backward','restart')):
            selected=x[:,cue::4];available=x[:,3::4].astype(bool)
            for positive in (False,True):
                v=selected[available & ((y==positive)[:,None])]
                rows.append(dict(kind=kind,cue=name,start_label=positive,count=len(v),mean=float(v.mean()),
                                 fraction_above_095=float((v>.95).mean()),std=float(v.std())))
    df=pd.DataFrame(rows);df.to_csv(out/'training_cue_distributions.csv',index=False)
    assert all(sha(p)==h for p,h in hashes.items())
    write(out/'audit.json',dict(status='complete',train_fold=0,pieces=ids,validation_predictions_used=False,
          labels_used_only_for_descriptive_grouping=True,sources_unchanged=True,
          interpretation='Beat/scale/staff pooled descriptions, not independent observations or a causal explanation.'))
    print(df.to_string(index=False))


if __name__=='__main__':main()
