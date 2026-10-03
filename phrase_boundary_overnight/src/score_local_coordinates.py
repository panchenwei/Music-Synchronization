"""Measure-local quarterbeat mapping, retaining events in alternative endings."""
import numpy as np
import pandas as pd
from .data import choose_measure_path,to_float
from .slice_energy_features import start_labels


def positioned_rows(frame,measures,n):
    path,mode,error=choose_measure_path(measures,n)
    if not path:raise ValueError('No compatible traversal')
    if 'mc_onset' not in frame:raise ValueError('Measure-local positions required')
    rows=[]
    for occurrence in path:
        for index,r in frame[frame.mc==occurrence['mc']].iterrows():
            q=occurrence['start_qb']+4*to_float(r.mc_onset)
            if not np.isfinite(q):raise ValueError('Nonfinite measure-local offset')
            rows.append((q,index,occurrence,r))
    return rows,mode,error


def local_events(notes,measures,n):
    rows,_,_=positioned_rows(notes,measures,n);events=[];ties=[]
    for q,_,_,r in rows:
        d=to_float(r.duration_qb)
        if not np.isfinite(d):raise ValueError('Nonfinite duration')
        events.append((q,max(d,0.),int(r.midi),int(r.staff),int(r.voice)))
        tie=r.get('tied',None)
        if pd.isna(tie):tie=None
        if tie not in (None,-1,0,1):raise ValueError('Invalid tie')
        ties.append(None if tie is None else int(tie))
    return events,ties


def local_labels(harmonies,measures,n):
    selected=harmonies[harmonies.phraseend.fillna('').astype(str).str.contains('{',regex=False)]
    rows,mode,error=positioned_rows(selected,measures,n)
    targets=[q for q,_,_,_ in rows]
    labels,mask,mapping=start_labels(targets,n)
    provenance=[dict(source_row=int(i),mc=int(o['mc']),visit=int(o['visit']),target_qb=q) for q,i,o,_ in rows]
    return labels,mask,mapping,provenance,mode,error
