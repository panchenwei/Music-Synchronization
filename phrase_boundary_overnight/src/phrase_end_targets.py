"""Independent structural-end targets; never drop the first end as a first start."""
import numpy as np
import pandas as pd
from .data import _nearest_beat
from .score_local_coordinates import positioned_rows


def end_targets(starts,ends,n):
    starts=sorted(set(float(q) for q in starts if np.isfinite(q) and 0<=q<n))
    ends=sorted(set(float(q) for q in ends if np.isfinite(q) and 0<=q<n))
    y=np.zeros(n,np.float32);mask=np.zeros(n,np.float32);rows=[]
    if not starts or not ends:return y,mask,rows
    lo,hi=starts[0],ends[-1]
    mask[(np.arange(n)>lo)&(np.arange(n)<=hi)]=1
    for q in ends:
        status,index,distance=_nearest_beat(q,n,.5)
        covered=lo<=q<=hi
        if covered and index is not None:y[index]=1;mask[index]=1
        rows.append(dict(target_qb=q,beat_index=index,status=status,distance=distance,covered=covered))
    for r in rows:
        if r['status']=='ambiguous_tie':
            for i in (int(np.floor(r['target_qb'])),int(np.ceil(r['target_qb']))):
                if 0<=i<n:mask[i]=0
    return y,mask,rows


def from_source(harmonies,measures,n):
    rows,_,_=positioned_rows(harmonies,measures,n);starts=[];ends=[]
    for q,_,_,r in rows:
        marker='' if pd.isna(r.get('phraseend')) else str(r.phraseend)
        if '{' in marker:starts.append(q)
        if '}' in marker:ends.append(q)
    return end_targets(starts,ends,n)
