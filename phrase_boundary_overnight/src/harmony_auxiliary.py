"""Train-only conservative harmonic degree supervision; never an inference input."""
import hashlib
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch import nn
from .data import to_float
from .score_local_coordinates import positioned_rows
from .recurrence_depth_models import make_model as cnn
from .phase7_models import CurvePieceBalancedSampler

DEGREES=('I','II','III','IV','V','VI','VII')


def degree(row):
    numeral=str(row.get('numeral','')).strip().upper()
    relative=row.get('relativeroot')
    if not pd.isna(relative) and str(relative).strip():return -1
    if numeral not in DEGREES:return -1
    if any(pd.isna(row.get(k)) or not str(row.get(k)).strip() for k in ('globalkey','localkey')):return -1
    return DEGREES.index(numeral)


def targets(harmonies,measures,n):
    """Only whole [b,b+1) cells with unambiguous same-degree coverage.

    Durations stop at the measure end and next annotation, even if that next
    annotation is invalid. No carrying a harmony across a repeat or a missing bar.
    Quality/inversion are deliberately not classes; applied/altered roots excluded.
    """
    rows,mode,error=positioned_rows(harmonies,measures,n)
    rows=sorted(rows,key=lambda r:(r[0],r[1]));cover=np.zeros((n,7),np.float64);provenance=[]
    multiplicity=Counter(round(row[0],8) for row in rows)
    for i,(q,index,o,r) in enumerate(rows):
        d=to_float(r.get('duration_qb'));label=degree(r) if multiplicity[round(q,8)]==1 else -1
        end=min(q+d,float(o['start_qb'])+float(o['duration_qb']),float(n)) if np.isfinite(d) and d>0 else q
        if i+1<len(rows):end=min(end,rows[i+1][0])
        provenance.append(dict(source_row=int(index),mc=int(o['mc']),visit=int(o['visit']),start=q,end=end,degree=label))
        if label<0 or end<=q:continue
        for b in range(max(0,int(np.floor(q))),min(n,int(np.ceil(end)))):
            cover[b,label]+=max(0.,min(end,b+1)-max(q,b))
    full=(np.abs(cover.sum(1)-1)<1e-5)&(cover.max(1)>1-1e-5)
    y=np.argmax(cover,axis=1).astype(np.int64);y[~full]=0
    return y,full.astype(np.float32),provenance,mode,error


def shuffled(y,mask,pid,seed):
    result=y.copy();idx=np.flatnonzero(mask>.5)
    salt=int(hashlib.sha256(f'{pid}:{seed}:harmony-control'.encode()).hexdigest()[:16],16)
    result[idx]=np.random.default_rng(salt).permutation(y[idx])
    assert np.array_equal(np.bincount(result[idx],minlength=7),np.bincount(y[idx],minlength=7))
    return result


class HarmonyCNN(nn.Module):
    def __init__(self,seed):
        super().__init__();self.core=cnn('C3',seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+20260914);self.harmony_head=nn.Linear(32,7)

    def forward(self,x,padding_mask=None,both=False):
        start,h=self.core(x,padding_mask=padding_mask,return_hidden=True)
        if not both:return start
        extra=self.harmony_head(h)
        if padding_mask is not None:extra=extra.masked_fill(padding_mask[...,None],0)
        return start,extra


def make_model(kind,seed):
    assert kind in ('G','D','Z')
    return HarmonyCNN(seed)


class HarmonySampler:
    def __init__(self,data,norm,seed,kind):
        self.start=CurvePieceBalancedSampler(data,norm,64,32,seed)
        aux={p:{**v,'labels':shuffled(v['harmony_labels'],v['harmony_mask'],p,seed) if kind=='D' else v['harmony_labels'],
                   'label_mask':v['harmony_mask']} for p,v in data.items()}
        self.harmony=CurvePieceBalancedSampler(aux,norm,64,32,seed)

    def state(self):return dict(start=self.start.state(),harmony=self.harmony.state())
    def load_state(self,state):self.start.load_state(state['start']);self.harmony.load_state(state['harmony'])
    def batch(self):
        a=self.start.batch();b=self.harmony.batch()
        assert torch.equal(a[0],b[0]) and torch.equal(a[3],b[3])
        return (*a,b[1].long(),b[2])
