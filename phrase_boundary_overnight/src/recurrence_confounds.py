"""Same atomic beat similarity, but average every cross-window correspondence."""
import numpy as np
from .motif_recurrence_features import SCALES,staff_sequences


def correspondence_average(x,w):
    n=len(x)
    if n<w:return np.empty((0,0),float)
    prefix=np.r_[np.zeros((1,x.shape[1])),np.cumsum(x,axis=0)]
    means=(prefix[w:]-prefix[:-w])/w
    # No window-level renormalization: exactly E[mean aligned cosine] under a
    # uniformly random permutation of one window's beat correspondence.
    return np.clip(means@means.T,0,1)


def control_features(events,n):
    streams=[]
    for x in staff_sequences(events,n):
        out=np.zeros((n,12),np.float32)
        for k,w in enumerate(SCALES):
            sim=correspondence_average(x,w);positions=np.arange(w,n-w+1)
            for b in positions:
                js=positions[abs(positions-b)>=2*w]
                if len(js)==0 or b>=n:continue
                f=sim[b,js];back=sim[b-w,js-w]
                out[b,k*4:k*4+4]=[f.max(),back.max(),np.max(f*(1-back)),1.]
        streams.append(out)
    return np.concatenate(streams,axis=1)
