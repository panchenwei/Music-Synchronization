"""Label-free ordered versus bagged within-work recurrence of score patterns."""
import numpy as np

SCALES=(3,6,12)


def normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def staff_sequences(events,n):
    """Two notated-staff streams, four subticks, onset+occupancy chroma.

    Input events are already tie-merged. Staff is NOT a recovered melody voice.
    """
    result=np.zeros((2,n,4,24),np.float64)
    staffs=sorted(set(float(e[3]) for e in events))
    if not staffs:raise ValueError('No staff events')
    for stream,staff in enumerate((staffs[0],staffs[-1])):
        for q,d,p,s,*_ in events:
            if s!=staff or d<=0:continue
            pc=int(p)%12;t=int(np.floor(q*4+1e-8))
            if 0<=t<n*4:result[stream,t//4,t%4,pc]=1
            for tick in range(max(0,int(np.floor(q*4))),min(n*4,int(np.ceil((q+d)*4)))):
                coverage=max(0.,min((tick+1)/4,q+d)-max(tick/4,q))*4
                result[stream,tick//4,tick%4,12+pc]=max(result[stream,tick//4,tick%4,12+pc],coverage)
    return normalize(result.reshape(2,n,96))


def window_similarity(x,w,ordered):
    """Same-length full windows; ordered mean same-offset cosine, bag cosine.

    Ordered and bag measures have different normalization, reported as part of
    representation change. No label-guided matching or inferred boundaries.
    """
    n=len(x);m=n-w+1
    if m<=0:return np.empty((0,0),float)
    if ordered:
        s=x@x.T;out=np.zeros((m,m),float)
        for offset in range(w):out+=s[offset:offset+m,offset:offset+m]
        return np.clip(out/w,0,1)
    prefix=np.r_[np.zeros((1,x.shape[1])),np.cumsum(x,axis=0)]
    y=normalize(prefix[w:]-prefix[:-w]);return np.clip(y@y.T,0,1)


def stream_features(x,ordered,scales=SCALES):
    n=len(x);out=np.zeros((n,len(scales)*4),np.float64)
    for k,w in enumerate(scales):
        sim=window_similarity(x,w,ordered)
        positions=np.arange(w,n-w+1)
        for b in positions:
            # Both complete [b-w,b+w) spans must be disjoint: no trivial match.
            js=positions[abs(positions-b)>=2*w]
            if len(js)==0 or b>=n:continue
            forward=sim[b,js];backward=sim[b-w,js-w]
            out[b,k*4:k*4+4]=[forward.max(),backward.max(),np.max(forward*(1-backward)),1.]
    return out.astype(np.float32)


def features(events,n,ordered):
    sequences=staff_sequences(events,n)
    return np.concatenate([stream_features(x,ordered) for x in sequences],axis=1)
