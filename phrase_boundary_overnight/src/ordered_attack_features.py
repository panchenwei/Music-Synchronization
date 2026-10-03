"""Ordered attack slots in two notated voices, not recovered perceptual melody."""
import numpy as np
from .relation_interval_features import reference_voices

def features(events,n):
    events=[tuple(e) for e in events if e[1]>0]
    voices=reference_voices(events);out=np.zeros((n,2,4,7),np.float32);overflow=np.zeros((n,2),np.float32)
    for stream,voice in enumerate(voices):
        groups={}
        for e in events:
            if e[3:5]==voice:groups.setdefault(round(float(e[0]),9),[]).append(e)
        previous=None;counts=np.zeros(n,int)
        for onset,notes in sorted(groups.items()):
            choose=max if stream==0 else min;pitch=choose(e[2] for e in notes)
            dur=max(e[1] for e in notes if e[2]==pitch)
            b=int(np.floor(onset+1e-9));validprev=previous is not None
            interval=0 if not validprev else np.clip(pitch-previous[1],-48,48)/48
            ioi=0 if not validprev else min(max(onset-previous[0],0),16)
            rest=0 if not validprev else min(max(onset-previous[0]-previous[2],0),16)
            vector=[interval,np.log1p(ioi)/np.log(17),np.log1p(min(dur,16))/np.log(17),rest/16,onset-b,1.,float(validprev)]
            if 0<=b<n:
                slot=counts[b];counts[b]+=1
                if slot<4:out[b,stream,slot]=vector
                else:overflow[b,stream]=1
            previous=(onset,pitch,dur)
    bag=out.copy()
    for b in range(n):
        for s in range(2):
            mask=out[b,s,:,5]>0
            if mask.any():bag[b,s,mask]=out[b,s,mask].mean(0)
    ordered=np.c_[out.reshape(n,56),overflow];unordered=np.c_[bag.reshape(n,56),overflow]
    assert np.isfinite(ordered).all() and abs(ordered).max()<=1+1e-6
    return ordered.astype(np.float32),unordered.astype(np.float32)
