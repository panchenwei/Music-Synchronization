"""Recurrence-era score-only key profiles; isolated from the older tonal study."""
import numpy as np
from music21.analysis.discrete import KrumhanslSchmuckler
from .score_context_study import pitch_profiles


def key_estimate(events,n):
    hist=np.zeros(12,float)
    for onset,duration,pitch,*_ in events:
        hist[int(pitch)%12]+=max(0.,min(float(n),onset+duration)-max(0.,onset))
    weights=KrumhanslSchmuckler()
    centered=hist-hist.mean();den=np.linalg.norm(centered)
    scores=[]
    for mode in ('major','minor'):
        w=np.asarray(weights.getWeights(mode));w=w-w.mean()
        scores.extend(float(centered@np.roll(w,t)/max(den*np.linalg.norm(w),1e-12)) for t in range(12))
    order=np.argsort(-np.asarray(scores),kind='stable');idx=int(order[0])
    return dict(tonic=idx%12,minor=idx//12,correlation=scores[idx],margin=scores[idx]-scores[int(order[1])])


def features(events,n):
    key=key_estimate(events,n);absolute=pitch_profiles(events,n)
    relative=np.concatenate([np.roll(absolute[:,:12],-key['tonic'],axis=1),np.roll(absolute[:,12:],-key['tonic'],axis=1)],axis=1)
    side=np.broadcast_to([key['minor'],key['margin']],(n,2)).astype(np.float32)
    return np.c_[absolute,side].astype(np.float32),np.c_[relative,side].astype(np.float32),key
