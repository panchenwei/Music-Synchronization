"""Train-work-balanced inter-start pair potential, not calibrated Bayesian posterior."""
import numpy as np
from .phase2_models import nms_probabilities


def fit_prior(data):
    hist=np.zeros(256,float);works=0;examples=0
    for item in data.values():
        mask=item['label_mask'].astype(bool)
        starts=np.flatnonzero((item['labels']>.5)&mask)
        gaps=[int(b-a) for a,b in zip(starts[:-1],starts[1:]) if mask[a:b+1].all()]
        if not gaps:continue
        if max(gaps)>256:raise ValueError('Training gap exceeds predeclared histogram support')
        hist+=np.bincount(np.asarray(gaps)-1,minlength=256)/len(gaps);works+=1;examples+=len(gaps)
    if not works:raise ValueError('No complete training inter-start intervals')
    hist/=works;mean=float(hist@np.arange(1,257))
    kernel=np.exp(-.5*np.arange(-3,4,dtype=float)**2);kernel/=kernel.sum()
    smooth=np.convolve(hist,kernel,'same');smooth/=smooth.sum()
    return dict(histogram=smooth.tolist(),mean=mean,works=works,intervals=examples,
                support=256,geometric_mixture=.5,smoothing_sigma=1.)


def log_ratio(prior,distance):
    d=np.asarray(distance,dtype=int)
    if (d<1).any():raise ValueError('Distances must be positive')
    rate=1/max(float(prior['mean']),1.00001)
    log_g=np.log(rate)+(d-1)*np.log1p(-rate)
    h=np.asarray(prior['histogram']);local=np.where(d<=len(h),h[np.minimum(d,len(h))-1],0.)
    log_local=np.full(d.shape,-np.inf);np.log(local,out=log_local,where=local>0)
    # Half of the unbounded geometric tail remains: never hard-delete short/long intervals.
    return np.logaddexp(log_g+np.log(.5),log_local+np.log(.5))-log_g


def decode(raw,threshold,prior,strength):
    p=nms_probabilities(np.asarray(raw,dtype=float))
    candidates=np.flatnonzero(p>0)
    if strength==0:return np.flatnonzero(p>=threshold)
    if not len(candidates):return np.array([],dtype=int)
    q=np.clip(p[candidates],1e-7,1-1e-7)
    emission=np.log(q)-np.log1p(-q)-np.log(threshold/(1-threshold))
    values=np.empty(len(q));parent=np.full(len(q),-1,dtype=int)
    for i,b in enumerate(candidates):
        values[i]=emission[i]
        if i:
            prior_score=values[:i]+strength*log_ratio(prior,b-candidates[:i])
            j=int(np.argmax(prior_score))
            if prior_score[j]>0:values[i]+=prior_score[j];parent[i]=j
    i=int(np.argmax(values))
    if values[i]<=0:return np.array([],dtype=int)
    path=[]
    while i>=0:path.append(int(candidates[i]));i=int(parent[i])
    return np.asarray(path[::-1],dtype=int)
