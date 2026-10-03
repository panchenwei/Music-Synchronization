"""Label-free, transposition-invariant predictive information from ordered voices."""
from collections import Counter,defaultdict
import numpy as np

IOI=np.array([.125,.25,1/3,.5,2/3,.75,1,1.5,2,3,4,6,8])


def streams(events):
    events=np.asarray(events,float)
    if events.size==0:return [[],[]]
    assert events.ndim==2 and events.shape[1]>=5
    # merged_events appends a sixth tie-origin flag; stream identity uses the first five fields.
    valid=events[(events[:,1]>0)&(events[:,0]>=0),:5]
    voices=sorted(set(tuple(map(int,r[3:5])) for r in valid))
    if not voices:return [[],[]]
    top=(1,1) if (1,1) in voices else voices[0];staff=max(s for s,v in voices)
    bass=max((v for v in voices if v[0]==staff),key=lambda k:sum(tuple(map(int,r[3:5]))==k for r in valid))
    result=[]
    for voice,lowest in [(top,False),(bass,True)]:
        groups=defaultdict(list)
        for q,d,p,s,v in valid:
            if (int(s),int(v))==voice:groups[round(q,8)].append(int(p))
        notes=[(q,(min if lowest else max)(p)) for q,p in sorted(groups.items())]
        seq=[]
        for (before,p0),(q,p1) in zip(notes[:-1],notes[1:]):
            if q<=before:continue
            pitch=int(np.clip(p1-p0,-24,24))+24
            rhythm=int(np.argmin(abs(np.log2(IOI)-np.log2(q-before))))
            seq.append((q,pitch,rhythm))
        result.append(seq)
    return result


class Predictor:
    def __init__(self,k,context=3):self.k=k;self.context=context;self.counts=defaultdict(lambda:np.zeros(k,np.float64))
    def fit(self,sequences):
        for sequence in sequences:
            history=[]
            for token in sequence:
                for depth in range(min(len(history),self.context)+1):
                    key=tuple(history[-depth:]) if depth else ();self.counts[key][int(token)]+=1
                history.append(int(token))
        return self
    def distribution(self,history,unigram=False):
        c=self.counts.get((),np.zeros(self.k));p=(c+1)/(c.sum()+self.k)
        if unigram:return p
        for depth in range(1,min(len(history),self.context)+1):
            c=self.counts.get(tuple(history[-depth:]))
            if c is not None:p=(c+10*p)/(c.sum()+10)
        assert np.isfinite(p).all() and abs(p.sum()-1)<1e-10 and (p>0).all()
        return p


def fit_predictors(voices_by_piece):
    return [[Predictor(k).fit([[r[field] for r in voices[v]] for voices in voices_by_piece]) for field,k in [(1,49),(2,len(IOI))]] for v in range(2)]


def features(voices,predictors,n,unigram=False):
    x=np.zeros((n,14),np.float32);nll=[]
    for voice,seq in enumerate(voices):
        histories=[[],[]]
        for q,pitch,rhythm in seq:
            b=int(np.floor(q+1e-7))
            if not 0<=b<n:continue
            values=[]
            for j,token in enumerate((pitch,rhythm)):
                model=predictors[voice][j];p=model.distribution(histories[j],unigram);h=-float((p*np.log(p)).sum());surprise=-float(np.log(p[token]));histories[j].append(token)
                after=model.distribution(histories[j],unigram);next_h=-float((after*np.log(after)).sum())
                values.extend([h/np.log(model.k),min(surprise/np.log(model.k),4),next_h/np.log(model.k)])
                nll.append(dict(voice=voice,field=j,nll=surprise))
            x[b,voice*7:voice*7+6]=np.maximum(x[b,voice*7:voice*7+6],values);x[b,voice*7+6]=1
    return x,nll
