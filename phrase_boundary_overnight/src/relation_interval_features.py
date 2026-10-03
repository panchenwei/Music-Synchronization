"""Label-free beat summaries of within/across-onset score pitch relationships."""
import numpy as np

def reference_voices(events):
    voices=sorted({e[3:5] for e in events if e[1]>0})
    if not voices:raise ValueError('No positive-duration score notes')
    melody=(1,1) if (1,1) in voices else voices[0]
    bottom=max(v[0] for v in voices)
    bass=max([v for v in voices if v[0]==bottom],key=lambda v:sum(e[3:5]==v and e[1]>0 for e in events))
    return melody,bass

def relation_features(events,n,voice,highest):
    """12 vertical + 12 horizontal directed intervals modulo 12, not chord labels."""
    valid=[e for e in events if e[1]>0];x=np.zeros((n,24),np.float64)
    choose=max if highest else min
    reference=[e for e in valid if e[3:5]==voice]
    # Preserve each distinct attack inside the beat, unlike a first-attack-only cue.
    attacks={}
    for o,d,p,*_ in reference:attacks.setdefault(round(float(o),9),[]).append(p)
    previous=None
    for onset,pitches in sorted(attacks.items()):
        pitch=choose(pitches);b=int(np.floor(onset+1e-9))
        if previous is not None and 0<=b<n:x[b,12+(pitch-previous)%12]+=1
        previous=pitch
    # Vertical interval pairs are genuinely simultaneous, not pairs pooled over a beat.
    for b in range(n):
        active=[e for e in valid if e[0]<b+1 and e[0]+e[1]>b]
        edges=sorted({float(b),float(b+1),*[v for e in active for v in (max(b,e[0]),min(b+1,e[0]+e[1]))]})
        for a,z in zip(edges[:-1],edges[1:]):
            mid=(a+z)/2;sounding=[e for e in active if e[0]<mid<e[0]+e[1]]
            ref=[e[2] for e in sounding if e[3:5]==voice]
            if not ref:continue
            anchor=choose(ref)
            for e in sounding:x[b,(e[2]-anchor)%12]+=z-a
    for a,b in ((0,12),(12,24)):
        den=x[:,a:b].sum(1,keepdims=True);x[:,a:b]=np.divide(x[:,a:b],den,out=np.zeros_like(x[:,a:b]),where=den>0)
    assert np.isfinite(x).all() and (x>=0).all()
    return x.astype(np.float32)
