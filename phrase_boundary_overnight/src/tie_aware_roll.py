"""Keep note occupancy; suppress only uniquely verified tied continuations."""
from collections import Counter
import numpy as np
from .score_piano_roll import piano_roll
from .resolve_tie_audit import predecessors


def build(events,ties,n):
    if len(events)!=len(ties):raise ValueError('One tie state per event required')
    if any(t not in (None,-1,0,1) for t in ties):raise ValueError('Invalid tie encoding')
    links={};unresolved=[]
    for i,tie in enumerate(ties):
        if tie not in (0,-1):continue
        tier,ids=predecessors(events,ties,i)
        if len(ids)==1:links[i]=ids[0]
        else:unresolved.append(i)
    counts=Counter(links.values())
    verified={i:j for i,j in links.items() if counts[j]==1}
    unresolved.extend(i for i,j in links.items() if counts[j]!=1)
    out=piano_roll(events,n);out[:,1]=0
    for i,(q,d,p,*_) in enumerate(events):
        if i not in verified and d>0 and 0<=q<n:
            tick=min(int(np.floor(q*4+1e-9)),n*4-1);out[tick//4,1,int(p),tick%4]=1
    return out,dict(verified_continuations=len(verified),unresolved_continuations=len(unresolved),unresolved_indices=sorted(unresolved))
