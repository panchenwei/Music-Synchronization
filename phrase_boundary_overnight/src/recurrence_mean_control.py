"""Analytic expectation of random messages and coverage-matched controls."""
import hashlib
import numpy as np
from .recurrence_message_probe import LEFT,RIGHT,MIN_DISTANCE

def build_mean_graph(graph,piece_id,kind):
    assert kind in ('M','U','S')
    n=len(graph);positions=np.arange(LEFT,n-RIGHT+1)
    eligible=np.zeros((n,n),bool)
    eligible[np.ix_(positions,positions)]=abs(positions[:,None]-positions[None,:])>=MIN_DISTANCE
    available=eligible.sum(1)>0;covered=graph.sum(1)>0
    assert not (covered&~available).any()
    if kind=='M':selected=covered
    elif kind=='U':selected=available
    else:
        seed=int.from_bytes(hashlib.sha256(('coverage-control-v1:'+piece_id).encode()).digest()[:8],'little')
        rng=np.random.default_rng(seed);selected=np.zeros(n,bool)
        selected[rng.choice(np.flatnonzero(available),int(covered.sum()),replace=False)]=True
    result=eligible.astype(np.float64)*selected[:,None]
    return result/np.maximum(result.sum(1,keepdims=True),1)
