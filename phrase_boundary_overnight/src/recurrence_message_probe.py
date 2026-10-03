"""Label-free within-score links and a fixed probability-message diagnostic."""
import hashlib
import numpy as np
from .motif_recurrence_features import staff_sequences,window_similarity

LEFT=3
RIGHT=6
MIN_DISTANCE=12
TOP_K=3
MIN_SIMILARITY=.8
MIX=.25

def build_graph(events,n):
    """Directed top-three matches of two-staff preceding/following patterns.

    No labels, quality masks or model probabilities participate in link choice.
    Coordinates are quarter-note units. Chroma loses octave; staff is not voice.
    """
    sequences=staff_sequences(events,n)
    positions=np.arange(LEFT,n-RIGHT+1)
    graph=np.zeros((n,n),np.float64)
    if not len(positions):return graph
    similarity=np.zeros((len(positions),len(positions)))
    for x in sequences:
        a=window_similarity(x,LEFT,True)
        b=window_similarity(x,RIGHT,True)
        similarity+=(a[np.ix_(positions-LEFT,positions-LEFT)]+b[np.ix_(positions,positions)])/4
    for i,pos in enumerate(positions):
        eligible=np.flatnonzero((abs(positions-pos)>=MIN_DISTANCE)&(similarity[i]>=MIN_SIMILARITY))
        chosen=eligible[np.argsort(-similarity[i,eligible],kind='stable')[:TOP_K]]
        if len(chosen):
            weights=similarity[i,chosen]**4
            graph[pos,positions[chosen]]=weights/weights.sum()
    return graph

def random_control(graph,piece_id):
    """Same source coverage, degree and weight multiset; random eligible targets."""
    n=len(graph);positions=np.arange(LEFT,n-RIGHT+1)
    seed=int.from_bytes(hashlib.sha256(('message-control-v1:'+piece_id).encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed);result=np.zeros_like(graph)
    for i in np.flatnonzero(graph.sum(1)>0):
        weights=graph[i,graph[i]>0]
        candidates=positions[abs(positions-i)>=MIN_DISTANCE]
        chosen=rng.choice(candidates,size=len(weights),replace=False)
        result[i,chosen]=weights
    return result

def mix_probabilities(probabilities,graph,mix=MIX):
    p=np.asarray(probabilities,dtype=np.float64)
    assert p.shape==(len(graph),) and graph.shape==(len(p),len(p))
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and 0<=mix<=1
    covered=graph.sum(1)>0
    np.testing.assert_allclose(graph.sum(1)[covered],1,atol=1e-12)
    out=p.copy();out[covered]=(1-mix)*p[covered]+mix*(graph@p)[covered]
    return out
