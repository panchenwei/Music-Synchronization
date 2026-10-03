import numpy as np
import torch
from src.note_relation_graph import build_graph,merged_events,tensors,NoteRelationBoundary

torch.set_num_threads(2)


def sample():
    e=[(0.,2.,60,1,1),(0.,1.,48,2,1),(1.,1.,64,1,1),(3.,1.,67,1,1)]
    return build_graph(e,[None]*4,np.arange(4)%3)


def test_ties_merge_without_losing_sound_duration():
    nodes,n=merged_events([(0.,1.,60,1,2),(1.,2.,60,1,1)],[1,-1])
    assert n==1 and nodes==[(0.,3.,60,1,2,0.)]


def test_graph_relations_and_pool_accounting():
    g=sample();ei=set(map(tuple,g['edge_index'].T.tolist()))
    assert all(i!=j and (j,i) in ei for i,j in ei)
    assert (0,1) in ei and (1,2) in ei and (2,3) in ei
    assert len(g['node_features'])==4 and g['node_features'].shape[1]==21
    assert np.isfinite(g['edge_features']).all()
    assert abs(g['pool_weight'][g['pool_kind']==1].sum()-5)<1e-7


def test_same_initial_weights_and_prediction():
    a=NoteRelationBoundary(False,42).eval();b=NoteRelationBoundary(True,42).eval()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    g=tensors(sample(),'cpu');x=torch.rand(2,4,58);x[...,34:]=0
    with torch.no_grad():torch.testing.assert_close(a(x,g),b(x,g),atol=0,rtol=0)


def test_neighbor_information_reaches_other_notes_only_in_graph():
    g=tensors(sample(),'cpu');changed={k:v.clone() for k,v in g.items()};changed['node_features'][0,0]+=.5
    for rel in (False,True):
        m=NoteRelationBoundary(rel,42).eval()
        with torch.no_grad():d=(m.encode_nodes(g)-m.encode_nodes(changed)).abs().sum(1)
        if rel:assert d[1]>1e-5
        else:assert d[1]==0


def test_node_reindexing_equivariance_and_finite_gradients():
    g=sample();order=np.array([2,0,3,1]);inverse=np.argsort(order);p={k:v.copy() for k,v in g.items()}
    p['node_features']=g['node_features'][order];p['edge_index']=inverse[g['edge_index']];p['pool_node']=inverse[g['pool_node']]
    m=NoteRelationBoundary(True,42).eval();a=tensors(g,'cpu');b=tensors(p,'cpu')
    torch.testing.assert_close(m.encode_beats(a),m.encode_beats(b),atol=2e-6,rtol=2e-6)
    m.encode_beats(a).square().sum().backward()
    grads=[p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(v).all() for v in grads)
