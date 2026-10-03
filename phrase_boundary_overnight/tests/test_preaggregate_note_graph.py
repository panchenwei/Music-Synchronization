import numpy as np
import torch
from src.note_relation_graph import RelationLayer,NoteRelationBoundary,build_graph,tensors
from src.preaggregate_note_graph import PreAggregateBoundary,preaggregate

torch.set_num_threads(2)


def graph():
    return build_graph([(0.,2.,60,1,1),(0.,1.,48,2,1),(1.,1.,64,1,1),(3.,1.,67,1,1)],[None]*4,np.arange(4)%3)


def test_identical_parameters_rng_and_initial_predictions():
    a=NoteRelationBoundary(True,42).eval();rng_a=torch.get_rng_state().clone()
    b=PreAggregateBoundary(42).eval();torch.testing.assert_close(rng_a,torch.get_rng_state(),rtol=0,atol=0)
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())==7737
    for k,p in a.state_dict().items():torch.testing.assert_close(p,b.state_dict()[k],rtol=0,atol=0)
    x=torch.rand(2,4,58);g=tensors(graph(),'cpu')
    with torch.no_grad():torch.testing.assert_close(a(x,g),b(x,g),rtol=0,atol=0)


def test_pair_association_is_preserved_in_counterexample():
    torch.manual_seed(192);layer=RelationLayer().eval();h=torch.randn(3,24)
    g={'edge_index':torch.tensor([[1,2],[0,0]]),'edge_features':torch.tensor([[1.,-.2,.1,0.],[-2.,.4,-.1,0.]])}
    other={**g,'edge_features':g['edge_features'].flip(0)}
    assert float((layer(h,g,True)-layer(h,other,True)).abs().max().detach())<1e-6
    assert float((preaggregate(layer,h,g)-preaggregate(layer,h,other)).abs().max().detach())>1e-4


def test_reindexing_and_gradients():
    g=graph();order=np.array([2,0,3,1]);inverse=np.argsort(order);p={k:v.copy() for k,v in g.items()}
    p['node_features']=g['node_features'][order];p['edge_index']=inverse[g['edge_index']];p['pool_node']=inverse[g['pool_node']]
    m=PreAggregateBoundary(42).eval();a=m.encode_beats(tensors(g,'cpu'));b=m.encode_beats(tensors(p,'cpu'))
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
    a[:,0].sum().backward();grads=[p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(v).all() for v in grads)
