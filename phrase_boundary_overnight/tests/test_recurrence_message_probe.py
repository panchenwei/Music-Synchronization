import numpy as np
from src.recurrence_message_probe import build_graph,random_control,mix_probabilities

def events():
    return np.array([(i,.8,60+(i%18)%7,staff) for i in range(72) for staff in (1,2)],float)

def test_graph_repetition_and_exclusion():
    g=build_graph(events(),72)
    assert g.sum()>0 and g[20,38]>0
    a,b=np.nonzero(g)
    assert (abs(a-b)>=12).all() and (a>=3).all() and (b<=66).all()
    assert ((g>0).sum(1)<=3).all()
    np.testing.assert_allclose(g.sum(1)[g.sum(1)>0],1)

def test_random_control_keeps_row_statistics():
    g=build_graph(events(),72);r=random_control(g,'synthetic')
    assert not np.array_equal(g,r)
    np.testing.assert_array_equal(r,random_control(g,'synthetic'))
    np.testing.assert_array_equal(np.sort(g,axis=1),np.sort(r,axis=1))
    a,b=np.nonzero(r);assert (abs(a-b)>=12).all()

def test_probability_message_and_identity():
    g=np.zeros((4,4));g[1,3]=1;p=np.array([.1,.2,.3,.9])
    q=mix_probabilities(p,g)
    np.testing.assert_allclose(q,[.1,.375,.3,.9])
    np.testing.assert_array_equal(mix_probabilities(p,g,0),p)
    np.testing.assert_array_equal(mix_probabilities(p,np.zeros((4,4))),p)

def test_empty_coverage_and_constant_invariance():
    g=build_graph(events(),72)
    np.testing.assert_allclose(mix_probabilities(np.full(72,.37),g),.37)
    assert build_graph(events()[:2],4).sum()==0
