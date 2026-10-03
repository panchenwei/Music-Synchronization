import numpy as np
from src.recurrence_mean_control import build_mean_graph
from src.recurrence_message_probe import mix_probabilities

def test_coverage_and_uniform_targets():
    g=np.zeros((64,64));g[12,40]=1;g[40,12]=1
    m=build_mean_graph(g,'test','M');u=build_mean_graph(g,'test','U');s=build_mean_graph(g,'test','S')
    np.testing.assert_array_equal(m.sum(1)>0,g.sum(1)>0)
    assert (s.sum(1)>0).sum()==2 and (u.sum(1)>0).sum()>2
    np.testing.assert_array_equal(s,build_mean_graph(g,'test','S'))
    for row in m[m.sum(1)>0]:assert len(np.unique(row[row>0]))==1
    for a in (m,u,s):
        i,j=np.nonzero(a);assert (abs(i-j)>=12).all() and (i>=3).all() and (j<=58).all()

def test_mean_is_expected_random_message():
    g=np.zeros((64,64));g[12,40]=1
    m=build_mean_graph(g,'test','M');p=np.linspace(.01,.99,64)
    eligible=np.flatnonzero(m[12]);q=mix_probabilities(p,m)
    assert abs(q[12]-(.75*p[12]+.25*p[eligible].mean()))<1e-12
    unchanged=np.ones(64,bool);unchanged[12]=False
    np.testing.assert_array_equal(q[unchanged],p[unchanged])
