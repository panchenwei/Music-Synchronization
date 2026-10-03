import numpy as np
from src.external_structure_probe import novelty,shuffled


def test_constant_and_right_hand_boundary():
    x=np.tile([1.,0.],(40,1));a=np.ones(40,bool)
    np.testing.assert_allclose(novelty(x,a),0,atol=1e-15)
    x[20:]=[0.,1.];p=novelty(x,a)
    assert p.argmax()==20 and abs(p[20]-.5)<1e-12 and p[0]==0


def test_availability_and_permutation():
    x=np.random.default_rng(1).normal(size=(30,7));a=np.ones(30,bool);a[[0,5,6,29]]=False
    s=shuffled(x,a,'test');np.testing.assert_array_equal(s[~a],x[~a]);np.testing.assert_allclose(np.sort(s[a],axis=0),np.sort(x[a],axis=0))
    np.testing.assert_array_equal(s,shuffled(x,a,'test'));assert not np.array_equal(s,x)
    changed=x.copy();changed[~a]=1000;np.testing.assert_array_equal(novelty(changed,a),novelty(x,a))
    assert np.isfinite(novelty(x,np.zeros(30,bool))).all() and novelty(x,np.zeros(30,bool)).sum()==0
