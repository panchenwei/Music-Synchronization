import numpy as np
from src.curve_arc_features import side_trends,transform
from src.curve_scale_features import contrast

def test_valley_distinguished_from_flat_and_signed():
    x=np.r_[np.arange(7,-1,-1),np.arange(8)].astype(float);ok=np.ones(16,bool)
    a,q=side_trends(x,ok,8);b,_=side_trends(-x,ok,8)
    old,_=contrast(x,ok,8)
    assert old[8]==0 and a[8,0]<0<a[8,1] and q[8]==1
    np.testing.assert_allclose(a[8,:2],-b[8,:2],atol=1e-6)
    np.testing.assert_allclose(a[8,2:],0,atol=1e-6)

def test_missing_and_edges_do_not_make_turns():
    x=np.arange(20,dtype=float);ok=np.ones(20,bool);x[8]=np.nan
    a,q=side_trends(x,ok,4)
    assert not q[:4].any() and not q[17:].any()
    assert not q[5:13].any() and np.isfinite(a).all()
    assert not a[q==0].any()

def test_estimated_last_tempo_is_excluded():
    x=np.zeros((1,40,9),np.float32);x[:,:,7:9]=1
    x[0,:,0]=np.arange(40);a=transform(x)
    x[0,-1,0]=99999;b=transform(x)
    np.testing.assert_array_equal(a,b)
