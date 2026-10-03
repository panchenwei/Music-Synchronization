import numpy as np
import torch
from src.ordered_attack_features import features

def test_order_transpose_and_bag():
    e=[(0,.2,60,1,1),(.25,.2,64,1,1),(.5,.4,62,1,1),(0,1,36,2,1),(1,1,43,2,1)]
    a,u=features(e,3);b,v=features([(o,d,p+7,s,w) for o,d,p,s,w in e],3)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(u,v)
    assert a.shape==(3,58) and not np.array_equal(a,u)
    slots=a[:,:56].reshape(3,2,4,7);bag=u[:,:56].reshape(3,2,4,7)
    assert slots[0,0,1,0]>0 and slots[0,0,2,0]<0
    np.testing.assert_allclose(slots.mean(2),bag.mean(2),atol=1e-7)
    assert not a[2].any()

def test_rest_chord_overflow():
    e=[(0,.25,60,1,1),(0,.5,67,1,1),(1,.2,65,1,1)]+[(2+i/8,.1,60+i,1,1) for i in range(6)]
    a,_=features(e,4);z=a[:,:56].reshape(4,2,4,7)
    assert z[0,0,:,5].sum()==1 and z[1,0,0,3]==.5/16
    assert a[2,56]==1 and z[2,0,:,5].sum()==4

def test_initial_control_equal():
    from src.ordered_attack_study import ArcBoundary
    from src.recurrence_depth_models import make_model
    torch.manual_seed(123);x=torch.randn(2,64,116)
    a=ArcBoundary('R',42).eval();b=make_model('C3',42).eval()
    assert sum(p.numel() for p in a.parameters())==7777
    with torch.no_grad():torch.testing.assert_close(a(x),b(x[:,:,:58]),atol=1e-6,rtol=1e-6)
