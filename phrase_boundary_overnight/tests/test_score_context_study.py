import numpy as np
import torch
from src.score_context_study import pitch_profiles,make_model

def test_duration_and_lowest_pitch():
    x=pitch_profiles([(0,1,60,1,1),(0,.5,48,2,1),(.5,.5,50,2,1)],1)[0]
    assert np.isclose(x[0],.75) and np.isclose(x[2],.25)
    assert np.isclose(x[12],.5) and np.isclose(x[14],.5)

def test_silence_grace_and_fractional_end():
    x=pitch_profiles([(.75,.5,61,1,1),(0,0,99,1,1)],3)
    assert x[0,1]==1 and x[1,1]==1 and x[2].sum()==0

def test_transposition_equivariance():
    a=pitch_profiles([(0,1,60,1,1),(0,1,48,2,1)],2)
    b=pitch_profiles([(0,1,62,1,1),(0,1,50,2,1)],2)
    np.testing.assert_allclose(np.roll(a[:,:12],2,axis=1),b[:,:12]);np.testing.assert_allclose(np.roll(a[:,12:],2,axis=1),b[:,12:])

def test_shared_capacity_and_zero_new_columns():
    models=[make_model(k,42).eval() for k in ('B','P','C','PC')]
    assert len({sum(p.numel() for p in m.parameters()) for m in models})==1
    x=torch.randn(2,24,58);z=x.clone();z[:,:,34:]=0
    for a,b in ((models[0],models[1]),(models[2],models[3])):
        with torch.no_grad():torch.testing.assert_close(a(z),b(x),atol=1e-6,rtol=1e-5)

def test_context_and_mask():
    for kind in ('B','C'):
        m=make_model(kind,42).eval();x=torch.randn(1,31,58,requires_grad=True);m(x)[0,15].backward();g=x.grad.abs().sum(-1)[0]
        if kind=='B':assert g[:13].sum()==0 and g[18:].sum()==0
        else:assert g[7]>0 and g[23]>0 and g[:7].sum()==0
        mask=torch.zeros(1,31,dtype=torch.bool);mask[:,25:]=True
        z=x.detach().clone();z[:,25:]=1e6
        with torch.no_grad():torch.testing.assert_close(m(x.detach(),padding_mask=mask),m(z,padding_mask=mask))
