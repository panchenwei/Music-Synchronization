import copy
import numpy as np
import torch
from src.models import Normalizer
from src.pitch_invariance_study import rotate_windows,Sampler

def test_rotation_preserves_other_inputs_and_time():
    x=torch.arange(2*8*58,dtype=torch.float32).reshape(2,8,58);z=rotate_windows(x,[2,7])
    torch.testing.assert_close(z[:,:,:34],x[:,:,:34])
    for i,s in enumerate((2,7)):
        for a,b in ((34,46),(46,58)):torch.testing.assert_close(z[i,:,a:b],torch.roll(x[i,:,a:b],s,dims=-1))
    torch.testing.assert_close(rotate_windows(z,[-2,-7]),x)

def test_rotation_commutes_with_shared_normalization():
    rng=np.random.default_rng(7);raw=rng.normal(size=(2,10,58)).astype(np.float32)
    mean=np.r_[np.zeros(34),np.full(12,.08),np.full(12,.07)].astype(np.float32)
    std=np.r_[np.ones(34),np.full(12,.2),np.full(12,.3)].astype(np.float32);norm=Normalizer(mean,std)
    a=rotate_windows(torch.from_numpy(norm.apply(raw)),[3,6]);b=norm.apply(rotate_windows(torch.from_numpy(raw),[3,6]).numpy())
    np.testing.assert_allclose(a.numpy(),b,atol=1e-6)

def fixture():
    rng=np.random.default_rng(8);x=rng.normal(size=(2,18,58)).astype(np.float32)
    return {'x':{'curves':x,'labels':np.arange(18,dtype=np.float32)%7==0,'label_mask':np.ones(18,np.float32)}}

def test_augmentation_rng_resume_and_mask():
    norm=Normalizer(np.zeros(58,np.float32),np.ones(58,np.float32));s=Sampler(fixture(),norm,42,True)
    state=copy.deepcopy(s.state());first=s.batch();s.load_state(state);second=s.batch()
    for a,b in zip(first,second):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert first[0][:,18:].abs().sum()==0 and first[3][:,18:].sum()==0
    control=Sampler(fixture(),norm,42,False).batch()
    for i in (1,2,3):torch.testing.assert_close(first[i],control[i],atol=0,rtol=0)
    torch.testing.assert_close(first[0][:,:,:34],control[0][:,:,:34],atol=0,rtol=0)

def test_shared_normalization_required():
    import pytest
    n=Normalizer(np.arange(58,dtype=np.float32),np.ones(58,np.float32))
    with pytest.raises(AssertionError):Sampler(fixture(),n,42,True)
