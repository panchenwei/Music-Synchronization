import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import numpy as np
from src.curve_scale_models import CurveScaleBoundary
from src.recurrence_depth_models import make_model
from src.curve_scale_study import normalizer


def test_common_initialization_and_baseline():
    torch.set_num_threads(2)
    x=torch.randn(2,21,78);old=make_model('C3',42).eval()
    models={k:CurveScaleBoundary(k,42).eval() for k in ('F','S','J')}
    for k,count in (('F',6401),('S',6657),('J',6833)):
        m=models[k];assert sum(p.numel() for p in m.parameters())==count
        torch.testing.assert_close(m(x),old(x[...,:58]),atol=2e-6,rtol=0)
        assert torch.count_nonzero(m.stem(torch.zeros(2,21,4,5),torch.zeros(2,21,dtype=torch.bool)))==0
    a,b=models['S'].stem,models['J'].stem
    for name,v in a.state_dict().items():
        if '.dw.' not in name:torch.testing.assert_close(v,b.state_dict()[name],atol=0,rtol=0)


def test_masks_locality_and_gradients():
    torch.set_num_threads(2)
    for kind in ('F','S','J'):
        m=CurveScaleBoundary(kind,42).cuda();x=torch.randn(2,30,78,device='cuda')
        mask=torch.arange(30,device='cuda')[None].expand(2,-1)>=25;dirty=x.clone();dirty[:,25:]=999
        m.eval()
        with torch.no_grad():m.core.input_projection.weight[:,58:].fill_(.1)
        torch.testing.assert_close(m(x,mask),m(dirty,mask),atol=0,rtol=0)
        torch.testing.assert_close(m(x,mask)[:,:25],m(x[:,:25]),atol=2e-5,rtol=0)
        # Stem max +/-8 time steps; leave >=8 halo, no global time pooling.
        z=x[...,58:].reshape(2,30,4,5);no=torch.zeros(2,30,dtype=torch.bool,device='cuda')
        torch.testing.assert_close(m.stem(z,no)[:,10:20],m.stem(z[:,2:28],no[:,2:28])[:,8:18],atol=1e-6,rtol=0)
        m.train();loss=m(x,mask)[:,:25].square().mean();loss.backward()
        assert m.stem.input.weight.grad.abs().max()>0
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_new_scaling_zero_and_quality_preserved(monkeypatch):
    import src.curve_scale_study as study
    from src.models import Normalizer
    monkeypatch.setattr(study,'base_normalizer',lambda data:Normalizer(np.zeros(58),np.ones(58)))
    z=np.zeros((1,4,4,5),np.float32);z[:,:,:2]=2;z[:,:,2:]=1;z[:,0,:]=0
    curves=np.concatenate([np.zeros((1,4,58)),z.reshape(1,4,20)],axis=-1)
    norm=normalizer({'toy':{'curves':curves,'time_scale':z}});out=norm.apply(curves)
    np.testing.assert_array_equal(out[0,0,58:],0)
    np.testing.assert_array_equal(out[:,:,68:],curves[:,:,68:])
    np.testing.assert_allclose(out[0,1,58:68],1)
