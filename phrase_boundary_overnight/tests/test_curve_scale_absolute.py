import torch
import numpy as np
from src.curve_scale_absolute_study import select_channels,AbsoluteBoundary
from src.curve_scale_quality_study import QualityBoundary


def test_only_coefficient_sign_changes():
    x=np.random.default_rng(4).normal(size=(2,12,78)).astype(np.float32);x[...,68:]=0
    meta=np.ones((2,12,4,5),np.float32);before=x.copy()
    out=select_channels({'curves':x,'time_scale':meta},'A')
    np.testing.assert_array_equal(out['curves'][...,:58],x[...,:58])
    np.testing.assert_array_equal(out['curves'][...,58:68],abs(x[...,58:68]))
    np.testing.assert_array_equal(out['curves'][...,58:68]**2,x[...,58:68]**2)
    np.testing.assert_array_equal(x,before);assert out['time_scale'] is meta
    assert not out['curves'][...,68:].any()


def test_shared_initialization_and_learning():
    torch.set_num_threads(2);a=AbsoluteBoundary('A',42).cuda();v=QualityBoundary('V',42).cuda()
    for k,x in a.state_dict().items():torch.testing.assert_close(x,v.state_dict()[k],atol=0,rtol=0)
    assert sum(p.numel() for p in a.parameters())==6657
    x=torch.randn(2,20,78,device='cuda');x[...,58:68]=x[...,58:68].abs();x[...,68:]=0
    with torch.no_grad():a.core.input_projection.weight[:,58:].fill_(.1)
    a(x).square().mean().backward()
    assert a.stem.input.weight.grad.abs().max()>0
    assert all(torch.isfinite(p.grad).all() for p in a.parameters() if p.grad is not None)
