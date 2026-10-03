import torch
import numpy as np
from src.curve_scale_quality_study import select_channels,QualityBoundary
from src.curve_scale_models import CurveScaleBoundary


def test_input_selection_preserves_fit_metadata_and_source():
    x=np.random.default_rng(2).normal(size=(2,30,78)).astype(np.float32)
    meta=np.ones((2,30,4,5),np.float32);original=x.copy()
    item={'curves':x,'time_scale':meta}
    v=select_channels(item,'V');q=select_channels(item,'Q')
    assert not v['curves'][...,68:].any() and not q['curves'][...,58:68].any()
    np.testing.assert_array_equal(v['curves'][...,:68],x[...,:68])
    np.testing.assert_array_equal(q['curves'][...,68:],x[...,68:])
    np.testing.assert_array_equal(q['curves'][...,:58],x[...,:58])
    np.testing.assert_array_equal(x,original)
    assert v['time_scale'] is meta and q['time_scale'] is meta


def test_same_architecture_initialization_and_unused_gradients():
    torch.set_num_threads(2);old=CurveScaleBoundary('S',42)
    for kind in ('V','Q'):
        m=QualityBoundary(kind,42).cuda()
        assert sum(p.numel() for p in m.parameters())==6657
        for k,v in old.state_dict().items():torch.testing.assert_close(v,m.state_dict()[k].cpu(),atol=0,rtol=0)
        x=torch.randn(2,20,78,device='cuda')
        if kind=='V':x[...,68:]=0;unused=slice(2,4)
        else:x[...,58:68]=0;unused=slice(0,2)
        with torch.no_grad():m.core.input_projection.weight[:,58:].fill_(.1)
        m(x).square().mean().backward()
        assert torch.count_nonzero(m.stem.input.weight.grad[:,unused])==0
        assert m.stem.input.weight.grad.abs().max()>0
