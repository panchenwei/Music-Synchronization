import torch
from src.curve_arc_study import ArcBoundary
from src.recurrence_depth_models import make_model

def test_initial_predictions_match_c3_with_arbitrary_extra_features():
    torch.set_num_threads(2)
    a=ArcBoundary('R',42).eval();b=make_model('C3',42).eval()
    x=torch.randn(2,64,88);mask=torch.zeros(2,64,dtype=torch.bool);mask[1,50:]=True
    assert sum(p.numel() for p in a.parameters())==6881
    torch.testing.assert_close(a(x,mask),b(x[...,:58],mask),rtol=0,atol=1e-6)
    a.train();a(x,mask).square().mean().backward()
    assert torch.isfinite(a.core.input_projection.weight.grad).all()
    assert torch.count_nonzero(a.core.input_projection.weight.grad[:,58:])>0

def test_zero_control_has_no_extra_feature_gradient():
    a=ArcBoundary('Z',42);x=torch.randn(2,64,88);x[...,58:]=0
    a(x).square().mean().backward()
    assert torch.count_nonzero(a.core.input_projection.weight.grad[:,58:])==0
