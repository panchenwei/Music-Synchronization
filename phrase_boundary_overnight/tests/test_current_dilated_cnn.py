import torch
from src.current_dilated_cnn import make_model
from src.recurrence_depth_models import make_model as original


def test_same_initial_function_parameters_and_rng():
    torch.set_num_threads(2)
    a=original('C3',42).eval();rng=torch.get_rng_state().clone();b=make_model('D',42).eval()
    torch.testing.assert_close(rng,torch.get_rng_state(),atol=0,rtol=0)
    assert sum(p.numel() for p in b.parameters())==5921
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
    x=torch.randn(2,45,58);mask=torch.zeros(2,45,dtype=torch.bool);mask[0,33:]=True
    torch.testing.assert_close(a(x,mask),b(x,mask),atol=0,rtol=0)


def test_actual_radius_14_and_padding():
    torch.set_num_threads(2);m=make_model('D',9).eval()
    with torch.no_grad():
        for layer in m.frontend.layers[1:]:layer.pointwise.weight.fill_(.02)
    x=torch.randn(1,49,58,requires_grad=True);m(x)[0,24].backward();g=x.grad.abs().sum(-1)[0]
    assert g[:10].sum()==0 and g[39:].sum()==0 and g[10]>0 and g[38]>0
    xx=torch.randn(2,49,58);mask=torch.zeros(2,49,dtype=torch.bool);mask[0,31:]=True
    altered=xx.clone();altered[mask]=10000
    torch.testing.assert_close(m(xx,mask),m(altered,mask),atol=0,rtol=0)
