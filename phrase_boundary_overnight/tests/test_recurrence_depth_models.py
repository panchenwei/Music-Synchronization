import torch
from src.recurrence_depth_models import make_model, DEPTHS

torch.set_num_threads(2)


def test_depth_parameter_count_shared_state_and_rng():
    base=make_model('O',42);state=torch.get_rng_state()
    for kind,params in [('C2',4609),('C3',5921)]:
        m=make_model(kind,42)
        torch.testing.assert_close(state,torch.get_rng_state(),rtol=0,atol=0)
        assert len(m.frontend.layers)==DEPTHS[kind] and len(m.blocks)==0
        assert sum(p.numel() for p in m.parameters())==params
        for name,value in base.state_dict().items():
            target=name.replace('frontend.','frontend.layers.0.',1) if name.startswith('frontend.') else name
            torch.testing.assert_close(value,m.state_dict()[target],rtol=0,atol=0)


def test_initial_output_preserved_and_padding_cannot_leak():
    x=torch.randn(2,20,58);mask=torch.zeros(2,20,dtype=torch.bool);mask[1,13:]=True
    base=make_model('O',43).eval()
    with torch.no_grad():reference=base(x,padding_mask=mask)
    for kind in ('C2','C3'):
        m=make_model(kind,43).eval()
        with torch.no_grad():
            actual=m(x,padding_mask=mask);bad=x.clone();bad[mask]=1e5
            torch.testing.assert_close(actual,reference,rtol=0,atol=1e-6)
            torch.testing.assert_close(actual,m(bad,padding_mask=mask),rtol=0,atol=0)
            assert not actual[mask].any()


def test_added_layers_receive_finite_gradients_and_learn():
    m=make_model('C3',42).train();opt=torch.optim.SGD(m.parameters(),lr=.01)
    x=torch.randn(2,20,58);y=torch.randn(2,20)
    for step in range(2):
        opt.zero_grad();loss=(m(x)-y).square().mean();loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        for layer in m.frontend.layers[1:]:
            assert layer.pointwise.weight.grad.abs().sum()>0
            if step==1:assert layer.convs[0].weight.grad.abs().sum()>0
        opt.step()
    assert all(layer.pointwise.weight.abs().sum()>0 for layer in m.frontend.layers[1:])
