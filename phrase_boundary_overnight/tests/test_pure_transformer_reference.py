import torch
from torch import nn
from src.pure_transformer_reference import make_model

def test_pure_architecture_and_seed():
    a,b=make_model('T',42),make_model('T',42)
    assert a.frontend is None and len(a.blocks)==2 and a.input_projection.in_features==58
    assert sum(isinstance(m,nn.MultiheadAttention) for m in a.modules())==2
    assert not any(isinstance(m,(nn.Conv1d,nn.Conv2d,nn.GRU,nn.LSTM)) for m in a.modules())
    for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)

def test_padding_gradient_and_tiny_overfit():
    torch.set_num_threads(2);m=make_model('T',42);x=torch.randn(2,16,58)
    mask=torch.zeros(2,16,dtype=torch.bool);mask[1,12:]=True
    m.eval();a=m(x,mask);z=x.clone();z[mask]=10000
    torch.testing.assert_close(a,m(z,mask),atol=1e-6,rtol=1e-6)
    assert a.shape==(2,16) and (a[mask]==0).all()
    target=torch.zeros(2,16);target[:,(3,9)]=1
    opt=torch.optim.AdamW(m.parameters(),lr=.01)
    def loss():return nn.functional.binary_cross_entropy_with_logits(m(x,mask)[~mask],target[~mask])
    initial=float(loss().detach())
    for _ in range(60):
        opt.zero_grad();v=loss();assert torch.isfinite(v);v.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        opt.step()
    assert float(loss().detach())<initial*.15
