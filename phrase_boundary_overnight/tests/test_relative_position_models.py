import pytest
import torch
from src.relative_position_models import make_model
from src.pure_transformer_reference import make_model as baseline

def test_shared_weights_and_nope_equivalence():
    a=baseline('T',42).eval();n=make_model('N',42).eval();r=make_model('R',42).eval()
    assert sum(p.numel() for p in r.parameters())==19073
    for k,v in a.state_dict().items():
        torch.testing.assert_close(v,n.state_dict()[k],atol=0,rtol=0)
        torch.testing.assert_close(v,r.state_dict()[k],atol=0,rtol=0)
    a.position.encoding.zero_();x=torch.randn(2,19,58)
    with torch.no_grad():torch.testing.assert_close(a(x),n(x),atol=1e-6,rtol=1e-6)

def test_bias_translation_and_distance():
    r=make_model('R',42);b=r.attention_bias(90,'cpu',torch.float32)
    torch.testing.assert_close(b[:,20:40,20:40],b[:,:20,:20],atol=0,rtol=0)
    assert (b[:,0,1]>b[:,0,20]).all() and (b.diagonal(dim1=1,dim2=2)==0).all()

@pytest.mark.parametrize('kind',['N','R'])
def test_padding_gradients_and_tiny_overfit(kind):
    torch.set_num_threads(2)
    m=make_model(kind,42).eval();x=torch.randn(2,18,58);mask=torch.zeros(2,18,dtype=torch.bool);mask[:,13:]=True
    changed=x.clone();changed[:,13:]=999
    with torch.no_grad():
        torch.testing.assert_close(m(x,mask),m(changed,mask),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(m(x,mask)[:,:13],m(x[:,:13]),atol=2e-6,rtol=2e-6)
    y=torch.zeros(2,18);y[:,[3,9]]=1;opt=torch.optim.Adam(m.parameters(),lr=.01)
    for _ in range(60):
        opt.zero_grad();logits=m(x,mask)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[~mask],y[~mask]);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        opt.step()
    assert loss.item()<.08
