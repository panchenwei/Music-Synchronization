import torch
from src.window_ranking_study import rank_loss, combined

def test_gradient_direction():
    z=torch.tensor([[0.,0.,0.]],requires_grad=True);y=torch.tensor([[1.,0.,0.]])
    rank_loss(z,y,torch.ones_like(y)).backward()
    assert z.grad[0,0]<0 and (z.grad[0,1:]>0).all()

def test_mask_and_shift():
    z=torch.tensor([[1.,-1.,99.]],requires_grad=True);y=torch.tensor([[1.,0.,0.]]);m=torch.tensor([[1.,1.,0.]])
    a=rank_loss(z,y,m);b=rank_loss(z+12,y,m);torch.testing.assert_close(a,b)
    a.backward();assert z.grad[0,2]==0
    torch.testing.assert_close(a,torch.nn.functional.softplus(torch.tensor(-2.)))

def test_empty_classes_and_mask():
    for y,m in [(torch.ones(2,4),torch.ones(2,4)),(torch.zeros(2,4),torch.ones(2,4)),(torch.ones(2,4),torch.zeros(2,4))]:
        z=torch.randn(2,4,requires_grad=True);a=rank_loss(z,y,m);a.backward();assert a==0 and torch.isfinite(z.grad).all() and (z.grad==0).all()

def test_zero_weight_identity():
    x=torch.tensor(.7,requires_grad=True);z=torch.randn(2,4);y=torch.zeros_like(z)
    assert combined(x,z,y,torch.ones_like(z),0) is x
