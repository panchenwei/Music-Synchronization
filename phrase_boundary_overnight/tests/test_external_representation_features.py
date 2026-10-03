import torch
from src.external_representation_features import make_model
from src.recurrence_depth_models import make_model as cnn


def test_zero_entry_preserves_target_function_and_allows_learning():
    torch.set_num_threads(2)
    original=cnn('C3',42).eval();rng=torch.get_rng_state().clone();m=make_model('E',42).eval()
    torch.testing.assert_close(rng,torch.get_rng_state(),atol=0,rtol=0)
    assert sum(p.numel() for p in m.parameters())==6945
    x=torch.randn(2,40,58);extra=torch.randn(2,40,32);xx=torch.cat([x,extra],-1)
    mask=torch.zeros(2,40,dtype=torch.bool);mask[0,31:]=True
    torch.testing.assert_close(original(x,mask),m(xx,mask),atol=2e-6,rtol=2e-5)
    m(xx,mask).square().mean().backward()
    assert m.input_projection.weight.grad[:,58:].abs().sum()>0
