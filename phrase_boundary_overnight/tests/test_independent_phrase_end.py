import torch
from src.independent_phrase_end import build_model
from src.phrase_end_auxiliary import DualBoundary


def test_same_initial_end_function_without_start_head():
    torch.set_num_threads(2);dual=DualBoundary(42).eval();single=build_model('D',42).eval()
    assert sum(p.numel() for p in single.parameters())==3297
    for k,v in single.state_dict().items():
        expected=dual.end_head.state_dict()[k.removeprefix('output.')] if k.startswith('output.') else dual.core.state_dict()[k]
        torch.testing.assert_close(v,expected,rtol=0,atol=0)
    x=torch.randn(2,20,58)
    torch.testing.assert_close(single(x),dual(x,both=True)[1],rtol=0,atol=0)


def test_independent_has_no_start_parameters():
    m=build_model('D',43);x=torch.randn(2,12,58);m(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
