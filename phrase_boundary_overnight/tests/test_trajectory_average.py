import torch
from src.trajectory_average_study import average_states

def test_average_and_no_mutation():
    a={'w':torch.tensor([1.,2.])};b={'w':torch.tensor([3.,6.])};c={'w':torch.tensor([5.,4.])}
    out=average_states([a,b,c]);torch.testing.assert_close(out['w'],torch.tensor([3.,4.]))
    torch.testing.assert_close(a['w'],torch.tensor([1.,2.]));assert out['w'].data_ptr()!=a['w'].data_ptr()

def test_single_state_exact():
    a={'x':torch.randn(3,5)};out=average_states([a]);torch.testing.assert_close(a['x'],out['x'],atol=0,rtol=0)
