"""Add frozen score-derived expectation information to the unchanged C3 backbone."""
import torch
from torch import nn
from .recurrence_depth_models import make_model as make_base


def make_model(kind,seed):
    assert kind in ('G','D','Z')
    model=make_base('C3',seed);old=model.input_projection
    # Preserve BOTH CPU and all CUDA streams when constructing the extra columns.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(seed+20260914);linear=nn.Linear(72,32)
    with torch.no_grad():
        linear.weight.zero_();linear.weight[:,:58].copy_(old.weight);linear.bias.copy_(old.bias)
    model.input_projection=linear
    return model
