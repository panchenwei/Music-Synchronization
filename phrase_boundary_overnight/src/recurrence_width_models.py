"""Uniform hidden-channel duplication, matching the narrow initial eval function.

Local adaptation of the function-preserving expansion idea, not a full Net2Net
reproduction. No pretrained checkpoint is used; both models start untrained.
"""
import copy
import torch
from torch import nn
from .recurrence_depth_models import make_model as narrow_factory
from .phase2_models import SinusoidalPositionEncoding


def duplicate_norm(norm):
    n=norm.normalized_shape[0]
    result=nn.LayerNorm(n*2,eps=norm.eps)
    with torch.no_grad():
        result.weight.copy_(norm.weight.repeat(2));result.bias.copy_(norm.bias.repeat(2))
    return result


def widen(model):
    assert model.kind=='C' and len(model.blocks)==0 and model.input_projection.out_features==32
    assert not model.frontend.wide and len(model.frontend.convs)==1
    # Construction must not alter the original CPU/CUDA RNG streams.
    with torch.random.fork_rng(devices=[]):
        m=copy.deepcopy(model);old=model.input_projection
        m.input_projection=nn.Linear(old.in_features,64)
        m.position=SinusoidalPositionEncoding(64)  # unused by kind C, kept dimensionally coherent
        m.frontend.norm=duplicate_norm(model.frontend.norm)
        m.frontend.convs=nn.ModuleList([nn.Conv1d(64,64,5,padding=2,groups=64)])
        m.frontend.pointwise=nn.Conv1d(64,64,1)
        m.final_norm=duplicate_norm(model.final_norm);m.output=nn.Linear(64,1)
        with torch.no_grad():
            m.input_projection.weight.copy_(old.weight.repeat(2,1));m.input_projection.bias.copy_(old.bias.repeat(2))
            a=model.frontend.convs[0];b=m.frontend.convs[0]
            b.weight.copy_(a.weight.repeat(2,1,1));b.bias.copy_(a.bias.repeat(2))
            a=model.frontend.pointwise;b=m.frontend.pointwise
            b.weight.copy_(a.weight.repeat(2,2,1)/2);b.bias.copy_(a.bias.repeat(2))
            m.output.weight.copy_(model.output.weight.repeat(1,2)/2);m.output.bias.copy_(model.output.bias)
    return m


def make_model(kind,seed):
    assert kind in ('O','W')
    base=narrow_factory('O',seed)
    return base if kind=='O' else widen(base)
