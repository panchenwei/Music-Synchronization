"""CNN depth controls with preserved shared initialization and identity additions."""
import copy
import torch
from torch import nn
from .score_context_study import make_model as original_factory

DEPTHS={'O':1,'C2':2,'C3':3}


class StackedFrontend(nn.Module):
    def __init__(self, original, depth):
        super().__init__()
        self.layers=nn.ModuleList([original]);self.wide=False
        for _ in range(depth-1):
            layer=copy.deepcopy(original)
            # New residual path starts at zero; depthwise weights still vary by
            # channel as in the original, and receive gradients after the first step.
            with torch.no_grad():layer.pointwise.weight.zero_();layer.pointwise.bias.zero_()
            self.layers.append(layer)

    def forward(self,h,padding_mask=None):
        for layer in self.layers:h=layer(h,padding_mask)
        return h


def make_model(kind,seed):
    assert kind in DEPTHS
    m=original_factory('P',seed)
    if kind!='O':m.frontend=StackedFrontend(m.frontend,DEPTHS[kind])
    return m
