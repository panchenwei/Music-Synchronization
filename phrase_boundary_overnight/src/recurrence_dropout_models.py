"""Matched stronger residual-path dropout in the retained three-block CNN."""
from .recurrence_depth_models import make_model as base_factory


def make_model(kind,seed):
    assert kind in ('C3','D4')
    model=base_factory('C3',seed)
    if kind=='D4':
        for layer in model.frontend.layers:layer.dropout.p=.4
    return model
