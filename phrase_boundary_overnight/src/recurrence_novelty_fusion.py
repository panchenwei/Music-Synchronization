"""Matched capacity/availability controls for frozen recurrence + local novelty."""
import numpy as np
import torch
from torch import nn
from .score_context_study import make_model as original_factory


def extra_features(ordered, novelty, kind):
    assert kind in ('A','F')
    assert ordered.shape == novelty.shape and ordered.shape[1] == 24
    extra = novelty[:, :16].copy()
    if kind == 'A': extra[:, :12] = 0
    return np.concatenate([ordered, extra], axis=1)


def make_model(kind, seed):
    assert kind in ('O','A','F')
    m = original_factory('P', seed)
    if kind == 'O': return m
    old = m.input_projection
    with torch.random.fork_rng(devices=[]): expanded = nn.Linear(74,32)
    with torch.no_grad():
        expanded.weight.zero_(); expanded.weight[:, :58].copy_(old.weight); expanded.bias.copy_(old.bias)
    m.input_projection = expanded
    return m
