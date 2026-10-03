"""Remove the learned blank-score response before fusion; optional scale control."""
import torch
from torch import nn
from torch.nn import functional as F
from .axis_split_music import AxisBoundary


class ContentStem(nn.Module):
    halo_beats=2

    def __init__(self,base,normalize):
        super().__init__();self.base=base;self.normalize=normalize

    def forward(self,roll,padding_mask=None):
        actual=self.base(roll,padding_mask)
        # Do not detach: cancellation removes gradients of content-independent
        # offsets, while gradients of the real-score difference remain trainable.
        blank=self.base(torch.zeros_like(roll),padding_mask)
        z=actual-blank
        if self.normalize:z=F.layer_norm(z,(z.shape[-1],),eps=1e-5)
        if padding_mask is not None:z=z.masked_fill(padding_mask[...,None],0)
        return z


class ContentBoundary(AxisBoundary):
    def __init__(self,kind,seed):
        assert kind in ('C','N')
        super().__init__('S',seed)
        self.stem=ContentStem(self.stem,kind=='N')
