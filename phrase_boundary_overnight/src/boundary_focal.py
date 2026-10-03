"""Hard-label focal modulation preserving the historical positive class weight."""
import torch
from torch import nn
from torch.nn import functional as F


class WeightedFocal(nn.Module):
    def __init__(self, positive_weight, gamma):
        super().__init__()
        if gamma < 0:
            raise ValueError('gamma must be nonnegative')
        self.register_buffer('positive_weight', torch.as_tensor(positive_weight).detach().clone())
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        ce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.positive_weight, reduction='none')
        if self.gamma == 0:
            return ce
        p = torch.sigmoid(logits)
        pt = torch.where(targets > .5, p, 1 - p)
        return ce * (1 - pt).pow(self.gamma)
