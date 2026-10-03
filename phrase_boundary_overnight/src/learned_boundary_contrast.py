"""Explicit left/right learned-state comparison, identity-initialized residual."""
import torch
from torch import nn
from torch.nn import functional as F
from .recurrence_depth_models import make_model as original


def side_features(h,valid,kind):
    assert kind in ('G','D','Z') and h.ndim==3 and valid.shape==h.shape[:2]
    n=h.shape[1];padded=F.pad(h.masked_fill(~valid[...,None],0),(0,0,4,3));mask=F.pad(valid.to(h.dtype),(4,3))
    parts=[]
    for offsets in (range(-4,0),range(0,4)):
        total=sum(padded[:,4+o:4+o+n] for o in offsets);count=sum(mask[:,4+o:4+o+n] for o in offsets)
        parts.append(total/count.clamp_min(1)[...,None])
    left,right=parts
    x=right-left if kind=='G' else (left+right)*.5
    if kind=='Z':x=x*0
    return torch.cat([x,x.abs()],-1).masked_fill(~valid[...,None],0)


class BoundaryContrast(nn.Module):
    def __init__(self,kind,seed):
        super().__init__();self.kind=kind;self.core=original('C3',seed)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):self.residual=nn.Linear(64,1,bias=False)
        with torch.no_grad():self.residual.weight.zero_()
    def forward(self,x,padding_mask=None,**kwargs):
        valid=torch.ones(x.shape[:2],dtype=torch.bool,device=x.device) if padding_mask is None else ~padding_mask.bool()
        logit,h=self.core(x,padding_mask=~valid,return_hidden=True)
        correction=self.residual(side_features(h,valid,self.kind)).squeeze(-1)
        return (logit+correction).masked_fill(~valid,0)


def make_model(kind,seed):
    model=BoundaryContrast(kind,seed);assert sum(p.numel() for p in model.parameters())==5985;return model
