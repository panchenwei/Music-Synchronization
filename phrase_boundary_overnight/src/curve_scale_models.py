"""Small time/scale front ends over signed performance-curve contrasts."""
import torch
from torch import nn
from torch.nn import functional as F
from .recurrence_depth_models import make_model as c3_model


class FlatScaleStem(nn.Module):
    def __init__(self,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+5400)
            self.input=nn.Linear(20,8,bias=False);self.readout=nn.Linear(8,8,bias=False)

    def forward(self,z,mask):
        z=z.masked_fill(mask[...,None,None],0)
        return self.readout(F.gelu(self.input(z.flatten(2)))).masked_fill(mask[...,None],0)


class ScaleBlock(nn.Module):
    def __init__(self,kind,seed):
        super().__init__();kernels=((3,1),(1,5)) if kind=='S' else ((3,5),(3,5))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.dw=nn.ModuleList([nn.Conv2d(4,4,k,padding=(k[0]//2,k[1]//2),groups=4,bias=False) for k in kernels])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+10)
            self.pw=nn.ModuleList([nn.Conv2d(4,4,1,bias=False) for _ in range(2)])

    def forward(self,x,mask):
        b,c,s,t=x.shape;x=x.reshape(b,2,4,s,t).transpose(1,2).reshape(b,8,s,t);out=[]
        for i,h in enumerate(x.chunk(2,1)):
            z=F.gelu(self.dw[i](h)).masked_fill(mask,0)
            z=z.mean(2,keepdim=True) if i==0 else F.avg_pool2d(z,(1,5),stride=1,padding=(0,2))
            out.append((h+F.gelu(self.pw[i](z))).masked_fill(mask,0))
        return torch.cat(out,1)


class AxisScaleStem(nn.Module):
    def __init__(self,kind,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+5400)
            self.input=nn.Conv2d(4,8,1,bias=False);self.readout=nn.Linear(40,8,bias=False)
        self.blocks=nn.ModuleList([ScaleBlock(kind,seed+5500+i*20) for i in range(2)])

    def forward(self,z,mask):
        z=z.masked_fill(mask[...,None,None],0).permute(0,2,3,1);tm=mask[:,None,None,:]
        z=F.gelu(self.input(z)).masked_fill(tm,0)
        for block in self.blocks:z=block(z,tm)
        z=z.permute(0,3,1,2).flatten(2)
        return self.readout(z).masked_fill(mask[...,None],0)


class CurveScaleBoundary(nn.Module):
    def __init__(self,kind,seed):
        super().__init__();assert kind in ('F','S','J');self.core=c3_model('C3',seed)
        old=self.core.input_projection
        with torch.random.fork_rng(devices=[]):new=nn.Linear(66,32)
        with torch.no_grad():
            new.weight.zero_();new.weight[:,:58].copy_(old.weight);new.bias.copy_(old.bias)
        self.core.input_projection=new
        self.stem=FlatScaleStem(seed) if kind=='F' else AxisScaleStem(kind,seed)

    def forward(self,x,padding_mask=None):
        assert x.shape[-1]==78
        mask=torch.zeros(x.shape[:2],dtype=torch.bool,device=x.device) if padding_mask is None else padding_mask
        z=self.stem(x[...,58:].reshape(*x.shape[:2],4,5),mask)
        return self.core(torch.cat([x[...,:58],z],-1),padding_mask=mask)
