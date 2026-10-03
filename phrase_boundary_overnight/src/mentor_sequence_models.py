"""Full-context supervision and a compact bidirectional RoPE Transformer."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .phase7_models import CurvePieceBalancedSampler
from .phase6_models import window_starts
from .recurrence_depth_models import make_model as cnn
from .current_gru_models import make_model as gru


class RotaryAttention(nn.Module):
    def __init__(self):
        super().__init__();self.qkv=nn.Linear(32,96);self.proj=nn.Linear(32,32)
    def forward(self,x,pad):
        b,t,_=x.shape;q,k,v=self.qkv(x).reshape(b,t,3,4,8).permute(2,0,3,1,4)
        angle=torch.arange(t,device=x.device,dtype=x.dtype)[:,None]*torch.exp(-math.log(10000)*torch.arange(4,device=x.device,dtype=x.dtype)/4)[None]
        co,si=angle.cos()[None,None],angle.sin()[None,None]
        def rotate(z):
            a,c=z[...,:4],z[...,4:];return torch.cat([a*co-c*si,a*si+c*co],dim=-1)
        z=F.scaled_dot_product_attention(rotate(q),rotate(k),v,attn_mask=(~pad)[:,None,None,:],dropout_p=0.)
        return self.proj(z.transpose(1,2).reshape(b,t,32)).masked_fill(pad[...,None],0)


class Block(nn.Module):
    def __init__(self):
        super().__init__();self.n1=nn.LayerNorm(32);self.att=RotaryAttention();self.n2=nn.LayerNorm(32)
        self.ff=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Dropout(.2),nn.Linear(64,32));self.drop=nn.Dropout(.2)
    def forward(self,x,pad):
        x=(x+self.drop(self.att(self.n1(x),pad))).masked_fill(pad[...,None],0)
        return (x+self.drop(self.ff(self.n2(x)))).masked_fill(pad[...,None],0)


class GlobalTransformer(nn.Module):
    def __init__(self,seed):
        super().__init__();self.input_projection=cnn('C3',seed).input_projection
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+3061);self.layers=nn.ModuleList([Block(),Block()]);self.norm=nn.LayerNorm(32);self.output=nn.Linear(32,1)
    def forward(self,x,padding_mask=None):
        pad=torch.zeros(x.shape[:2],dtype=torch.bool,device=x.device) if padding_mask is None else padding_mask.bool()
        assert x.shape[-1]==58 and (~pad).any(1).all()
        h=self.input_projection(x.masked_fill(pad[...,None],0)).masked_fill(pad[...,None],0)
        for block in self.layers:h=block(h,pad)
        return self.output(self.norm(h)).squeeze(-1).masked_fill(pad,0)


def make_model(kind,seed):
    if kind=='C_full':return cnn('C3',seed)
    if kind in ('G64','Gfull'):return gru('G',seed)
    if kind in ('T64','Tfull'):return GlobalTransformer(seed)
    raise ValueError(kind)


class ContextSampler(CurvePieceBalancedSampler):
    def __init__(self,data,norm,seed,full,batch_size=8):
        super().__init__(data,norm,64,batch_size,seed);self.full=full
    def batch(self):
        samples=[];self.last_selection=[]
        for _ in range(self.batch_size):
            pid=str(self.rng.choice(self.pieces));item=self.data[pid];n=len(item['labels'])
            perf=int(self.rng.integers(len(item['curves'])));start=int(self.rng.choice(window_starts(n,64,32)))
            a,z=(0,n) if self.full else (start,min(n,start+64))
            x=self.normalizer.apply(item['curves'][perf,a:z]).astype('float32');y=item['labels'][a:z].astype('float32')
            mask=np.zeros(z-a,'float32');lo=start-a;length=min(64,n-start);mask[lo:lo+length]=item['label_mask'][start:start+length]
            samples.append((x,y,mask,np.ones(z-a,'float32')));self.last_selection.append((pid,perf,start,lo,length))
        size=max(len(s[0]) for s in samples)
        def pad(v):return np.pad(v,((0,size-len(v)),(0,0)) if v.ndim==2 else (0,size-len(v)))
        return tuple(torch.as_tensor(np.stack([pad(s[i]) for s in samples])) for i in range(4))
