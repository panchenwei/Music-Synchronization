"""Clean-room TF-SepNet-inspired pitch/time front end for per-beat detection.

Not a reproduction of the scene classifier: local temporal broadcast, pixelwise
channel normalization, no temporal downsampling or clip-level pooling head.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .recurrence_depth_models import make_model


class PixelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__(); self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)


class AxisBlock(nn.Module):
    def __init__(self, mode, channels=8):
        super().__init__()
        assert mode in ('S','J') and channels % 2 == 0
        half=channels//2
        kernels=((5,1),(1,5)) if mode=='S' else ((5,5),(5,5))
        self.dw=nn.ModuleList([nn.Conv2d(half,half,k,padding=(k[0]//2,k[1]//2),groups=half) for k in kernels])
        self.norm=nn.ModuleList([PixelNorm(half) for _ in range(2)])
        self.pw=nn.ModuleList([nn.Conv2d(half,half,1) for _ in range(2)])

    def forward(self, x, mask):
        b,c,p,t=x.shape
        x=x.reshape(b,2,c//2,p,t).transpose(1,2).reshape(b,c,p,t)
        halves=x.chunk(2,1); out=[]
        for i,h in enumerate(halves):
            z=F.gelu(self.norm[i](self.dw[i](h))).masked_fill(mask,0)
            # Pitch context broadcasts over register; temporal context is LOCAL.
            z=z.mean(2,keepdim=True) if i==0 else F.avg_pool2d(z,(1,5),stride=1,padding=(0,2))
            out.append((h+F.gelu(self.pw[i](z))).masked_fill(mask,0))
        return torch.cat(out,1)


class AxisStem(nn.Module):
    halo_beats=2  # two blocks: at most +/-8 quarter-beat subticks

    def __init__(self, mode, seed):
        super().__init__()
        # Common modules exactly match between arms despite different DW shapes.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+3100)
            self.input=nn.Conv2d(2,8,1); self.norm=PixelNorm(8)
            self.readout=nn.Sequential(nn.Linear(256,8),nn.LayerNorm(8))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+3200)
            self.blocks=nn.ModuleList([AxisBlock(mode) for _ in range(2)])

    def forward(self, roll, padding_mask=None):
        b,t=roll.shape[:2]
        assert roll.shape[2:]==(2,128,4)
        mask=torch.zeros((b,t),dtype=torch.bool,device=roll.device) if padding_mask is None else padding_mask
        roll=roll.masked_fill(mask[...,None,None,None],0)
        tm=mask.repeat_interleave(4,1)[:,None,None,:]
        x=roll.permute(0,2,3,1,4).reshape(b,2,128,t*4)
        x=F.gelu(self.norm(self.input(x))).masked_fill(tm,0)
        for block in self.blocks: x=block(x,tm)
        # Preserve eight register bands and ordered within-beat subticks.
        x=x.reshape(b,8,8,16,t,4).mean(3).permute(0,3,1,2,4).reshape(b,t,256)
        return self.readout(x).masked_fill(mask[...,None],0)


class AxisBoundary(nn.Module):
    def __init__(self, mode, seed):
        super().__init__(); self.core=make_model('C3',seed)
        old=self.core.input_projection
        with torch.random.fork_rng(devices=[]): new=nn.Linear(66,32)
        with torch.no_grad():
            new.weight.zero_(); new.weight[:,:58].copy_(old.weight); new.bias.copy_(old.bias)
        self.core.input_projection=new; self.stem=AxisStem(mode,seed)

    def forward_embedded(self,x,features,padding_mask=None):
        return self.core(torch.cat([x,features],-1),padding_mask=padding_mask)

    def forward(self,x,roll,padding_mask=None):
        return self.forward_embedded(x,self.stem(roll,padding_mask),padding_mask)


def features_in_blocks(model,roll,block_size=128):
    blocks=[]; n=roll.shape[1]; halo=model.stem.halo_beats
    for start in range(0,n,block_size):
        stop=min(n,start+block_size); left=max(0,start-halo); right=min(n,stop+halo)
        blocks.append(model.stem(roll[:,left:right])[:,start-left:stop-left])
    return torch.cat(blocks,1)


def predictions(model,data,norm,device):
    was=model.training; model.eval(); out={}
    try:
        with torch.no_grad():
            for pid,item in sorted(data.items()):
                features=features_in_blocks(model,torch.from_numpy(item['piano_roll'][None]).to(device)); rows=[]
                for start in range(0,len(item['curves']),8):
                    x=torch.from_numpy(norm.apply(item['curves'][start:start+8]).astype(np.float32)).to(device)
                    rows.extend(torch.sigmoid(model.forward_embedded(x,features.expand(x.shape[0],-1,-1))).cpu().numpy())
                out[pid]={str(k):v for k,v in zip(item['performance_ids'],rows)}
    finally: model.train(was)
    return out
