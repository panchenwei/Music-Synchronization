"""Small CNN plus bounded relative-bias self-attention; not a Conformer reproduction."""
import torch
from torch import nn
from .recurrence_depth_models import make_model as cnn_factory


class LocalBlock(nn.Module):
    def __init__(self, radius=8):
        super().__init__()
        self.radius=radius;self.heads=4;self.dim=8
        self.norm1=nn.LayerNorm(32);self.qkv=nn.Linear(32,96)
        self.relative_bias=nn.Parameter(torch.zeros(4,2*radius+1))
        self.proj=nn.Linear(32,32);self.norm2=nn.LayerNorm(32)
        self.ffn=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,32))
        self.dropout=nn.Dropout(.2)
        nn.init.zeros_(self.proj.weight);nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.ffn[-1].weight);nn.init.zeros_(self.ffn[-1].bias)

    def forward(self,h,mask):
        b,t,_=h.shape
        q,k,v=self.qkv(self.norm1(h)).reshape(b,t,3,4,8).unbind(2)
        index=torch.arange(t,device=h.device)[:,None]+torch.arange(-self.radius,self.radius+1,device=h.device)[None,:]
        inside=(index>=0)&(index<t);index=index.clamp(0,t-1)
        valid=inside[None,:,:]&~mask[:,index]
        logits=torch.einsum('bthd,btwhd->bthw',q,k[:,index])/(8**.5)
        logits=logits+self.relative_bias[None,None,:,:]
        logits=logits.masked_fill(~valid[:,:,None,:],float('-inf'))
        # Padded query rows may have no unmasked key. Avoid all-minus-inf softmax.
        logits=torch.where(valid.any(-1)[:,:,None,None],logits,torch.zeros_like(logits))
        weights=logits.softmax(-1).masked_fill(~valid[:,:,None,:],0.)
        mixed=torch.einsum('bthw,btwhd->bthd',self.dropout(weights),v[:,index]).reshape(b,t,32)
        h=h+self.dropout(self.proj(mixed));h=h+self.dropout(self.ffn(self.norm2(h)))
        return h.masked_fill(mask[:,:,None],0.)


class Hybrid(nn.Module):
    def __init__(self,base,layers):
        super().__init__()
        self.base=base
        self.attention=nn.ModuleList([LocalBlock() for _ in range(layers)])

    def forward(self,x,padding_mask=None):
        if padding_mask is None:padding_mask=torch.zeros(x.shape[:2],dtype=torch.bool,device=x.device)
        padding_mask=padding_mask.bool()
        if padding_mask.all(1).any():raise ValueError('Fully masked sequence is prohibited')
        h=self.base.input_projection(x).masked_fill(padding_mask[:,:,None],0.)
        h=self.base.frontend(h,padding_mask)
        for layer in self.attention:h=layer(h,padding_mask)
        return self.base.output(self.base.final_norm(h)).squeeze(-1).masked_fill(padding_mask,0.)


def make_model(kind,seed):
    assert kind in ('C3','A1','A2')
    base=cnn_factory('C3',seed)
    if kind=='C3':return base
    # New modules do not shift the original global training RNG at construction.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+871)
        return Hybrid(base,int(kind[-1]))
