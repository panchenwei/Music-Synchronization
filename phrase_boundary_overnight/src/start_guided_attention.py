"""Training-only local attention supervision for inter-start regions, not phrase ends."""
import torch
from torch import nn
from .recurrence_local_attention import LocalBlock
from .recurrence_depth_models import make_model as cnn


def local_indices(length,radius,device):
    idx=torch.arange(length,device=device)[:,None]+torch.arange(-radius,radius+1,device=device)[None]
    return idx.clamp(0,length-1),(idx>=0)&(idx<length)


def attention_target(labels,known,radius,kind):
    assert kind in ('G','D','Z')
    known=known.bool();idx,inside=local_indices(labels.shape[1],radius,labels.device)
    # Unknown positions interrupt supervision; never bridge ambiguous label holes.
    runs=(~known).long().cumsum(1)
    legal=inside[None]&known[:,:,None]&known[:,idx]&(runs[:,:,None]==runs[:,idx])
    if kind!='D':
        groups=(labels>.5).long().cumsum(1)
        legal=legal&(groups[:,:,None]==groups[:,idx])
    return legal.float()/legal.sum(-1,keepdim=True).clamp_min(1),known


class GuidedBlock(LocalBlock):
    def forward(self,h,mask):
        b,t,_=h.shape
        q,k,v=self.qkv(self.norm1(h)).reshape(b,t,3,4,8).unbind(2)
        index,inside=local_indices(t,self.radius,h.device)
        valid=inside[None]&~mask[:,index]
        logits=torch.einsum('bthd,btwhd->bthw',q,k[:,index])/(8**.5)
        logits=logits+self.relative_bias[None,None]
        logits=logits.masked_fill(~valid[:,:,None],float('-inf'))
        logits=torch.where(valid.any(-1)[:,:,None,None],logits,torch.zeros_like(logits))
        weights=logits.softmax(-1).masked_fill(~valid[:,:,None],0.)
        self.last_weights=weights
        mixed=torch.einsum('bthw,btwhd->bthd',self.dropout(weights),v[:,index]).reshape(b,t,32)
        h=h+self.dropout(self.proj(mixed));h=h+self.dropout(self.ffn(self.norm2(h)))
        return h.masked_fill(mask[:,:,None],0.)


class GuidedHybrid(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;self.attention=nn.ModuleList([GuidedBlock()])

    def forward(self,x,padding_mask=None):
        if padding_mask is None:padding_mask=torch.zeros(x.shape[:2],dtype=torch.bool,device=x.device)
        padding_mask=padding_mask.bool()
        if padding_mask.all(1).any():raise ValueError('Fully masked sequence is prohibited')
        x=x.masked_fill(padding_mask[:,:,None],0.)
        h=self.base.input_projection(x).masked_fill(padding_mask[:,:,None],0.)
        h=self.base.frontend(h,padding_mask)
        for layer in self.attention:h=layer(h,padding_mask)
        return self.base.output(self.base.final_norm(h)).squeeze(-1).masked_fill(padding_mask,0.)

    def guidance(self,labels,known,kind):
        target,queries=attention_target(labels,known,8,kind)
        weights=self.attention[0].last_weights[:,:,:2,:]
        term=target[:,:,None,:]*(target[:,:,None,:].clamp_min(1e-8).log()-weights.clamp_min(1e-8).log())
        return (term.sum(-1)*queries[:,:,None]).sum()/(queries.sum()*2).clamp_min(1)


def make_model(kind,seed):
    assert kind in ('G','D','Z')
    base=cnn('C3',seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+871)
        return GuidedHybrid(base)
