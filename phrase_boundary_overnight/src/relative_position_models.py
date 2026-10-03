"""No-absolute-position and symmetric ALiBi controls on the frozen pure T family."""
import torch
from .phase7_models import Phase7BoundaryModel
from .models import seed_everything

class RelativePositionBoundaryModel(Phase7BoundaryModel):
    def __init__(self,position_kind):
        super().__init__('A',input_dim=58,d_model=32,layers=2,heads=4,ffn_dim=64,dropout=.2)
        if position_kind not in ('N','R'):raise ValueError(position_kind)
        self.position_kind=position_kind
        # Four-head fixed slopes from ALiBi. Buffer does not change parameter count.
        self.register_buffer('slopes',torch.tensor([.25,.0625,.015625,.00390625]),persistent=False)

    def attention_bias(self,length,device,dtype):
        pos=torch.arange(length,device=device)
        distance=abs(pos[:,None]-pos[None,:]).to(dtype)
        return -self.slopes.to(dtype)[:,None,None]*distance[None]

    def forward(self,inputs,padding_mask=None,capture_attention=False,return_hidden=False):
        if inputs.ndim!=3:raise ValueError('Expected B,T,C')
        if padding_mask is None:padding_mask=torch.zeros(inputs.shape[:2],dtype=torch.bool,device=inputs.device)
        if padding_mask.shape!=inputs.shape[:2] or padding_mask.all(1).any():raise ValueError('Invalid padding')
        padding_mask=padding_mask.bool()
        h=self.input_projection(inputs).masked_fill(padding_mask[...,None],0)
        # N and R both remove the absolute sinusoidal addition; R alone adds distance bias.
        if self.position_kind=='R':
            batch,length=h.shape[:2]
            bias=self.attention_bias(length,h.device,h.dtype)[None].expand(batch,-1,-1,-1).clone()
            bias=bias.masked_fill(padding_mask[:,None,None,:],float('-inf')).reshape(batch*4,length,length)
        for block in self.blocks:
            if self.position_kind=='N':h=block(h,padding_mask,capture_attention=capture_attention)
            else:
                z=block.norm1(h)
                attended,weights=block.attention(z,z,z,attn_mask=bias,need_weights=capture_attention,average_attn_weights=False)
                block.last_attention=weights.detach() if capture_attention else None
                h=h+block.dropout1(attended)
                h=h+block.dropout2(block.ffn(block.norm2(h)))
            h=h.masked_fill(padding_mask[...,None],0)
        h=self.final_norm(h)
        logits=self.output(h).squeeze(-1).masked_fill(padding_mask,0)
        return (logits,h) if return_hidden else logits

def make_model(kind,seed):
    seed_everything(seed)
    return RelativePositionBoundaryModel(kind)
