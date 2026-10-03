"""Training-only phrase-start supervision for the centered pitch/time stem."""
import torch
from torch import nn
from .axis_content_centering import ContentBoundary


class AuxiliaryBoundary(ContentBoundary):
    def __init__(self,kind,seed):
        assert kind=='A'
        super().__init__('C',seed)
        # Preserve the old model's initialization and future dropout RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+4400)
            self.auxiliary=nn.Sequential(nn.Conv1d(8,8,5,padding=2),nn.GELU(),nn.Conv1d(8,1,1))

    def forward(self,x,roll,padding_mask=None):
        z=self.stem(roll,padding_mask)
        main=self.forward_embedded(x,z,padding_mask)
        if not self.training:return main
        aux=self.auxiliary(z.transpose(1,2)).squeeze(1)
        if padding_mask is not None:aux=aux.masked_fill(padding_mask,0)
        return main,aux


class AuxiliaryBCE(nn.BCEWithLogitsLoss):
    weight_auxiliary=.25

    def forward(self,logits,target):
        if not isinstance(logits,tuple):return super().forward(logits,target)
        main,aux=logits
        # Return an elementwise loss; the unchanged engine applies label masks.
        main_loss=super().forward(main,target)
        if self.weight_auxiliary==0:return main_loss
        return main_loss+self.weight_auxiliary*super().forward(aux,target)
