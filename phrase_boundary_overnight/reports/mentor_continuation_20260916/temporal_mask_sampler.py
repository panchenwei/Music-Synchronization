"""Versioned input augmentation; independent checkpointed randomness."""
import copy
import numpy as np
import torch
from src.mentor_sequence_models import ContextSampler


class MaskSampler(ContextSampler):
    def __init__(self,data,norm,seed,full,batch_size=8,arm='B'):
        assert not full and arm in ('B','D')
        super().__init__(data,norm,seed,full,batch_size)
        self.arm=arm
        self.augmentation_rng=np.random.default_rng(seed+50001)

    def state(self):
        return {**super().state(),'augmentation_rng':copy.deepcopy(self.augmentation_rng.bit_generator.state),'augmentation_arm':self.arm}

    def load_state(self,state):
        assert state['augmentation_arm']==self.arm
        super().load_state(state)
        self.augmentation_rng.bit_generator.state=copy.deepcopy(state['augmentation_rng'])

    def batch(self):
        x,y,mask,valid=super().batch()
        x=x.clone();self.last_mask=torch.zeros_like(valid,dtype=torch.bool)
        for i in range(len(x)):
            n=int(valid[i].sum());k=min(4,max(1,n//4));u=self.augmentation_rng.random(n)
            if self.arm=='B':
                start=min(int(u[0]*(n-k+1)),n-k);indices=np.arange(start,start+k)
            else:indices=np.argsort(u)[:k]
            self.last_mask[i,indices]=True
        x[self.last_mask]=0
        return x,y,mask,valid


def base_state(state):
    return {k:v for k,v in state.items() if k not in ('augmentation_rng','augmentation_arm')}
