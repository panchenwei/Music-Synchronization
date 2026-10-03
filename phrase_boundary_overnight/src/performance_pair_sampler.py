"""Preserve primary sampling while pairing a different performance of same work/time."""
import copy
import numpy as np
import torch
from .phase7_models import CurvePieceBalancedSampler,window_starts

class PairSampler(CurvePieceBalancedSampler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.pair_rng=np.random.default_rng(71421)
    def state(self):return {**super().state(),'pair_rng':copy.deepcopy(self.pair_rng.bit_generator.state)}
    def load_state(self,s):
        super().load_state(s);self.pair_rng.bit_generator.state=copy.deepcopy(s['pair_rng'])
    def batch(self):
        replay=np.random.default_rng();replay.bit_generator.state=copy.deepcopy(self.rng.bit_generator.state)
        first=super().batch();pairs=[];self.pair_keys=[]
        for _ in range(self.batch_size):
            pid=str(replay.choice(self.pieces));item=self.data[pid];curves=item['curves'];assert len(curves)>0
            original=int(replay.integers(len(curves)))
            start=int(replay.choice(window_starts(len(item['labels']),self.window,max(self.window//2,1))))
            partner=(original+int(self.pair_rng.integers(1,len(curves))))%len(curves) if len(curves)>1 else original
            x=self.normalizer.apply(curves[partner,start:start+self.window]).astype(np.float32)
            if len(x)<self.window:x=np.pad(x,((0,self.window-len(x)),(0,0)))
            pairs.append(x);self.pair_keys.append((pid,original,partner,start))
        self.pair_batch=torch.as_tensor(np.stack(pairs));return first

def consistency_forward(model,first_logits,paired,valid,loss_mask):
    devices=[paired.device.index] if paired.is_cuda else []
    # Keep the primary trajectory's future dropout RNG identical to its control.
    with torch.random.fork_rng(devices=devices):
        other=model(paired,padding_mask=~valid.bool())
    return (((first_logits-other)**2)*loss_mask).sum()/loss_mask.sum().clamp_min(1)
