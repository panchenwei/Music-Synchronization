"""Keep the original 64-beat supervised draws, add real left/right input context."""
import numpy as np
import torch
from .phase7_models import CurvePieceBalancedSampler
from .phase6_models import window_starts
from .recurrence_local_attention import make_model as original_factory

HALO=22


def make_model(kind,seed):
    return original_factory({'C3':'C3','HC3':'C3','A2':'A2','HA2':'A2'}[kind],seed)


class HaloSampler(CurvePieceBalancedSampler):
    def batch(self):
        samples=[];self.last_selection=[]
        for _ in range(self.batch_size):
            pid=str(self.rng.choice(self.pieces));item=self.data[pid];curves=item['curves']
            assert len(curves)>0
            perf=int(self.rng.integers(len(curves)));curve=curves[perf];n=len(item['labels'])
            start=int(self.rng.choice(window_starts(n,self.window,max(self.window//2,1))))
            length=self.window+2*HALO;left=start-HALO;a=max(0,left);z=min(n,left+length);offset=a-left
            x=np.zeros((length,curve.shape[-1]),np.float32);y=np.zeros(length,np.float32)
            loss=np.zeros(length,np.float32);valid=np.zeros(length,np.float32)
            x[offset:offset+z-a]=self.normalizer.apply(curve[a:z]).astype(np.float32)
            y[offset:offset+z-a]=item['labels'][a:z]
            valid[offset:offset+z-a]=1.
            core=min(self.window,n-start)
            loss[HALO:HALO+core]=item['label_mask'][start:start+core]
            samples.append((x,y,loss,valid));self.last_selection.append((pid,perf,start))
        return tuple(torch.as_tensor(np.stack([row[i] for row in samples])) for i in range(4))
