"""Same roll weights/parameters, with optional continuous-time convolution."""
import numpy as np
import torch
from torch import nn
from .score_roll_branch import RollBoundary


class CrossbeatStem(nn.Module):
    def __init__(self, original, continuous):
        super().__init__()
        self.conv = original.conv
        self.projection = original.projection
        self.continuous = continuous

    def forward(self, roll, padding_mask=None):
        b, t = roll.shape[:2]
        mask = torch.zeros((b,t), dtype=torch.bool, device=roll.device) if padding_mask is None else padding_mask
        roll = roll.masked_fill(mask[...,None,None,None], 0)
        if not self.continuous:
            z = self.conv(roll.reshape(-1,2,128,4))
            z = torch.cat([z.mean(2), z.amax(2)], 1).flatten(1).reshape(b,t,64)
        else:
            z = roll.permute(0,2,3,1,4).reshape(b,2,128,t*4)
            tm = mask.repeat_interleave(4, dim=1)[:,None,None,:]
            for layer in self.conv:
                z = layer(z).masked_fill(tm, 0)
            z = torch.cat([z.mean(2), z.amax(2)], 1)
            z = z.reshape(b,16,t,4).permute(0,2,1,3).reshape(b,t,64)
        return self.projection(z).masked_fill(mask[...,None], 0)


class CrossbeatBoundary(RollBoundary):
    def __init__(self, continuous, seed):
        super().__init__('R', seed)
        # Reuse exact frozen-random stem, no additional initialization or RNG change.
        self.stem = CrossbeatStem(self.stem, continuous)

    def forward(self, x, roll, padding_mask=None):
        return self.forward_embedded(x, self.stem(roll, padding_mask), padding_mask)


def features_in_blocks(model, roll, block_size=128):
    """One beat halo covers the two-convolution +/-2-subtick receptive field."""
    blocks=[]; n=roll.shape[1]
    for first in range(0,n,block_size):
        last=min(n,first+block_size); left=max(0,first-1); right=min(n,last+1)
        z=model.stem(roll[:,left:right])
        blocks.append(z[:,first-left:last-left])
    return torch.cat(blocks,1)


def predictions(model, data, norm, device):
    was_training=model.training; model.eval(); result={}
    try:
        with torch.no_grad():
            for pid,item in sorted(data.items()):
                roll=torch.from_numpy(item['piano_roll'][None]).to(device)
                features=features_in_blocks(model,roll); rows=[]
                for first in range(0,len(item['curves']),8):
                    x=torch.from_numpy(norm.apply(item['curves'][first:first+8]).astype(np.float32)).to(device)
                    rows.extend(torch.sigmoid(model.forward_embedded(x,features.expand(x.shape[0],-1,-1))).cpu().numpy())
                result[pid]={str(k):v for k,v in zip(item['performance_ids'],rows)}
    finally: model.train(was_training)
    return result
