"""Small trainable/frozen-random piano-roll branch, with the old CNN core retained."""
import copy
import numpy as np
import torch
from torch import nn
from .score_context_study import make_model
from .phase7_models import window_starts


class RollStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(2,8,(5,3),padding=(2,1)), nn.GELU(),
                                  nn.Conv2d(8,8,(5,3),padding=(2,1)), nn.GELU())
        self.projection = nn.Sequential(nn.Linear(64,24), nn.LayerNorm(24))

    def forward(self, roll):
        shape = roll.shape[:2]
        z = self.conv(roll.reshape(-1,2,128,4))
        z = torch.cat([z.mean(dim=2),z.amax(dim=2)],dim=1).flatten(1)
        return self.projection(z).reshape(*shape,24)


class RollBoundary(nn.Module):
    def __init__(self, kind, seed):
        super().__init__()
        if kind not in ('L','R'):
            raise ValueError('L learned, R frozen random')
        self.core = make_model('P', seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1000)
            self.stem = RollStem()
        if kind == 'R':
            self.stem.requires_grad_(False)

    def forward_embedded(self, x, features, padding_mask=None):
        return self.core(torch.cat([x[...,:34],features],dim=-1),padding_mask=padding_mask)

    def forward(self, x, roll, padding_mask=None):
        return self.forward_embedded(x,self.stem(roll),padding_mask)


class RollSampler:
    """Exactly the historical piece/performance/window RNG sequence, with score roll added."""
    def __init__(self, data, norm, window, batch_size, seed):
        self.data, self.norm = data, norm
        self.window, self.batch_size = int(window), int(batch_size)
        self.pieces = sorted(data)
        self.rng = np.random.default_rng(seed)

    def state(self):
        return {'rng':copy.deepcopy(self.rng.bit_generator.state)}

    def load_state(self, state):
        self.rng.bit_generator.state=state['rng']

    def batch(self):
        rows=[]
        for _ in range(self.batch_size):
            item=self.data[str(self.rng.choice(self.pieces))]
            curve=item['curves'][int(self.rng.integers(len(item['curves'])))]
            start=int(self.rng.choice(window_starts(len(item['labels']),self.window,max(self.window//2,1))))
            stop=min(len(item['labels']),start+self.window)
            x=self.norm.apply(curve[start:stop]).astype(np.float32)
            y=item['labels'][start:stop].astype(np.float32)
            mask=item['label_mask'][start:stop].astype(np.float32)
            valid=np.ones(stop-start,np.float32)
            roll=item['piano_roll'][start:stop]
            pad=self.window-len(y)
            if pad:
                x=np.pad(x,((0,pad),(0,0)));y=np.pad(y,(0,pad));mask=np.pad(mask,(0,pad))
                valid=np.pad(valid,(0,pad));roll=np.pad(roll,((0,pad),(0,0),(0,0),(0,0)))
            rows.append((x,y,mask,valid,roll))
        return tuple(torch.as_tensor(np.stack([r[i] for r in rows])) for i in range(5))


def roll_predictions(model,data,norm,device):
    was_training=model.training
    model.eval()
    out={}
    try:
        with torch.no_grad():
            for pid,item in sorted(data.items()):
                blocks=[]
                for start in range(0,len(item['labels']),128):
                    roll=torch.from_numpy(item['piano_roll'][None,start:start+128]).to(device)
                    blocks.append(model.stem(roll))
                features=torch.cat(blocks,dim=1)
                rows=[]
                for start in range(0,len(item['curves']),8):
                    x=torch.from_numpy(norm.apply(item['curves'][start:start+8]).astype(np.float32)).to(device)
                    rows.extend(torch.sigmoid(model.forward_embedded(x,features.expand(x.shape[0],-1,-1))).cpu().numpy())
                out[pid]={str(k):v for k,v in zip(item['performance_ids'],rows)}
    finally:
        model.train(was_training)
    return out
