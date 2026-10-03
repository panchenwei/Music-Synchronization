"""Frozen source-domain recurrence embeddings as optional target input features."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from .score_context_study import ROOT,read,write,sha,normalizer as target_normalizer
from .recurrence_depth_models import make_model as cnn
from . import external_recurrence_transfer as source
from . import run_recurrence_depth_study as base
from .three_round_round2 import split_ids
from .models import Normalizer
from .phase3_models import fit_train_normalizer

OUT=ROOT/'reports/external_representation_features';ART=ROOT/'artifacts/external_representation_features'
ACTIVE_SEED=None


def make_model(kind,seed):
    assert kind in ('E','S','Z')
    model=cnn('C3',seed);original=model.input_projection
    with torch.random.fork_rng(devices=[]):expanded=nn.Linear(90,32)
    with torch.no_grad():
        expanded.weight.zero_();expanded.weight[:,:58].copy_(original.weight);expanded.bias.copy_(original.bias)
    model.input_projection=expanded
    return model


def dataset(ids,kind):
    assert kind in ('E','S','Z') and ACTIVE_SEED in (42,43)
    result=base.dataset(ids,'C3')
    for pid,item in result.items():
        with np.load(ART/f'seed{ACTIVE_SEED}'/f'{pid}.npz',allow_pickle=False) as z:feat=z['E' if kind=='Z' else kind].copy()
        if kind=='Z':feat.fill(0.)
        item['source_embedding']=feat
        item['curves']=np.concatenate([item['curves'],np.broadcast_to(feat,(*item['curves'].shape[:2],32))],axis=-1)
    return result


def normalizer(data):
    n=target_normalizer(data);extra=fit_train_normalizer([v['source_embedding'] for v in data.values()])
    assert len(n.mean)==58 and len(extra.mean)==32
    return Normalizer(np.r_[n.mean,extra.mean],np.r_[n.std,extra.std])


def extract(model,norm,recurrence):
    x=np.zeros((1,len(recurrence),58),np.float32);x[0,:,34:]=recurrence
    tensor=torch.from_numpy(norm.apply(x)).cuda();mask=torch.zeros(tensor.shape[:2],dtype=torch.bool,device='cuda')
    with torch.no_grad():
        h=model.input_projection(tensor);h=model.frontend(h,mask);h=model.final_norm(h)
    return h[0].cpu().numpy().astype(np.float32)


def prepare_features():
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    assert read(source.OUT/'completion_audit.json')['status']=='complete'
    hashes=dict(read(source.OUT/'contract.json')['hashes'])
    hashes[str(Path(__file__))]=sha(Path(__file__))
    ids=sorted(set(p for f in (0,1) for split in ('train','validation') for p in split_ids(f)[split]))
    data=base.dataset(ids,'C3');rows=[]
    for seed in (42,43):
        folder=ART/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True);models={}
        for kind in ('E','S'):
            path=source.ART/'external/checkpoints'/f'{kind}_seed{seed}_fold2/best.pt';hashes[str(path)]=sha(path)
            cp=torch.load(path,map_location='cpu',weights_only=False);m=cnn('C3',seed).cuda().eval();m.load_state_dict(cp['model']);m.requires_grad_(False)
            models[kind]=(m,Normalizer(cp['mean'],cp['std']))
        for pid,item in data.items():
            assert time.monotonic()-began<900
            feat={k:extract(m,n,item['curves'][0,:,34:58]) for k,(m,n) in models.items()}
            path=folder/f'{pid}.npz'
            if path.exists():
                with np.load(path,allow_pickle=False) as z:
                    for k in feat:np.testing.assert_allclose(z[k],feat[k],atol=2e-5,rtol=2e-5)
            else:np.savez_compressed(path,**feat)
            assert all(v.shape==(len(item['labels']),32) and np.isfinite(v).all() for v in feat.values())
            hashes[str(path)]=sha(path)
            rows.append(dict(seed=seed,piece_id=pid,positions=len(item['labels']),embedding_dim=32,
                real_shuffle_mean_absolute_difference=float(abs(feat['E']-feat['S']).mean())))
        print('FEATURES',seed,len(data),flush=True)
    assert all(sha(p)==h for p,h in hashes.items())
    pd.DataFrame(rows).to_csv(OUT/'feature_audit.csv',index=False);write(OUT/'source_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='prepared',score_files=len(data),seed_count=2,source_encoders=4,
        source_encoder_parameters=5921,trainable_target_parameters=6945,labels_passed_to_encoder=False,
        normalized_under_source_statistics=True,whole_score_offline_context=True,hashes_unchanged=True,seconds=time.monotonic()-began))


if __name__=='__main__':
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with threadpool_limits(2):prepare_features()
