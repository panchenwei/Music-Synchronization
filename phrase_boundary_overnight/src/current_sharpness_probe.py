"""TRAIN-only local directional sensitivity, not a proof of generalization cause."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from . import run_recurrence_depth_study as source
from .score_context_study import ROOT,normalizer,sha,write,CurvePieceBalancedSampler,positive_weight
from .three_round_round2 import split_ids

OUT=ROOT/'reports/current_sharpness_probe'

def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True);rows=[];hashes={}
    for fold in (0,1):
        tr=source.dataset(split_ids(fold)['train'],'C3');norm=normalizer(tr)
        for seed in (42,43):
            for point in ('best','latest'):
                path=source.ART/'checkpoints'/f'C3_seed{seed}_fold{fold}'/f'{point}.pt';hashes[str(path)]=sha(path)
                cp=torch.load(path,map_location='cpu',weights_only=False);model=source.make_model('C3',seed).cuda().eval();model.load_state_dict(cp['model'])
                sampler=CurvePieceBalancedSampler(tr,norm,64,32,915);params=list(model.parameters())
                crit=nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(positive_weight(tr,10),device='cuda'))
                for batch in range(8):
                    x,y,mask,valid=(a.cuda() for a in sampler.batch())
                    def loss():return (crit(model(x,padding_mask=~valid.bool()),y)*mask).sum()/mask.sum().clamp_min(1)
                    l0=loss();grads=torch.autograd.grad(l0,params);gn=torch.stack([g.square().sum() for g in grads]).sum().sqrt()
                    originals=[p.detach().clone() for p in params]
                    for radius in (.01,.05):
                        losses=[]
                        for sign in (1.,-1.):
                            with torch.no_grad():
                                for p,v,g in zip(params,originals,grads):p.copy_(v+sign*radius*g/(gn+1e-12))
                                losses.append(float(loss()))
                        with torch.no_grad():
                            for p,v in zip(params,originals):p.copy_(v)
                        rows.append(dict(fold=fold,seed=seed,point=point,step=cp['step'],batch=batch,radius=radius,base_loss=float(l0.detach()),plus_loss=losses[0],minus_loss=losses[1],increase=losses[0]-float(l0.detach()),symmetric_curvature=(losses[0]+losses[1]-2*float(l0.detach()))/radius**2,grad_norm=float(gn)))
                for k,v in cp['model'].items():torch.testing.assert_close(v,model.state_dict()[k].cpu(),atol=0,rtol=0)
    df=pd.DataFrame(rows);df.to_csv(OUT/'probe.csv',index=False)
    means=df.groupby(['point','radius'])[['base_loss','plus_loss','increase','symmetric_curvature','grad_norm']].mean();means.to_csv(OUT/'means.csv')
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'audit.json',dict(status='complete',train_only=True,checkpoints=8,batches=64,radii=[.01,.05],direction='normalized current-batch gradient',dropout_disabled=True,source_hashes_unchanged=True,weights_exactly_restored=True,limitations='Directional finite differences, not maximum sharpness or Hessian eigenvalues; parameterization-dependent; no validation data used and no proof of causal generalization mechanism.',hashes=hashes))
    print(means.to_string())

if __name__=='__main__':main()
