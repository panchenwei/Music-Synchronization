"""Paired TRAIN-only behavior/gradient probes after ranking study; no new selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from torch import nn
from . import window_ranking_study as study

def main():
    torch.set_num_threads(2);rows=[]
    for fold in (0,1):
        data=study.source.dataset(study.split_ids(fold)['train'],'C3');norm=study.base.normalizer(data)
        for seed in (42,43):
            for kind,folder in [('C3',study.source.ART),('R',study.ART)]:
                cp=torch.load(folder/'checkpoints'/f'{kind}_seed{seed}_fold{fold}'/'best.pt',map_location='cpu',weights_only=False)
                model=study.make_model('C3',seed).cuda().eval();model.load_state_dict(cp['model'])
                sampler=study.base.CurvePieceBalancedSampler(data,norm,64,32,9026)
                crit=nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(study.base.positive_weight(data,10),device='cuda'))
                params=list(model.parameters())
                for batch in range(12):
                    x,y,mask,valid=(a.cuda() for a in sampler.batch());z=model(x,padding_mask=~valid.bool())
                    bce=(crit(z,y)*mask).sum()/mask.sum().clamp_min(1);rank=study.rank_loss(z,y,mask)
                    ga=torch.cat([a.flatten() for a in torch.autograd.grad(bce,params,retain_graph=True)])
                    gb=torch.cat([a.flatten() for a in torch.autograd.grad(rank,params)])
                    pairs=(y.bool()&mask.bool())[:,:,None]&((~y.bool())&mask.bool())[:,None,:]
                    count=pairs.sum((1,2));good=count>0;diff=z.detach()[:,:,None]-z.detach()[:,None,:]
                    accuracy=((diff>0)*pairs).sum((1,2))/count.clamp_min(1)
                    rows.append(dict(kind=kind,fold=fold,seed=seed,batch=batch,best_step=cp['step'],bce=float(bce.detach()),rank_loss=float(rank.detach()),pair_accuracy=float(accuracy[good].mean()),bce_grad_norm=float(ga.norm()),rank_grad_norm=float(gb.norm()),gradient_cosine=float(torch.dot(ga,gb)/(ga.norm()*gb.norm()).clamp_min(1e-12))))
    df=pd.DataFrame(rows);df.to_csv(study.OUT/'train_gradient_probe.csv',index=False)
    means=df.groupby('kind')[['bce','rank_loss','pair_accuracy','bce_grad_norm','rank_grad_norm','gradient_cosine']].mean();means.to_csv(study.OUT/'train_gradient_means.csv')
    study.base.write(study.OUT/'train_probe_audit.json',dict(status='complete',train_only=True,batches=len(df),paired_sampler_seed=9026,dropout_disabled=True,selection_used=False,limits='Best checkpoints already selected on reused development set; gradient probe is diagnostic, not causal evidence or independent validation.'))
    print(means.to_string())

if __name__=='__main__':main()
