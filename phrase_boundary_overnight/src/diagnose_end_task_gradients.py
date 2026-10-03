"""Read-only train-example gradient alignment at fixed E checkpoints, not causal proof."""
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from .phrase_end_auxiliary import OUT,ART,read,write,sha,dataset,end_data,DualBoundary,DualSampler,split_ids,Normalizer,positive_weight


def main():
    began=time.monotonic();torch.set_num_threads(2);rows=[];hashes={}
    assert read(OUT/'completion_audit.json')['status']=='complete'
    for fold in (0,1):
        data=dataset(split_ids(fold)['train'])
        for seed in (42,43):
            run=f'E_seed{seed}_fold{fold}'
            for version in ('best','latest'):
                p=ART/'checkpoints'/run/f'{version}.pt';hashes[str(p)]=sha(p)
                s=torch.load(p,map_location='cpu',weights_only=False);model=DualBoundary(seed).eval();model.load_state_dict(s['model'])
                norm=Normalizer(s['mean'],s['std']);sampler=DualSampler(data,norm,1234)
                cs=nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(positive_weight(data,10)))
                ce=nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(positive_weight(end_data(data),10)))
                # Exclude task-specific heads; compare gradients on shared encoder only.
                params=[v for k,v in model.core.named_parameters() if not k.startswith('output.')]
                for batch in range(6):
                    x,y,m,v,ey,em=sampler.batch();a,b=model(x,padding_mask=~v.bool(),both=True)
                    ls=(cs(a,y)*m).sum()/m.sum().clamp_min(1);le=(ce(b,ey)*em).sum()/em.sum().clamp_min(1)
                    gs=torch.autograd.grad(ls,params,retain_graph=True);ge=torch.autograd.grad(le,params)
                    gs=torch.cat([g.flatten() for g in gs]);ge=torch.cat([g.flatten() for g in ge])
                    ns=float(gs.norm());ne=float(ge.norm());cos=float(torch.dot(gs,ge)/(gs.norm()*ge.norm()).clamp_min(1e-12))
                    rows.append(dict(run_id=run,checkpoint=version,step=s['step'],batch=batch,start_loss=float(ls.detach()),end_loss=float(le.detach()),start_grad_norm=ns,end_grad_norm=ne,cosine=cos,weighted_end_to_start_norm=.25*ne/max(ns,1e-12)))
                print('DIAGNOSED',run,version,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'gradient_alignment.csv',index=False)
    summary=df.groupby(['run_id','checkpoint']).agg(mean_cosine=('cosine','mean'),negative_fraction=('cosine',lambda x:float((x<0).mean())),weighted_end_to_start_norm=('weighted_end_to_start_norm','mean'));summary.to_csv(OUT/'gradient_alignment_summary.csv')
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'gradient_alignment_manifest.json',dict(status='complete',training_updates=0,checkpoint_hashes=hashes,source_sha256=sha(__file__),examples='train only; six fixed equal-piece batches per checkpoint',dropout=False,scope='shared encoder gradients at selected and final checkpoints, NOT full training trajectory or causal mediation test',seconds=time.monotonic()-began))
    print(summary.to_string(),flush=True)


if __name__=='__main__':main()
