"""Post-hoc gradient conflict diagnostic; fixed checkpoints, no optimizer steps."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from . import harmony_auxiliary_study as study
from .harmony_auxiliary import HarmonySampler,make_model
from .score_context_study import ROOT,read,write,sha,normalizer,positive_weight
from .three_round_round2 import split_ids

OUT=ROOT/'reports/harmony_gradient_probe'


def main():
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);rows=[];hashes=dict(read(study.OUT/'contract.json')['hashes'])
    assert read(study.OUT/'completion_audit.json')['status']=='complete'
    for p in (Path(__file__),OUT/'PROTOCOL.md'):hashes[str(p)]=sha(p)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    for fold in (0,1):
        ids=split_ids(fold);train=study.dataset(ids['train'],'G');val=study.dataset(ids['validation'],'G');norm=normalizer(train)
        weight=torch.tensor(positive_weight(train,10),device='cuda')
        for seed in (42,43):
            for kind in ('G','D'):
                for checkpoint in ('best','latest'):
                    cp=study.ART/'checkpoints'/f'{kind}_seed{seed}_fold{fold}'/f'{checkpoint}.pt';hashes[str(cp)]=sha(cp)
                    saved=torch.load(cp,map_location='cpu',weights_only=False);model=make_model(kind,seed).cuda().eval();model.load_state_dict(saved['model'])
                    params=list(model.parameters())
                    for split,data in [('train',train),('validation',val)]:
                        sampler=HarmonySampler(data,norm,20260914,kind if split=='train' else 'G')
                        for batch in range(4):
                            assert time.monotonic()-began<600
                            x,y,m,v,hy,hm=(z.cuda() for z in sampler.batch());a,b=model(x,padding_mask=~v.bool(),both=True)
                            main=(torch.nn.functional.binary_cross_entropy_with_logits(a,y,reduction='none',pos_weight=weight)*m).sum()/m.sum().clamp_min(1)
                            aux=(torch.nn.functional.cross_entropy(b.transpose(1,2),hy,reduction='none')*hm).sum()/hm.sum().clamp_min(1)/np.log(7)
                            gm=torch.autograd.grad(main,params,retain_graph=True,allow_unused=True);ga=torch.autograd.grad(aux,params,allow_unused=True)
                            shared=[(g,h) for g,h in zip(gm,ga) if g is not None and h is not None]
                            u=torch.cat([g.flatten() for g,h in shared]);z=.25*torch.cat([h.flatten() for g,h in shared])
                            dot=float(u@z);nu=float(u.norm());nz=float(z.norm());cos=dot/max(nu*nz,1e-20)
                            rows.append(dict(fold=fold,seed=seed,kind=kind,checkpoint=checkpoint,step=saved['step'],split=split,batch=batch,
                                main_loss=float(main.detach()),aux_loss=float(aux.detach()),cosine=cos,conflicting=cos<0,
                                weighted_aux_to_main_norm=nz/max(nu,1e-20),sum_dot_main=float((u+z)@u),shared_parameters=len(u)))
                    print('PROBED',fold,seed,kind,checkpoint,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'gradients.csv',index=False)
    means=df.groupby(['kind','checkpoint','split'])[['cosine','conflicting','weighted_aux_to_main_norm','main_loss','aux_loss']].mean();means.to_csv(OUT/'means.csv')
    selected=df[(df.kind=='G')&(df.split=='train')]
    gate=bool(selected.conflicting.mean()>.5 and selected.weighted_aux_to_main_norm.median()>=.1)
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='complete',checkpoints=16,windows=len(df),training_runs=0,test_used=False,hashes_unchanged=True,
        g_train_conflict_fraction=float(selected.conflicting.mean()),g_train_median_aux_main_norm=float(selected.weighted_aux_to_main_norm.median()),
        projected_gradient_followup_gate=gate,seconds=time.monotonic()-began))
    print(means.to_string(),flush=True);print('Follow-up gate:',gate,flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
