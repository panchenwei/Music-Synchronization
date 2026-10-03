"""Frozen-model diagnostic, no fitting: attention behavior and head sensitivity."""
import argparse,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from . import start_guided_attention_study as study
from .start_guided_attention import attention_target,make_model
from .score_context_study import ROOT,read,write,sha,normalizer,CurvePieceBalancedSampler,positive_weight
from .three_round_round2 import split_ids


def zero_heads(first):
    def hook(module,args):
        x=args[0].clone();x[...,0:16 if first else 0]=x[...,0:16 if first else 0]
        if first:x[...,:16]=0
        else:x[...,16:]=0
        return (x,)
    return hook


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--early',action='store_true');args=ap.parse_args()
    out=ROOT/'reports/start_guidance_probe'/('early_fold0_seed42' if args.early else 'complete')
    out.mkdir(parents=True,exist_ok=True);began=time.monotonic();rows=[];hashes={}
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if not args.early:assert read(study.OUT/'completion_audit.json')['status']=='complete'
    contract=read(study.OUT/'contract.json');assert all(sha(p)==h for p,h in contract['hashes'].items())
    hashes.update(contract['hashes']);hashes[str(Path(__file__))]=sha(Path(__file__))
    for f in ((0,) if args.early else (0,1)):
        ids=split_ids(f);train=study.dataset(ids['train'],'G');val=study.dataset(ids['validation'],'G');norm=normalizer(train)
        pw=positive_weight(train,10)
        for s in ((42,) if args.early else (42,43)):
            for k in ('G','D','Z'):
                name=f'{k}_seed{s}_fold{f}';meta=study.ART/'metrics'/f'{name}.json'
                assert meta.exists();cp=study.ART/'checkpoints'/name/'best.pt';hashes[str(cp)]=sha(cp);hashes[str(meta)]=sha(meta)
                state=torch.load(cp,map_location='cpu',weights_only=False);model=make_model(k,s).cuda().eval();model.load_state_dict(state['model'])
                for split,data in [('train',train),('validation',val)]:
                    sampler=CurvePieceBalancedSampler(data,norm,64,32,20260913)
                    for batch in range(4):
                        assert time.monotonic()-began<1200
                        x,y,known,valid=(v.cuda() for v in sampler.batch());mask=~valid.bool()
                        with torch.no_grad():
                            logits=model(x,mask);weights=model.attention[0].last_weights.detach().clone()
                            g,_=attention_target(y,known,8,'G');d,_=attention_target(y,known,8,'D')
                            cross=(d>0)&(g==0);queries=cross.any(-1)&known.bool()
                            klg=float(model.guidance(y,known,'G'));kld=float(model.guidance(y,known,'D'))
                            masses={};entropy={}
                            for label,heads in [('guided',slice(0,2)),('free',slice(2,4))]:
                                w=weights[:,:,heads,:]
                                masses[label]=float(((w*cross[:,:,None]).sum(-1)*queries[:,:,None]).sum()/(queries.sum()*2).clamp_min(1))
                                entropy[label]=float((-(w*w.clamp_min(1e-8).log()).sum(-1)*known[:,:,None]).sum()/(known.sum()*2).clamp_min(1))
                            sensitivity={}
                            for first,label in [(True,'guided'),(False,'free')]:
                                handle=model.attention[0].proj.register_forward_pre_hook(zero_heads(first))
                                try:ablated=model(x,mask)
                                finally:handle.remove()
                                sensitivity[label]=float(((ablated-logits).abs()*known).sum()/known.sum().clamp_min(1))
                        row=dict(run=name,kind=k,fold=f,seed=s,split=split,batch=batch,kl_start_target=klg,kl_distance_target=kld,
                            cross_mass_guided=masses['guided'],cross_mass_free=masses['free'],entropy_guided=entropy['guided'],entropy_free=entropy['free'],
                            zero_guided_mean_abs_logit_delta=sensitivity['guided'],zero_free_mean_abs_logit_delta=sensitivity['free'],
                            queries_near_start=int(queries.sum()),known_queries=int(known.sum()))
                        if batch==0:
                            z=model(x,mask);bce=torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none',pos_weight=torch.tensor(pw,device='cuda'))
                            bce=(bce*known).sum()/known.sum().clamp_min(1);aux=model.guidance(y,known,k)
                            params=list(model.parameters())
                            gb=torch.autograd.grad(bce,params,retain_graph=True,allow_unused=True)
                            ga=torch.autograd.grad(aux,params,allow_unused=True)
                            bn=sum(float(a.square().sum()) for a in gb if a is not None)**.5
                            an=sum(float(a.square().sum()) for a in ga if a is not None)**.5
                            row.update(bce_grad_norm=bn,aux_grad_norm=an,weighted_aux_to_bce_grad_ratio=(0 if k=='Z' else .1)*an/max(bn,1e-12))
                        rows.append(row)
                pd.DataFrame(rows).to_csv(out/'windows.csv',index=False);print('PROBED',name,flush=True)
    frame=pd.DataFrame(rows);means=frame.groupby(['kind','split']).mean(numeric_only=True)
    means.to_csv(out/'means.csv');assert all(sha(p)==h for p,h in hashes.items())
    write(out/'source_hashes.json',hashes);write(out/'completion_audit.json',dict(status='complete',partial=args.early,training_runs=0,
        models_probed=3 if args.early else 12,batches_per_model_per_split=4,window=64,weights_unchanged=True,source_hashes_unchanged=True,
        seconds=time.monotonic()-began,caution='Diagnostic on fixed sampled windows, not a new F1 or independent test; head ablation is not proof of a named musical concept.'))
    print(means[['kl_start_target','cross_mass_guided','zero_guided_mean_abs_logit_delta','weighted_aux_to_bce_grad_ratio']].to_string(),flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
