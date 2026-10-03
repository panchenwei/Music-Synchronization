"""Read-only one-step check of the two-backward unprojected control."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy
from pathlib import Path
import torch
import numpy as np
from threadpoolctl import threadpool_limits
from .harmony_auxiliary import make_model,HarmonySampler
from .harmony_auxiliary_study import dataset
from .three_round_round2 import split_ids
from .score_context_study import ROOT,write,sha,normalizer,positive_weight
from .auxiliary_gradient_projection import merge


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    data=dataset(split_ids(0)['train'],'G');norm=normalizer(data);a=make_model('G',42).cuda().train();b=copy.deepcopy(a)
    x,y,m,v,hy,hm=(z.cuda() for z in HarmonySampler(data,norm,42,'G').batch())
    weight=torch.tensor(positive_weight(data,10),device='cuda');rng=torch.cuda.get_rng_state_all();cpurng=torch.get_rng_state();results=[]
    for model,separate in [(a,False),(b,True)]:
        torch.cuda.set_rng_state_all(rng);torch.set_rng_state(cpurng);params=list(model.parameters());opt=torch.optim.AdamW(params,lr=.001,weight_decay=.0001)
        logits,hlogits=model(x,padding_mask=~v.bool(),both=True)
        main=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=weight,reduction='none')*m).sum()/m.sum().clamp_min(1)
        ce=(torch.nn.functional.cross_entropy(hlogits.transpose(1,2),hy,reduction='none')*hm).sum()
        aux=(.25*ce/hm.sum().clamp_min(1)/np.log(7)) if separate else (.25*(ce/hm.sum().clamp_min(1)/np.log(7)))
        if separate:
            gm=torch.autograd.grad(main,params,retain_graph=True,allow_unused=True);ga=torch.autograd.grad(aux,params,allow_unused=True);gs,_=merge(gm,ga,False)
            for p,g in zip(params,gs):p.grad=g
        else:(main+aux).backward()
        gradients=[p.grad.clone() for p in params];torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
        results.append((logits.detach(),gradients,torch.cuda.get_rng_state_all()))
    logit=float((results[0][0]-results[1][0]).abs().max());gradient=max(float((x-y).abs().max()) for x,y in zip(results[0][1],results[1][1]))
    parameter=max(float((x-y).abs().max()) for x,y in zip(a.parameters(),b.parameters()))
    assert logit==0 and gradient<1e-5 and parameter<1e-5
    assert all(torch.equal(x,y) for x,y in zip(results[0][2],results[1][2]))
    result=dict(status='passed',one_real_batch=True,new_persisted_training_runs=0,old_model_files_modified=False,logit_max_error=logit,
        gradient_max_error=gradient,one_step_parameter_max_error=parameter,dropout_rng_equal=True,
        caveat='One-step equivalence within float error, not proof of identical 300-step trajectories.',script_sha256=sha(Path(__file__)))
    write(ROOT/'reports/harmony_projection/gradient_equivalence.json',result);print(result,flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
