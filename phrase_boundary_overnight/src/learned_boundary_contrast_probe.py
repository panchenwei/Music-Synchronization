"""Frozen-checkpoint residual removal; descriptive train/validation probe only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from . import learned_boundary_contrast_study as study
from .learned_boundary_contrast import make_model
from .score_context_study import ROOT,read,write,sha,normalizer,CurvePieceBalancedSampler,positive_weight
from .three_round_round2 import split_ids

OUT=ROOT/'reports/learned_boundary_contrast_probe'


def main():
    began=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    assert read(study.OUT/'completion_audit.json')['status']=='complete'
    hashes=dict(read(study.OUT/'contract.json')['hashes'])
    for p in (Path(__file__),OUT/'PROTOCOL.md'):hashes[str(p)]=sha(p)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);rows=[]
    for fold in (0,1):
        ids=split_ids(fold)
        for kind in ('G','D','Z'):
            train=study.dataset(ids['train'],kind);val=study.dataset(ids['validation'],kind);norm=normalizer(train)
            pos=torch.tensor(positive_weight(train,10),device='cuda')
            for seed in (42,43):
                cp=study.ART/'checkpoints'/f'{kind}_seed{seed}_fold{fold}'/'best.pt';hashes[str(cp)]=sha(cp)
                saved=torch.load(cp,map_location='cpu',weights_only=False);m=make_model(kind,seed).cuda().eval();m.load_state_dict(saved['model'])
                np.testing.assert_array_equal(norm.mean,saved['mean']);np.testing.assert_array_equal(norm.std,saved['std'])
                for split,data in [('train',train),('validation',val)]:
                    sampler=CurvePieceBalancedSampler(data,norm,64,32,20260914)
                    for batch in range(4):
                        assert time.monotonic()-began<600
                        x,y,mask,valid=(z.cuda() for z in sampler.batch());use=mask>.5;truth=y[use].cpu().numpy()
                        with torch.no_grad():
                            full=m(x,padding_mask=~valid.bool());core=m.core(x,padding_mask=~valid.bool())
                        if kind=='Z':torch.testing.assert_close(full,core,rtol=0,atol=0)
                        baseline=None
                        for condition,logit in [('full',full),('no_residual',core)]:
                            prob=logit.sigmoid();bce=float((torch.nn.functional.binary_cross_entropy_with_logits(logit,y,pos_weight=pos,reduction='none')*mask).sum()/mask.sum().clamp_min(1))
                            ap=float(average_precision_score(truth,prob[use].cpu().numpy()))
                            if baseline is None:baseline=(bce,ap)
                            rows.append(dict(kind=kind,fold=fold,seed=seed,split=split,batch=batch,condition=condition,
                                bce=bce,ap=ap,bce_delta=bce-baseline[0],ap_delta=ap-baseline[1],
                                probability_change=float((prob-full.sigmoid()).abs()[use].mean()),
                                residual_abs=float((full-core).abs()[use].mean()),core_logit_abs=float(core.abs()[use].mean()),
                                residual_weight_norm=float(m.residual.weight.detach().norm())))
                print('PROBED',kind,fold,seed,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'metrics.csv',index=False)
    cols=['bce','ap','bce_delta','ap_delta','probability_change','residual_abs','core_logit_abs','residual_weight_norm']
    means=df.groupby(['kind','split','condition'])[cols].mean();means.to_csv(OUT/'means.csv')
    assert all(sha(p)==h for p,h in hashes.items());write(OUT/'source_hashes.json',hashes)
    write(OUT/'completion_audit.json',dict(status='complete',models=12,fixed_batches=96,training_runs=0,test_used=False,
        hashes_unchanged=True,zero_control_exact=True,seconds=time.monotonic()-began))
    print(means.to_string(),flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
